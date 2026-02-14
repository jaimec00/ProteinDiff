# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		training_run.py
description:	runs training
'''
# ----------------------------------------------------------------------------------------------------------------------

import gc
import os
import time
import hydra
import mlflow
from tqdm import tqdm
from pathlib import Path
import importlib
from dataclasses import dataclass, field
from omegaconf import OmegaConf as om, DictConfig, MISSING

import torch

from proteus.data.data_loader import DataHolder, DataHolderCfg
from proteus.data.construct_registry import ConstructRegistry
from proteus.data.data_utils import DataBatch
from proteus.training.logger import Logger, LoggerCfg
from proteus.losses.training_loss import TrainingRunLosses, LossFnCfg
from proteus.training.optim import OptimCfg, setup_optim
from proteus.training.scheduler import SchedulerCfg, setup_scheduler
from proteus.utils.profiling import ProfilerCfg, Profiler
from proteus.types import Any, Iterator, Tuple

# ----------------------------------------------------------------------------------------------------------------------

# detect anomolies in training and dont tf32 matmuls 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class TrainingParamsCfg:
	max_steps: int = 100_000        # Stop after this many steps
	val_interval: int = 1_000       # Run validation every N steps
	accumulation_steps: int = 1
	grad_clip_norm: float = 0.0
	compile_model: bool = False
	load_from_checkpoint: str = ""
	checkpoint_interval: int = 1_000
	gc_interval: int = 500

@dataclass
class TrainingRunCfg:
	model: Any = MISSING
	construct_function: str = MISSING
	data: DataHolderCfg = MISSING
	logger: LoggerCfg = MISSING
	losses: LossFnCfg = MISSING
	optim: OptimCfg = MISSING
	scheduler: SchedulerCfg = MISSING
	profiler: ProfilerCfg = MISSING
	training_params: TrainingParamsCfg = MISSING

class TrainingRun:

	def __init__(self, cfg: TrainingRunCfg) -> None:

		om.resolve(cfg)

		self.gpu = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.cpu = torch.device("cpu")

		# tells the data holder what data to create
		ConstructRegistry.set_construct_function(cfg.construct_function)

		self.data = DataHolder(cfg.data)
		self.losses = TrainingRunLosses(cfg.losses)
		self.logger = Logger(cfg.logger)
		self.profiler = Profiler(cfg.profiler, self.logger.out_path)

		self.max_steps = cfg.training_params.max_steps
		self.val_interval = cfg.training_params.val_interval
		self.accumulation_steps = cfg.training_params.accumulation_steps
		self.grad_clip_norm = cfg.training_params.grad_clip_norm
		self.batch_counter = 0
		self.gc_interval = cfg.training_params.gc_interval
		
		self.model, self.optim, self.scheduler, cfg = self.maybe_load_checkpoint(cfg)
		self.checkpoint_interval = cfg.training_params.checkpoint_interval

		if cfg.training_params.compile_model:
			self.log("compiling model...")
			self.model = torch.compile(self.model, dynamic=True)

		with mlflow.start_run():
			self.last_ts = time.perf_counter()
			self.log_params(cfg)
			self.train()
			self.test()
			self.log("fin", fancy=True)

	def maybe_load_checkpoint(self, cfg: TrainingRunCfg) -> Tuple[Any, Any, Any, Any]:

		def load_model_cls(model_cfg: Any):
			if hasattr(model_cfg, "model_cls"):
				model_cls = model_cfg.model_cls
			else:
				raise RuntimeError

			module_str, model_cls = model_cls.rsplit(".", 1)
			module = importlib.import_module(module_str)
			return getattr(module, model_cls)

		if cfg.training_params.load_from_checkpoint:

			weights_path = Path(cfg.training_params.load_from_checkpoint)

			checkpoint_path = weights_path.parent
			
			# TODO warn user if the configs mismatch
			cfg.model = om.load(checkpoint_path / "model_cfg.yaml")
			cfg.optim = om.load(checkpoint_path / "optim_cfg.yaml")
			cfg.scheduler = om.load(checkpoint_path / "scheduler_cfg.yaml")
		
			weights = torch.load(str(weights_path), map_location=self.cpu, weights_only=True)

			model_cls = load_model_cls(cfg.model)

			model = model_cls(cfg.model)
			model.load_state_dict(weights["model"], strict=False)
			optim = setup_optim(cfg.optim, model)
			optim.load_state_dict(weights["optim"])
			scheduler = setup_scheduler(cfg.scheduler, optim)
			scheduler.load_state_dict(weights["scheduler"])

			# TODO: handle step, epoch and other training state stuff (along with dataloader being in sync with this)

		else:
			model_cls = load_model_cls(cfg.model)
			model = model_cls(cfg.model)
			optim = setup_optim(cfg.optim, model)
			scheduler = setup_scheduler(cfg.scheduler, optim)

		# move to gpu
		model = model.to(self.gpu)
		for state in optim.state.values():
			for k, v in state.items():
				if torch.is_tensor(v):
					state[k] = v.to(self.gpu)

		# save the fully built model 
		om.save(cfg.model, self.logger.out_path / "model_cfg.yaml")
		om.save(cfg.optim, self.logger.out_path / "optim_cfg.yaml")
		om.save(cfg.scheduler, self.logger.out_path / "scheduler_cfg.yaml")
		
		return model, optim, scheduler, cfg

	def maybe_save_checkpoint(self) -> None:
		if (
			not self.learn_step
			or self.last_step % self.checkpoint_interval != 0 
			or self.last_step==0
		):
			return
		
		weights = {
			"model": self.model.state_dict(),
			"optim": self.optim.state_dict(),
			"scheduler": self.scheduler.state_dict(),
		}

		checkpoint_path = str(self.logger.out_path / f"checkpoint_step-{self.last_step:,}.pt")
		self.log(f"saved checkpoint to {checkpoint_path}")
		torch.save(weights, checkpoint_path)

	def set_training(self) -> None:
		self.model.train()
		self.losses.tmp.clear()

	def run_val(self) -> bool:
		return (
			self.last_step % self.val_interval == 0 
			and self.last_step > 0
			and self.learn_step
		)

	def get_batch(self, train_iter: Iterator) -> DataBatch:
		try:
			return next(train_iter), train_iter
		except StopIteration:
			train_iter = iter(self.data.train)
			return next(train_iter), train_iter

	def garbage_collect(self) -> None:
		if (
			self.gc_interval 
			and self.last_step >= self.gc_interval 
			and self.last_step % self.gc_interval == 0
		):
			torch.cuda.empty_cache()
			gc.collect()

	def train(self) -> None:

		self.log(f"initializing training for {self.max_steps} steps...")

		self.set_training()
		train_iter = iter(self.data.train)

		with self.profiler as profiler:
			with self.create_pbar(self.max_steps, "training progress") as pbar:
				while self.last_step < self.max_steps:

					# next batch
					data_batch, train_iter = self.get_batch(train_iter)

					# learn 
					self.batch_learn(data_batch)

					# update progress bar
					if self.learn_step:
						
						# update progress
						self.update_pbar(pbar, data_batch)

						# log step metrics
						self.log_step(self.losses.tmp, data_batch)

					# profiler step
					profiler.step()

					# save the checkpoint
					self.maybe_save_checkpoint()

					# garbage collection
					self.garbage_collect()

					# validation at intervals
					if self.run_val():

						# log
						self.log(f"step: {self.last_step}\nlr: {self.last_lr}", fancy=True)

						# losses
						losses = self.losses.tmp.get_avg()
						self.logger.log_losses(losses)
						self.losses.train.add_losses(losses)

						# validation and continue
						self.validation()
						self.set_training()

		self.log(f"training finished after {self.last_step} steps", fancy=True)

	@torch.no_grad()
	def validation(self) -> None:

		# switch to evaluation mode to perform validation
		self.model.eval()

		# clear losses for this run
		self.losses.tmp.clear()

		# pbar
		with self.create_pbar(len(self.data.val), "validation progress") as pbar:

			# loop through validation batches
			for data_batch in self.data.val:

				# run the model
				self.batch_forward(data_batch)

				# update progress bar
				self.update_pbar(pbar, data_batch, step=data_batch.samples)

		# log the losses
		val_losses = self.losses.tmp.get_avg()
		self.log_val(val_losses)
		self.losses.val.add_losses(val_losses)

	@torch.no_grad()
	def test(self) -> None:

		# switch to evaluation mode
		self.model.eval()

		# log
		self.log("starting testing", fancy=True)

		# clear losses for this run
		self.losses.tmp.clear()

		with self.create_pbar(len(self.data.test), "test progress") as pbar:

			# loop through testing batches
			for data_batch in self.data.test:

				# run the model
				self.batch_forward(data_batch)

				# update pbar
				self.update_pbar(pbar, data_batch, step=data_batch.samples)

		# log the losses
		test_losses = self.losses.tmp.get_avg()
		self.logger.log_losses(test_losses, mode="test")
		self.losses.test.extend_losses(self.losses.tmp)

	def batch_learn(self, data_batch: DataBatch) -> None:
		self.batch_forward(data_batch)
		self.batch_backward(data_batch)
		self.batch_counter += 1

	def batch_forward(self, data_batch: DataBatch) -> None:
		
		# move batch to gpu
		data_batch.move_to(self.gpu)
		outputs = self.model(data_batch)
		losses = self.losses.loss_fn(outputs)
		self.losses.tmp.add_losses(losses, valid_toks=data_batch.loss_tokens)

	def batch_backward(self, data_batch: DataBatch) -> None:

		loss = self.losses.tmp.get_last_loss()
		loss.backward()

		if self.learn_step:

			# grad clip
			if self.grad_clip_norm:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

			# step
			self.optim.step()
			self.scheduler.step()
			self.optim.zero_grad()

	def log_step(self, losses: TrainingRunLosses, data_batch: DataBatch) -> None:
		'''
		lots of approximations here
		- sample from the last accum step only (first rank only when do distributed)
		- samples / toks is counted from current step and mult by accum steps (and dp world_size when do dist)
		not meant to be extensive, just a decent enough approximation for monitoring
		'''
		
		# get delta time or update last time
		cur_ts = time.perf_counter()
		if self.last_step % self.logger.log_interval != 0:
			self.last_ts = cur_ts
			return 
		delta_ts = cur_ts - self.last_ts

		# create the metrics
		tokens = data_batch.tokens
		loss_tokens = data_batch.loss_tokens.item()
		samples = data_batch.samples
		losses_dict = losses.get_last_losses(scale=1/loss_tokens)
		data_dict = {
			"toks_per_batch": tokens*self.accumulation_steps, 
			"loss_toks_per_batch": loss_tokens*self.accumulation_steps, 
			"loss_toks_per_sample": loss_tokens / samples,
			"loss_toks_to_total_toks_ratio": loss_tokens / tokens,
			"samples_per_batch": samples*self.accumulation_steps,
			"toks_per_sample": tokens / samples,
		}
		throughput_dict = {
			"toks_per_sec": tokens*self.accumulation_steps / delta_ts, 
			"updates_per_sec": 1 / delta_ts,
			"fwd_bwd_per_Sec": self.accumulation_steps / delta_ts,
			"loss_toks_per_seq": loss_tokens / delta_ts
		}

		# format
		losses_dict = {f"train/loss/{k}": v for k, v in losses_dict.items()}
		data_dict = {f"train/data/{k}": v for k, v in data_dict.items()}
		throughput_dict = {f"train/throughput/{k}": v for k, v in throughput_dict.items()}
		scheduler_dict = {f"train/lr": self.last_lr}
		
		# combine
		step_dict = losses_dict | data_dict | scheduler_dict | throughput_dict

		self.logger.log_step(step_dict, self.last_step)

	def log_val(self, losses: TrainingRunLosses) -> None:
		self.logger.log_losses(losses, mode="val")
		self.logger.log_step({f"val/loss/{k}": v for k, v in losses.items()}, self.last_step)

	def log_params(self, cfg: TrainingRunCfg) -> None:
		self.logger.log_param("configuration", om.to_yaml(cfg))
		num_params = sum(p.numel() for p in self.model.parameters())
		self.logger.log_param("parameters", num_params)
		self.logger.log_param("run_dir", self.logger.out_path)


	def log(self, message: str, fancy=False) -> None:
		if fancy:
			message = f"\n\n{'-'*80}\n{message}\n{'-'*80}\n"
		self.logger.log.info(message)

	def create_pbar(self, total: int, desc: str) -> tqdm:
		"""Create a progress bar for training/validation/test loops."""
		return tqdm(total=total, desc=desc, unit="steps")

	def update_pbar(self, pbar: tqdm, data_batch, step=1) -> None:
		"""Update progress bar with loss and advance by 1 step."""
		if self.last_step % 10 == 0:
			pbar.set_postfix(loss=self.losses.tmp.get_last_loss() / data_batch.loss_tokens)
		pbar.update(step)

	@property
	def last_lr(self) -> int:
		return self.scheduler.get_last_lr()[0]

	@property
	def last_step(self) -> int:
		'''just for clearer naming'''
		return int(self.scheduler.last_epoch)

	@property
	def learn_step(self) -> bool:
		return (self.batch_counter + 1) % self.accumulation_steps == 0
