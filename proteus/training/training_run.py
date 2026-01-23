# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		training_run.py
description:	runs training
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch

import os
import hydra
import mlflow
from tqdm import tqdm
from hydra.utils import instantiate
from dataclasses import dataclass, field
from omegaconf import OmegaConf as om, DictConfig
import time
from pathlib import Path

from proteus.model import Proteus
from proteus.model.Proteus import ProteusCfg
from proteus.training.logger import Logger, LoggerCfg
from proteus.training.data.data_loader import DataHolder, DataHolderCfg
from proteus.training.losses.training_loss import TrainingRunLosses, TrainingRunLossesCfg
from proteus.training.optim import OptimCfg, setup_optim
from proteus.training.scheduler import SchedulerCfg, setup_scheduler
from proteus.static.constants import TrainingStage
from proteus.utils.profiling import ProfilerCfg, Profiler

# ----------------------------------------------------------------------------------------------------------------------

# detect anomolies in training and dont tf32 matmuls 
# torch.autograd.set_detect_anomaly(True, check_nan=True) # throws error when nan encountered
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class TrainingRunCfg:
	model: ProteusCfg
	data: DataHolderCfg
	logger: LoggerCfg
	losses: TrainingRunLossesCfg
	optim: OptimCfg
	scheduler: SchedulerCfg
	profiler: ProfilerCfg

	train_stage: TrainingStage = TrainingStage.VAE
	max_steps: int = 100_000        # Stop after this many steps
	val_interval: int = 1_000       # Run validation every N steps
	accumulation_steps: int = 1
	grad_clip_norm: float = 5.0
	compile_model: bool = True
	load_from_checkpoint: str = ""
	checkpoint_interval: int = 1_000

class TrainingRun:

	def __init__(self, cfg: TrainingRunCfg) -> None:

		self.gpu = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.cpu = torch.device("cpu")

		self.data = DataHolder(cfg.data)
		self.losses = TrainingRunLosses(cfg.losses)
		self.logger = Logger(cfg.logger)
		self.profiler = Profiler(cfg.profiler, self.logger.out_path)

		self.train_stage = cfg.train_stage
		self.max_steps = cfg.max_steps
		self.val_interval = cfg.val_interval
		self.accumulation_steps = cfg.accumulation_steps
		self.grad_clip_norm = cfg.grad_clip_norm
		self.batch_counter = 0
		
		self.model, self.optim, self.scheduler, cfg = self.maybe_load_checkpoint(cfg)
		self.checkpoint_interval = cfg.checkpoint_interval

		self.log(f"configuration:\n\n{om.to_yaml(cfg)}", fancy=True)
		num_params = sum(p.numel() for p in self.model.parameters())
		self.log(f"model initialized with {num_params:,} parameters", fancy=True)

		if cfg.compile_model:
			self.log("compiling model...")
			self.model = torch.compile(self.model, dynamic=True)

		with mlflow.start_run():
			self.last_ts = time.perf_counter()
			self.train()
			self.test()
			self.log("fin", fancy=True)

	def maybe_load_checkpoint(self, cfg: TrainingRunCfg):

		if cfg.load_from_checkpoint:

			weights_path = Path(cfg.load_from_checkpoint)

			checkpoint_path = weights_path.parent
			
			# TODO warn user if the configs mismatch
			cfg.model = om.load(checkpoint_path / "model_cfg.yaml")
			cfg.optim = om.load(checkpoint_path / "optim_cfg.yaml")
			cfg.scheduler = om.load(checkpoint_path / "scheduler_cfg.yaml")
		
			weights = torch.load(str(weights_path), map_location=self.cpu, weights_only=True)

			# TODO: add _target_ to model cfg (was built from model factory, so it is missing)

			model = Proteus(cfg.model)
			model.load_state_dict(weights["model"], strict=False)
			optim = setup_optim(cfg.optim, model)
			optim.load_state_dict(weights["optim"])
			scheduler = setup_scheduler(cfg.scheduler, optim)
			scheduler.load_state_dict(weights["scheduler"])

			# TODO: handle step, epoch and other training state stuff (along with dataloader being in sync with this)

		else:
			model = Proteus(cfg.model)
			optim = setup_optim(cfg.optim, model)
			scheduler = setup_scheduler(cfg.scheduler, optim)

		# move to gpu
		model = model.to(self.gpu)
		for state in optim.state.values():
			for k, v in state.items():
				if torch.is_tensor(v):
					state[k] = v.to(self.gpu)

		# TODO: add _target_ to the model

		# save the fully built model 
		om.save(cfg.model, self.logger.out_path / "model_cfg.yaml")
		om.save(cfg.optim, self.logger.out_path / "optim_cfg.yaml")
		om.save(cfg.scheduler, self.logger.out_path / "scheduler_cfg.yaml")
		
		return model, optim, scheduler, cfg

	def maybe_save_checkpoint(self):
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

	def set_training(self):
		self.model.train()
		self.losses.clear_tmp_losses()

	def run_val(self):
		return (
			self.last_step % self.val_interval == 0 
			and self.last_step > 0
			and self.learn_step
		)

	def get_batch(self, train_iter):
		try:
			return next(train_iter), train_iter
		except StopIteration:
			train_iter = iter(self.data.train)
			return next(train_iter), train_iter

	def train(self):

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

					# validation at intervals
					if self.run_val():
						self.log(f"step: {self.last_step}\nlr: {self.last_lr}", fancy=True)
						losses = self.losses.tmp.get_avg()
						self.logger.log_losses(losses)
						self.losses.train.add_losses(losses)
						self.validation()
						self.set_training()

		self.log(f"training finished after {self.last_step} steps", fancy=True)

	@torch.no_grad()
	def validation(self):

		# switch to evaluation mode to perform validation
		self.model.eval()

		# clear losses for this run
		self.losses.clear_tmp_losses()

		# progress bar
		val_pbar = self.create_pbar(len(self.data.val), "validation progress")

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
	def test(self):

		# switch to evaluation mode
		self.model.eval()

		# log
		self.log("starting testing", fancy=True)

		# clear losses for this run
		self.losses.clear_tmp_losses()

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

	def batch_learn(self, data_batch):
		self.batch_forward(data_batch)
		self.batch_backward(data_batch)

	def batch_forward(self, data_batch):
		
		# move batch to gpu
		data_batch.move_to(self.gpu)

		# for vae training
		if self.train_stage==TrainingStage.VAE:

			(
				latent,
				latent_mu, latent_logvar,
				divergence_pred, divergence_true,
				seq_pred,
				struct_logits, struct_head,
			) = self.model(
				coords=data_batch.coords,
				labels=data_batch.labels,
				atom_mask=data_batch.atom_mask,
				seq_idx=data_batch.seq_idx,
				chain_idx=data_batch.chain_idx,
				sample_idx=data_batch.sample_idx,
				cu_seqlens=data_batch.cu_seqlens,
				max_seqlen=data_batch.max_seqlen,
				stage=TrainingStage.VAE,
			)

			# compute loss
			losses = self.losses.loss_fn.vae_loss(
				latent_mu=latent_mu,
				latent_logvar=latent_logvar,
				divergence_pred=divergence_pred,
				divergence_true=divergence_true,
				seq_pred=seq_pred,
				seq_true=data_batch.labels,
				struct_logits=struct_logits,
				struct_head=struct_head,
				coords=data_batch.coords,
				cu_seqlens=data_batch.cu_seqlens,
				atom_mask=data_batch.atom_mask,
			)

		# diffusion training
		elif self.train_stage==TrainingStage.DIFFUSION:
			# TODO: implement
			pass

		# add the losses to the temporary losses
		self.losses.tmp.add_losses(losses, valid_toks=len(data_batch))

	def batch_backward(self, data_batch):

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
		self.batch_counter += 1

	def log_step(self, losses, data_batch):
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
		losses_dict = losses.get_last_losses(scale=1/len(data_batch))
		data_dict = {"toks_per_batch": len(data_batch)*self.accumulation_steps, "samples_per_batch": data_batch.samples*self.accumulation_steps}
		throughput_dict = {"toks_per_sec": len(data_batch)*self.accumulation_steps / delta_ts, "updates_per_sec": 1 / delta_ts}

		# format
		losses_dict = {f"train/loss/{k}": v for k, v in losses_dict.items()}
		data_dict = {f"train/data/{k}": v for k, v in data_dict.items()}
		throughput_dict = {f"train/throughput/{k}": v for k, v in throughput_dict.items()}
		scheduler_dict = {f"train/lr": self.last_lr}
		
		# combine
		step_dict = losses_dict | data_dict | scheduler_dict | throughput_dict

		self.logger.log_step(step_dict, self.last_step)

	def log_val(self, losses):
		self.logger.log_losses(losses, mode="val")
		self.logger.log_step({f"val/loss/{k}": v for k, v in losses.items()}, self.last_step)

	def log(self, message, fancy=False):
		if fancy:
			message = f"\n\n{'-'*80}\n{message}\n{'-'*80}\n"
		self.logger.log.info(message)

	def create_pbar(self, total: int, desc: str) -> tqdm:
		"""Create a progress bar for training/validation/test loops."""
		return tqdm(total=total, desc=desc, unit="steps")

	def update_pbar(self, pbar: tqdm, data_batch, step=1):
		"""Update progress bar with loss and advance by 1 step."""
		if self.last_step % 10 == 0:
			pbar.set_postfix(loss=self.losses.tmp.get_last_loss().item() / len(data_batch))
		pbar.update(step)

	@property
	def last_lr(self):
		return self.scheduler.get_last_lr()[0]

	@property
	def last_step(self):
		'''just for clearer naming'''
		return int(self.scheduler.last_epoch)

	@property
	def learn_step(self):
		return (self.batch_counter + 1) % self.accumulation_steps == 0



@hydra.main(version_base=None, config_path="../../configs", config_name="debug")
def main(cfg: DictConfig):
	# build model from simple cfg
	import importlib
	simple_model_cfg = instantiate(cfg.model)
	try:
		model_factory_path = simple_model_cfg.model_factory.split(".")                                                                        
		module_path = ".".join(model_factory_path[:-1])                                                                                       
		function_name = model_factory_path[-1]                                                                                                
		module = importlib.import_module(module_path)                                                                                         
		model_factory = getattr(module, function_name)
	except (ImportError, AttributeError) as e:
          raise ImportError(                                                                                                                    
              f"failed to import model_factory at {simple_model_cfg.model_factory}. "                                                           
              "make sure the path is correct"                                                                                                   
          ) from e     
		  
	model_cfg = model_factory(simple_model_cfg)
	cfg.model = om.structured(model_cfg)
	
	TrainingRun(cfg)

if __name__ == "__main__":
	main()