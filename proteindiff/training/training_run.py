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

from proteindiff.model import ProteinDiff
from proteindiff.model.ProteinDiff import ProteinDiffCfg
from proteindiff.training.logger import Logger, LoggerCfg
from proteindiff.training.data.data_loader import DataHolder, DataHolderCfg
from proteindiff.training.losses.training_loss import TrainingRunLosses, TrainingRunLossesCfg
from proteindiff.training.optim import OptimCfg, setup_optim
from proteindiff.training.scheduler import SchedulerCfg, setup_scheduler
from proteindiff.static.constants import TrainingStage

# ----------------------------------------------------------------------------------------------------------------------

# detect anomolies in training and allow tf32 matmuls (TODO: reconsider tf32?)
torch.autograd.set_detect_anomaly(True, check_nan=True) # throws error when nan encountered
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

@dataclass
class TrainingRunCfg:
	model: ProteinDiffCfg
	data: DataHolderCfg
	logger: LoggerCfg
	losses: TrainingRunLossesCfg
	optim: OptimCfg
	scheduler: SchedulerCfg

	train_stage: TrainingStage
	epochs: int = 10 # TODO: change to steps
	accumulation_steps: int = 1
	grad_clip_norm: float = 5.0
	compile_model: bool = True
	#TODO: add other stuff relevant to train (keep simple for now)

class TrainingRun:

	def __init__(self, cfg: TrainingRunCfg) -> None:

		self.gpu = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.cpu = torch.device("cpu")

		self.data = DataHolder(cfg.data)
		self.losses = TrainingRunLosses(cfg.losses)
		self.logger = Logger(cfg.logger)
		self.model = ProteinDiff(cfg.model).to(self.gpu)
		self.optim = setup_optim(cfg.optim, self.model)
		self.scheduler = setup_scheduler(cfg.scheduler, self.optim)

		self.train_stage = cfg.train_stage
		self.epochs = cfg.epochs
		self.accumulation_steps = cfg.accumulation_steps
		self.grad_clip_norm = cfg.grad_clip_norm
		self.cfg = cfg

		self.log(f"configuration:\n\n{om.to_yaml(self.cfg)}", fancy=True)
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


	def train(self):

		self.log(
			f"\n\ninitializing training. "
			f"training on approximately {len(self.data.train)} clusters "
			f"for {self.epochs} epochs.\n" 
		)
		
		# loop through epochs
		for epoch_idx in range(self.epochs):

			# TODO: mae a helper to only set non-frozen models to train
			# make sure in training mode
			self.model.train()

			# log some info
			self.logger.log_epoch(epoch_idx, self.last_step, self.last_lr)

			# clear temp losses
			self.losses.clear_tmp_losses()

			epoch_pbar = self._create_pbar(len(self.data.train), "training progress")

			# loop through batches
			for b_idx, data_batch in enumerate(self.data.train):

				# learn
				self.batch_learn(data_batch, b_idx)

				# update pbar
				self._update_pbar(epoch_pbar, data_batch, b_idx)

				# log the step
				self.log_step(self.losses.tmp, data_batch)

			# log epoch losses and save avg 
			epoch_losses = self.losses.tmp.get_avg()
			self.logger.log_losses(epoch_losses, mode="train")
			self.losses.train.add_losses(epoch_losses)

			# run validation
			self.validation()
			
		# announce trainnig is done
		self.log(f"training finished after {epoch_idx} epochs", fancy=True)

		# plot training losses
		self.logger.plot_training(self.losses)


	@torch.no_grad()
	def validation(self):
		
		# switch to evaluation mode to perform validation
		self.model.eval()

		# clear losses for this run
		self.losses.clear_tmp_losses()

		# progress bar
		val_pbar = self._create_pbar(len(self.data.val), "validation progress")

		# loop through validation batches
		for b_idx, data_batch in enumerate(self.data.val):

			# run the model
			self.batch_forward(data_batch)

			self._update_pbar(val_pbar, data_batch, b_idx)		

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
		
		# progress bar
		test_pbar = self._create_pbar(len(self.data.test), "test progress")

		# loop through testing batches
		for b_idx, data_batch in enumerate(self.data.test):

			# run the model
			self.batch_forward(data_batch)

			# update pbar
			self._update_pbar(test_pbar, data_batch, b_idx)
		
		# log the losses
		test_losses = self.losses.tmp.get_avg()
		self.logger.log_losses(test_losses, mode="test")
		self.losses.test.extend_losses(self.losses.tmp)

	def batch_learn(self, data_batch, b_idx):
		self.batch_forward(data_batch)
		self.batch_backward(data_batch, b_idx)

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

	def batch_backward(self, data_batch, b_idx):

		loss = self.losses.tmp.get_last_loss() 
		loss.backward()

		learn_step = (b_idx + 1) % self.accumulation_steps == 0
		if learn_step:
		
			# grad clip
			if self.grad_clip_norm:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

			# step
			self.optim.step()
			self.scheduler.step()
			self.optim.zero_grad()

	def log_step(self, losses, data_batch):
		
		# get delta time or update last time
		cur_ts = time.perf_counter()
		if self.last_step % self.logger.log_interval != 0:
			self.last_ts = cur_ts
			return 
		delta_ts = cur_ts - self.last_ts

		# create the metrics
		losses_dict = losses.get_last_losses(scale=1/len(data_batch))
		data_dict = {"toks_per_batch": len(data_batch), "samples_per_batch": data_batch.seqlens.size(0)}
		throughput_dict = {"toks_per_sec": len(data_batch) / delta_ts, "updates_per_sec": 1 / delta_ts}

		# format
		losses_dict = {f"train/loss/{k}": v for k, v in losses_dict.items()}
		data_dict = {f"train/data/{k}": v for k, v in data_dict.items()}
		throughput_dict = {f"train/throughput/{k}": v for k, v in throughput_dict.items()}
		scheduler_dict = {f"train/lr": self.last_lr}
		
		# combine
		step_dict = losses_dict | data_dict | scheduler_dict| throughput_dict

		self.logger.log_step(step_dict, self.last_step)

	def log_val(self, losses):
		self.logger.log_losses(losses, mode="val")
		self.logger.log_step({f"val/loss/{k}": v for k, v in losses.items()}, self.last_step)

	def log(self, message, fancy=False):
		if fancy:
			message = f"\n\n{'-'*80}\n{message}\n{'-'*80}\n"
		self.logger.log.info(message)

	def _create_pbar(self, total: int, desc: str) -> tqdm:
		"""Create a progress bar for training/validation/test loops."""
		return tqdm(total=total, desc=desc, unit="samples")

	def _update_pbar(self, pbar: tqdm, data_batch, b_idx: int):
		"""Update progress bar with loss and advance by batch samples."""
		if b_idx % 10 == 0:
			pbar.set_postfix(loss=self.losses.tmp.get_last_loss().item() / len(data_batch))
		pbar.update(data_batch.samples)

	@property
	def last_lr(self):
		return self.scheduler.get_last_lr()[0]

	@property
	def last_step(self):
		'''just for clearer naming'''
		return int(self.scheduler.last_epoch)

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