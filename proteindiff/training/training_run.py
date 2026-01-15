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
from tqdm import tqdm
from hydra.utils import instantiate
from dataclasses import dataclass, field
from omegaconf import OmegaConf as om, DictConfig

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
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
	#TODO: add other stuff relevant to train (keep simple for now)

class TrainingRun:

	def __init__(self, cfg: TrainingRunCfg) -> None:

		self.gpu = torch.device('cuda' if torch.cuda.is_available() else "cpu")
		self.cpu = torch.device("cpu")

		self.data = DataHolder(cfg.data)
		self.losses = TrainingRunLosses(cfg.losses)
		self.logger = Logger(cfg.logger)
		self.model = ProteinDiff(cfg.model).to(self.gpu)
		# self.model = torch.compile(self.model, dynamic=True)
		self.optim = setup_optim(cfg.optim, self.model)
		self.scheduler = setup_scheduler(cfg.scheduler, self.optim)

		self.train_stage = cfg.train_stage
		self.epochs = cfg.epochs
		self.accumulation_steps = cfg.accumulation_steps
		self.grad_clip_norm = cfg.grad_clip_norm

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

			# setup the epoch # TODO: helper for get last lr
			self.logger.log_epoch(epoch_idx, self.scheduler.last_epoch, self.scheduler.get_last_lr()[0])

			# clear temp losses
			self.losses.clear_tmp_losses()

			epoch_pbar = tqdm(total=len(self.data.train), desc="training progress", unit="samples")

			# loop through batches
			for b_idx, data_batch in enumerate(self.data.train):

				# learn
				self.batch_learn(data_batch, b_idx)

				# update pbar
				if b_idx % 10 ==0:
					epoch_pbar.set_postfix(loss=self.losses.tmp.get_last_loss().item() / len(data_batch))
				epoch_pbar.update(data_batch.samples)
			
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
		val_pbar = tqdm(total=len(self.data.val), desc="validation progress", unit="samples")

		# loop through validation batches
		for b_idx, data_batch in enumerate(self.data.val):
				
			# run the model
			self.batch_forward(data_batch)

			if b_idx % 10 == 0:
				val_pbar.set_postfix(loss=self.losses.tmp.get_last_loss().item() / len(data_batch))
			val_pbar.update(data_batch.samples)

		# log the losses
		val_losses = self.losses.tmp.get_avg()
		self.logger.log_losses(val_losses, mode="val")
		self.losses.val.add_losses(val_losses)

	@torch.no_grad()
	def test(self):

		# switch to evaluation mode
		self.model.eval()

		# log
		self.log("starting testing", fancy=True)
		
		# progress bar
		test_pbar = tqdm(total=len(self.data.test), desc="test progress", unit="samples")

		# loop through testing batches
		for b_idx, data_batch in enumerate(self.data.test):
				
			# run the model
			self.batch_forward(data_batch)

			# update pbar
			if b_idx % 10 == 0:
				test_pbar.set_postfix(loss=self.losses.tmp.get_last_loss().item() / len(data_batch))
			test_pbar.update(data_batch.samples)
		
		# log the losses
		test_losses = self.losses.tmp.get_avg()
		self.logger.log_losses(test_losses, mode="test")
		self.losses.test.extend_losses(self.losses.tmp)

	def log(self, message, fancy=False):
		if fancy:
			message = f"\n\n{'-'*80}\n{message}\n{'-'*80}\n"
		self.logger.log.info(message)


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
				coords_bb, frames
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
				coords_bb=coords_bb, 
				frames=frames,
				cu_seqlens=data_batch.cu_seqlens
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


@hydra.main(version_base=None, config_path="../../configs", config_name="debug")
def main(cfg: DictConfig):
	# build model from simple cfg
	from proteindiff.utils.cfg_utils.model_cfg_utils import build_model_cfg_from_simple_cfg
	simple_model_cfg = instantiate(cfg.model)
	model_cfg = build_model_cfg_from_simple_cfg(simple_model_cfg)
	cfg.model = om.structured(model_cfg)
	
	TrainingRun(cfg)

if __name__ == "__main__":
	main()