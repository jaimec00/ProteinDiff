# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		train_utils.py
description:	utility classes for training proteusAI
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.optim.lr_scheduler as lr_scheduler

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
from proteindiff.training.losses import TrainingRunLosses, TrainingRunLossesCfg
from proteindiff.training.optim import OptimCfg
from proteindiff.training.scheduler import SchedulerCfg
from proteindiff.static.constants import TrainingStage

# ----------------------------------------------------------------------------------------------------------------------

# detect anomolies in training and allow tf32 matmuls
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
		self.cfg = cfg

		self.setup_training()
		self.train()
		self.test()

		# done
		self.log("fin", fancy=True)

	def setup_training(self):

		self.data = DataHolder(self.cfg.data)
		self.losses = TrainingRunLosses(self.cfg.losses)
		self.logger = Logger(self.cfg.logger)

		self.setup_model()
		self.setup_optim()
		self.setup_scheduler()

	def setup_model(self):
		
		self.log("loading model...")
		with self.gpu:
			self.model = ProteinDiff(self.cfg.model)

	def setup_optim(self):
		self.log("loading optimizer...")
		self.optim = torch.optim.AdamW(
			self.model.parameters(),
			lr=1.0, # will use LambaLR instead
			betas=(self.cfg.optim.beta1, self.cfg.optim.beta2), 
			eps=self.cfg.optim.eps, 
			weight_decay=self.cfg.optim.weight_decay
		)
		self.optim.zero_grad()

	def setup_scheduler(self):

		self.log("loading scheduler...")

		if self.cfg.scheduler.lr_type == "attn":

			# compute the scale
			if self.cfg.scheduler.lr_step == 0.0:
				scale = self.cfg.scheduler.d_model**(-0.5)
			else:
				scale = self.cfg.scheduler.warmup_steps**(0.5) * self.cfg.scheduler.lr_step # scale needed so max lr is what was specified

			def attn(step):
				'''lr scheduler from attn paper'''
				step = step # in case job gets cancelled and want to start from where left off
				return scale * min((step+1)**(-0.5), (step+1)*(self.scheduler.warmup_steps**(-1.5)))

			self.scheduler = lr_scheduler.LambdaLR(self.optim, attn)

		elif self.cfg.scheduler.lr_type == "static":
			def static(step):
				return self.cfg.scheduler.lr_step
			self.scheduler = lr_scheduler.LambdaLR(self.optim, static)

		else:
			raise ValueError(f"invalid lr_type: {self.cfg.scheduler.lr_type}. options are ['attn', 'static']")

	def model_checkpoint(self, epoch_idx):
		if (epoch_idx+1) % self.logger.model_checkpoints == 0: # model checkpointing
			self.output.save_checkpoint(appended_str=f"e{epoch_idx}_s{round(self.losses.val.get_last_loss(),2)}")

	def save_checkpoint(self, appended_str=""):
		# TODO: implement checkpointing
		checkpoint = {}
		checkpoint_path = self.logger.out_path / f"checkpoint_{appended_str}.pth"
		torch.save(checkpoint, checkpoint_path)
		self.log.info(f"checkpoint saved to {checkpoint_path}")

	def training_converged(self, epoch_idx):
		# TODO check this out

		criteria = self.losses.val.losses[list(self.losses.val.losses.keys())[0]]

		choose_best = min # choose best
		best = float("inf")
		converged = lambda best, thresh: best > thresh

		# val losses are already in avg seq sim format per epoch
		if self.training_parameters.early_stopping.tolerance+1 > len(criteria):
			return False

		current_n = criteria[-(self.training_parameters.early_stopping.tolerance):]
		old = criteria[-(self.training_parameters.early_stopping.tolerance+1)]

		for current in current_n:
			delta = current - old
			best = choose_best(best, delta) 

		has_converged = converged(best, self.training_parameters.early_stopping.thresh)

		return has_converged

	def train(self):
		'''
		entry point for training the model. loads train and validation data, loops through epochs, plots training, 
		runs testing and saves the model
		'''

		self.log(
			f"\n\nInitializing training. "
			f"Training on approx. {len(self.data.train)} clusters "
			f"for {self.cfg.epochs} epochs.\n" 
		)
		
		# loop through epochs
		for epoch_idx in range(self.cfg.epochs):

			# TODO: mae a helper to only set non-frozen models to train
			# make sure in training mode
			self.model.train()

			# setup the epoch # TODO: helper for get last lr
			self.logger.log_epoch(epoch_idx, self.scheduler.last_epoch, self.scheduler.get_last_lr()[0])

			# clear temp losses
			self.losses.clear_tmp_losses()

			# init epoch pbar # TODO check the total arg is right
			epoch_pbar = tqdm(total=len(self.data.train), desc="training progress", unit="step")

			# loop through batches
			for b_idx, data_batch in enumerate(self.data.train):

				# learn
				self.batch_learn(data_batch)

				# update pbar
				epoch_pbar.update(1)
			
			# log epoch losses and save avg 
			epoch_losses = self.losses.tmp.get_avg()
			self.logger.log_losses(epoch_losses)
			losses.train.add_losses(losses_dict)

			# run validation
			self.validation()
			
			# checkpoint
			self.model_checkpoint(epoch_idx)

			# early stopping
			if self.training_converged(epoch_idx): break
			
		# announce trainnig is done
		self.log(f"training finished after {epoch.epoch} epochs", fancy=True)

		# plot training losses
		self.output.plot_training(self.losses)

		# save the model
		self.output.save_checkpoint(self.model, self.optim, self.scheduler, appended_str="final")

	@torch.no_grad()
	def validation(self):
		
		# switch to evaluation mode to perform validation
		self.model.eval()

		# clear losses for this run
		self.losses.clear_tmp_losses()

		# progress bar # TODO: check len is right here
		val_pbar = tqdm(total=len(self.data.val), desc="validation progress", unit="step")

		# loop through validation batches
		for data_batch in self.data.val:
				
			# run the model
			self.batch_forward(data_batch)

			val_pbar.update(1)

		# log the losses
		val_losses = self.losses.tmp.get_avg()
		self.output.log_losses(test_losses)
		self.losses.add_losses(val_losses)

	@torch.no_grad()
	def test(self):

		# switch to evaluation mode
		self.model.eval()

		# log
		self.log("starting testing", fancy=True)
		
		# init losses
		self.losses.set_inference_losses(self.training_parameters.train_type)

		# progress bar
		test_pbar = tqdm(total=len(self.data.test), desc="test progress", unit="step")

		# loop through testing batches
		for data_batch in self.data.test:
				
			# run the model
			self.batch_forward(data_batch)

			# update pbar
			test_pbar.update(1)
		
		# log the losses
		test_losses = self.losses.tmp.get_avg()
		self.output.log_losses(test_losses)
		self.losses.extend_losses(test_losses)

	def log(self, message, fancy=False):
		if fancy:
			message = f"\n\n{'-'*80}\n{message}\n{'-'*80}\n"
		self.logger.log.info(message)


	def batch_learn(self, data_batch, b_idx):
		'''
		a single iteration over a batch.
		'''

		# forward pass
		self.batch_forward(data_batch)

		# backward pass
		self.batch_backward(data_batch, b_idx)

	def batch_forward(self, data_batch):
		'''
		performs the forward pass, gets the outputs and computes the losses of a batch. 
		'''
		
		# move batch to gpu
		data_batch.move_to(self.model.device)

		# for vae training
		if self.cfg.train_type==TrainingStage.VAE:
			
			coords_bb, divergence, local_frames = self.model.tokenizer(data_batch.coords, data_batch.labels, data_batch.atom_mask)
			
			(
				latent,
				mu, logvar,
				divergence_pred, 
				seq_pred,
				distogram, 
				anglogram, 
				t, x, y, sin, cos,
				plddt, pae, 
			) = model.vae(
				divergence=divergence,
				coords_bb=coords_bb,
				frames=local_frames,
				seq_idx=data_batch.seq_idx,
				chain_idx=data_batch.chain_idx,
				sample_idx=data_batch.sample_idx,
				cu_seqlens=data_batch.cu_seqlens,
				max_seqlen=data_batch.max_seqlen,
			)
				
			# compute loss
			losses = self.losses.loss_fn.vae(
				latent_mu, 
				latent_logvar, 
				divergence_pred, 
				divergence, 
				seq_pred, data_batch.labels,
				data_batch.coords, coords_bb, frames,
				t, x, y, 
				sin, cos,
				plddt, pae,
				data_batch.sample_idx
			)

		# diffusion training
		elif self.cfg.train_type==TrainingStage.DIFFUSION:
			# TODO: implement
			pass

		# add the losses to the temporary losses
		self.losses.tmp.add_losses(losses, valid_toks=self.loss_mask.sum())

	def batch_backward(self, data_batch, b_idx):

		loss = self.losses.tmp.get_last_loss() 
		loss.backward()

		learn_step = (b_idx + 1) % self.cfg.accumulation_steps == 0
		if learn_step:
		
			# grad clip
			if self.grad_clip_norm:
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_norm)

			# step
			self.optim.step()
			self.scheduler.step()
			self.optim.zero_grad()


@hydra.main(version_base=None, config_path="../../../configs", config_name="debug")
def main(cfg: DictConfig):
	# build model from simple cfg
	from proteindiff.utils.cfg_utils.model_cfg_utils import build_model_cfg_from_simple_cfg
	simple_model_cfg = instantiate(cfg.model)
	model_cfg = build_model_cfg_from_simple_cfg(simple_model_cfg)
	cfg.model = om.structured(model_cfg)
	
	TrainingRun(cfg)

if __name__ == "__main__":
	main()