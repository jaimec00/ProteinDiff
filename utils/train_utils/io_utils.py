# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		io_utils.py
description:	utility classes for input/output operations during training 
'''
# ----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
import logging
import torch
import math
import sys

# ----------------------------------------------------------------------------------------------------------------------

class Output():

	def __init__(self, out_path, model_checkpoints=10, rank=0, world_size=1):\

		self.out_path = Path(out_path)
		self.out_path.mkdir(parents=True, exist_ok=True)

		self.plot_path = self.out_path / Path("plots")

		self.log = self.setup_logging(self.out_path / Path("log.txt"))
		self.model_checkpoints = model_checkpoints
		self.rank = rank 
		self.world_size = world_size

	def setup_logging(self, log_file):

		logger = logging.getLogger("proteusAI_log")
		logger.setLevel(logging.DEBUG)

		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

		file_handler = logging.FileHandler(log_file, mode="w")
		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(formatter)

		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setLevel(logging.DEBUG)
		console_handler.setFormatter(formatter)

		logger.addHandler(file_handler)
		logger.addHandler(console_handler)

		return logger

	def log_trainingrun(self, training_parameters, hyper_parameters, data):
		'''basically just prints the config file w/ a little more info'''

		log = 	textwrap.dedent(f'''

		total parameters: {training_parameters.num_params:,}
		
		model hyper-parameters:

		data:
  			data_path: {data.data_path}
			dataset split ({data.num_train + data.num_val + data.num_test:,} clusters total): 
				train clusters: {data.num_train:,}
				validation clusters: {data.num_val:,}
				test clusters: {data.num_test:,}
			batch size (tokens): {data.batch_tokens:,}
			max batch size (samples): {data.max_batch_size:,}
			min sequence length (tokens): {data.min_seq_size:,}
			max sequence length (tokens): {data.max_seq_size:,}
			effective batch size (tokens): {training_parameters.loss.accumulation_steps if training_parameters.loss.token_based_step else data.batch_tokens * training_parameters.loss.accumulation_steps}

		training-parameters:
			epochs: {training_parameters.epochs}
			rng: {training_parameters.rng}
			checkpoint:
				checkpoint_path: {training_parameters.checkpoint.path}
			inference:
				temperature: {training_parameters.inference.temperature}
			early_stopping:
				thresh: {training_parameters.early_stopping.thresh} 
				tolerance: {training_parameters.early_stopping.tolerance}
			adam:
				beta1: {training_parameters.adam.beta1}
				beta2: {training_parameters.adam.beta2}
				epsilon: {training_parameters.adam.epsilon}
			regularization:
				dropout: {training_parameters.regularization.dropout}
				noise_coords_std: {training_parameters.regularization.noise_coords_std}
				homo_thresh: {training_parameters.regularization.homo_thresh}
				label_smoothing: {training_parameters.regularization.label_smoothing}
			loss:
				accumulation_steps: {training_parameters.loss.accumulation_steps} batches
				grad_clip_norm: {training_parameters.loss.grad_clip_norm}
			lr:
				lr_type: {training_parameters.lr.lr_type}
				lr_step: {training_parameters.lr.lr_step}
				warmup_steps: {training_parameters.lr.warmup_steps}
		
		output directory: {self.out_path}
		''')

		if self.rank==0:
			self.log.info(log)

	def log_epoch(self, epoch, step, current_lr):

		if self.rank==0:
			self.log.info(textwrap.dedent(f'''
			
				{'-'*80}
				epoch {epoch}, step {step:,}: 
				{'-'*80}
				
				current learning rate: {current_lr}
			''')
			)

	def log_losses(self, losses, mode):

		# workers pickle their loss objects, send to master, master extends the loss
		if self.rank == 0:
			loss_list = [None for _ in range(self.world_size)]
		else:
			loss_list = None


		torch.distributed.gather_object(
			obj=losses.tmp,
			object_gather_list=loss_list,
			dst=0,
			group=None
		)

		if self.rank!=0: 
			return # workers are done after gather

		for worker_loss in loss_list[1:]: # exclude the master loss object, as that is what we are extending
			losses.tmp.extend_losses(worker_loss) 

		losses_dict = losses.tmp.get_avg()
		for loss_type, loss in losses_dict.items():
			self.log.info(f"{mode} {loss_type} per token: {str(loss)}")	
		
		if mode == "train":
			losses.train.add_losses(losses_dict)
		elif mode == "validation":	
			losses.val.add_losses(losses_dict)
		else: # testing
			losses.test.extend_losses(losses.tmp)

	def log_epoch_losses(self, losses):
		self.log_losses(losses, "train")

	def log_val_losses(self, losses):
		self.log_losses(losses, "validation")

	def log_test_losses(self, losses):
		self.log_losses(losses, "test")

	def plot_training(self, losses):

		# convert to numpy arrays
		losses.to_numpy()

		# make the output directory
		self.plot_path.mkdir(exist_ok=True)

		# specify number of epochs
		epochs = np.arange(len(losses.train))

		# extract the keys and iterate
		loss_types = losses.val.losses.keys()
		for loss_type in loss_types:
			plt.plot(epochs, losses.train.losses[loss_type], marker='o', color='red', label="Training")
			plt.plot(epochs, losses.val.losses[loss_type], marker='o', color='blue', label="Validation")
			plt.title(f'{loss_type} vs. Epochs')
			plt.xlabel('Epochs')
			plt.ylabel(loss_type)
			plt.legend()
			plt.grid(True)
			loss_path = self.plot_path / Path(f"{'_'.join(loss_type.lower().split(' '))}.png")
			plt.savefig(loss_path)
			self.log.info(f"Plot of {loss_type} vs. Epochs saved to {loss_path}")
			plt.figure()

	def save_checkpoint(self, model, adam=None, scheduler=None, appended_str=""):

		checkpoint = {	"model": {	"vae": model.module.vae.state_dict(), 
									"diffusion": model.module.diffusion.state_dict(), 
									"classifier": model.module.classifier.state_dict(), 
						}
						"adam": (None if adam is None else adam.state_dict()), 
						"scheduler": (None if scheduler is None else scheduler.state_dict())
					}
		checkpoint_path = self.out_path / Path(f"checkpoint_{appended_str}.pth")
		torch.save(checkpoint, checkpoint_path)
		self.log.info(f"checkpoint saved to {checkpoint_path}")

# ----------------------------------------------------------------------------------------------------------------------