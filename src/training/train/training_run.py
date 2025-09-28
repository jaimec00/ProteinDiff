# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		train_utils.py
description:	utility classes for training proteusAI
'''
# ----------------------------------------------------------------------------------------------------------------------

import torch
import torch.optim.lr_scheduler as lr_scheduler

from tqdm import tqdm
import os

from model import ProteinDiff
from training.logger import Output
from training.data.data_loader import DataHolder
from training.train.training_run_utils import Epoch, Batch
from training.losses import TrainingRunLosses

# ----------------------------------------------------------------------------------------------------------------------

# detect anomolies in training and allow tf32 matmuls
torch.autograd.set_detect_anomaly(True, check_nan=True) # throws error when nan encountered
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class TrainingRun:

	def __init__(self, args: Box) -> None:

		self.gpu = torch.device(f'cuda')
		self.cpu = torch.device("cpu")
		self.debug = args.debug_grad

		self.setup_training(args)
		self.train()
		self.test()

		# done
		self.log("Fin", fancy=True)

	def setup_training(self, args):
		'''
		sets up the training by setting up the model, optimizer, scheduler, loss 
		function, scaler (if using AMP), and losses

		Args:
			None

		Returns:
			None
		'''

		self.hyper_parameters = args.hyper_parameters
		self.training_parameters = args.training_parameters
		
		self.data = DataHolder(	args.data.data_path, # single chain or multichain
								num_train=args.data.num_train, num_val=args.data.num_val, num_test=args.data.num_test, 
								batch_tokens=args.data.batch_tokens, min_seq_size=args.data.min_seq_size, max_seq_size=args.data.max_seq_size, 
								max_resolution=args.data.max_resolution, homo_thresh=args.training_parameters.regularization.homo_thresh, 
								asymmetric_units_only=self.training_parameters.train_type=="vae", # vae and classifier dont have residues communicate, no need for copies
								num_workers=args.data.num_workers, prefetch_factor=args.data.prefetch_factor, rng_seed=args.data.rng_seed, buffer_size=args.data.buffer_size
							)
		
		self.losses = TrainingRunLosses(	args.training_parameters.train_type,
											args.training_parameters.loss.label_smoothing, 
											args.training_parameters.loss.beta,
										)

		self.output = Output(args.output.out_path, model_checkpoints=args.output.model_checkpoints)

		self.checkpoint = torch.load(self.training_parameters.checkpoint.path, weights_only=True, map_location=self.gpu) if self.training_parameters.checkpoint.path else ""

		self.scaler = torch.GradScaler("cuda", init_scale=32) if args.training_parameters.use_amp else None # start small, have it adjust later to avoid overflow early on 

		self.setup_model()
		self.setup_optim()
		self.setup_scheduler()

		self.output.log_trainingrun(self.training_parameters, self.hyper_parameters, self.data)

	def setup_model(self):
		'''
		instantiates proteusAI with given Hyper-Parameters, moves it to gpu, 
		optionally loads model weights from pre-trained models, and freezes modules
		depending on train type

		Args:
			None
		
		Returns:
			None
		'''
		
		self.log("loading model...")
		
		self.model = ProteinDiff(	d_model=self.hyper_parameters.d_model, 
									d_diffusion=self.hyper_parameters.d_diffusion, 
									d_latent=self.hyper_parameters.d_latent, 
									top_k=self.hyper_parameters.top_k, 
									voxel_dims=self.hyper_parameters.voxel_dims, 
									cell_dim=self.hyper_parameters.cell_dim,
									vae_layers=self.hyper_parameters.vae_layers,
									diff_layers=self.hyper_parameters.diff_layers,
									diff_parameterization=self.hyper_parameters.diff_parameterization,
									class_layers=self.hyper_parameters.class_layers
								)
		# parallelize the model
		self.model.to(self.gpu)

		# load any checkpoints
		if self.checkpoint:
			state_dicts = self.checkpoint["model"]
			if self.training_parameters.checkpoint.vae:
				self.model.vae.load_state_dict(state_dicts["vae"])
				self.model.classifier.load_state_dict(state_dicts["classifier"])
			if self.training_parameters.checkpoint.diff:
				self.model.diffusion.load_state_dict(state_dicts["diffusion"])

		self.model.eval()
		if self.training_parameters.train_type=="diffusion": # freeze vae if in diffusion
			for param in self.model.vae.parameters():
				param.requires_grad = False


		# get number of parameters for logging
		self.training_parameters.num_params = sum(p.numel() for p in self.model.parameters())

	def setup_optim(self):
		'''
		sets up the optimizer, zeros out the gradient

		Args:
			None
		
		Returns:
			None
		'''

		self.log("Loading Optimizer...")
		self.optim = torch.optim.AdamW(	self.model.parameters(), lr=1.0,
										betas=(self.training_parameters.adam.beta1, self.training_parameters.adam.beta2), 
										eps=float(self.training_parameters.adam.epsilon), weight_decay=self.training_parameters.adam.weight_decay)
		self.optim.zero_grad()
		if self.checkpoint and self.training_parameters.checkpoint.adam:
			self.optim.load_state_dict(self.checkpoint["adam"])

	def setup_scheduler(self):
		'''
		sets up the loss scheduler, right now only using ReduceLROnPlateu, but planning on making this configurable

		Args:
			None
		
		Returns:
			None
		'''

		self.log("Loading Scheduler...")

		if self.training_parameters.lr.lr_type == "attn":

			# compute the scale
			if self.training_parameters.lr.lr_step == 0.0:
				scale = self.hyper_parameters.d_model**(-0.5)
			else:
				scale = self.training_parameters.lr.warmup_steps**(0.5) * self.training_parameters.lr.lr_step # scale needed so max lr is what was specified

			def attn(step):
				'''lr scheduler from attn paper'''
				step = step # in case job gets cancelled and want to start from where left off
				return scale * min((step+1)**(-0.5), (step+1)*(self.training_parameters.lr.warmup_steps**(-1.5)))

			self.scheduler = lr_scheduler.LambdaLR(self.optim, attn)

		elif self.training_parameters.lr.lr_type == "static":
			def static(step):
				return self.training_parameters.lr.lr_step
			self.scheduler = lr_scheduler.LambdaLR(self.optim, static)

		else:
			raise ValueError(f"invalid lr_type: {self.training_parameters.lr.lr_type}. options are ['attn', 'static']")

		if self.checkpoint and self.training_parameters.checkpoint.sched:
			self.scheduler.load_state_dict(self.checkpoint["scheduler"])

	def model_checkpoint(self, epoch_idx):
		if (epoch_idx+1) % self.output.model_checkpoints == 0: # model checkpointing
			self.output.save_checkpoint(self.model, adam=self.optim, scheduler=self.scheduler, appended_str=f"e{epoch_idx}_s{round(self.losses.val.get_last_loss(),2)}")

	def training_converged(self, epoch_idx):

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

		# load the data, note that all gpus are required to load all the data with the same random seeds, so they get unique data compared to other gpus each epoch
		self.log("Loading Training Data...")
		self.data.load("train")
		self.log("Loading Validation Data...")
		self.data.load("val")

		# log training info
		self.log(f"\n\nInitializing training. "\
					f"Training on approx. {len(self.data.train)} batches "\
					f"of batch size {self.data.batch_tokens} tokens "\
					f"for {self.training_parameters.epochs} epochs.\n" 
				)
		
		# loop through epochs
		for epoch_idx in range(self.training_parameters.epochs):

			# initialize epoch and loop through batches
			epoch = Epoch(self, epoch_idx)
			epoch.epoch_loop()
			
			self.model_checkpoint(epoch_idx)
			if self.training_converged(epoch_idx): break
			
		# announce trainnig is done
		self.log(f"Training Done After {epoch.epoch} Epochs", fancy=True)

		# plot training losses
		self.output.plot_training(self.losses)

		# save the model
		self.output.save_checkpoint(self.model, self.optim, self.scheduler, appended_str="final")

	def validation(self):
		
		# switch to evaluation mode to perform validation
		self.model.eval()

		# clear losses for this run
		self.losses.clear_tmp_losses()

		# dummy epoch so can still access training run parent
		dummy_epoch = Epoch(self)

		# progress bar
		val_pbar = tqdm(total=len(self.data.val_data), desc="Validation Progress", unit="Step")
				
		# turn off gradient calculation
		with torch.no_grad():

			# loop through validation batches
			for data_batch in self.data.val_data:
					
				# init batch
				batch = Batch(data_batch, epoch=dummy_epoch)

				# run the model
				batch.batch_forward()

				val_pbar.update(1)

			# add the avg losses to the global loss and log
			self.output.log_val_losses(self.losses)

	def test(self):

		# switch to evaluation mode
		self.model.eval()

		self.log("Starting Testing", fancy=True)
		
		# load testing data
		self.log("Loading Testing Data...")
		self.data.load("test")

		# init losses
		self.losses.set_inference_losses(self.training_parameters.train_type)

		# dummy epoch so can still access training run parent
		dummy_epoch = Epoch(self)

		# progress bar
		test_pbar = tqdm(total=len(self.data.test_data), desc="Test Progress", unit="Step")

		# turn off gradient calculation
		with torch.no_grad():

			# loop through testing batches
			for data_batch in self.data.test_data:
					
				# init batch
				batch = Batch(  data_batch,
								temp=self.training_parameters.inference.temperature,
								inference=True, epoch=dummy_epoch
							)

				# run the model
				batch.batch_forward()

				# update pbar
				test_pbar.update(1)
		
		# log the losses
		self.output.log_test_losses(self.losses)

	def log(self, message, fancy=False):
		if fancy:
			message = f"\n\n{'-'*80}\n{message}\n{'-'*80}\n"

		self.output.log.info(message)
