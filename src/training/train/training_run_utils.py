# ----------------------------------------------------------------------------------------------------------------------

import torch
from tqdm import tqdm
from data.constants import aa_2_lbl

# ----------------------------------------------------------------------------------------------------------------------

class Epoch():	
	def __init__(self, training_run, epoch=None):

		self.training_run_parent = training_run
		self.epoch = epoch
		self.epochs = training_run.training_parameters.epochs
		self.train_type = self.training_run_parent.training_parameters.train_type

	def training(self):
		if self.train_type=="vae":
			self.training_run_parent.model.vae.train()
			self.training_run_parent.model.classifier.train()
		elif self.train_type=="diffusion":
			self.training_run_parent.model.diffusion.train()

	def epoch_loop(self):
		'''
		a single training loop through one epoch. loops through batches, logs the losses, and runs validation
		'''

		# make sure in training mode
		self.training()

		# setup the epoch
		self.training_run_parent.output.log_epoch(self.epoch, self.training_run_parent.scheduler.last_epoch, self.training_run_parent.scheduler.get_last_lr()[0])

		# clear temp losses
		self.training_run_parent.losses.clear_tmp_losses()

		# init epoch pbar
		epoch_pbar = tqdm(total=len(self.training_run_parent.data.train_data), desc="Training Progress", unit="step")

		# loop through batches
		for b_idx, data_batch in enumerate(self.training_run_parent.data.train_data):

			# instantiate this batch
			batch = Batch(data_batch, b_idx=b_idx, epoch=self)

			# learn
			batch.batch_learn()

			# update pbar
			epoch_pbar.update(1)
		
		# log epoch losses and save avg
		self.training_run_parent.output.log_epoch_losses(self.training_run_parent.losses)

		# run validation
		self.training_run_parent.validation()

		# switch representative cluster samples
		if self.epoch < (self.epochs - 1):
			self.training_run_parent.output.log.info("Loading Next Epoch's Training Data...")
			self.training_run_parent.data.train_data.rotate_data()
			self.training_run_parent.data.val_data.rotate_data()

# ----------------------------------------------------------------------------------------------------------------------

class Batch():
	def __init__(self, data_batch, b_idx=None, epoch=None, inference=False, temp=1e-6):

		# data
		self.coords = data_batch.coords 
		self.labels = data_batch.labels
		self.seq_pos = data_batch.seq_pos
		self.chain_pos = data_batch.chain_pos
		self.sample_idx = data_batch.sample_idx

		# define masks
		self.atom_mask = data_batch.atom_mask
		self.valid_mask = data_batch.coords_mask & data_batch.seq_mask
		self.loss_mask = self.valid_mask  & data_batch.canonical_seq_mask & data_batch.chain_mask

		# other stuff
		self.b_idx = b_idx
		self.epoch_parent = epoch
		self.inference = inference
		self.temp = temp
		self.train_type = epoch.train_type
		self.scaler =epoch.training_run_parent.scaler 

	def move_to(self, device):

		self.coords = self.coords.to(device)
		self.labels = self.labels.to(device)
		self.atom_mask = self.atom_mask.to(device)
		self.valid_mask = self.valid_mask.to(device)
		self.loss_mask = self.loss_mask.to(device)

	def batch_learn(self):
		'''
		a single iteration over a batch.
		'''

		# add random noise to the coordinates (batch learn only used in training)
		self.noise_coords()

		# forward pass
		self.batch_forward()

		# backward pass
		self.batch_backward()

	def batch_forward(self):
		'''
		performs the forward pass, gets the outputs and computes the losses of a batch. 
		'''
		
		# move batch to gpu
		self.move_to(self.epoch_parent.training_run_parent.gpu)

		# utils
		model = self.epoch_parent.training_run_parent.model	
		loss_function = self.epoch_parent.training_run_parent.losses.loss_function

		# for vae training
		if self.train_type=="vae":
			
			latent_mu, latent_logvar, divergence_pred, divergence, seq_pred = model(self.coords, self.labels, self.atom_mask, self.valid_mask, run_type="vae")
				
			# compute loss
			losses = loss_function.vae(latent_mu, latent_logvar, divergence_pred, divergence, seq_pred, self.labels, self.loss_mask)

		# diffusion training
		elif self.train_type=="diffusion":

			# inference only applicable after train diffusion
			if self.inference:

				seq_pred = model(self.coords, self.labels, self.atom_mask, self.valid_mask, run_type="inference")
				
				# compute loss
				losses = loss_function.inference(seq_pred, self.labels, self.loss_mask)
			
			else:

				pred, trgt = model(self.coords, self.labels, self.seq_pos, self.chain_pos, self.atom_mask, self.valid_mask, run_type="diffusion")
				
				# compute loss
				losses = loss_function.diff(pred, trgt, self.loss_mask)

		# add the losses to the temporary losses
		self.epoch_parent.training_run_parent.losses.tmp.add_losses(losses, valid_toks=self.loss_mask.sum())

	def batch_backward(self):

		# utils
		accumulation_steps = self.epoch_parent.training_run_parent.training_parameters.loss.accumulation_steps
		optim = self.epoch_parent.training_run_parent.optim
		scheduler = self.epoch_parent.training_run_parent.scheduler
		learn_step = (self.b_idx + 1) % accumulation_steps == 0

		loss = self.epoch_parent.training_run_parent.losses.tmp.get_last_loss() 

		# perform backward pass to accum grads
		if self.scaler is None:
			loss.backward()
		else:
			self.scaler.scale(loss).backward()

		if learn_step:
		
			if self.scaler is None:
				# grad clip
				if self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm:
					torch.nn.utils.clip_grad_norm_(self.epoch_parent.training_run_parent.model.parameters(), max_norm=self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm)
				# step
				optim.step()
			else:
				if self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm:
					self.scaler.unscale_(optim) # scaler sees that grads already unscaled when call step, no issue
					torch.nn.utils.clip_grad_norm_(self.epoch_parent.training_run_parent.model.parameters(), max_norm=self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm)

				self.scaler.step(optim)
				self.scaler.update()

			scheduler.step()
			optim.zero_grad()

	def noise_coords(self):

		'''data augmentation via gaussian noise injection into coords, default is 0.02 A standard deviation, centered around 0'''

		# define noise
		noise = torch.randn_like(self.coords) * self.epoch_parent.training_run_parent.training_parameters.regularization.noise_coords_std

		# add noise
		self.coords = self.coords + noise


