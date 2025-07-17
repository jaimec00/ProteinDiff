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
			self.training_run_parent.model.module.vae.train()
			self.training_run_parent.model.module.classifier.train()
		elif self.train_type=="diffusion":
			self.training_run_parent.model.module.diffusion.train()

	def epoch_loop(self):
		'''
		a single training loop through one epoch. loops through batches, logs the losses, and runs validation
		'''

		# make sure in training mode
		self.training()

		# setup the epoch
		if self.training_run_parent.rank==0:
			self.training_run_parent.output.log_epoch(self.epoch, self.training_run_parent.step, self.training_run_parent.scheduler.get_last_lr()[0])

		# clear temp losses
		self.training_run_parent.losses.clear_tmp_losses()

		# init epoch pbar
		if self.training_run_parent.rank==0:
			epoch_pbar = tqdm(total=len(self.training_run_parent.data.train_data), desc="epoch_progress", unit="step")

		# loop through batches
		for b_idx, data_batch in enumerate(self.training_run_parent.data.train_data):

			# instantiate this batch
			batch = Batch(data_batch, b_idx=b_idx, epoch=self)

			# learn
			batch.batch_learn()

			# update pbar
			if self.training_run_parent.rank==0:
				epoch_pbar.update(self.training_run_parent.world_size)
		
		# log epoch losses and save avg
		self.training_run_parent.output.log_epoch_losses(self.training_run_parent.losses)

		# run validation
		self.training_run_parent.validation()

		# switch representative cluster samples
		if self.epoch < (self.epochs - 1):
			if self.training_run_parent.rank==0:
				self.training_run_parent.output.log.info("loading next epoch's training data...")
			self.training_run_parent.data.train_data.rotate_data()
			self.training_run_parent.data.val_data.rotate_data()

# ----------------------------------------------------------------------------------------------------------------------

class Batch():
	def __init__(self, data_batch, b_idx=None, epoch=None, inference=False, temp=1e-6):

		# data
		self.coords = data_batch.coords 
		self.labels = data_batch.labels

		# define masks
		self.atom_mask = data_batch.atom_mask
		self.valid_mask = ~data_batch.pad_mask & data_batch.coords_mask & data_batch.seq_mask
		self.loss_mask = self.valid_mask  & data_batch.canonical_seq_mask & data_batch.chain_mask

		# other stuff
		self.b_idx = b_idx
		self.epoch_parent = epoch
		self.inference = inference
		self.temp = temp
		self.train_type = epoch.train_type
		self.world_size = epoch.training_run_parent.world_size
		self.rank = epoch.training_run_parent.rank

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
		model = self.epoch_parent.training_run_parent.model.module			
		loss_function = self.epoch_parent.training_run_parent.losses.loss_function

		# for vae training
		if self.train_type=="vae":
			
			# get the fields
			_, fields = model.prep(self.coords, self.labels, self.atom_mask)

			# get the encoder latents and decoder field predictions 
			latent, latent_mu, latent_logvar, fields_pred = model.vae(fields)

			# predict sequence from fields
			seq_pred = model.classifier(fields_pred.detach()) # detach so classifier loss doesnt affect vae

			# compute loss
			losses = loss_function.vae(latent_mu, latent_logvar, fields_pred, fields, seq_pred, self.labels, self.loss_mask)

		# diffusion training
		elif self.train_type=="diffusion":

			# inference only applicable after train diffusion
			if self.inference:

				# get coords
				coords_bb = model.prep.get_backbone(self.coords)

				# generate a latent from white noise
				generated_latent = model.diffusion.generate(coords_bb, self.valid_mask)

				# predict the fields from generated latent
				fields_pred = model.vae.dec(generated_latent)

				# predict sequence
				seq_pred = model.classifier(fields_pred)

				# compute loss
				losses = loss_function.inference(seq_pred, self.labels, self.loss_mask)
			
			else:

				# run the prep to get the fields
				coords_bb, fields = model.prep(self.coords, self.labels, self.atom_mask)

				# sample a latent
				latent, latent_mu, latent_logvar = model.vae.enc(fields)

				# get random timesteps and noise the latent
				t = model.diffusion.get_rand_t_for(latent)
				latent_noised, noise = model.diffusion.noise(latent, t)

				# predict noise
				noise_pred = model.diffusion(coords_bb, latent_noised, t, self.valid_mask)

				# compute loss
				losses = loss_function.diff(noise_pred, noise, self.loss_mask)

		# add the losses to the temporary losses
		self.epoch_parent.training_run_parent.losses.tmp.add_losses(losses, valid_toks=self.loss_mask.sum())

	def batch_backward(self):

		# utils
		accumulation_steps = self.epoch_parent.training_run_parent.training_parameters.loss.accumulation_steps
		optim = self.epoch_parent.training_run_parent.optim
		scheduler = self.epoch_parent.training_run_parent.scheduler
		learn_step = (self.b_idx + 1) % accumulation_steps == 0

		# get last loss (ddp avgs the gradients, i want the sum, so mult by world size)
		loss = self.epoch_parent.training_run_parent.losses.tmp.get_last_loss() * self.epoch_parent.training_run_parent.world_size # no scaling by accumulation steps, as already handled by grad clipping and scaling would introduce batch size biases

		# perform backward pass to accum grads
		loss.backward()

		if learn_step:
		
			# grad clip
			if self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm:
				torch.nn.utils.clip_grad_norm_(self.epoch_parent.training_run_parent.model.parameters(), max_norm=self.epoch_parent.training_run_parent.training_parameters.loss.grad_clip_norm)

			# step
			optim.step()
			optim.zero_grad()
			scheduler.step()

			self.epoch_parent.training_run_parent.step += 1

	def noise_coords(self):

		'''data augmentation via gaussian noise injection into coords, default is 0.02 A standard deviation, centered around 0'''

		# define noise
		noise = torch.randn_like(self.coords) * self.epoch_parent.training_run_parent.training_parameters.regularization.noise_coords_std

		# add noise
		self.coords = self.coords + noise
