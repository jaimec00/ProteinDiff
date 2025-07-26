import torch
import torch.nn as nn
from collections import OrderedDict
import math

from utils.model_utils.preprocesser import PreProcesser
from utils.model_utils.vae import VAE
from utils.model_utils.latent_diffusion.diffusion import Diffusion
from utils.model_utils.classifier import Classifier

class ProteinDiff(nn.Module):
	def __init__(self,  d_model=256, d_diffusion=256, d_latent=16, top_k=16, 
						voxel_dims=16, cell_dim=0.75,
						min_rbf=2.0, max_rbf=22.0, num_rbf=16,
						vae_layers=3, diff_layers=3, class_layers=3
						):
		super().__init__()
		'''
		this is basically just a wrapper to hold all of the individual models together.
		training run handles how to use them efficiently. 
		'''

		# just to make it easier for now
		assert math.log(voxel_dims, 2).is_integer()

		self.prep = PreProcesser(voxel_dims=(voxel_dims,)*3, cell_dim=cell_dim)
		self.vae = VAE(voxel_dim=voxel_dims, d_model=d_model, d_latent=d_latent, resnet_layers=vae_layers)
		self.diffusion = Diffusion(d_model=d_diffusion, d_latent=d_latent, layers=diff_layers, t_max=1000, num_rbf=num_rbf)
		self.classifier = Classifier(voxel_dim=voxel_dims, d_model=d_model, resnet_layers=class_layers)

	def forward(self, C, L, atom_mask=None, valid_mask=None, run_type="inference", temp=1e-6):
		'''
		'''
		if run_type=="inference":
			C_backbone = self.prep.get_backbone(C)
			latent = self.diffusion.generate(C_backbone, valid_mask)
			voxel = self.vae.dec(latent)
			seq = self.classifier(voxel)
			return seq

		if run_type=="vae":
			_, voxels = self.prep(C, L, atom_mask, valid_mask)
			latent, latent_mu, latent_logvar, decoded_voxels = self.vae(voxels)
			seq = self.classifier(decoded_voxels.detach()) # classifier does not affect vae gradients

			return latent_mu, latent_logvar, decoded_voxels, voxels, seq

		if run_type=="diffusion":
			with torch.no_grad():
				C_backbone, voxels = self.prep(C, L, atom_mask, valid_mask)
				latent, latent_mu, latent_logvar = self.vae.enc(voxels)
				t = self.diffusion.get_rand_t_for(latent)
				latent_noised, velocity = self.diffusion.noise(latent, t)
			velocity_pred = self.diffusion(latent_noised, t, C_backbone, valid_mask)

			return velocity_pred, velocity
