
import torch
import torch.nn as nn
from utils.model_utils.base_modules import ResNet
import math

class VAE(nn.Module):
	def __init__(self, voxel_dim=16, d_model=256, d_latent=16, resnet_layers=1):
		super().__init__()

		self.enc = VAEEncoder(voxel_dim=voxel_dim, d_model=d_model, d_latent=d_latent, resnet_layers=resnet_layers)
		self.dec = VAEDecoder(voxel_dim=voxel_dim, d_model=d_model, d_latent=d_latent, resnet_layers=resnet_layers)

	def forward(self, voxels):
		latent, latent_mu, latent_logvar = self.enc(voxels)
		voxels_pred = self.dec(latent)

		return latent, latent_mu, latent_logvar, voxels_pred

class VAEEncoder(nn.Module):
	def __init__(self, voxel_dim=16, d_model=256, d_latent=16, resnet_layers=1):
		super().__init__()

		num_downsample = math.log(voxel_dim, 2)
		d_start = d_model // voxel_dim

		self.featurizer = nn.Sequential(   # increase channels, keep spatial res
											nn.Conv3d(1, d_start, 2, stride=1, padding='same', bias=False),
											nn.GroupNorm(d_start//8, d_start),
											nn.SiLU(),
											ResNet(d_model=d_start,kernel_size=2,layers=resnet_layers)
										)

		# halve spatial dims, double feature channels each time
		self.downsamples = nn.ModuleList([	nn.Sequential(
															nn.Conv3d(d_start*(2**i), d_start*(2**(i+1)), 2, stride=2, padding=0, bias=False),
															nn.GroupNorm(d_start*(2**(i+1))//8, d_start*(2**(i+1))),
															nn.SiLU(),
															ResNet(d_model=d_start*(2**(i+1)),kernel_size=2,layers=resnet_layers),
														) 
											for i in range(int(num_downsample))
										])

		# project to latent params
		self.latent_proj = nn.Conv3d(d_model, 2*d_latent, 1, stride=1, padding="same", bias=False)	

	def forward(self, voxels):
		'''
		voxels (torch.Tensor): full voxels of each residue, of shape Z,N,1,Vx,Vy,Vz
		'''

		Z, N, Cin, Vx, Vy, Vz = voxels.shape 

		# reshape to be compatible w/ torch convolutions
		voxels = voxels.reshape(Z*N, Cin, Vx, Vy, Vz)

		# add channels
		features = self.featurizer(voxels)

		# downsample
		for downsample in self.downsamples:
			features = downsample(features)

		# project to latent params
		latent_params = self.latent_proj(features).reshape(Z, N, -1, 1, 1, 1)

		# split into mu and logvar
		latent_mu, latent_logvar = torch.chunk(latent_params, chunks=2, dim=2)

		# sample a latent
		latent = latent_mu + torch.randn_like(latent_logvar)*torch.exp(0.5*latent_logvar)

		return latent, latent_mu, latent_logvar


class VAEDecoder(nn.Module):
	def __init__(self, voxel_dim=16, d_model=256, d_latent=16, resnet_layers=1):
		super().__init__()

		num_upsample = math.log(voxel_dim, 2)
		d_final = d_model // voxel_dim

		self.featurizer = nn.Sequential(   # increase channels, keep spatial res
											nn.Conv3d(d_latent, d_model, 2, stride=1, padding='same', bias=False),
											nn.GroupNorm(d_model//8, d_model),
											nn.SiLU(),
											ResNet(d_model=d_model,kernel_size=2,layers=resnet_layers)
										)
				
		self.upsamples = nn.ModuleList([nn.Sequential(
														nn.ConvTranspose3d(d_model//(2**i), d_model//(2**(i+1)), 2, stride=2, padding=0, output_padding=0, bias=False),
														nn.GroupNorm(d_model//(2**(i+1))//8, d_model//(2**(i+1))),
														nn.SiLU(),
														ResNet(d_model=d_model//(2**(i+1)),kernel_size=2,layers=resnet_layers),
													) 
										for i in range(int(num_upsample))
									])

		# reconstruct the final voxel
		self.voxel_proj = nn.Conv3d(d_final, 1, 1, stride=1, padding="same", bias=False)

	def forward(self, latent):
		'''
		latent (torch.Tensor): latent voxels of each residue, of shape Z,N,4,4,4,4
		'''

		Z, N, Cin, Vx, Vy, Vz = latent.shape 

		# reshape to be compatible w/ torch convolutions, no cross talk, so simply flattent the Z,N part,
		latent = latent.reshape(Z*N, Cin, Vx, Vy, Vz)

		features = self.featurizer(latent)

		for upsample in self.upsamples:
			features = upsample(features)

		voxels = self.voxel_proj(features)

		ZN, Cout, Vx, Vy, Vz = voxels.shape

		voxels = voxels.reshape(Z, N, Cout, Vx, Vy, Vz)

		return voxels