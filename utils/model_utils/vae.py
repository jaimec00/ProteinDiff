
import torch
import torch.nn as nn
from utils.model_utils.base_modules import ResNet

class VAE(nn.Module):
	def __init__(self):
		super().__init__()

		self.enc = VAEEncoder()
		self.dec = VAEDecoder()

	def forward(self, divergence):
		latent, latent_mu, latent_logvar = self.enc(divergence)
		divergence_pred = self.dec(latent)

		return latent, latent_mu, latent_logvar, divergence_pred

class VAEEncoder(nn.Module):
	def __init__(self):
		super().__init__()


		self.encoder = nn.Sequential(   
										# increase channels, keep spatial res
										nn.Conv3d(1, 16, 2, stride=1, padding='same', bias=False),
										nn.GroupNorm(2, 16),
										nn.SiLU(),

										ResNet(d_model=16,kernel_size=2,layers=1),

										# downsample
										nn.Conv3d(16, 32, 2, stride=2, padding=0, bias=False),
										nn.GroupNorm(4, 32),
										nn.SiLU(),

										ResNet(d_model=32,kernel_size=2,layers=1),

										# downsample
										nn.Conv3d(32, 64, 2, stride=2, padding=0, bias=False),
										nn.GroupNorm(8, 64),
										nn.SiLU(),

										ResNet(d_model=64,kernel_size=2,layers=1),

										# downsample
										nn.Conv3d(64, 128, 2, stride=2, padding=0, bias=False),
										nn.GroupNorm(16, 128),
										nn.SiLU(),

										ResNet(d_model=128,kernel_size=2,layers=1),

										# downsample
										nn.Conv3d(128, 256, 2, stride=2, padding=0, bias=False),
										nn.GroupNorm(32, 256),
										nn.SiLU(),

										ResNet(d_model=256,kernel_size=1,layers=1),

										# project to latent params
										nn.Conv3d(256, 32, 1, stride=1, padding="same", bias=False)
									)

	def forward(self, fields):
		'''
		fields (torch.Tensor): full voxels of each residue, of shape Z,N,3,Vx,Vy,Vz
		there is no cross talk, each residue is operated on independantly
		'''

		Z, N, Cin, Vx, Vy, Vz = fields.shape 

		# reshape to be compatible w/ torch convolutions, no cross talk, so simply flattent the Z,N part,
		fields = fields.view(Z*N, Cin, Vx, Vy, Vz)

		# get latent params
		latent_params = self.encoder(fields)
		
		# reshape to Z,N,2*Cout,4x4x4
		_, two_C_out, zx, zy, zz = latent_params.shape
		latent_params = latent_params.view(Z, N, two_C_out, zx, zy, zz)

		# split into mu and logvar
		z_mu, z_logvar = torch.chunk(latent_params, chunks=2, dim=2)

		# sample a latent
		z = z_mu + torch.randn_like(z_logvar)*torch.exp(0.5*z_logvar)

		return z, z_mu, z_logvar


class VAEDecoder(nn.Module):
	def __init__(self):
		super().__init__()


		self.decoder = nn.Sequential(   
										# increase channels, keep spatial res at 1x1x1
										nn.Conv3d(16, 256, 1, stride=1, padding='same', bias=False),
										nn.GroupNorm(32, 256),
										nn.SiLU(),

										ResNet(d_model=256,kernel_size=1,layers=1),

										# upsample to 2x2x2
										nn.ConvTranspose3d(256, 128, 2, stride=2, padding=0, output_padding=0, bias=False),
										nn.GroupNorm(16, 128),
										nn.SiLU(),

										ResNet(d_model=128,kernel_size=2,layers=1),

										# upsample to 4x4x4
										nn.ConvTranspose3d(128, 64, 2, stride=2, padding=0, output_padding=0, bias=False),
										nn.GroupNorm(8, 64),
										nn.SiLU(),

										ResNet(d_model=64,kernel_size=2,layers=1),

										# upsample to 8x8x8
										nn.ConvTranspose3d(64, 32, 2, stride=2, padding=0, output_padding=0, bias=False),
										nn.GroupNorm(4, 32),
										nn.SiLU(),

										ResNet(d_model=32,kernel_size=2,layers=1),

										# upsample to 8x8x8
										nn.ConvTranspose3d(32, 16, 2, stride=2, padding=0, output_padding=0, bias=False),
										nn.GroupNorm(2, 16),
										nn.SiLU(),

										ResNet(d_model=16,kernel_size=2,layers=1),

										# reconstruct the final voxel
										nn.Conv3d(16, 1, 1, stride=1, padding="same", bias=False)
									)

	def forward(self, latent):
		'''
		latent (torch.Tensor): latent voxels of each residue, of shape Z,N,4,4,4,4
		there is no cross talk, each residue is operated on independantly
		'''

		Z, N, Cin, Vx, Vy, Vz = latent.shape 

		# reshape to be compatible w/ torch convolutions, no cross talk, so simply flattent the Z,N part,
		latent = latent.view(Z*N, Cin, Vx, Vy, Vz)

		# reconstruct the fields and reshape
		fields = self.decoder(latent)
		ZN, Cout, Vx, Vy, Vz = fields.shape 
		fields = fields.view(Z, N, Cout, Vx, Vy, Vz)
		
		return fields

