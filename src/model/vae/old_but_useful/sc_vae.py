
import torch
import torch.nn as nn
from model.vae.vae_utils import ResNet
from static.constants import canonical_aas
import math


class SideChainVAE(nn.Module):
	def __init__(self, voxel_dim=16, d_model=256, d_latent=16, resnet_enc_layers=3, resnet_dec_layers=3, resnet_class_layers=3):
		super().__init__()

		self.enc = SideChainEncoder(voxel_dim=voxel_dim, d_model=d_model, d_latent=d_latent, resnet_layers=resnet_enc_layers)
		self.dec = SideChainDecoder(voxel_dim=voxel_dim, d_model=d_model, d_latent=d_latent, resnet_layers=resnet_dec_layers)
		self.classifier = SideChainClassifier(voxel_dim=voxel_dim, d_model=d_model, resnet_layers=resnet_class_layers)

	def forward(self, voxels):

		# get side chain latent
		latent, latent_mu, latent_logvar = self.enc(voxels)

		# reconstruct voxels
		voxels_pred = self.dec(latent)

		# predict seq from reconstructed voxels. detach dec output so classifier optimized independantly
		seq_pred = self.classifier(voxels_pred.detach())

		return latent, latent_mu, latent_logvar, voxels_pred, seq_pred

class SideChainEncoder(nn.Module):
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
		voxels (torch.Tensor): full voxels of each residue, of shape ZN,1,Vx,Vy,Vz
		'''

		# add channels
		features = self.featurizer(voxels)

		# downsample
		for downsample in self.downsamples:
			features = downsample(features)

		# project to latent params
		latent_params = self.latent_proj(features)

		# split into mu and logvar
		latent_mu, latent_logvar = torch.chunk(latent_params, chunks=2, dim=1)

		# sample a latent
		latent = latent_mu + torch.randn_like(latent_logvar)*torch.exp(0.5*latent_logvar)

		return latent, latent_mu, latent_logvar


class SideChainDecoder(nn.Module):
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
		latent (torch.Tensor): latent voxels of each residue, of shape ZN,d_latent,1,1,1
		'''

		features = self.featurizer(latent)

		for upsample in self.upsamples:
			features = upsample(features)

		voxels = self.voxel_proj(features)

		return voxels

class SideChainClassifier(nn.Module):
	def __init__(self, voxel_dim=16, d_model=256, resnet_layers=3):
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

		self.classify = nn.Linear(d_model, len(canonical_aas))

	def forward(self, voxels):

		ZN = voxels.size(0)
		features = self.featurizer(voxels)

		for downsample in self.downsamples:
			features = downsample(features)

		# output is Z*N, Cout, 1,1,1 so reshape to ZN,Cout
		features = features.reshape(ZN, -1)

		# project to amino acids, ZN,20
		aas = self.classify(features)

		return aas