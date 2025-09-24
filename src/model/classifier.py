import torch
import torch.nn as nn
from data.constants import canonical_aas
from utils.model_utils.base_modules import ResNet
import math

class Classifier(nn.Module):
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

		Z, N, Cin, Vx, Vy, Vz = voxels.shape
		features = self.featurizer(voxels.reshape(Z*N, Cin, Vx, Vy, Vz))

		for downsample in self.downsamples:
			features = downsample(features)

		# output is Z*N, Cout, 1,1,1 so reshape to Z,N,Cout
		features = features.reshape(Z, N, -1)

		# project to amino acids, Z,N,20
		aas = self.classify(features)

		return aas