import torch
import torch.nn as nn
from data.constants import canonical_aas
from utils.model_utils.base_modules import ResNet

class Classifier(nn.Module):
	def __init__(self):
		super().__init__()

		# start at 3x16x16x16
		self.classifier = nn.Sequential(
										# increase channels, keep spatial res at 16x16x16
										nn.Conv3d(1, 16, 2, stride=1, padding='same', bias=False),
										nn.GroupNorm(2, 16),
										nn.SiLU(),

										ResNet(d_model=16, kernel_size=2, layers=1),

										# downsample 8x8x8
										nn.Conv3d(16, 32, 2, stride=2, padding=0, bias=False),
										nn.GroupNorm(4, 32),
										nn.SiLU(),

										ResNet(d_model=32, kernel_size=2, layers=1),

										# downsample 4x4x4
										nn.Conv3d(32, 64, 2, stride=2, padding=0, bias=False),
										nn.GroupNorm(8, 64),
										nn.SiLU(),

										ResNet(d_model=64, kernel_size=2, layers=1),

										# downsample 2x2x2
										nn.Conv3d(64, 128, 2, stride=2, padding=0, bias=False),
										nn.GroupNorm(16, 128),
										nn.SiLU(),
										
										ResNet(d_model=128, kernel_size=2, layers=1),

										# downsample 1x1x1
										nn.Conv3d(128, 256, 2, stride=2, padding=0, bias=False),
										nn.GroupNorm(16, 256),

										ResNet(d_model=256, kernel_size=2, layers=1),

									) 

		self.classify = nn.Linear(256, len(canonical_aas))


	def forward(self, fields):

		Z, N, Cin, Vx, Vy, Vz = fields.shape
		fields = self.classifier(fields.reshape(Z*N, Cin, Vx, Vy, Vz))

		# output is Z*N, Cout, 1,1,1 so reshape to Z,N,Cout
		fields = fields.reshape(Z, N, -1)

		# project to amino acids, Z,N,20
		aas = self.classify(fields)

		return aas