
import torch
import torch.nn as nn
from model.utils.base_modules import MLP, FlashMHA

class ResNet(nn.Module):
	def __init__(self, d_model=128, kernel_size=2, layers=1):

		super().__init__()

		self.convs = nn.ModuleList([	nn.Sequential(	nn.Conv3d(d_model, d_model, kernel_size, stride=1, padding="same", bias=False),
														nn.GroupNorm(max(d_model//16,1), d_model),
														nn.SiLU()
													) 
										for _ in range(layers)
									])

	def forward(self, latent):
		
		for conv in self.convs:
			latent = latent + conv(latent)

		return latent






class PairwiseProjHead(nn.Module):
    def __init__(self, d_model=256, d_down=128, dist_bins=64, angle_bins=16):
        super().__init__()
        self.downsample = nn.Linear(d_model, d_down)
        self.bin = MLP(d_in=d_down, d_hidden=dist_bins+angle_bins, d_out=dist_bins+angle_bins, hidden_layers=1, act="silu")
        self._dist_bins = dist_bins
        self._angle_bins = angle_bins

    def forward(self, x):
		# this will turn into a triton kernel soon, and then a cuda kernel
        q, k = torch.chunk(self.downsample(x), chunks=2, dim=-1) # ZN x D//2
        q_i, k_j = q.unsqueeze(0), k.unsqueeze(1)
        prod, diff = q_i*k_j, k_j-q_i
        pw = torch.cat([prod, diff], dim=-1) # ZN x ZN x D
        distogram, anglogram = torch.split(self.bin(pw), [self._dist_bins, self._angle_bins], dim=-1)
        return distogram, anglogram