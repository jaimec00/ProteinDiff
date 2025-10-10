import torch
import torch.nn as nn
from utils.model_utils.base_modules import MLP, FlashMHA
import math 

class NodeDenoiser(nn.Module):
	def __init__(self, d_model=128, heads=8, layers=3):
		super().__init__()

		# encoder
		self.encs = nn.ModuleList([DiT(d_model=d_model, heads=heads) for _ in range(layers)])

	def forward(self, latent, condition, cu_seqlens, max_seqlen):

		for enc in self.encs:
			latent = enc(latent, condition, cu_seqlens, max_seqlen)

		return latent

class DiT(nn.Module):
	def __init__(self, d_model=128, heads=4):
		super().__init__()

		self.attn = FlashMHA(d_model=d_model, heads=heads)
		self.attn_norm = adaLN_Zero(d_in=d_model, d_gammabeta=d_model, d_alpha=d_model)

		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=4*d_model, hidden_layers=0, act="silu", dropout=0.0)
		self.ffn_norm = adaLN_Zero(d_in=d_model, d_gammabeta=d_model, d_alpha=d_model)

		self.norm = StaticLayerNorm(d_model)

	def forward(self, latent, condition, cu_seqlens, max_seqlen):

		# conditioning
		alpha1, gamma1, beta1 = self.attn_norm(condition)
		alpha2, gamma2, beta2 = self.ffn_norm(condition)

		# norm and conditioning
		latent = gamma1*self.norm(latent) + beta1
		
		# attn
		latent = latent + alpha1*self.attn(latent, cu_seqlens, max_seqlen)

		# ffn
		latent2 = gamma2*self.norm(latent) + beta2
		latent = latent + alpha2*self.ffn(latent2)

		return latent

class FiLM(nn.Module):
	def __init__(self, d_in=256, d_model=256):
		super().__init__()
		self.gamma_beta = MLP(d_in=d_in, d_out=2*d_model, d_hidden=d_model, hidden_layers=1, dropout=0.0, act="silu", zeros=False)
		
	def forward(self, x, condition):
		gamma_beta = self.gamma_beta(condition)
		gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
		x = gamma*x + beta
		return x

class adaLN_Zero(nn.Module):
	'''adaptive layer norm to perform affine transformation conditioned on timestep and nodes. adaLNzero, where initialized to all zeros'''
	def __init__(self, d_in=128, d_gammabeta=128, d_alpha=128):
		super().__init__()
		self.gamma_beta = MLP(d_in=d_in, d_out=2*d_gammabeta, d_hidden=d_gammabeta, hidden_layers=1, dropout=0.0, act="silu", zeros=False)
		self.alpha = MLP(d_in=d_in, d_out=d_alpha, d_hidden=d_alpha, hidden_layers=1, dropout=0.0, act="silu", zeros=True)

	def forward(self, x):
		gamma, beta = torch.chunk(self.gamma_beta(x), chunks=2, dim=-1)
		alpha = self.alpha(x)
		return gamma, beta, alpha

class StaticLayerNorm(nn.Module):
	'''just normalizes each token to have a mean of 0 and var of 1, no scaling and shifting'''
	def __init__(self, d_model):
		super().__init__()
		self.d_model = d_model
	def forward(self, x):
		centered = x - x.mean(dim=-1, keepdim=True) 
		std = centered.std(dim=-1, keepdim=True)
		normed = centered / std.masked_fill(std==0, 1)
		return normed

class CosineScheduler(nn.Module):
	def __init__(self, t_max, s=0.008):
		super().__init__()
		self.t_max = t_max
		self.s = s
		
	def get_abars(self, t):
		stage = (t+1)/self.t_max
		abars = torch.cos(torch.pi*0.5*(stage+self.s)/(1+self.s))**2 / math.cos(torch.pi*0.5*self.s/(1+self.s))**2
		return abars

	def get_betas(self, t):
		abars = self.get_abars(t)
		abars_tminus1 = self.get_abars(t-1)
		betas = 1 - (abars/abars_tminus1)
		return betas

	def forward(self, t):
		abars = self.get_abars(t)
		betas = self.get_betas(t)
		return abars, betas

