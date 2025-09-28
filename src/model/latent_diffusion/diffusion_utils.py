import torch
import torch.nn as nn
from utils.model_utils.base_modules import MLP
from data.constants import alphabet
import math 

class NodeDenoiser(nn.Module):
	def __init__(self, d_model=128, layers=3):
		super().__init__()

		# will cat the node, edge, and t embeddings and pass through mlp so each node only updates its view of its nbrs (including self)
		self.view_mlp = MLP(d_in=d_model*3, d_out=d_model, d_hidden=d_model, hidden_layers=2, ac="silu")

		# encoder
		self.encs = nn.ModuleList([DiT(d_model, heads=4) for _ in range(layers)])

	def forward(self, nodes, t, edges, nbrs, nbr_mask):

		# convert ZxNxD to ZxNxKxD, ie each node gets a view of its nearest neighbors, based on edge and t embeddings
		nodes, t = self.get_views(nodes, edges, t, nbrs)

		for enc in self.encs:
			# nodes = torch.utils.checkpoint.checkpoint(enc, nodes, t, nbr_mask, use_reentrant=False)
			nodes = enc(nodes, t, nbr_mask)

		# get the 0th dim (self)
		nodes = torch.gather(nodes, 2, torch.zeros_like(nodes))

		return nodes
	


class DiT(nn.Module):
	def __init__(self, d_model=128, heads=4):
		super().__init__()

		self.attn = GraphAttention(d_model=d_model, heads=heads)
		self.attn_norm = adaLN_Zero(d_in=d_model, d_gammabeta=d_model, d_alpha=d_model)

		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=4*d_model, hidden_layers=0, act="silu", dropout=0.0)
		self.ffn_norm = adaLN_Zero(d_in=d_model, d_gammabeta=d_model, d_alpha=d_model)

		self.norm = StaticLayerNorm(d_model)

	def forward(self, nodes, edges, nbrs, t, nbr_mask):

		# conditioning
		alpha1, gamma1, beta1 = self.attn_norm(t)
		alpha2, gamma2, beta2 = self.ffn_norm(t)

		# norm and conditioning
		nodes_i = gamma1*self.norm(nodes) + beta1
		
		# gather neighbors, add edges as positional encoding
		_, nodes_j = self.gather_nodes(nodes_i, nbrs)
		
		# attn
		nodes = nodes + alpha1*self.attn(nodes_i, edges, nodes_j, nbr_mask)

		# ffn
		nodes2 = gamma2*self.norm(nodes) + beta2
		nodes = nodes + alpha2*self.ffn(nodes2)

		return nodes


class GraphAttention(nn.Module):
	def __init__(self, d_model=256, heads=4):
		super().__init__()

		self.H = heads
		self.Dm = d_model
		self.Dk = d_model // heads

		xavier_scale = (6/(self.Dk + d_model))**0.5

		self.edge_conditionK = FiLM(d_in=d_model, d_model=d_model)
		self.edge_conditionV = FiLM(d_in=d_model, d_model=d_model)

		self.q_proj = nn.Parameter(-xavier_scale + torch.rand(self.H, self.Dm, self.Dk) * (2*xavier_scale)) # H x Dm x Dk
		self.k_proj = nn.Parameter(-xavier_scale + torch.rand(self.H, self.Dm, self.Dk) * (2*xavier_scale)) # H x Dm x Dk
		self.v_proj = nn.Parameter(-xavier_scale + torch.rand(self.H, self.Dm, self.Dk) * (2*xavier_scale)) # H x Dm x Dk

		self.q_bias = nn.Parameter(torch.zeros(self.H, self.Dk)) # H x Dk
		self.k_bias = nn.Parameter(torch.zeros(self.H, self.Dk)) # H x Dk
		self.v_bias = nn.Parameter(torch.zeros(self.H, self.Dk)) # H x Dk

		self.out_proj = nn.Linear(d_model, d_model, bias=False)

	def forward(self, nodes_i, edges, nodes_j, nbr_mask):

		# convenience
		Z, N, Ki, Dm = nodes_j.shape
		H = self.H
		Dk = self.Dk

		q = nodes_i.reshape(Z,1,N,Dm)
		k = self.edge_conditionK(nodes_j, edges).reshape(Z,1,N,Ki,Dm)
		v = self.edge_conditionV(nodes_j, edges).reshape(Z,1,N,Ki,Dm)

		# project the tensors, doing reshape for readability
		Q = torch.matmul(q, self.q_proj.reshape(1,H,Dm,Dk)) + self.q_bias.reshape(1,H,1,Dk) # Z,1,N,Dm@1,H,Dm,Dk->Z,H,N,Dk
		K = torch.matmul(k, self.k_proj.reshape(1,H,1,Dm,Dk)) + self.k_bias.reshape(1,H,1,1,Dk) # Z,1,N,K,Dm@1,H,1,Dm,Dk->Z,H,N,K,Dk
		V = torch.matmul(v, self.v_proj.reshape(1,H,1,Dm,Dk)) + self.v_bias.reshape(1,H,1,1,Dk) # Z,1,N,K,Dm@1,H,1,Dm,Dk->Z,H,N,K,Dk

		# cretae attn mat
		S = torch.matmul(Q.unsqueeze(-2), K.transpose(-2,-1)) / ((Dk)**0.5) # Z,H,N,1,Dk @ Z,H,N,Dk,K -> Z,H,N,1,K
		attn_mask = ((~nbr_mask) & ~((~nbr_mask).all(dim=-1, keepdim=True))).reshape(Z,1,N,1,Ki) # attn mask, invalid tokens have themselves as neighbors, so deal with that
		P = torch.softmax(S.masked_fill(attn_mask, float("-inf")), dim=-1) # Z,H,N,1,K
		out = torch.matmul(P, V).squeeze(-2) # Z,H,N,1,K @ Z,H,N,K,Dk -> Z,H,N,1,Dk -> Z,H,N,Dk

		# cat heads and reshape
		out = out.permute(0,2,3,1).reshape(Z, N, Dm) # Z,N,Dk,H -> Z,N,Dm

		# project through final linear layer
		out = self.out_proj(out)

		# return
		return out # Z,N,Dm

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
		gamma_beta = self.gamma_beta(x)
		gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
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

