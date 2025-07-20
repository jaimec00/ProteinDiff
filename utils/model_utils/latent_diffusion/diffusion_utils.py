import torch
import torch.nn as nn
from utils.model_utils.base_modules import MLP
from data.constants import alphabet
import math 

class NodeDenoiser(nn.Module):
	def __init__(self, d_model=128, top_k=32, layers=3):
		super().__init__()

		min_rbf, max_rbf, num_rbf = 2.0,22.0,16
		self.register_buffer("rbf_centers", torch.linspace(min_rbf, max_rbf, num_rbf))
		self.spread = (max_rbf - min_rbf) / num_rbf
		self.edge_norm = nn.LayerNorm(4*4*num_rbf)
		self.edge_proj = nn.Linear(4*4*num_rbf, d_model)
		# self.encs = nn.ModuleList([MPNN(d_model, update_edges=(i<(layers-1))) for i in range(layers)])
		self.encs = nn.ModuleList([DiT(d_model, heads=4, update_edges=(i<(layers-1))) for i in range(layers)])
		self.top_k = top_k

	def forward(self, nodes, t, edges, nbrs, nbr_mask):

		for enc in self.encs:
			nodes, edges = torch.utils.checkpoint.checkpoint(enc, nodes, edges, nbrs, t, nbr_mask, use_reentrant=False)

		return nodes

	def get_constants(self, C_backbone, valid_mask):
		nbrs, nbr_mask = self.get_neighbors(C_backbone, valid_mask)
		edges = self.get_edges(C_backbone, nbrs)

		return edges, nbrs, nbr_mask

	def get_neighbors(self, C, valid_mask):

		# prep
		Ca = C[:, :, 1, :]
		Z, N, S = Ca.shape
		assert N > self.top_k

		# get distances
		dists = torch.sqrt(torch.sum((Ca.unsqueeze(1) - Ca.unsqueeze(2))**2, dim=3)) # Z x N x N
		dists = torch.where((dists==0) | (~valid_mask).unsqueeze(2), float("inf"), dists) # Z x N x N
		
		# get topk 
		nbrs = dists.topk(self.top_k, dim=2, largest=False) # Z x N x K

		# masked nodes have themselves as edges, masked edges are the corresponding node
		node_idxs = torch.arange(N, device=dists.device).view(1,-1,1) # 1 x N x 1
		nbr_mask = valid_mask.unsqueeze(2) & torch.gather(valid_mask.unsqueeze(2).expand(-1,-1,self.top_k), 1, nbrs.indices)
		nbr_mask = nbr_mask & (nbrs.values!=0) # exclude self and distant neighbors
		nbrs = torch.where(nbr_mask, nbrs.indices, node_idxs) # Z x N x K

		return nbrs, nbr_mask

	def get_edges(self, C_backbone, nbrs):

		Z, N, A, S = C_backbone.shape
		_, _, K = nbrs.shape

		C_nbrs = torch.gather(C_backbone.unsqueeze(2).expand(Z, N, K, A, S), 1, nbrs.view(Z,N,K,1,1).expand(Z, N, K, A, S)) # Z,N,K,A,S

		dists = torch.sqrt(torch.sum((C_backbone.view(Z,N,1,A,1,S) - C_nbrs.view(Z,N,K,1,A,S))**2, dim=-1)) # Z,N,1,A,1,S - Z,N,K,1,A,S --> Z,N,K,A,A

		rbf_numerator = (dists.view(Z, N, K, A, A, 1) - self.rbf_centers.view(1,1,1,1,1,-1))**2 # Z,N,K,A,A,S

		rbf = torch.exp(-rbf_numerator / (self.spread**2)).reshape(Z,N,K,-1)

		edges = self.edge_proj(self.edge_norm(rbf))

		return edges

class DiT(nn.Module):
	def __init__(self, d_model=128, update_edges=True, heads=4):
		super().__init__()

		self.attn = SelfAttention(d_model=d_model, heads=heads)
		self.attn_norm = adaLN_Zero(d_in=d_model, d_gammabeta=d_model, d_alpha=d_model)

		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=4*d_model, hidden_layers=0, act="silu", dropout=0.0)
		self.ffn_norm = adaLN_Zero(d_in=d_model, d_gammabeta=d_model, d_alpha=d_model)

		self.update_edges = update_edges
		if update_edges:
			self.edge_update = MLP(d_in=d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, act="silu", dropout=0.0)
			self.edge_norm = adaLN_Zero(d_in=d_model, d_gammabeta=d_model, d_alpha=d_model)
	
		self.norm = StaticLayerNorm(d_model)

	def forward(self, nodes, edges, nbrs, t, nbr_mask):

		# conditioning
		alpha1, gamma1, beta1 = self.attn_norm(t)
		alpha2, gamma2, beta2 = self.ffn_norm(t)

		# norm and conditioning
		nodes_i = gamma1*self.norm(nodes) + beta1
		
		# gather neighbors, add edges as positional encoding
		_, nodes_j = self.gather_nodes(nodes_i, nbrs)
		nodes_j = nodes_j + edges
		
		# attn
		nodes = nodes + alpha1*self.attn(nodes_i, nodes_j, nbr_mask)

		# ffn
		nodes2 = gamma2*self.norm(nodes) + beta2
		nodes = nodes + alpha2*self.ffn(nodes2)

		# update edges
		if self.update_edges:
			alpha3, gamma3, beta3 = self.edge_norm(t.unsqueeze(2))
			nodes_i, nodes_j = self.gather_nodes(nodes, nbrs)
			# edge_pre = torch.cat([nodes_i, nodes_j, gamma3*self.norm(edges) + beta3], dim=-1)
			edge_pre = gamma3*self.norm(edges + nodes_i + nodes_j) + beta3
			edges = edges + alpha3*self.edge_update(edge_pre)

		return nodes, edges

	def gather_nodes(self, V, K):

		dimZ, dimN, dimDv = V.shape
		_, _, dimK = K.shape

		# gather neighbor nodes
		Vi = V.unsqueeze(2).expand(-1,-1,dimK,-1) # Z x N x K x Dv
		Ki = K.unsqueeze(3).expand(-1,-1,-1,dimDv) # Z x N x K x Dv
		Vj = torch.gather(Vi, 1, Ki) # Z x N x K x Dv

		return Vi, Vj

class SelfAttention(nn.Module):
	def __init__(self, d_model=256, heads=4):
		super().__init__()

		self.H = heads
		self.Dm = d_model
		self.Dk = d_model // heads

		xavier_scale = (6/(self.Dk + d_model))**0.5

		self.q_proj = nn.Parameter(-xavier_scale + torch.rand(self.H, self.Dm, self.Dk) * (2*xavier_scale)) # H x Dm x Dk
		self.k_proj = nn.Parameter(-xavier_scale + torch.rand(self.H, self.Dm, self.Dk) * (2*xavier_scale)) # H x Dm x Dk
		self.v_proj = nn.Parameter(-xavier_scale + torch.rand(self.H, self.Dm, self.Dk) * (2*xavier_scale)) # H x Dm x Dk

		self.q_bias = nn.Parameter(torch.zeros(self.H, self.Dk)) # H x Dk
		self.k_bias = nn.Parameter(torch.zeros(self.H, self.Dk)) # H x Dk
		self.v_bias = nn.Parameter(torch.zeros(self.H, self.Dk)) # H x Dk

		self.out_proj = nn.Linear(d_model, d_model, bias=False)

	def forward(self, nodes_i, nodes_j, nbr_mask):

		# convenience
		Z, N, Ki, Dm = nodes_j.shape
		H = self.H
		Dk = self.Dk

		# project the tensors, doing reshape for readability
		Q = torch.matmul(nodes_i.reshape(Z,1,N,Dm), self.q_proj.reshape(1,H,Dm,Dk)) + self.q_bias.reshape(1,H,1,Dk) # Z,1,N,Dm@1,H,Dm,Dk->Z,H,N,Dk
		K = torch.matmul(nodes_j.reshape(Z,1,N,Ki,Dm), self.k_proj.reshape(1,H,1,Dm,Dk)) + self.k_bias.reshape(1,H,1,1,Dk) # Z,1,N,K,Dm@1,H,1,Dm,Dk->Z,H,N,K,Dk
		V = torch.matmul(nodes_j.reshape(Z,1,N,Ki,Dm), self.v_proj.reshape(1,H,1,Dm,Dk)) + self.v_bias.reshape(1,H,1,1,Dk) # Z,1,N,K,Dm@1,H,1,Dm,Dk->Z,H,N,K,Dk

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

class BetaScheduler(nn.Module):
	def __init__(self, t_max, beta_min=1e-4, beta_max=2e-2):
		super().__init__()
		self.t_max = t_max
		self.beta_min = beta_min 
		self.beta_max = beta_max 
		self.register_buffer("abars", torch.cumprod(1 - torch.linspace(self.beta_min, self.beta_max, self.t_max), dim=0))

	def get_abars(self, t):
		abars = torch.gather(self.abars.unsqueeze(1).expand(-1, t.size(0)), 0, t.unsqueeze(0).expand(self.abars.size(0), -1)).max(dim=0).values
		return abars

	def get_betas(self, t):
		betas = self.beta_min + (t/self.t_max)*(self.beta_max - self.beta_min)
		return betas

	def forward(self, t):
		abars = self.get_abars(t)
		betas = self.get_betas(t)
		return abars, betas


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

class MPNN(nn.Module):
	def __init__(self, d_model=128, update_edges=True):
		super().__init__()

		self.node_msgr = MLP(d_in=3*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=0.0, act="silu", zeros=False)
		self.node_msgr_norm = adaLN_Zero(d_in=d_model, d_gammabeta=3*d_model, d_alpha=d_model)

		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=d_model*4, hidden_layers=0, dropout=0.0, act="silu", zeros=False)
		self.ffn_norm = adaLN_Zero(d_in=d_model, d_gammabeta=d_model, d_alpha=d_model)

		self.update_edges = update_edges
		if update_edges:
			self.edge_msgr = MLP(d_in=3*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=0.0, act="silu", zeros=False)
			self.edge_msgr_norm = adaLN_Zero(d_in=d_model, d_gammabeta=3*d_model, d_alpha=d_model)

		self.norm = StaticLayerNorm(d_model)

	def forward(self, V, E, K, t, nbr_mask):

		gamma1, beta1, alpha1 = self.node_msgr_norm(t)
		gamma2, beta2, alpha2 = self.ffn_norm(t)

		# prepare message
		Mv_pre = gamma1.unsqueeze(2) * self.norm(self.prepare_message(V, E, K)) + beta1.unsqueeze(2) # Z x N x K x 3*d_model

		# process the message
		Mv = alpha1 * torch.sum(self.node_msgr(Mv_pre) * nbr_mask.unsqueeze(3), dim=2) # Z x N x d_model

		# send the message
		V = gamma2 * self.norm(V + Mv) + beta2 # Z x N x d_model

		# process the updated node
		V = V + alpha2 * self.ffn(V)

		if not self.update_edges:
			return V, E

		gamma3, beta3, alpha3 = self.edge_msgr_norm(t.unsqueeze(2)) # add the nbr dimension

		# prepare message
		Me_pre = gamma3 * self.norm(self.prepare_message(V, E, K)) + beta3

		# process the message
		Me = alpha3 * self.edge_msgr(Me_pre) * nbr_mask.unsqueeze(3) # Z x N x K x d_model

		# update the edges
		E = E + Me # Z x N x K x d_model

		return V, E

	def prepare_message(self, V, E, K):

		# gathe neighbor nodes
		Vi, Vj = self.gather_nodes(V, K) # Z x N x K x d_model

		# cat the node and edge tensors to create the message
		Mv_pre = torch.cat([Vi, Vj, E], dim=3) # Z x N x K x (3*d_model)

		return Mv_pre

	def gather_nodes(self, V, K):

		dimZ, dimN, dimDv = V.shape
		_, _, dimK = K.shape

		# gather neighbor nodes
		Vi = V.unsqueeze(2).expand(-1,-1,dimK,-1) # Z x N x K x Dv
		Ki = K.unsqueeze(3).expand(-1,-1,-1,dimDv) # Z x N x K x Dv
		Vj = torch.gather(Vi, 1, Ki) # Z x N x K x Dv

		return Vi, Vj

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
		std = std.masked_fill(std==0, 1)
		return centered / std