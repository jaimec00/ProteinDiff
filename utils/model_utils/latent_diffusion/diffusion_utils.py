import torch
import torch.nn as nn
from utils.model_utils.base_modules import MLP
from data.constants import alphabet
import math 

class NodeDenoiser(nn.Module):
	def __init__(self, d_model=128, top_k=32, layers=3, min_rbf=2.0, max_rbf=22.0, num_rbf=16):
		super().__init__()

		self.rbf_norm = nn.LayerNorm(num_rbf*4*4)
		self.edge_proj = nn.Linear(num_rbf*4*4 + 9, d_model)
		cluster_size = 6
		self.encs = nn.ModuleList([DiTCluster(d_model, heads=4, cluster_size=min(layers, cluster_size*(i+1)) - cluster_size*i) for i in range(0,layers,cluster_size)])
		self.top_k = top_k
		self.register_buffer("rbf_centers", torch.linspace(min_rbf, max_rbf, num_rbf))
		self.spread = (max_rbf - min_rbf) / num_rbf

	def forward(self, nodes, t, edges, nbrs, nbr_mask):

		for enc in self.encs:
			# nodes = torch.utils.checkpoint.checkpoint(enc, nodes, edges, nbrs, t, nbr_mask, use_reentrant=False)
			nodes = enc(nodes, edges, nbrs, t, nbr_mask)

		return nodes

	def get_constants(self, C_backbone, frames, valid_mask):

		nbrs, nbr_mask = self.get_neighbors(C_backbone, valid_mask)
		edges = self.get_edges(C_backbone, frames, nbrs)

		return edges, nbrs, nbr_mask

	def get_neighbors(self, C, valid_mask):

		# prep
		Ca = C[:, :, 1, :]
		Z, N, S = Ca.shape
		assert N > self.top_k

		# get distances
		dists = torch.sqrt(torch.sum((Ca.unsqueeze(1) - Ca.unsqueeze(2))**2, dim=3)) # Z x N x N
		dists = torch.where(valid_mask.unsqueeze(1) | valid_mask.unsqueeze(2), dists, float("inf")) # Z x N x N
		
		# get topk 
		nbrs = dists.topk(self.top_k, dim=2, largest=False) # Z x N x K

		# masked nodes have themselves as edges, masked edges are the corresponding node
		node_idxs = torch.arange(N, device=dists.device).view(1,-1,1) # 1 x N x 1
		nbr_mask = valid_mask.unsqueeze(2) & torch.gather(valid_mask.unsqueeze(2).expand(-1,-1,self.top_k), 1, nbrs.indices)
		nbrs = torch.where(nbr_mask, nbrs.indices, node_idxs) # Z x N x K

		return nbrs, nbr_mask

	def get_edges(self, C_backbone, frames, nbrs):

		Z, N, A, S = C_backbone.shape
		_, _, K = nbrs.shape

		C_nbrs = torch.gather(C_backbone.unsqueeze(2).expand(Z, N, K, A, S), 1, nbrs.view(Z,N,K,1,1).expand(Z, N, K, A, S)) # Z,N,K,A,S

		dists = torch.sqrt(torch.sum((C_backbone.view(Z,N,1,A,1,S) - C_nbrs.view(Z,N,K,1,A,S))**2, dim=-1)) # Z,N,1,A,1,S - Z,N,K,1,A,S --> Z,N,K,A,A

		rbf_numerator = (dists.view(Z, N, K, A, A, 1) - self.rbf_centers.view(1,1,1,1,1,-1))**2 # Z,N,K,A,A,S

		rbf = torch.exp(-rbf_numerator / (self.spread**2)).reshape(Z,N,K,-1)

		# testing if including the relative frames helps
		frame_nbrs = torch.gather(frames.unsqueeze(2).expand(Z, N, K, 3, 3), 1, nbrs.reshape(Z, N, K, 1, 1).expand(Z, N, K, 3, 3))
		relative_frames = torch.matmul(frames.unsqueeze(2).transpose(-2,-1), frame_nbrs).reshape(Z, N, K, -1)

		edges = torch.cat([self.rbf_norm(rbf), relative_frames], dim=3) # Z,N,K,256+9=265

		edges = self.edge_proj(edges)

		return edges

class DiTCluster(nn.Module):
	'''
	meant to group DiT layers so that checkpoint only saves layers//clustersize inputs for recomputation in bwd pass
	'''
	def __init__(self, d_model=256, heads=4, cluster_size=4):
		super().__init__()
		
		self.DiTs = nn.ModuleList([DiT(d_model, heads=4) for _ in range(cluster_size)])

	def forward(self, nodes, edges, nbrs, t, nbr_mask):
		for DiT in self.DiTs:
			nodes = DiT(nodes, edges, nbrs, t, nbr_mask)
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

	def gather_nodes(self, V, K):

		dimZ, dimN, dimDv = V.shape
		_, _, dimK = K.shape

		# gather neighbor nodes
		Vi = V.unsqueeze(2).expand(-1,-1,dimK,-1) # Z x N x K x Dv
		Ki = K.unsqueeze(3).expand(-1,-1,-1,dimDv) # Z x N x K x Dv
		Vj = torch.gather(Vi, 1, Ki) # Z x N x K x Dv

		return Vi, Vj

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

