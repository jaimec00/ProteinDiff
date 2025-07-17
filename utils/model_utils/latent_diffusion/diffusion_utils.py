import torch
import torch.nn as nn
from utils.model_utils.base_modules import MLP
from data.constants import alphabet

class StructureEncoder(nn.Module):
	def __init__(self, d_model=128, top_k=32, layers=3, dropout=0.0):
		super().__init__()

		# might try to intiialize context nodes to seq instead of zeros as conditioning, but not for now
		# seq_emb = nn.Embedding(len(alphabet), d_model)
		self.register_buffer("V_start", torch.zeros(d_model))

		min_rbf, max_rbf, num_rbf = 2.0,22.0,16
		self.register_buffer("rbf_centers", torch.linspace(min_rbf, max_rbf, num_rbf))
		self.spread = (max_rbf - min_rbf) / num_rbf

		self.edge_norm = nn.LayerNorm(4*4*num_rbf)
		self.edge_proj = nn.Linear(4*4*num_rbf, d_model)

		self.encs = nn.ModuleList([MPNN(d_model, update_edges=True, dropout=dropout) for _ in range(layers)])

		self.top_k = top_k

	def forward(self, C_backbone, valid_mask):

		Z, N, _, _ = C_backbone.shape

		V = self.V_start.view(1,1,-1).expand(Z, N, -1)
		nbrs, nbr_mask = self.get_neighbors(C_backbone, valid_mask)
		E = self.get_edges(C_backbone, nbrs)

		for enc in self.encs:
			V, E = enc(V, E, nbrs, nbr_mask)

		return V, E, nbrs, nbr_mask

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

class MPNN(nn.Module):
	def __init__(self, d_model=128, update_edges=True, dropout=0.0):
		super().__init__()

		self.node_messenger = MLP(d_in=3*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=dropout, act="gelu", zeros=False)
		self.node_messenger_norm = nn.LayerNorm(d_model)

		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=d_model*4, hidden_layers=0, dropout=dropout, act="gelu", zeros=False)
		self.ffn_norm = nn.LayerNorm(d_model)

		self.update_edges = update_edges
		if update_edges:
			self.edge_messenger = MLP(d_in=3*d_model, d_out=d_model, d_hidden=d_model, hidden_layers=1, dropout=dropout, act="gelu", zeros=False)
			self.edge_messenger_norm = nn.LayerNorm(d_model)

		self.dropout = nn.Dropout(dropout)

	def forward(self, V, E, K, nbr_mask):

		# prepare message
		Mv_pre = self.prepare_message(V, E, K)

		# process the message
		Mv = self.node_messenger(Mv_pre) * nbr_mask.unsqueeze(3) # Z x N x K x d_model

		# send the message
		V = self.node_messenger_norm(V + self.dropout(Mv.sum(dim=2))) # Z x N x d_model

		# process the updated node
		V = self.ffn_norm(V + self.dropout(self.ffn(V)))

		if not self.update_edges:
			return V

		# prepare message
		Me_pre = self.prepare_message(V, E, K)

		# process the message
		Me = self.edge_messenger(Me_pre) * nbr_mask.unsqueeze(3) # Z x N x K x d_model

		# update the edges
		E = self.edge_messenger_norm(E + self.dropout(Me)) # Z x N x K x d_model

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


class LatentEncoder(nn.Module):
	def __init__(self, d_latent=4, d_model=128):
		super().__init__()

		self.feature_conv = nn.Sequential(
											nn.Conv3d(d_latent, d_model//4, 2, stride=1, padding="same", bias=False),
											nn.GroupNorm(d_model//64, d_model//4),
											nn.SiLU(),

											nn.Conv3d(d_model//4, d_model//2, 2, stride=1, padding="same", bias=False),
											nn.GroupNorm(d_model//32, d_model//2),
											nn.SiLU(),

											nn.Conv3d(d_model//2, d_model, 2, stride=1, padding="same", bias=False),
											nn.GroupNorm(d_model//16, d_model),
											nn.SiLU(),

											nn.Conv3d(d_model, d_model, 2, stride=1, padding="same", bias=False)
										)

	def forward(self, latent):
		latent = self.feature_conv(latent)
		return latent


class GraphUpdater(nn.Module):
	def __init__(self, d_model=128):
		super().__init__()

		self.collapse_latent = nn.Sequential(   
												# downsample to 1x1x1
												nn.Conv3d(d_model, d_model, 2, stride=2, padding=0, bias=False),
												nn.GroupNorm(d_model//16, d_model),
												nn.SiLU(),

												# downsample to 1x1x1
												nn.Conv3d(d_model, d_model, 2, stride=2, padding=0, bias=False),
												nn.GroupNorm(d_model//16, d_model),
												nn.SiLU(),

												# project one last time
												nn.Conv3d(d_model, d_model, 1, stride=1, padding="same", bias=True)
									)

		self.latent_conditioning = FiLM(d_model=d_model, d_condition=d_model)

		self.mpnn = MPNN(d_model=d_model, update_edges=True)

	def forward(self, latent, t, nodes, edges, nbrs, nbr_mask):
		
		# prep
		ZN, Cin, Vx, Vy, Vz = latent.shape
		Z, N, Cout = nodes.shape

		# reshape
		latent = self.collapse_latent(latent)
		latent = latent.view(Z, N, Cout) + t.reshape(Z,N,Cout)

		# condition nodes on latent
		nodes = self.latent_conditioning(nodes, latent)

		# message passing
		nodes, edges = self.mpnn(nodes, edges, nbrs, nbr_mask)

		return nodes, edges


class DiT(nn.Module):
	def __init__(self, d_model=128, heads=4):
		super().__init__()

		self.attn = SelfAttention(d_model=d_model, heads=heads)
		self.attn_norm = adaLN(d_model=d_model, d_in=d_model, d_hidden=d_model, hidden_layers=1, dropout=0.0)

		self.ffn = MLP(d_in=d_model, d_out=d_model, d_hidden=4*d_model, hidden_layers=0, act="silu", dropout=0.0)
		self.ffn_norm = adaLN(d_model=d_model, d_in=d_model, d_hidden=d_model, hidden_layers=1, dropout=0.0)

		self.norm = StaticLayerNorm(d_model)

	def forward(self, latent, nodes, t):

		# reshape
		ZN, d_model, Vx, Vy, Vz = latent.shape
		latent = latent.reshape(ZN, d_model, Vx*Vy*Vz).permute(0,2,1) #Z*N, Vx*Vy*Vz, d_model
		nodes = nodes.reshape(ZN, 1, d_model)
		t = t.reshape(ZN,1,d_model)

		# conditioning
		conditioning = nodes + t
		alpha1, gamma1, beta1 = self.attn_norm(conditioning)
		alpha2, gamma2, beta2 = self.ffn_norm(conditioning)
		
		# attn
		latent2 = gamma1*self.norm(latent) + beta1
		latent = latent + alpha1*self.attn(latent2)

		# ffn
		latent2 = gamma2*self.norm(latent) + beta2
		latent = latent + alpha2*self.ffn(latent2)

		# reshape
		latent = latent.permute(0,2,1).reshape(ZN, d_model, Vx, Vy, Vz)

		return latent

class SelfAttention(nn.Module):
	def __init__(self, d_model=128, heads=4):
		super().__init__()

		self.heads = heads
		self.d_model = d_model
		self.d_k = self.d_model // self.heads

		xavier_scale = (6/(self.d_k + d_model))**0.5

		self.q_proj = nn.Parameter(-xavier_scale + torch.rand(self.heads, self.d_model, self.d_k) * (2*xavier_scale)) # heads x d_model x d_k
		self.k_proj = nn.Parameter(-xavier_scale + torch.rand(self.heads, self.d_model, self.d_k) * (2*xavier_scale)) # heads x d_model x d_k
		self.v_proj = nn.Parameter(-xavier_scale + torch.rand(self.heads, self.d_model, self.d_k) * (2*xavier_scale)) # heads x d_model x d_k

		self.q_bias = nn.Parameter(torch.zeros(self.heads, self.d_k)) # heads x d_k
		self.k_bias = nn.Parameter(torch.zeros(self.heads, self.d_k)) # heads x d_k
		self.v_bias = nn.Parameter(torch.zeros(self.heads, self.d_k)) # heads x d_k

		self.out_proj = nn.Linear(d_model, d_model, bias=False)

	def forward(self, latent):

		Z, N, d_model = latent.shape

		# project the tensors
		Q = torch.matmul(latent.unsqueeze(1), self.q_proj.unsqueeze(0)) + self.q_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k
		K = torch.matmul(latent.unsqueeze(1), self.k_proj.unsqueeze(0)) + self.k_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k
		V = torch.matmul(latent.unsqueeze(1), self.v_proj.unsqueeze(0)) + self.v_bias.unsqueeze(0).unsqueeze(2) # batch x heads x N x d_k

		# perform attention
		S = torch.matmul(Q, K.transpose(2,3)) / (self.d_k**0.5)
		P = torch.softmax(S, dim=3)
		out = torch.matmul(P, V)

		# cat heads
		out = out.permute(0,2,3,1) # batch x N x d_k x heads
		out = out.reshape(Z, N, self.d_model) # batch x N x d_k x heads --> batch x N x d_model

		# project through final linear layer
		out = self.out_proj(out) # batch x N x d_model --> batch x N x d_model

		# return
		return out # batch x N x d_model

class adaLN(nn.Module):
	'''adaptive layer norm to perform affine transformation conditioned on timestep and nodes. adaLNzero, where initialized to all zeros'''
	def __init__(self, d_in=128, d_model=128, d_hidden=128, hidden_layers=1, dropout=0.0):
		super(adaLN, self).__init__()
		self.gamma_beta = MLP(d_in=d_in, d_out=2*d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout, act="silu", zeros=False)
		self.alpha = MLP(d_in=d_in, d_out=d_model, d_hidden=d_hidden, hidden_layers=hidden_layers, dropout=dropout, act="silu", zeros=True)

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

class UNet(nn.Module):
	'''
	downsamples the latent from 4x4x4x4 to 128x1x1x1, then upsamples back to 4x4x4x4
	each resolution is conditioned on catted tensor of timestep and node state from structure encoder
	at bottleneck, the node is conditioned on catted tensor of bottleneck and timestep, and then does
	multiple layers of message passing with other nodes. allows each latent to get an updated conditioning
	vector (the node) based on its neighbor's latent states and the timestep
	this updated node is then used as conditioining with the timestep for the upsampling stage
	'''
	def __init__(self, d_latent=4, d_model=128, mpnn_layers=3, resnet_layers=2):
		super().__init__()

		# increase channels, same spatial res
		self.conv_feature = nn.Conv3d(d_latent, d_model//8, 2, stride=1, padding="same", bias=True) #16,4,4,4
		self.condition_feature = ResNet(d_model=d_model//8, d_condition=2*d_model, kernel_size=2, layers=resnet_layers) #16,4,4,4

		# preprocess w/ resnets and downsample to 2x2x2, double channels
		self.conv_down1 = nn.Sequential(	nn.SiLU(),
											nn.GroupNorm(d_model//128, d_model//8),
											nn.Conv3d(d_model//8, d_model//4, 2, stride=2, padding=0, bias=True)
										) 
		self.condition_down1 = ResNet(d_model=d_model//4, d_condition=2*d_model, kernel_size=2, layers=resnet_layers)
		
		# preprocess w/ resnets and downsample to 1x1x1, double channels
		self.conv_down2 = nn.Sequential(	nn.SiLU(),
											nn.GroupNorm(d_model//64, d_model//4),
											nn.Conv3d(d_model//4, d_model//2, 2, stride=2, padding=0, bias=True)									
										) 
		self.condition_down2 = ResNet(d_model=d_model//2, d_condition=2*d_model, kernel_size=2, layers=resnet_layers)
		
		# bottleneck
		self.conv_bottleneck = nn.Sequential(	nn.SiLU(),
												nn.GroupNorm(d_model//32, d_model//2),
												nn.Conv3d(d_model//2, d_model, 1, stride=1, padding=0, bias=True),											
										)
		self.condition_bottleneck = ResNet(d_model=d_model, d_condition=2*d_model, kernel_size=1, layers=resnet_layers)

		# condition the nodes before message passing and after each layer
		self.node_conditionings = nn.ModuleList([FiLM_Node(d_model=d_model, d_condition=2*d_model) for _ in range(mpnn_layers+1)])
		self.mpnns = nn.ModuleList([MPNN(d_model=d_model, update_edges=False) for _ in range(mpnn_layers)])

		# prep and up conv to 2x2x2
		self.conv_up1 = nn.Sequential(	nn.SiLU(),
										nn.GroupNorm(d_model//16, d_model),
										nn.ConvTranspose3d(d_model, d_model//4, 2, stride=2, padding=0, output_padding=0, bias=True)
									)
		self.convpost_up1 = nn.Sequential(	nn.SiLU(),
											nn.GroupNorm(d_model//32, d_model//2),
											nn.Conv3d(d_model//2, d_model//4, 3, stride=1, padding=1, bias=True),											
										)
		self.condition_up1 = ResNet(d_model=d_model//4, d_condition=2*d_model, kernel_size=2, layers=resnet_layers)

		# prep and up conv to 4x4x4
		self.conv_up2 = nn.Sequential(	nn.SiLU(),
										nn.GroupNorm(d_model//64, d_model//4),
										nn.ConvTranspose3d(d_model//4, d_model//8, 2, stride=2, padding=0, output_padding=0, bias=True)
										
									)
		self.convpost_up2 = nn.Sequential(	nn.SiLU(),
											nn.GroupNorm(d_model//64, d_model//4),
											nn.Conv3d(d_model//4, d_model//8, 3, stride=1, padding=1, bias=True)
										)									
		self.condition_up2 = ResNet(d_model=d_model//8, d_condition=2*d_model, kernel_size=2, layers=resnet_layers)


		# final projection to original latent dim, essentially a linear layer since kernel=1
		self.conv_noise = nn.Conv3d(d_model//8, d_latent, 1, stride=1, padding="same")

	def forward(self, latent, t, nodes, edges, nbrs, nbr_mask):

		# ------------------------------------------------------------------------

		# prep
		ZN, d_latent, Vx, Vy, Vz = latent.shape
		Z, N, d_model = nodes.shape
		nodes = nodes.reshape(ZN, d_model) # reshape to make conditioning easier

		# define conditioning using orig node states and timestep
		condition = torch.cat([nodes, t], dim=1) # ZN,2*d_model

		# increase channels
		conv_feature = self.conv_feature(latent) # ZN,16,4,4,4
		condition_feature = self.condition_feature(conv_feature, condition) # ZN,16,4,4,4

		# ------------------------------------------------------------------------

		# downsample with strided conv and condition
		conv_down1 = self.conv_down1(condition_feature) # ZN,32,2,2,2
		condition_down1 = self.condition_down1(conv_down1, condition) # ZN,32,2,2,2

		conv_down2 = self.conv_down2(condition_down1) # ZN,64,1,1,1
		condition_down2 = self.condition_down2(conv_down2, condition) # ZN,64,1,1,1

		# ------------------------------------------------------------------------

		# bottleneck 
		conv_bottleneck = self.conv_bottleneck(condition_down2) # ZN,128,1,1,1
		condition_bottleneck = self.condition_bottleneck(conv_bottleneck, condition) # ZN,128,1,1,1

		# ------------------------------------------------------------------------

		# # reshape for the mpnn layer and condition nodes on latent state
		# node_condition = torch.cat([condition_bottleneck.reshape(Z, N, d_model), t.reshape(Z, N, d_model)], dim=2)
		# nodes = nodes.reshape(Z,N,d_model)

		# # first conditioning so the first layer sees latent states
		# nodes = self.node_conditionings[0](nodes, node_condition) # Z,N,128
		
		# # do message passing to see the other latents' states, continue conditioning at each layer
		# for mpnn, node_conditioning in zip(self.mpnns, self.node_conditionings[1:]):
		# 	nodes = mpnn(nodes, edges, nbrs, nbr_mask) # Z,N,128
		# 	nodes = node_conditioning(nodes, node_condition)
		# nodes = nodes.reshape(ZN,d_model) # reshape nodes back to be compatible with latents

		# # update the conditioning, timestep and UPDATED nodes
		# condition = torch.cat([nodes, t], dim=1)

		# ------------------------------------------------------------------------

		# upsample
		conv_up1 = torch.cat([self.conv_up1(condition_bottleneck), condition_down1], dim=1) # ZN, 32+32=64, 2, 2, 2
		convpost_up1 = self.convpost_up1(conv_up1) # ZN,32,2,2,2
		condition_up1 = self.condition_up1(convpost_up1, condition) # ZN,32,2,2,2

		conv_up2 = torch.cat([self.conv_up2(condition_up1), condition_feature], dim=1) # ZN, 16+16=32, 4, 4, 4
		convpost_up2 = self.convpost_up2(conv_up2) # ZN,16,4,4,4
		condition_up2 = self.condition_up2(convpost_up2, condition) # ZN, 16, 4, 4, 4

		# ------------------------------------------------------------------------

		# project to latent dim to get noise predictions
		noise_pred = self.conv_noise(condition_up2) # ZN,4,4,4,4

		# reshape
		noise_pred = noise_pred.reshape(Z, N, d_latent, Vx, Vy, Vz)

		return noise_pred

		# ------------------------------------------------------------------------

class ResNet(nn.Module):
	def __init__(self, d_model=128, d_condition=128, kernel_size=2, layers=1):

		super().__init__()

		self.convs = nn.ModuleList([	nn.Sequential(	nn.SiLU(),
														nn.GroupNorm(max(d_model//16,1), d_model),
														nn.Conv3d(d_model, d_model, kernel_size, stride=1, padding="same")
													) 
										for _ in range(layers)
									])
		self.conditionings = nn.ModuleList([FiLM_Latent(d_model=d_model, d_condition=d_condition) for _ in range(layers)])

	def forward(self, latent, condition):
		
		for conv, conditioning in zip(self.convs, self.conditionings):
			latent = latent + conditioning(conv(latent), condition)

		return latent

# FiLM for different shapes 
class FiLM_Latent(nn.Module):
	def __init__(self, d_model=128, d_condition=128):
		super().__init__()
		self.gamma_beta = MLP(d_in=d_condition, d_out=2*d_model, d_hidden=d_model, hidden_layers=1, act="silu", dropout=0.0)
	def forward(self, latent, condition): # assumes they are the same shape
		gamma_beta = self.gamma_beta(condition) # ZN,d_condituib
		ZN, two_d_model = gamma_beta.shape
		gamma, beta = torch.chunk(gamma_beta.reshape(ZN,two_d_model,1,1,1), dim=1, chunks=2)
		return gamma*latent + beta

class FiLM_Node(nn.Module):
	def __init__(self, d_model=128, d_condition=128):
		super().__init__()
		self.gamma_beta = MLP(d_in=d_condition, d_out=2*d_model, d_hidden=d_model, hidden_layers=1, act="silu", dropout=0.0)
	def forward(self, nodes, condition): # assumes they are the same shape
		gamma_beta = self.gamma_beta(condition) # Z, N, 2dmodel
		gamma, beta = torch.chunk(gamma_beta, dim=-1, chunks=2)
		return gamma*nodes + beta

class FiLM(nn.Module):
	def __init__(self, d_model=128, d_condition=128):
		super().__init__()
		self.gamma_beta = MLP(d_in=d_condition, d_out=2*d_model, d_hidden=d_model, hidden_layers=1, act="silu", dropout=0.0)
	def forward(self, nodes, condition): # assumes they are the same shape
		gamma_beta = self.gamma_beta(condition) # Z, N, 2dmodel
		gamma, beta = torch.chunk(gamma_beta, dim=-1, chunks=2)
		return gamma*nodes + beta