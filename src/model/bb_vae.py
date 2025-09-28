'''
this is the backbone encoder
plan is to pass the frames, bb coords, 
'''

import torch
import torch.nn as nn

from model.base_modules import MLP


class BackBoneVae(nn.Module):
    def __init__(self, d_model=256, top_k=16):
        super().__init__()
        self.enc = BackBoneEncoder()
        self.dec = BackBoneDecoder()

class MPNN(nn.Module):
    def __init__(self, d_model=256):

        self.

    def forward(self, nodes, edges, nbrs, nbr_mask):
        
        
        nodes1 = self._message(nodes, edges, nbrs, nbrs_mask)
        nodes = self.ln(nodes + nodes1)


    def message(self, nodes, edges, nbrs, nbr_mask):

        Z, N, K, D = nbrs.shape
        nodes_j = torch.gather(nodes.unsqueeze(2).expand(Z, N, K, D), 1, nbrs)

        

class BackBoneEncoder(nn.Module):
    def __init__(self, d_model=256, top_k=16, layers=3):
        super().__init__()
        self.edge_enc = EdgeEncoder()
        self.start_nodes = nn.Parameter(torch.randn((d_model,)))
        self.mpnn = nn.ModuleList([MPNN(d_model) for _ in range(layers)])

    def forward(self):

        Z, N = valid_mask.shape
        edges, nbrs, nbr_mask = self.edge_enc(coords_bb, frames, seq_pos, chain_pos, valid_mask)
        nodes = self.start_nodes.reshape(1,1,-1)


class EdgeEncoder(nn.Module):
    def __init__(self, d_model=256, top_k=16):
        super().__init__()

        # rbf stuff (rbfs w/ linearly spaced centers of inter-residue backbone atom pairs)
		self.register_buffer("rbf_centers", torch.linspace(min_rbf, max_rbf, num_rbf))
		self.spread = (max_rbf - min_rbf) / num_rbf
		self.rbf_norm = nn.LayerNorm(num_rbf*4*4)
		self.rbf_proj = nn.Linear(num_rbf*4*4, d_model)

		# relative frames (flattened 3x3 to 9)
		self.frame_proj = nn.Linear(9, d_model)

		# relative seq pos (relative difference in sequence)
		self.seq_emb = nn.Embedding(66, d_model) # [-32,33] 33 is diff chains, 0 is self

		# combine the rbf, frames, and seq emb into a single edge embedding
		self.edge_mlp = MLP(d_in=d_model*3, d_out=d_model, d_hidden=d_model, hidden_layers=2, ac="silu")

    def forward(self, coords_bb, frames, seq_pos, chain_pos, valid_mask):

        nbrs, nbr_mask = self.get_neighbors(coords_bb, valid_mask)
		edges = self.get_edges(coords_bb, frames, seq_pos, chain_pos, nbrs)
        return edges, nbrs, nbr_mask


    @torch.no_grad()
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

	def get_edges(self, C_backbone, frames, seq_idxs, chain_idxs, nbrs):

		rel_rbf = self.get_rbfs(C_backbone, nbrs)
		rel_frames = self.get_relative_frames(frames, nbrs)
		rel_idxs = self.get_relative_idxs(frames, nbrs)

		edges = self.edge_mlp(torch.cat([rel_rbf, rel_frames, rel_idxs], dim=3)) # Z,N,K,d_model

		return edges

	def get_rbfs(self, C_backbone, nbrs):

		Z, N, A, S = C_backbone.shape
		_, _, K = nbrs.shape

		C_nbrs = torch.gather(C_backbone.unsqueeze(2).expand(Z, N, K, A, S), 1, nbrs.view(Z,N,K,1,1).expand(Z, N, K, A, S)) # Z,N,K,A,S

		dists = torch.sqrt(torch.sum((C_backbone.view(Z,N,1,A,1,S) - C_nbrs.view(Z,N,K,1,A,S))**2, dim=-1)) # Z,N,1,A,1,S - Z,N,K,1,A,S --> Z,N,K,A,A

		rbf_numerator = (dists.view(Z, N, K, A, A, 1) - self.rbf_centers.view(1,1,1,1,1,-1))**2 # Z,N,K,A,A,S

		rbf = torch.exp(-rbf_numerator / (self.spread**2)).reshape(Z,N,K,-1)
		
		rbf = self.rbf_proj(self.rbf_norm(rbf))

		return rbf

	def get_frames(self, frames, nbrs):

		Z, N, K = nbrs.shape

		frames = frames.unsqueeze(2).expand(Z, N, K, 3, 3)
		nbrs = nbrs.reshape(Z, N, K, 1, 1).expand(Z, N, K, 3, 3)
		
		frame_nbrs = torch.gather(frames, 1, nbrs)
		rel_frames = torch.matmul(frames.transpose(-2,-1), frame_nbrs).reshape(Z, N, K, -1)

		return self.frame_proj(rel_frames)

	def get_seq_pos(self, seq_idxs, chain_idxs, nbrs):

		Z, N = seq_idxs.shape

		seq_idxs = seq_idxs.unsqueeze(2).expand(Z, N, K)		
		seq_nbrs = torch.gather(seq_idxs, 1, nbrs) # Z,N,K

		chain_idxs = chain_idxs.unsqueeze(2).expand(Z, N, K)
		chain_nbrs = torch.gather(chain_idxs, 1, nbrs) # Z,N,K
		diff_chain = chain_idxs!=chain_nbrs

		rel_idx = torch.clamp(seq_nbrs - seq_idxs, min=-32, max=32)
		rel_idx = rel_idx.masked_fill(diff_chain, 33) + 32 # starts at 0

		return self.seq_emb(rel_idx)

