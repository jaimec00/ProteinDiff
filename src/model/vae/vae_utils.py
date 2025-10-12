
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

class MPNN(nn.Module):
	def __init__(self, d_model=256, update_edges=True):
		super().__init__()

		self.node_mlp = MLP(d_in=3*d_model, d_hidden=d_model, d_out=d_model, hidden_layers=2, act="silu")
		self.ln1 = nn.LayerNorm(d_model)
		
		self.ffn = MLP(d_in=d_model, d_hidden=4*d_model, d_out=d_model, hidden_layers=0, act="silu")
		self.ln2 = nn.LayerNorm(d_model)

		self._update_edges = update_edges
		if update_edges:
			self.edge_mlp = MLP(d_in=3*d_model, d_hidden=d_model, d_out=d_model, hidden_layers=2, act="silu")
			self.edge_ln = nn.LayerNorm(d_model)

	def forward(self, nodes, edges, nbrs, nbr_mask):
		
		nodes = self._node_msg(nodes, edges, nbrs, nbr_mask)
		edges = self._edge_msg(nodes, edges, nbrs)
		return nodes, edges

	def _node_msg(self, nodes, edges, nbrs, nbr_mask):

		message = self._create_msg(nodes, edges, nbrs)
		nodes1 = torch.sum(self.node_mlp(message) * nbr_mask.unsqueeze(-1), dim=1)
		nodes = self.ln1(nodes + nodes1)
		nodes = self.ln2(self.ffn(nodes) + nodes)
		return nodes

	def _edge_msg(self, nodes, edges, nbrs):
		
		if self._update_edges:
			message = self._create_msg(nodes, edges, nbrs)
			edges = self.edge_ln(edges + self.edge_mlp(message))

		return edges

	def _create_msg(self, nodes, edges, nbrs):
		ZN, K, D = edges.shape
		nodes_i = nodes.unsqueeze(1).expand(ZN, K, D)
		nodes_j = torch.gather(nodes_i, 0, nbrs.unsqueeze(-1).expand(ZN, K, D))
		message = torch.cat([nodes_i, nodes_j, edges], dim=-1)
		return message


class EdgeEncoder(nn.Module):
	def __init__(self, d_model=256, top_k=16):
		super().__init__()

		# rbf stuff (rbfs w/ linearly spaced centers of inter-residue backbone atom pairs)
		min_rbf, max_rbf, num_rbf = 2.0, 22.0, 16
		self.top_k = top_k
		self.register_buffer("rbf_centers", torch.linspace(min_rbf, max_rbf, num_rbf))
		self.spread = (max_rbf - min_rbf) / num_rbf
		self.rbf_norm = nn.LayerNorm(num_rbf*4*4)
		self.rbf_proj = nn.Linear(num_rbf*4*4, d_model)

		# relative frames (flattened 3x3 to 9)
		self.frame_proj = nn.Linear(9, d_model)

		# relative seq pos (relative difference in sequence)
		self.seq_emb = nn.Embedding(66, d_model) # [-32,33] 33 is diff chains, 0 is self

		# combine the rbf, frames, and seq emb into a single edge embedding
		self.edge_mlp = MLP(d_in=d_model*3, d_out=d_model, d_hidden=d_model, hidden_layers=2, act="silu")
		self.edge_ln = nn.LayerNorm(d_model*3)

	def forward(self, coords_bb, frames, seq_pos, chain_pos, sample_idx):

		nbrs, nbr_mask = self._get_neighbors(coords_bb, sample_idx)
		edges = self._get_edges(coords_bb, frames, seq_pos, chain_pos, nbrs)
		return edges, nbrs, nbr_mask

	@torch.no_grad()
	def _get_neighbors(self, C, sample_idx):

		# prep
		Ca = C[:, 1, :]
		ZN, S = Ca.shape

		# get distances
		dists = torch.sqrt(torch.sum((Ca.unsqueeze(0) - Ca.unsqueeze(1))**2, dim=-1)) # ZN x ZN
		
		# sequences from other samples dont count
		dists.masked_fill_(sample_idx.unsqueeze(0) != sample_idx.unsqueeze(1), float("inf")) # ZN x ZN
		
		# get topk 
		nbrs = dists.topk(self.top_k, dim=1, largest=False) # ZN x K

		# some samples might have less than K tokens, so create a nbr mask
		nbr_sample_idxs = torch.gather(sample_idx.unsqueeze(1).expand(-1,self.top_k), 0, nbrs.indices)
		nbr_mask = nbr_sample_idxs == sample_idx.unsqueeze(1)

		# masked edge idxs are the idx corresponding to the self node
		node_idxs = torch.arange(ZN, device=dists.device).unsqueeze(1) # ZN x 1
		nbrs = torch.where(nbr_mask, nbrs.indices, node_idxs) # ZN x K

		return nbrs, nbr_mask

	def _get_edges(self, C_backbone, frames, seq_pos, chain_pos, nbrs):

		rel_rbf = self.rbf_proj(self.rbf_norm(self._get_rbfs(C_backbone, nbrs)))
		rel_frames = self.frame_proj(self._get_frames(frames, nbrs))
		rel_idxs = self.seq_emb(self._get_seq_pos(seq_pos, chain_pos, nbrs))

		edges = self.edge_mlp(self.edge_ln(torch.cat([rel_rbf, rel_frames, rel_idxs], dim=-1))) # Z,N,K,d_model

		return edges

	@torch.no_grad()
	def _get_rbfs(self, C_backbone, nbrs):

		ZN, A, S = C_backbone.shape
		_, K = nbrs.shape

		C_nbrs = torch.gather(C_backbone.unsqueeze(1).expand(ZN, K, A, S), 0, nbrs.reshape(ZN,K,1,1).expand(ZN, K, A, S)) # ZN,K,A,S

		dists = torch.sqrt(torch.sum((C_backbone.reshape(ZN,1,A,1,S) - C_nbrs.reshape(ZN,K,1,A,S))**2, dim=-1)) # ZN,1,A,1,S - ZN,K,1,A,S --> ZN,K,A,A

		rbf_numerator = (dists.reshape(ZN, K, A, A, 1) - self.rbf_centers.reshape(1,1,1,1,-1))**2 # ZN,K,A,A,R

		rbf = torch.exp(-rbf_numerator / (self.spread**2)).reshape(ZN,K,-1)
		
		return rbf

	@torch.no_grad()
	def _get_frames(self, frames, nbrs):

		ZN, K = nbrs.shape

		frames = frames.unsqueeze(1).expand(ZN, K, 3, 3)
		nbrs = nbrs.reshape(ZN, K, 1, 1).expand(ZN, K, 3, 3)
		
		frame_nbrs = torch.gather(frames, 0, nbrs)
		rel_frames = torch.matmul(frames.transpose(-2,-1), frame_nbrs).reshape(ZN, K, -1)

		return rel_frames

	@torch.no_grad()
	def _get_seq_pos(self, seq_pos, chain_pos, nbrs):

		ZN, K = nbrs.shape

		seq_pos = seq_pos.unsqueeze(1).expand(ZN, K)		
		seq_nbrs = torch.gather(seq_pos, 0, nbrs) # ZN,K

		chain_pos = chain_pos.unsqueeze(1).expand(ZN, K)
		chain_nbrs = torch.gather(chain_pos, 0, nbrs) # ZN,K
		diff_chain = chain_pos!=chain_nbrs

		rel_idx = torch.clamp(seq_nbrs - seq_pos, min=-32, max=32)
		rel_idx = rel_idx.masked_fill(diff_chain, 33) + 32 # starts at 0

		return rel_idx


class Transformer(nn.Module):
	def __init__(self, d_model=256, heads=8):
		super().__init__()
		self.attn = FlashMHA(d_model=d_model, heads=heads)
		self.attn_norm = nn.LayerNorm(d_model)
		self.ffn = MLP(d_in=d_model, d_hidden=4*d_model, d_out=d_model, hidden_layers=0, act="silu")
		self.ffn_norm = nn.LayerNorm(d_model)

	def forward(self, x, cu_seqlens, max_seqlen):
		x1 = self.attn(x, cu_seqlens, max_seqlen)
		x = self.attn_norm(x+x1)
		x1 = self.ffn(x)
		x = self.ffn_norm(x+x1)
		return x


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