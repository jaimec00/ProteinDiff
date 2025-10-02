
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

		self.node_mlp = MLP(d_in=3*d_model, d_hidden=d_model, d_out=d_model, num_hidden=2, act="silu")
		self.ln1 = nn.LayerNorm(d_model)
		
		self.ffn = MLP(d_in=d_model, d_hidden=4*d_model, d_out=d_model, num_hidden=0, act="silu")
		self.ln2 = nn.LayerNorm(d_model)

		self._update_edges = update_edges
		if update_edges:
			self.edge_mlp = MLP(d_in=3*d_model, d_hidden=d_model, d_out=d_model, num_hidden=2, act="silu")
			self.edge_ln = nn.LayerNorm(d_model)

	def forward(self, nodes, edges, nbrs, nbr_mask):
		
		nodes = self._node_msg(nodes, edges, nbrs, nbrs_mask)
		edges = self._edge_msg(nodes, edges, nbrs)
		return nodes, edges

	def _node_msg(self, nodes, edges, nbrs, nbr_mask):

		message = self._create_msg(nodes, edges, nbrs, nbr_mask)
		nodes1 = torch.sum(self.node_mlp(message) * nbr_mask.unsqueeze(-1), dim=2)
		nodes = self.ln1(nodes + nodes1)
		nodes = self.ln2(self.ffn(nodes) + nodes)
		return nodes

	def _edge_msg(self, nodes, edges, nbrs):
		
		if self._update_edges:
			message = self._create_msg(nodes, edges, nbrs)
			edges = self.edge_ln(edges + self.edge_mlp(message))

		return edges


	def _create_msg(nodes, edges, nbrs):
		Z, N, K, D = edges.shape
		nodes_i = nodes.unsqueeze(2).expand(Z, N, K, D)
		nodes_j = torch.gather(nodes_i, 1, nbrs)
		message = torch.cat([nodes_i, nodes_j, edges], dim=-1)
		return message


class EdgeEncoder(nn.Module):
	def __init__(self, d_model=256, top_k=16):
		super().__init__()

		# rbf stuff (rbfs w/ linearly spaced centers of inter-residue backbone atom pairs)
		min_rbf, max_rbf, num_rbf = 2.0, 22.0, 16
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
		self.edge_ln = nn.LayerNorm(d_model)

	def forward(self, coords_bb, frames, seq_pos, chain_pos, valid_mask):

		nbrs, nbr_mask = self._get_neighbors(coords_bb, valid_mask)
		edges = self._get_edges(coords_bb, frames, seq_pos, chain_pos, nbrs)
		return edges, nbrs, nbr_mask


	@torch.no_grad()
	def _get_neighbors(self, C, valid_mask):

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
		node_idxs = torch.arange(N, device=dists.device).reshape(1,-1,1) # 1 x N x 1
		nbr_mask = valid_mask.unsqueeze(2) & torch.gather(valid_mask.unsqueeze(2).expand(-1,-1,self.top_k), 1, nbrs.indices)
		nbrs = torch.where(nbr_mask, nbrs.indices, node_idxs) # Z x N x K

		return nbrs, nbr_mask

	def _get_edges(self, C_backbone, frames, seq_idxs, chain_idxs, nbrs):

		rel_rbf = self.rbf_proj(self.rbf_norm(self._get_rbfs(C_backbone, nbrs)))
		rel_frames = self.frame_proj(self._get_frames(frames, nbrs))
		rel_idxs = self.seq_emb(self._get_seq_pos(seq_idxs, chain_idxs, nbrs))

		edges = self.edge_mlp(self.edge_ln(torch.cat([rel_rbf, rel_frames, rel_idxs], dim=3))) # Z,N,K,d_model

		return edges

	@torch.no_grad()
	def _get_rbfs(self, C_backbone, nbrs):

		Z, N, A, S = C_backbone.shape
		_, _, K = nbrs.shape

		C_nbrs = torch.gather(C_backbone.unsqueeze(2).expand(Z, N, K, A, S), 1, nbrs.view(Z,N,K,1,1).expand(Z, N, K, A, S)) # Z,N,K,A,S

		dists = torch.sqrt(torch.sum((C_backbone.view(Z,N,1,A,1,S) - C_nbrs.view(Z,N,K,1,A,S))**2, dim=-1)) # Z,N,1,A,1,S - Z,N,K,1,A,S --> Z,N,K,A,A

		rbf_numerator = (dists.view(Z, N, K, A, A, 1) - self.rbf_centers.view(1,1,1,1,1,-1))**2 # Z,N,K,A,A,S

		rbf = torch.exp(-rbf_numerator / (self.spread**2)).reshape(Z,N,K,-1)
		
		return rbf

	@torch.no_grad()
	def _get_frames(self, frames, nbrs):

		Z, N, K = nbrs.shape

		frames = frames.unsqueeze(2).expand(Z, N, K, 3, 3)
		nbrs = nbrs.reshape(Z, N, K, 1, 1).expand(Z, N, K, 3, 3)
		
		frame_nbrs = torch.gather(frames, 1, nbrs)
		rel_frames = torch.matmul(frames.transpose(-2,-1), frame_nbrs).reshape(Z, N, K, -1)

		return rel_frames

	@torch.no_grad()
	def _get_seq_pos(self, seq_idxs, chain_idxs, nbrs):

		Z, N = seq_idxs.shape

		seq_idxs = seq_idxs.unsqueeze(2).expand(Z, N, K)		
		seq_nbrs = torch.gather(seq_idxs, 1, nbrs) # Z,N,K

		chain_idxs = chain_idxs.unsqueeze(2).expand(Z, N, K)
		chain_nbrs = torch.gather(chain_idxs, 1, nbrs) # Z,N,K
		diff_chain = chain_idxs!=chain_nbrs

		rel_idx = torch.clamp(seq_nbrs - seq_idxs, min=-32, max=32)
		rel_idx = rel_idx.masked_fill(diff_chain, 33) + 32 # starts at 0

		return rel_idx


class Transformer(nn.Module):
	def __init__(self, d_model=256, heads=8):
		super().__init__()
		self.attn = FlashMHA(d_model=d_model, heads=heads)
		self.attn_norm = nn.LayerNorm(d_model)
		self.ffn = MLP(d_in=d_model, d_hidden=4*d_model, d_out=d_model, num_hidden=0, act="silu")
		self.ffn_norm = nn.LayerNorm(d_model)

	def forward(self, x, valid_mask):
		x1 = self.attn(x, valid_mask)
		x = self.attn_norm(x+x1)
		x1 = self.ffn(x)
		x = self.ffn_norm(x+x1)
		return x


class PairwiseProjHead(nn.Module):
    def __init__(self, d_model, d_down, dist_bins, angle_bins):
        super().__init__()
        self.downsample = nn.Linear(d_model, d_down)
        self.bin = MLP(d_in=2*d_down, d_hidden=dist_bins+angle_bins, d_out=dist_bins+angle_bins, num_hidden=1, act="silu")
        self._dist_bins = dist_bins
        self._angle_bins = angle_bins

    def forward(self, x):

        q, k = torch.chunk(self.downsample(x), chunks=2, dim=-1) # Z x N x D//2
        q_i, k_j = q.unsqueeze(2), k.unsqueeze(1)
        prod, diff = q_i*k_j, k_j-q_i
        pw = torch.cat([prod, diff], dim=-1) # Z x N x N x D
        distogram, anglogram = torch.chunk(self.bin(pw), chunks=[self._dist_bins, self._angle_bins], dim=-1)
        return distogram, anglogram