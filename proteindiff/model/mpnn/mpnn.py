
import torch
import torch.nn as nn

from dataclasses import dataclass, field

from proteindiff.model.base import Base
from proteindiff.model.model_utils.mlp import (
	MLP, MLPCfg,
	MPNNMLP, MPNNMLPCfg,
	FFN, FFNCfg
)
from proteindiff.types import Float, Int, Bool, T, Tuple, List



@dataclass
class MPNNBlockCfg:
	d_model: int = 256
	node_mlp: MPNNMLPCfg = field(default_factory=MPNNMLPCfg)
	ffn_mlp: FFNCfg = field(default_factory=FFNCfg)
	edge_mlp: MPNNMLPCfg | None = field(default_factory=MPNNMLPCfg)


class MPNNBlock(Base):
	def __init__(self, cfg: MPNNBlockCfg):
		super().__init__()

		self.node_mlp = MPNNMLP(cfg.node_mlp)
		self.ln1 = nn.LayerNorm(cfg.d_model)

		self.ffn = FFN(cfg.ffn_mlp)
		self.ln2 = nn.LayerNorm(cfg.d_model)

		self._update_edges = isinstance(cfg.edge_mlp, MPNNMLPCfg)
		if self._update_edges:
			self.edge_mlp = MPNNMLP(cfg.edge_mlp)
			self.edge_ln = nn.LayerNorm(cfg.d_model)

	def forward(
		self,
		nodes: Float[T, "ZN d_model"],
		edges: Float[T, "ZN K d_model"],
		nbrs: Int[T, "ZN K"],
		nbr_mask: Bool[T, "ZN K"],
	) -> Tuple[Float[T, "ZN d_model"], Float[T, "ZN K d_model"]]:
		nodes = self._node_msg(nodes, edges, nbrs, nbr_mask)
		edges = self._edge_msg(nodes, edges, nbrs)
		return nodes, edges

	def _node_msg(
		self,
		nodes: Float[T, "ZN d_model"],
		edges: Float[T, "ZN K d_model"],
		nbrs: Int[T, "ZN K"],
		nbr_mask: Bool[T, "ZN K"],
	) -> Float[T, "ZN d_model"]:
		message = self._create_msg(nodes, edges, nbrs)
		mlp_out = self.node_mlp(message)
		masked = mlp_out * nbr_mask.unsqueeze(-1)
		nodes1 = torch.sum(masked, dim=1)
		nodes = self.ln1(nodes + nodes1)
		nodes = self.ln2(self.ffn(nodes) + nodes)
		return nodes

	def _edge_msg(
		self,
		nodes: Float[T, "ZN d_model"],
		edges: Float[T, "ZN K d_model"],
		nbrs: Int[T, "ZN K"],
	) -> Float[T, "ZN K d_model"]:
		if self._update_edges:
			message = self._create_msg(nodes, edges, nbrs)
			edges = self.edge_ln(edges + self.edge_mlp(message))
		return edges

	def _create_msg(
		self,
		nodes: Float[T, "ZN d_model"],
		edges: Float[T, "ZN K d_model"],
		nbrs: Int[T, "ZN K"],
	) -> Float[T, "ZN K 3d_model"]:
		ZN, K, D = edges.shape
		nodes_i = nodes.unsqueeze(1).expand(ZN, K, D)
		nodes_j = nodes[nbrs]  # Simpler indexing: ZN,K,D
		message = torch.cat([nodes_i, nodes_j, edges], dim=-1)
		return message


@dataclass
class EdgeEncoderCfg:
	d_model: int = 256
	top_k: int = 16
	min_rbf: float = 2.0
	max_rbf: float = 22.0
	num_rbf: int = 16
	edge_mlp: MPNNMLPCfg = field(default_factory = MPNNMLPCfg)

class EdgeEncoder(Base):
	def __init__(self, cfg: EdgeEncoderCfg):
		super().__init__()

		self.top_k = cfg.top_k
		self.register_buffer("rbf_centers", torch.linspace(cfg.min_rbf, cfg.max_rbf, cfg.num_rbf))
		self.spread = (cfg.max_rbf - cfg.min_rbf) / cfg.num_rbf
		self.rbf_norm = nn.LayerNorm(cfg.num_rbf*4*4)
		self.rbf_proj = nn.Linear(cfg.num_rbf*4*4, cfg.d_model)

		# relative frames (flattened 3x3 to 9)
		self.frame_proj = nn.Linear(9, cfg.d_model)

		# relative seq pos (relative difference in sequence)
		self.seq_emb = nn.Embedding(66, cfg.d_model) # [-32,33] 33 is diff chains, 0 is self

		# combine the rbf, frames, and seq emb into a single edge embedding
		self.edge_mlp = MPNNMLP(cfg.edge_mlp)
		self.edge_ln = nn.LayerNorm(cfg.d_model*3)

	def forward(
		self,
		coords_bb: Float[T, "ZN 4 3"],
		frames: Float[T, "ZN 3 3"],
		seq_pos: Int[T, "ZN"],
		chain_pos: Int[T, "ZN"],
		sample_idx: Int[T, "ZN"],
	) -> Tuple[Float[T, "ZN K d_model"], Int[T, "ZN K"], Bool[T, "ZN K"]]:
		nbrs, nbr_mask = self._get_neighbors(coords_bb, sample_idx)
		edges = self._get_edges(coords_bb, frames, seq_pos, chain_pos, nbrs)
		return edges, nbrs, nbr_mask

	@torch.no_grad()
	def _get_neighbors(
		self,
		C: Float[T, "ZN 4 3"],
		sample_idx: Int[T, "ZN"],
	) -> Tuple[Int[T, "ZN K"], Bool[T, "ZN K"]]:
		# prep
		Ca = C[:, 1, :]
		ZN, S = Ca.shape

		# get distances: compute squared distances directly (avoid sqrt for topk since ordering is preserved)
		diff = Ca.unsqueeze(0) - Ca.unsqueeze(1)  # ZN x ZN x S
		dists_sq = torch.sum(diff * diff, dim=-1)  # ZN x ZN

		# sequences from other samples dont count (in-place)
		dists_sq.masked_fill_(sample_idx.unsqueeze(0) != sample_idx.unsqueeze(1), float("inf"))

		# get topk (clamp K to ZN to avoid out of range error)
		k = min(self.top_k, ZN)
		topk_result = dists_sq.topk(k, dim=1, largest=False)
		nbrs_indices = topk_result.indices

		# some samples might have less than K tokens, so create a nbr mask
		nbr_mask = sample_idx[nbrs_indices] == sample_idx.unsqueeze(1)

		# masked edge idxs are the idx corresponding to the self node
		node_idxs = torch.arange(ZN, device=dists_sq.device).unsqueeze(1)
		nbrs = nbrs_indices.where(nbr_mask, node_idxs)

		return nbrs, nbr_mask

	def _get_edges(
		self,
		C_backbone: Float[T, "ZN 4 3"],
		frames: Float[T, "ZN 3 3"],
		seq_pos: Int[T, "ZN"],
		chain_pos: Int[T, "ZN"],
		nbrs: Int[T, "ZN K"],
	) -> Float[T, "ZN K d_model"]:
		rel_rbf = self.rbf_proj(self.rbf_norm(self._get_rbfs(C_backbone, nbrs)))
		rel_frames = self.frame_proj(self._get_frames(frames, nbrs))
		rel_idxs = self.seq_emb(self._get_seq_pos(seq_pos, chain_pos, nbrs))

		edges = self.edge_mlp(self.edge_ln(torch.cat([rel_rbf, rel_frames, rel_idxs], dim=-1)))
		return edges

	@torch.no_grad()
	def _get_rbfs(
		self,
		C_backbone: Float[T, "ZN 4 3"],
		nbrs: Int[T, "ZN K"],
	) -> Float[T, "ZN K num_rbf*16"]:
		ZN, A, S = C_backbone.shape
		_, K = nbrs.shape

		C_nbrs = torch.gather(C_backbone.reshape(ZN, 1, A, S).expand(ZN, K, A, S), 0, nbrs.reshape(ZN,K,1,1).expand(ZN, K, A, S)) # ZN,K,A,S

		# Use squared distances for RBF (equivalent up to scaling)
		diff = C_backbone.reshape(ZN,1,A,1,S) - C_nbrs.reshape(ZN,K,1,A,S) # ZN,K,A,A,S
		dists_sq = torch.sum(diff * diff, dim=-1).sqrt() # ZN,K,A,A

		spread_sq = self.spread**2
		rbf_numerator = dists_sq.unsqueeze(-1) - self.rbf_centers.reshape(1,1,1,1,-1) # ZN,K,A,A,R
		rbf_numerator  = rbf_numerator**2
		rbf_numerator.mul_(-1).div_(spread_sq)

		rbf = torch.exp(rbf_numerator).reshape(ZN,K,-1)
		return rbf

	@torch.no_grad()
	def _get_frames(
		self,
		frames: Float[T, "ZN 3 3"],
		nbrs: Int[T, "ZN K"],
	) -> Float[T, "ZN K 9"]:
		ZN, K = nbrs.shape

		nbrs_idx = nbrs.reshape(ZN, K, 1, 1).expand(ZN, K, 3, 3)
		frame_nbrs = torch.gather(frames.reshape(ZN, 1, 3, 3).expand(ZN, K, 3, 3), 0, nbrs_idx)

		frames_exp = frames.unsqueeze(1).expand(ZN, K, 3, 3)
		rel_frames = torch.matmul(frames_exp.transpose(-2,-1), frame_nbrs).reshape(ZN, K, -1)
		return rel_frames

	@torch.no_grad()
	def _get_seq_pos(
		self,
		seq_pos: Int[T, "ZN"],
		chain_pos: Int[T, "ZN"],
		nbrs: Int[T, "ZN K"],
	) -> Int[T, "ZN K"]:
		ZN, K = nbrs.shape

		seq_nbrs = torch.gather(seq_pos.reshape(ZN, 1).expand(ZN, K), 0, nbrs) # ZN,K
		chain_nbrs = torch.gather(chain_pos.reshape(ZN, 1).expand(ZN, K), 0, nbrs) # ZN,K
		diff_chain = chain_pos.unsqueeze(1) != chain_nbrs

		rel_idx = torch.clamp(seq_nbrs - seq_pos.unsqueeze(1), min=-32, max=32)
		rel_idx.masked_fill_(diff_chain, 33)
		rel_idx.add_(32)  # starts at 0
		return rel_idx


@dataclass
class MPNNModelCfg:
	edge_encoder: EdgeEncoderCfg = field(default_factory=EdgeEncoderCfg)
	mpnn_block: MPNNBlockCfg = field(default_factory=MPNNBlockCfg)
	layers: int = 4


class MPNNModel(Base):
	def __init__(self, cfg: MPNNModelCfg):
		super().__init__()

		self.edge_encoder = EdgeEncoder(cfg=cfg.edge_encoder)
		self.mpnn_blocks = nn.ModuleList([
			MPNNBlock(cfg.mpnn_block)
			for _ in range(cfg.layers)
		])

	def forward(
		self,
		coords_bb: Float[T, "ZN 4 3"],
		frames: Float[T, "ZN 3 3"],
		seq_pos: Int[T, "ZN"],
		chain_pos: Int[T, "ZN"],
		sample_idx: Int[T, "ZN"],
		nodes: Float[T, "ZN d_model"],
	) -> Float[T, "ZN d_model"]:
		"""
		Process protein graph through edge encoding and MPNN blocks.

		Args:
			coords_bb: Backbone coordinates (ZN, 4, 3)
			frames: Local frames (ZN, 3, 3)
			seq_pos: Sequence positions (ZN,)
			chain_pos: Chain positions (ZN,)
			sample_idx: Sample indices (ZN,)
			nodes: Node features (ZN, d_model)

		Returns:
			nodes: Updated node features (ZN, d_model)
		"""
		# Encode edges from structure
		edges, nbrs, nbr_mask = self.edge_encoder(coords_bb, frames, seq_pos, chain_pos, sample_idx)

		# Apply MPNN blocks
		for mpnn_block in self.mpnn_blocks:
			nodes, edges = mpnn_block(nodes, edges, nbrs, nbr_mask)

		return nodes