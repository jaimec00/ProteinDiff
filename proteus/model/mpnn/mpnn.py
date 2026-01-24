
import torch
import torch.nn as nn

from dataclasses import dataclass, field

from proteus.model.base import Base
from proteus.model.model_utils.mlp import (
	MLP, MLPCfg,
	MPNNMLP, MPNNMLPCfg,
	FFN, FFNCfg
)
from proteus.model.mpnn.get_neighbors import get_neighbors
from proteus.types import Float, Int, Bool, T, Tuple, List, Optional
from proteus.utils.tensor import unpad, repad



@dataclass
class MPNNBlockCfg:
	d_model: int = 256
	node_mlp: MPNNMLPCfg = field(default_factory=MPNNMLPCfg)
	ffn_mlp: FFNCfg = field(default_factory=FFNCfg)
	edge_mlp: Optional[MPNNMLPCfg] = field(default_factory=MPNNMLPCfg)


class MPNNBlock(Base):
	def __init__(self, cfg: MPNNBlockCfg) -> None:
		super().__init__()

		self.node_mlp: MPNNMLP = MPNNMLP(cfg.node_mlp)
		self.ln1: nn.LayerNorm = nn.LayerNorm(cfg.d_model)

		self.ffn: FFN = FFN(cfg.ffn_mlp)
		self.ln2: nn.LayerNorm = nn.LayerNorm(cfg.d_model)

		self._update_edges: bool = isinstance(cfg.edge_mlp, MPNNMLPCfg)
		if self._update_edges:
			self.edge_mlp: MPNNMLP = MPNNMLP(cfg.edge_mlp)
			self.edge_ln: nn.LayerNorm = nn.LayerNorm(cfg.d_model)

	def forward(
		self,
		nodes: Float[T, "BL d_model"],
		edges: Float[T, "BL K d_model"],
		nbrs: Int[T, "BL K"],
		nbr_mask: Bool[T, "BL K"],
	) -> Tuple[Float[T, "BL d_model"], Float[T, "BL K d_model"]]:
		nodes = self._node_msg(nodes, edges, nbrs, nbr_mask)
		edges = self._edge_msg(nodes, edges, nbrs)
		return nodes, edges

	def _node_msg(
		self,
		nodes: Float[T, "BL d_model"],
		edges: Float[T, "BL K d_model"],
		nbrs: Int[T, "BL K"],
		nbr_mask: Bool[T, "BL K"],
	) -> Float[T, "BL d_model"]:
		message = self._create_msg(nodes, edges, nbrs)
		mlp_out = self.node_mlp(message)
		masked = mlp_out * nbr_mask.unsqueeze(-1)
		nodes1 = torch.sum(masked, dim=1)
		nodes = self.ln1(nodes + nodes1)
		nodes = self.ln2(self.ffn(nodes) + nodes)
		return nodes

	def _edge_msg(
		self,
		nodes: Float[T, "BL d_model"],
		edges: Float[T, "BL K d_model"],
		nbrs: Int[T, "BL K"],
	) -> Float[T, "BL K d_model"]:
		if self._update_edges:
			message = self._create_msg(nodes, edges, nbrs)
			edges = self.edge_ln(edges + self.edge_mlp(message))
		return edges

	def _create_msg(
		self,
		nodes: Float[T, "BL d_model"],
		edges: Float[T, "BL K d_model"],
		nbrs: Int[T, "BL K"],
	) -> Float[T, "BL K 3d_model"]:
		BL, K, D = edges.shape
		nodes_i = nodes.unsqueeze(1).expand(BL, K, D)
		nodes_j = nodes[nbrs]  # Simpler indexing: BL,K,D
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
	def __init__(self, cfg: EdgeEncoderCfg) -> None:
		super().__init__()

		self.top_k: int = cfg.top_k
		self.register_buffer("rbf_centers", torch.linspace(cfg.min_rbf, cfg.max_rbf, cfg.num_rbf))
		self.spread: float = (cfg.max_rbf - cfg.min_rbf) / cfg.num_rbf
		self.rbf_norm: nn.LayerNorm = nn.LayerNorm(cfg.num_rbf*4*4)
		self.rbf_proj: nn.Linear = nn.Linear(cfg.num_rbf*4*4, cfg.d_model)

		# relative frames (flattened 3x3 to 9)
		self.frame_proj: nn.Linear = nn.Linear(9, cfg.d_model)

		# relative seq pos (relative difference in sequence)
		self.seq_emb: nn.Embedding = nn.Embedding(66, cfg.d_model) # [-32,33] 33 is diff chains, 0 is self

		# combine the rbf, frames, and seq emb into a single edge embedding
		self.edge_mlp: MPNNMLP = MPNNMLP(cfg.edge_mlp)
		self.edge_ln: nn.LayerNorm = nn.LayerNorm(cfg.d_model*3)

	def forward(
		self,
		coords_bb: Float[T, "BL 4 3"],
		frames: Float[T, "BL 3 3"],
		seq_pos: Int[T, "BL"],
		chain_pos: Int[T, "BL"],
		cu_seqlens: Int[T, "B+1"],
	) -> Tuple[Float[T, "BL K d_model"], Int[T, "BL K"], Bool[T, "BL K"]]:
		nbrs, nbr_mask = get_neighbors(coords_bb[:, 1, :], self.top_k, cu_seqlens)
		edges = self._get_edges(coords_bb, frames, seq_pos, chain_pos, nbrs)
		return edges, nbrs, nbr_mask

	def _get_edges(
		self,
		C_backbone: Float[T, "BL 4 3"],
		frames: Float[T, "BL 3 3"],
		seq_pos: Int[T, "BL"],
		chain_pos: Int[T, "BL"],
		nbrs: Int[T, "BL K"],
	) -> Float[T, "BL K d_model"]:
		rel_rbf = self.rbf_proj(self.rbf_norm(self._get_rbfs(C_backbone, nbrs)))
		rel_frames = self.frame_proj(self._get_frames(frames, nbrs))
		rel_idxs = self.seq_emb(self._get_seq_pos(seq_pos, chain_pos, nbrs))

		edges = self.edge_mlp(self.edge_ln(torch.cat([rel_rbf, rel_frames, rel_idxs], dim=-1)))
		return edges

	@torch.no_grad()
	def _get_rbfs(
		self,
		C_backbone: Float[T, "BL 4 3"],
		nbrs: Int[T, "BL K"],
	) -> Float[T, "BL K num_rbf*16"]:
		BL, A, S = C_backbone.shape
		_, K = nbrs.shape

		C_nbrs = torch.gather(C_backbone.reshape(BL, 1, A, S).expand(BL, K, A, S), 0, nbrs.reshape(BL,K,1,1).expand(BL, K, A, S)) # BL,K,A,S

		# Use squared distances for RBF (equivalent up to scaling)
		diff = C_backbone.reshape(BL,1,A,1,S) - C_nbrs.reshape(BL,K,1,A,S) # BL,K,A,A,S
		dists_sq = torch.sum(diff * diff, dim=-1).sqrt() # BL,K,A,A

		spread_sq = self.spread**2
		rbf_numerator = dists_sq.unsqueeze(-1) - self.rbf_centers.reshape(1,1,1,1,-1) # BL,K,A,A,R
		rbf_numerator  = rbf_numerator**2
		rbf_numerator.mul_(-1).div_(spread_sq)

		rbf = torch.exp(rbf_numerator).reshape(BL,K,-1)
		return rbf

	@torch.no_grad()
	def _get_frames(
		self,
		frames: Float[T, "BL 3 3"],
		nbrs: Int[T, "BL K"],
	) -> Float[T, "BL K 9"]:
		BL, K = nbrs.shape

		nbrs_idx = nbrs.reshape(BL, K, 1, 1).expand(BL, K, 3, 3)
		frame_nbrs = torch.gather(frames.reshape(BL, 1, 3, 3).expand(BL, K, 3, 3), 0, nbrs_idx)

		frames_exp = frames.unsqueeze(1).expand(BL, K, 3, 3)
		rel_frames = torch.matmul(frames_exp.transpose(-2,-1), frame_nbrs).reshape(BL, K, -1)
		return rel_frames

	@torch.no_grad()
	def _get_seq_pos(
		self,
		seq_pos: Int[T, "BL"],
		chain_pos: Int[T, "BL"],
		nbrs: Int[T, "BL K"],
	) -> Int[T, "BL K"]:
		BL, K = nbrs.shape

		seq_nbrs = torch.gather(seq_pos.reshape(BL, 1).expand(BL, K), 0, nbrs) # BL,K
		chain_nbrs = torch.gather(chain_pos.reshape(BL, 1).expand(BL, K), 0, nbrs) # BL,K
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
	def __init__(self, cfg: MPNNModelCfg) -> None:
		super().__init__()

		self.edge_encoder: EdgeEncoder = EdgeEncoder(cfg=cfg.edge_encoder)
		self.mpnn_blocks: nn.ModuleList = nn.ModuleList([
			MPNNBlock(cfg.mpnn_block)
			for _ in range(cfg.layers)
		])

	def forward(
		self,
		bb_coords: Float[T, "B L 4 3"],
		frames: Float[T, "B L 3 3"],
		seq_idx: Int[T, "B L"],
		chain_idx: Int[T, "B L"],
		pad_mask: Bool[T, "B L"],
		x: Float[T, "B L d_model"],
	) -> Float[T, "B L d_model"]:
		"""
		Process protein graph through edge encoding and MPNN blocks.

		Args:
			bb_coords: Backbone coordinates (B, L, 4, 3)
			frames: Local frames (B, L, 3, 3)
			seq_idx: Sequence positions (B, L)
			chain_idx: Chain positions (B, L)
			pad_mask: Padding mask (B, L)
			x: Node features (B, L, d_model)

		Returns:
			nodes: Updated node features (B, L, d_model)
		"""
		# Unpad all inputs
		[bb_u, frames_u, seq_u, chain_u, x_u], cu_seqlens, max_seqlen = unpad(
			bb_coords, frames, seq_idx, chain_idx, x, pad_mask=pad_mask
		)

		# Encode edges from structure
		edges, nbrs, nbr_mask = self.edge_encoder(bb_u, frames_u, seq_u, chain_u, cu_seqlens)

		# Apply MPNN blocks
		for mpnn_block in self.mpnn_blocks:
			x_u, edges = mpnn_block(x_u, edges, nbrs, nbr_mask)

		# Repad output
		[x_padded] = repad(x_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

		return x_padded