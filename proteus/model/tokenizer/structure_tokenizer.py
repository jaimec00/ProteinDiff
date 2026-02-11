
from dataclasses import dataclass, field
from omegaconf import II

import torch
import torch.nn as nn

from proteus.model.base import Base
from proteus.static.constants import alphabet
from proteus.types import Float, Int, Bool, T
from proteus.model.model_utils import EdgeMLP, EdgeMLPCfg

@dataclass
class StructureTokenizerCfg:
	edge_mlp: EdgeMLPCfg = field(default_factory=EdgeMLPCfg)
	d_model: int = II("model.d_pair")
	min_rbf: float = 2.0
	max_rbf: float = 22.0
	num_rbf: int = 16

class StructureTokenizer(Base):
	def __init__(self, cfg: StructureTokenizerCfg) -> None:
		super().__init__()

		# rbf computation 
		self.register_buffer("rbf_centers", torch.linspace(cfg.min_rbf, cfg.max_rbf, cfg.num_rbf))
		self.spread = (cfg.max_rbf - cfg.min_rbf) / cfg.num_rbf
		self.rbf_proj = nn.Linear(cfg.num_rbf*4*4, cfg.d_model)

		# relative frames (flattened 3x3 to 9)
		self.frame_proj = nn.Linear(9, cfg.d_model)

		# relative seq pos (relative difference in sequence)
		self.seq_emb = nn.Embedding(66, cfg.d_model) # [-32,33] 33 is diff chains, 0 is self

		# combine the rbf, frames, and seq emb into a single edge embedding
		self.edge_mlp = EdgeMLP(cfg.edge_mlp)
		self.edge_ln = nn.LayerNorm(cfg.d_model)


	def forward(
		self, 
		coords_bb_dist: Float[T, "BLL 4 4"],
		rel_frames: Int[T, "BLL 3 3"],
		rel_seq_idx: Int[T, "BLL"], 
		diff_chain: Bool[T, "BLL"],
	) -> Float[T, "BLL D"]:

		BLL = coords_bb_dist.size(0)
		rel_rbf = self.rbf_proj(self._get_rbfs(coords_bb_dist).reshape(BLL,-1))
		rel_frames = self.frame_proj(rel_frames.reshape(BLL,-1))
		rel_idxs = self.seq_emb(self._get_seq_idx(rel_seq_idx, diff_chain))

		edges = self.edge_mlp(self.edge_ln(rel_rbf + rel_frames + rel_idxs))
		return edges

	@torch.no_grad()
	def _get_rbfs(self, coords_bb_dist: Float[T, "BLL 4 4"]) -> Float[T, "BLL num_rbf*4*4"]:
		rbf_numerator = coords_bb_dist.unsqueeze(-1) - self.rbf_centers.reshape(1,1,1,-1) # BLL,A,A,R
		rbf = rbf_numerator.pow_(2).mul_(-1).div_(self.spread**2).exp_()
		return rbf

	@torch.no_grad()
	def _get_seq_idx(
		self,
		rel_seq_idx: Int[T, "BLL"],
		diff_chain: Bool[T, "BLL"],
	) -> Int[T, "B L L"]:

		rel_seq_idx = rel_seq_idx.clamp_(min=-32, max=32).masked_fill_(diff_chain, 33).add_(32)
		return rel_seq_idx