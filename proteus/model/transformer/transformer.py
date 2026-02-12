from dataclasses import dataclass, field
from omegaconf import II

import torch
import torch.nn as nn

from proteus.model.transformer.attention import MHA, MHACfg
from proteus.model.model_utils.mlp import FFN, FFNCfg
from proteus.types import Float, Int, Bool, T, List
from proteus.model.base import Base


@dataclass
class TransformerBlockCfg:
	d_model: int = II("model.d_model")
	attn: MHACfg = field(default_factory=MHACfg)
	ffn: FFNCfg = field(default_factory=FFNCfg)


class TransformerBlock(Base):
	def __init__(self, cfg: TransformerBlockCfg) -> None:
		super().__init__()
		self.attn = MHA(cfg.attn)
		self.attn_norm_q = nn.LayerNorm(cfg.d_model)
		self.attn_norm_kv = nn.LayerNorm(cfg.d_model)
		self.ffn = FFN(cfg.ffn)
		self.ffn_norm = nn.LayerNorm(cfg.d_model)

	def forward(
		self,
		q: Float[T, "BL D"],
		kv: Float[T, "BL D"],
		cu_seqlens_q: Int[T, "B+1"],
		cu_seqlens_kv: Int[T, "B+1"],
		max_seqlen_q: int,
		max_seqlen_kv: int,
	) -> Float[T, "BL D"]:
		# TODO: fix this to support cross attention
		q = q + self.attn(self.attn_norm_q(q), self.attn_norm_kv(kv), cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
		q = q + self.ffn(self.ffn_norm(q))
		return q


@dataclass
class TransformerModelCfg:
	transformer_block: TransformerBlockCfg = field(default_factory = TransformerBlockCfg)
	layers: int = 1

class TransformerModel(Base):
	def __init__(self, cfg: TransformerModelCfg) -> None:
		super().__init__()
		self.blocks: nn.ModuleList = nn.ModuleList([
			TransformerBlock(cfg.transformer_block)
			for _ in range(cfg.layers)
		])

	def forward(
		self,
		q: Float[T, "BL d_model"],
		kv: Float[T, "BL d_model"],
		cu_seqlens_q: Bool[T, "BL"],
		cu_seqlens_kv: Bool[T, "BL"],
		max_seqlen_q: int,
		max_seqlen_kv: int,
		) -> Float[T, "BL d_model"]:

		for block in self.blocks:
			x = block(q, kv, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)

		return x

# just to make interpolation cleaner, packed pairs logic dealt with in dataloader
@dataclass
class PairMHACfg(MHACfg):
	d_model: int = II("model.d_pair")

@dataclass
class PairFFNCfg(FFNCfg):
	d_model: int = II("model.d_pair")
	expansion_factor: int = 2

@dataclass 
class PairformerBlockCfg(TransformerBlockCfg):
	d_model: int = II("model.d_pair")
	attn: PairMHACfg = field(default_factory=PairMHACfg)
	ffn: PairFFNCfg = field(default_factory=PairFFNCfg)

@dataclass
class PairformerModelCfg(TransformerModelCfg):
	transformer_block: PairformerBlockCfg = field(default_factory=PairformerBlockCfg)
