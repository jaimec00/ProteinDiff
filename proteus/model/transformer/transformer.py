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
		self.attn: MHA = MHA(cfg.attn)
		self.attn_norm: nn.LayerNorm = nn.LayerNorm(cfg.d_model)
		self.ffn: FFN = FFN(cfg.ffn)
		self.ffn_norm: nn.LayerNorm = nn.LayerNorm(cfg.d_model)

	def forward(
		self,
		x: Float[T, "BL D"],
		cu_seqlens: Int[T, "B+1"],
		max_seqlen: int,
	) -> Float[T, "BL D"]:
		# TODO: fix this to support cross attention
		x1 = self.attn_norm(x)
		x = x + self.attn(x1, x1, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
		x = x + self.ffn(self.ffn_norm(x))
		return x


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
		x: Float[T, "BL d_model"],
		cu_seqlens: Bool[T, "BL"],
		max_seqlen: int
		) -> Float[T, "BL d_model"]:

		for block in self.blocks:
			x = block(x, cu_seqlens, max_seqlen)

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
