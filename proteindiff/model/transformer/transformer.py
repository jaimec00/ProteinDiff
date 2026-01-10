import torch
import torch.nn as nn

from dataclasses import dataclass, field

from proteindiff.model import Base
from proteindiff.model.transformer import MHA, MHACfg
from proteindiff.model.utils import FFN, FFNCfg
from proteindiff.typing import Float, Int, T, List


@dataclass
class TransformerBlockCfg:
	d_model: int = 256
	attn: MHACfg = field(default_factory=MHACfg) 
	ffn: FFNCfg = field(default_factory=FFNCfg)


class TransformerBlock(Base):
	def __init__(self, cfg: TransformerBlockCfg):
		super().__init__()
		self.attn = MHA(cfg.attn)
		self.attn_norm = nn.LayerNorm(cfg.d_model)
		self.ffn = FFN(cfg.ffn)
		self.ffn_norm = nn.LayerNorm(cfg.d_model)

	def forward(
		self, 
		x: Float[T, "ZN d_model"], 
		cu_seqlens: Int[T, "Z"], 
		max_seqlen: int,
		) -> Float[T, "ZN d_model"]:
		x1 = self.attn(x, cu_seqlens, max_seqlen)
		x = self.attn_norm(x+x1)
		x1 = self.ffn(x)
		x = self.ffn_norm(x+x1)
		return x

@dataclass
class TransformerModelCfg:
	blocks: List[TransformerBlockCfg] = field(default_factory=lambda: [TransformerBlockCfg])

class TransformerModel(Base):
	def __init__(self, cfg: TransformerModelCfg):
		super().__init__()
		self.blocks = nn.ModuleList([
			TransformerBlock(block)
			for block in cfg.blocks
		])

	def forward(
		self, 
		x: Float[T, "ZN d_model"], 
		cu_seqlens: Int[T, "Z"], 
		max_seqlen: int,
		) -> Float[T, "ZN d_model"]:

		for block in self.blocks:
			x = block(x, cu_seqlens, max_seqlen)

		return x