import torch
import torch.nn as nn

from dataclasses import dataclass, field

from proteus.model.base import Base
from proteus.model.transformer.attention import MHA, MHACfg
from proteus.model.model_utils.mlp import FFN, FFNCfg
from proteus.types import Float, Int, Bool, T, List
from proteus.utils.tensor import unpad, repad


@dataclass
class TransformerBlockCfg:
	d_model: int = 256
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
		x: Float[T, "BL d_model"],
		cu_seqlens: Int[T, "B+1"],
		max_seqlen: int,
		) -> Float[T, "BL d_model"]:
		x1 = self.attn(x, cu_seqlens, max_seqlen)
		x = self.attn_norm(x+x1)
		x1 = self.ffn(x)
		x = self.ffn_norm(x+x1)
		return x


@dataclass
class TransformerModelCfg:
	transformer_block: TransformerBlockCfg = field(default_factory = TransformerBlockCfg)
	layers: int = 4

class TransformerModel(Base):
	def __init__(self, cfg: TransformerModelCfg) -> None:
		super().__init__()
		self.blocks: nn.ModuleList = nn.ModuleList([
			TransformerBlock(cfg.transformer_block)
			for _ in range(cfg.layers)
		])

	def forward(
		self,
		x: Float[T, "B L d_model"],
		pad_mask: Bool[T, "B L"]
		) -> Float[T, "B L d_model"]:

		# Unpad for flash attention (requires BL)
		[x_unpacked], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)

		# Process with TransformerBlocks (unchanged)
		for block in self.blocks:
			x_unpacked = block(x_unpacked, cu_seqlens, max_seqlen)

		# Repad output
		[x_padded] = repad(x_unpacked, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

		return x_padded

