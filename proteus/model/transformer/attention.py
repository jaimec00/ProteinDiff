import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass, field
from unittest.mock import MagicMock

if torch.cuda.is_available():
	from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
else:
	flash_attn_varlen_qkvpacked_func = MagicMock()

from proteus.types import T, Float, Int
from proteus.model.base import Base


@dataclass
class MHACfg:
	d_model: int = 256
	heads: int = 16
	dropout_p: float = 0.0

	def __postinit__(self) -> None:
		assert self.d_model%self.heads == 0

class MHA(Base):
	"""
	works for self and cross attention
	"""
	def __init__(self, cfg: MHACfg) -> None:
		super().__init__()

		self.d_k: int = cfg.d_model // cfg.heads
		self.heads: int = cfg.heads
		self.dropout_p: float = cfg.dropout_p
		self.Wqkv: nn.Linear = nn.Linear(cfg.d_model, 3*cfg.d_model, bias=True)
		self.out_proj: nn.Linear = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

	def forward(
		self,
		qkv: Float[T, "BL d_model"],
		cu_seqlens: Int[T, "B+1"],
		max_seqlen: int,
	) -> Float[T, "BL d_model"]:

		# convenience
		BL, Dm = qkv.shape
		H, Dk = self.heads, self.d_k
		qkv_dtype = qkv.dtype

		# project the tensors
		QKV = self.Wqkv(qkv).reshape(BL, 3, H, Dk).to(torch.float16).contiguous()

		# dropout if in training
		dropout_p = self.dropout_p if self.training else 0.0

		# flash attention 2
		out = flash_attn_varlen_qkvpacked_func( # BL x H x Dk
			QKV,
			cu_seqlens,
			max_seqlen,
			dropout_p=dropout_p, # dropout
			softmax_scale=Dk**-0.5, # sm scale
			deterministic=dropout_p>0.0 # for deterministic bwd, only when dropout is used
		).to(qkv_dtype)

		# output projection
		out = self.out_proj(out.reshape(BL, Dm))

		return out