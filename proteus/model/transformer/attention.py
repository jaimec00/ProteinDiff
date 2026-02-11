from dataclasses import dataclass, field
from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
	from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
else:
	from unittest.mock import MagicMock
	flash_attn_varlen_kvpacked_func = MagicMock()

from proteus.types import T, Float, Int
from proteus.model.base import Base


@dataclass
class MHACfg:
	d_model: int = II("model.d_model")
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
		self.Wq: nn.Linear = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
		self.Wkv: nn.Linear = nn.Linear(cfg.d_model, 2*cfg.d_model, bias=True)
		self.out_proj: nn.Linear = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
		self.reset_parameters()

	def reset_parameters(self) -> None:
		nn.init.kaiming_uniform_(self.Wq.weight)
		nn.init.kaiming_uniform_(self.Wkv.weight) # technically incorrect but whatevs
		nn.init.zeros_(self.Wq.bias)
		nn.init.zeros_(self.Wkv.bias)
		nn.init.zeros_(self.out_proj.weight)

	def forward(
		self,
		q: Float[T, "BL d_model"],
		kv: Float[T, "BL d_model"],
		cu_seqlens_q: Int[T, "B+1"],
		cu_seqlens_kv: Int[T, "B+1"],
		max_seqlen_q: int,
		max_seqlen_kv: int,
	) -> Float[T, "BL d_model"]:
		'''
		supports self-attn, cross-attn, gqa
		'''

		# convenience
		BL, Dm = q.shape
		H, Dk = self.heads, self.d_k
		q_dtype = q.dtype

		# project the tensors
		Q = self.Wq(q).reshape(BL, H, Dk).half().contiguous()
		KV = self.Wkv(kv).reshape(BL, 2, H, Dk).half().contiguous()

		# dropout if in training
		dropout_p = self.dropout_p if self.training else 0.0

		# flash attention 2
		out = flash_attn_varlen_kvpacked_func( # BL x H x Dk
			Q,
			KV,
			cu_seqlens_q,
			cu_seqlens_kv,
			max_seqlen_q,
			max_seqlen_kv,
			dropout_p=dropout_p, # dropout
			softmax_scale=Dk**-0.5, # sm scale
			deterministic=dropout_p>0.0 # for deterministic bwd, only when dropout is used
		).to(q_dtype)

		# output projection
		out = self.out_proj(out.reshape(BL, Dm))

		return out