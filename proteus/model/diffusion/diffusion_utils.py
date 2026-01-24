import torch
import torch.nn as nn

from dataclasses import dataclass, field
from enum import StrEnum
import math

from proteus.model.base import Base
from proteus.model.transformer.attention import MHA, MHACfg
from proteus.model.transformer.transformer import TransformerModel, TransformerModelCfg
from proteus.model.model_utils.mlp import MLP, MLPCfg, ProjectionHead, ProjectionHeadCfg, FFN, FFNCfg, ActivationFn
from proteus.model.mpnn.mpnn import MPNNModel, MPNNModelCfg
from proteus.static.constants import alphabet
from proteus.types import Float, Int, T, List

@dataclass
class DiTBlockCfg:
	d_model: int = 256
	attn: MHACfg = field(default_factory=MHACfg)
	ffn: FFNCfg = field(default_factory=FFNCfg)

class DiTBlock(Base):
	def __init__(self, cfg: DiTBlockCfg):
		super().__init__()

		self.attn = MHA(cfg.attn)
		self.attn_norm = adaLN_Zero(d_in=cfg.d_model, d_gammabeta=cfg.d_model, d_alpha=cfg.d_model)

		self.ffn = FFN(cfg.ffn)
		self.ffn_norm = adaLN_Zero(d_in=cfg.d_model, d_gammabeta=cfg.d_model, d_alpha=cfg.d_model)

		self.norm = StaticLayerNorm(cfg.d_model)

	def forward(
		self,
		latent: Float[T, "BL d_model"],
		condition: Float[T, "BL d_model"],
		cu_seqlens: Int[T, "B+1"],
		max_seqlen: int,
	) -> Float[T, "BL d_model"]:

		# conditioning
		gamma1, beta1, alpha1 = self.attn_norm(condition)
		gamma2, beta2, alpha2 = self.ffn_norm(condition)

		# norm and conditioning
		latent = gamma1*self.norm(latent) + beta1
		
		# attn
		latent = latent + alpha1*self.attn(latent, cu_seqlens, max_seqlen)

		# ffn
		latent2 = gamma2*self.norm(latent) + beta2
		latent = latent + alpha2*self.ffn(latent2)

		return latent

@dataclass
class DiTModelCfg:
	dit_block: DiTBlockCfg = field(default_factory=DiTBlockCfg)
	layers: int = 4


class DiTModel(Base):
	def __init__(self, cfg: DiTModelCfg):
		super().__init__()
		self.dit_blocks = nn.ModuleList([
			DiTBlock(cfg.dit_block)
			for _ in range(cfg.layers)
		])

	def forward(
		self,
		latent: Float[T, "BL d_model"],
		condition: Float[T, "BL d_model"],
		cu_seqlens: Int[T, "B+1"],
		max_seqlen: int,
	) -> Float[T, "BL d_model"]:
		for block in self.dit_blocks:
			latent = block(latent, condition, cu_seqlens, max_seqlen)
		return latent

class adaLN_Zero(nn.Module):
	'''adaptive layer norm to perform affine transformation conditioned on timestep and nodes. adaLNzero, where initialized to all zeros'''
	def __init__(self, d_in: int = 256, d_gammabeta: int = 256, d_alpha: int = 256):
		super().__init__()
		# TODO: creating MLPCfg instead of MLP instances
		self.gamma_beta = MLP(MLPCfg(d_in=d_in, d_out=2*d_gammabeta, d_hidden=d_gammabeta, hidden_layers=1, dropout=0.0, act=ActivationFn.SILU, zeros=False))
		self.alpha = MLP(MLPCfg(d_in=d_in, d_out=d_alpha, d_hidden=d_alpha, hidden_layers=1, dropout=0.0, act=ActivationFn.SILU, zeros=True))

	def forward(self, x: Float[T, "ZN d_model"]) -> tuple[Float[T, "ZN d_model"], Float[T, "ZN d_model"], Float[T, "ZN d_model"]]:
		gamma, beta = torch.chunk(self.gamma_beta(x), chunks=2, dim=-1)
		alpha = self.alpha(x)
		return gamma, beta, alpha

class StaticLayerNorm(nn.Module):
	'''just normalizes each token to have a mean of 0 and var of 1, no scaling and shifting'''
	def __init__(self, d_model: int):
		super().__init__()
		self.d_model = d_model
	def forward(self, x: Float[T, "ZN d_model"]) -> Float[T, "ZN d_model"]:
		centered = x - x.mean(dim=-1, keepdim=True)
		std = centered.std(dim=-1, keepdim=True)
		normed = centered / std.masked_fill(std==0, 1)
		return normed

class CosineScheduler(nn.Module):
	def __init__(self, t_max: int, s: float = 0.008):
		super().__init__()
		self.t_max = t_max
		self.s = s

	def get_abars(self, t: Int[T, "..."]) -> Float[T, "..."]:
		stage = (t+1)/self.t_max
		abars = torch.cos(torch.pi*0.5*(stage+self.s)/(1+self.s))**2 / math.cos(torch.pi*0.5*self.s/(1+self.s))**2
		return abars

	def get_betas(self, t: Int[T, "..."]) -> Float[T, "..."]:
		abars = self.get_abars(t)
		abars_tminus1 = self.get_abars(t-1)
		betas = 1 - (abars/abars_tminus1)
		return betas

	def forward(self, t: Int[T, "..."]) -> tuple[Float[T, "..."], Float[T, "..."]]:
		abars = self.get_abars(t)
		betas = self.get_betas(t)
		return abars, betas

@dataclass
class ConditionerCfg:
	d_model: int = 256
	d_conditioning: int = 256
	conditioning_mpnn: MPNNModelCfg = field(default_factory=MPNNModelCfg)
	conditioning_transformer: TransformerModelCfg = field(default_factory = TransformerModelCfg)
	

class Conditioner(Base):
	def __init__(self, cfg: ConditionerCfg):
		super().__init__()
		self.register_buffer("t_wavenumbers", 10000**-(torch.arange(0, cfg.d_conditioning//2, 2)/cfg.d_conditioning))
		self.conditioning_seq = nn.Embedding(len(alphabet), cfg.d_conditioning)
		self.conditioning_mpnn = MPNNModel(cfg.conditioning_mpnn)
		self.conditioning_transformer = TransformerModel(cfg.conditioning_transformer)
		self.condition_proj = ProjectionHead(ProjectionHeadCfg(d_in=cfg.d_conditioning*2, d_out=cfg.d_model))


	def forward(
		self,
		seq: Int[T, "BL"],
		coords_bb: Float[T, "BL 4 3"],
		frames: Float[T, "BL 3 3"],
		seq_idx: Int[T, "BL"],
		chain_idx: Int[T, "BL"],
		sample_idx: Int[T, "BL"],
		t: Int[T, "BL"],
		cu_seqlens: Int[T, "B+1"],
		max_seqlen: int,
	) -> Float[T, "BL d_model"]:

		seq_coords_conditioning = self.seq_coords_conditioning(
			seq,
			coords_bb,
			frames,
			seq_idx,
			chain_idx,
			sample_idx,
			cu_seqlens,
			max_seqlen,
		)
		t_conditioning = self.featurize_t(t)
		conditioning = self.combine_conditioning(seq_coords_conditioning, t_conditioning)
		return conditioning

	def combine_conditioning(
		self,
		seq_coords_conditioning: Float[T, "BL d_conditioning"],
		t_conditioning: Float[T, "BL d_conditioning"]
	) -> Float[T, "BL d_model"]:
		return self.condition_proj(torch.cat([seq_coords_conditioning, t_conditioning], dim=-1))

	def seq_coords_conditioning(
		self,
		seq: Int[T, "BL"],
		coords_bb: Float[T, "BL 4 3"],
		frames: Float[T, "BL 3 3"],
		seq_idx: Int[T, "BL"],
		chain_idx: Int[T, "BL"],
		sample_idx: Int[T, "BL"],
		cu_seqlens: Int[T, "B+1"],
		max_seqlen: int,
	) -> Float[T, "BL d_conditioning"]:
		seq_conditioning = self.conditioning_seq(seq)
		seq_coords_conditioning = self.conditioning_mpnn(coords_bb, frames, seq_idx, chain_idx, sample_idx, seq_conditioning)
		seq_coords_conditioning = self.conditioning_transformer(seq_coords_conditioning, cu_seqlens, max_seqlen)
		return seq_coords_conditioning
	
	def featurize_t(self, t: Int[T, "BL"]) -> Float[T, "BL d_conditioning"]:
		phase = self.t_wavenumbers.unsqueeze(0) * t.unsqueeze(1)
		sines = torch.sin(phase)
		cosines = torch.cos(phase)
		t_feats = torch.stack([sines, cosines], dim=-1).reshape(t.size(0), -1)
		return t_feats


class Parameterization(StrEnum):
	EPS = "eps"
	X0 = "x0"
	VPRED = "vpred"
	DEFAULT = "eps"