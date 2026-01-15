'''
title: 			ProteinDiff.py
author: 		jaime cardenas
description:	
'''


import torch
import torch.nn as nn

from dataclasses import dataclass, field
from enum import StrEnum

from proteindiff.model.base import Base
from proteindiff.model.vae.vae import VAEModel, VAEModelCfg
from proteindiff.model.diffusion.diffusion import DiffusionModel, DiffusionModelCfg
from proteindiff.model.tokenizer.tokenizer import Tokenizer, TokenizerCfg 
from proteindiff.types import Float, Int, Bool, T

from proteindiff.static.constants import TrainingStage

@dataclass
class ProteinDiffCfg:
	tokenizer: TokenizerCfg = field(default_factory = TokenizerCfg)
	vae: VAEModelCfg = field(default_factory = VAEModelCfg)
	diffusion: DiffusionModelCfg = field(default_factory = DiffusionModelCfg)

class ProteinDiff(Base):
	def __init__(self, cfg: ProteinDiffCfg):
		super().__init__()
		self.tokenizer = Tokenizer(cfg.tokenizer)
		self.vae = VAEModel(cfg.vae)
		self.diffusion = DiffusionModel(cfg.diffusion)

	def forward(
		self,
		coords: Float[T, "ZN 14 3"],
		labels: Int[T, "ZN"],
		atom_mask: Bool[T, "ZN 14"],
		seq_idx: Int[T, "ZN"],
		chain_idx: Int[T, "ZN"],
		sample_idx: Int[T, "ZN"],
		cu_seqlens: Int[T, "Z+1"],
		max_seqlen: int,
		aa_conditioning: Int[T, "ZN"] | None = None,
		bb_conditioning: Float[T, "ZN"] | None = None,
		stage: TrainingStage = TrainingStage.VAE
	):
		"""meant for training only, use `generate` for inference"""

		coords_bb, divergence, frames = self.tokenizer(coords, labels, atom_mask)

		match stage:
			case TrainingStage.VAE:
				(
					latent, latent_mu, latent_logvar,
					divergence_pred, 
					seq_pred, 
					struct_logits,
					struct_head,
				) = self.vae_fwd(divergence, coords_bb, frames, seq_idx, chain_idx, sample_idx, cu_seqlens, max_seqlen)
				return ( 
					latent, latent_mu, latent_logvar,
					divergence_pred, divergence,
					seq_pred, 
					struct_logits, struct_head,
					coords_bb, frames, # return these two as theyre needed for loss
				) 
			case TrainingStage.DIFFUSION:
				pass
		
	def vae_fwd(
		self,
		divergence: Int[T, "ZN 1 Vx Vy Vz"],
		coords_bb: Float[T, "ZN 14 3"],
		frames: Float[T, "ZN 3 3"],
		seq_idx: Int[T, "ZN"],
		chain_idx: Int[T, "ZN"],
		sample_idx: Int[T, "ZN"],
		cu_seqlens: Int[T, "Z+1"],
		max_seqlen: int,
	):
		(
            latent, latent_mu, latent_logvar,
            divergence_pred, 
            seq_pred,
            struct_logits, struct_head,
        ) = self.vae(
			divergence,
			coords_bb,
			frames,
			seq_idx,
			chain_idx,
			sample_idx,
			cu_seqlens,
			max_seqlen,
		)

		return (
            latent, latent_mu, latent_logvar,
            divergence_pred, 
            seq_pred, 
            struct_logits, struct_head
		)

	def diffusion_fwd(self):
		pass

	def generate(self):
		pass