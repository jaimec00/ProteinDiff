'''
title: 			proteus.py
author: 		jaime cardenas
description:
'''


import torch
import torch.nn as nn

from dataclasses import dataclass, field
from enum import StrEnum

from proteus.model.base import Base
from proteus.model.vae.vae import VAEModel, VAEModelCfg
from proteus.model.diffusion.diffusion import DiffusionModel, DiffusionModelCfg
from proteus.model.tokenizer.tokenizer import Tokenizer, TokenizerCfg 
from proteus.types import Float, Int, Bool, T, Dict, Any, Optional

from proteus.static.constants import TrainingStage

@dataclass
class proteusCfg:
	tokenizer: TokenizerCfg = field(default_factory = TokenizerCfg)
	vae: VAEModelCfg = field(default_factory = VAEModelCfg)
	diffusion: DiffusionModelCfg = field(default_factory = DiffusionModelCfg)

class proteus(Base):
	def __init__(self, cfg: proteusCfg) -> None:
		super().__init__()
		self.tokenizer: Tokenizer = Tokenizer(cfg.tokenizer)
		self.vae: VAEModel = VAEModel(cfg.vae)
		self.diffusion: DiffusionModel = DiffusionModel(cfg.diffusion)

	def forward(
		self,
		coords: Float[T, "B L 14 3"],
		labels: Int[T, "B L"],
		atom_mask: Bool[T, "B L 14"],
		seq_idx: Int[T, "B L"],
		chain_idx: Int[T, "B L"],
		pad_mask: Bool[T, "B L"],
		aa_conditioning: Optional[Int[T, "B L"]] = None,
		bb_conditioning: Optional[Float[T, "B L"]] = None,
		stage: TrainingStage = TrainingStage.VAE
	) -> Dict[str, Any]:
		"""meant for training only, use `generate` for inference"""

		coords_bb, divergence, frames = self.tokenizer(coords, labels, atom_mask, pad_mask)

		match stage:
			case TrainingStage.VAE:
				outputs = self.vae_fwd(divergence, coords_bb, frames, seq_idx, chain_idx, pad_mask)
				outputs["divergence_true"] = divergence
				return outputs
			case TrainingStage.DIFFUSION:
				pass
		
	def vae_fwd(
		self,
		divergence: Float[T, "B L 1 Vx Vy Vz"],
		coords_bb: Float[T, "B L 4 3"],
		frames: Float[T, "B L 3 3"],
		seq_idx: Int[T, "B L"],
		chain_idx: Int[T, "B L"],
		pad_mask: Bool[T, "B L"],
	) -> Dict[str, Any]:
		return self.vae(
			divergence,
			coords_bb,
			frames,
			seq_idx,
			chain_idx,
			pad_mask,
		)

	def diffusion_fwd(self) -> None:
		pass

	def generate(self) -> None:
		pass