
from dataclasses import dataclass, field
from omegaconf import II

import torch
import torch.nn as nn

from proteus.model.base import Base
from proteus.model.tokenizer.wave_func_tokenizer.learn_aa.wf_embedding_learn_aa import wf_embedding_learn_aa 
from proteus.model.tokenizer.wave_func_tokenizer.static_aa.wf_embedding_static_aa import wf_embedding_static_aa
from proteus.model.model_utils.mlp import UpsampleMLP, UpsampleMLPCfg 
from proteus.static.constants import alphabet
from proteus.types import T, Float, Int

@dataclass
class WaveFunctionTokenizerCfg:
	d_model: int = II("model.d_model")
	d_wf: int = II(".d_model")
	min_wl: float = 3.0
	max_wl: float = 15.0
	base_wl: float = 20.0
	up_proj: UpsampleMLPCfg = field(default_factory=lambda: UpsampleMLPCfg(d_in=II("..d_wf"), d_out=II("..d_model")))

class WaveFunctionTokenizer(Base):

	def __init__(self, cfg: WaveFunctionTokenizerCfg):
		super().__init__()

		wl = cfg.min_wl + ((cfg.max_wl-cfg.min_wl) * ((torch.logspace(0,1,cfg.d_wf//2, cfg.base_wl) - 1) / (cfg.base_wl-1)))
		wn = torch.log(torch.pi*2/wl)
		self.register_buffer("wavenumbers", wn)
		# self.aa_magnitudes = nn.Embedding(len(alphabet), cfg.d_wf//2)
		self.aa_magnitudes = nn.Parameter(torch.zeros(cfg.d_wf//2, len(alphabet)), requires_grad=True)
		self.up_proj = UpsampleMLP(cfg.up_proj)

	def _get_wavenumbers(self):
		return torch.exp(self.wavenumbers) # learns log of wavenumbers, ie log(2pi/lambda) = log(2pi) - log(lambda), log(2pi) is constant, so learning -log(lambda)

	def forward(
		self, 
		coords_alpha: Float[T, "BL 3"], 
		coords_beta: Float[T, "BL 3"], 
		aas: Int[T, "BL"],
		cu_seqlens: Int[T, "B+1"],
	):

		if self.training and torch.is_grad_enabled() and self.aa_magnitudes.requires_grad:
			# wf = wf_embedding_learn_aa(coords_alpha, coords_beta, aas, self.aa_magnitudes.weight.T, self._get_wavenumbers(), cu_seqlens)
			wf = wf_embedding_learn_aa(coords_alpha, coords_beta, aas, self.aa_magnitudes, self._get_wavenumbers(), cu_seqlens)
		else:
			# wf = wf_embedding_static_aa(coords_alpha, coords_beta, aas, self.aa_magnitudes.weight.T, self._get_wavenumbers(), cu_seqlens)
			wf = wf_embedding_static_aa(coords_alpha, coords_beta, aas, self.aa_magnitudes, self._get_wavenumbers(), cu_seqlens)

		tokens = self.up_proj(wf)

		return tokens