import torch
import torch.nn as nn

from dataclasses import dataclass, field

from proteus.model.diffusion.diffusion_utils import (
	CosineScheduler, 
	Parameterization,
	DiTModel, DiTModelCfg, 
	Conditioner, ConditionerCfg
)
from proteus.model.model_utils.mlp import ProjectionHead, ProjectionHeadCfg
from proteus.model.base import Base
from proteus.types import Float, Int, T

@dataclass
class DiffusionModelCfg:
	d_model: int = 256
	d_latent: int = 16
	t_max: int = 1000
	parameterization: Parameterization = Parameterization.DEFAULT
	conditioner: ConditionerCfg = field(default_factory = ConditionerCfg)
	denoiser: DiTModelCfg = field(default_factory = DiTModelCfg)

class DiffusionModel(Base):
	def __init__(self, cfg: DiffusionModelCfg):
		super().__init__()

		self.d_latent = cfg.d_latent
		self.noise_scheduler = CosineScheduler(cfg.t_max)
		self.parameterization = cfg.parameterization
		self.latent_proj = ProjectionHead(ProjectionHeadCfg(d_in=cfg.d_latent, d_out=cfg.d_model))
		self.conditioner = Conditioner(cfg.conditioner)
		self.denoiser = DiTModel(cfg.denoiser)
		self.pred_proj = nn.Linear(cfg.d_model, cfg.d_latent, bias=False)

	
	def forward(
		self,
		latent: Float[T, "ZN d_latent"],
		seq: Int[T, "ZN"],
		coords_bb: Float[T, "ZN 4 3"],
		frames: Float[T, "ZN 3 3"],
		seq_idx: Int[T, "ZN"],
		chain_idx: Int[T, "ZN"],
		sample_idx: Int[T, "ZN"],
		t: Int[T, "ZN"],
		cu_seqlens: Int[T, "Z+1"],
		max_seqlen: int,
	) -> Float[T, "ZN d_latent"]:

		conditioning = self.conditioner(seq, coords_bb, frames, seq_idx, chain_idx, sample_idx, t, cu_seqlens, max_seqlen)
		return self.denoise(latent, conditioning, cu_seqlens, max_seqlen)

	def denoise(
		self,
		latent: Float[T, "ZN d_latent"],
		conditioning: Float[T, "ZN d_model"],
		cu_seqlens: Int[T, "Z+1"],
		max_seqlen: int,
	) -> Float[T, "ZN d_latent"]:
		latent = self.latent_proj(latent)
		latent = self.denoiser(latent, conditioning, cu_seqlens, max_seqlen)
		noise_pred = self.pred_proj(latent)

		return noise_pred

	def noise(
		self,
		latent: Float[T, "ZN d_latent"],
		t: Int[T, "ZN"]
	) -> tuple[Float[T, "ZN d_latent"], Float[T, "ZN d_latent"]]:
		abars = self.noise_scheduler.get_abars(t).unsqueeze(-1)
		noise = torch.randn_like(latent)
		noised_latent = (abars**0.5)*latent + ((1 - abars)**0.5)*noise

		match self.parameterization:
			case Parameterization.EPS:
				trgt = noise
			case Parameterization.X0:
				trgt = latent
			case Parameterization.VPRED:
				trgt = (abars**0.5)*noise - ((1 - abars)**0.5)*latent

		return noised_latent, trgt

	def generate(
		self,
		seq: Int[T, "ZN"],
		coords_bb: Float[T, "ZN 4 3"],
		frames: Float[T, "ZN 3 3"],
		seq_idx: Int[T, "ZN"],
		chain_idx: Int[T, "ZN"],
		sample_idx: Int[T, "ZN"],
		cu_seqlens: Int[T, "Z+1"],
		max_seqlen: int,
		step_size: int = 1,
	) -> Float[T, "ZN d_latent"]:
		# prep and start from white noise latents
		device = seq.device
		ZN = seq.shape[0]
		seq_coords_condition = self.conditioner.seq_coords_conditioning(
			seq,
			coords_bb,
			frames,
			seq_idx,
			chain_idx,
			sample_idx,
			cu_seqlens,
			max_seqlen,
		)
		latent = torch.randn((ZN, self.d_latent), device=device)

		# initialize t
		t = torch.tensor([self.noise_scheduler.t_max-1], device=device)

		while t.item() >= 0:
			
			# get conditioning
			t_condition = self.conditioner.featurize_t(t)
			condition = self.conditioner.combine_conditioning(seq_coords_condition, t_condition.expand(ZN, -1))

			# predict noise
			pred = self.denoise(latent, condition, cu_seqlens, max_seqlen)

			# remove noise
			latent = self.nudge(latent, pred, t, step_size=step_size)

			# next timestep
			t = t - step_size

		return latent

	def nudge(
		self,
		latent: Float[T, "ZN d_latent"],
		pred: Float[T, "ZN d_latent"],
		t: Int[T, "..."],
		step_size: int = 1
	) -> Float[T, "ZN d_latent"]: 
		'''
		uses DDIM
		'''

		abars = self.noise_scheduler.get_abars(t).reshape(-1, 1)
		alphas = abars**0.5
		sigmas = (1-abars)**0.5

		if self.parameterization==Parameterization.EPS:
			eps_t = pred
			x0 = (latent - eps_t*sigmas)/alphas
		elif self.parameterization==Parameterization.X0:
			x0 = pred
			eps_t = (latent - x0*alphas)/sigmas
		elif self.parameterization==Parameterization.VPRED:
			x0 = (alphas*latent - sigmas*pred) /  (alphas**2 + sigmas**2)
			eps_t = (sigmas*latent + alphas*pred) /  (alphas**2 + sigmas**2)

		abars_tminus_step = self.noise_scheduler.get_abars((t-step_size).clamp(min=0)).reshape(-1, 1)
		alphas_tminus_step = abars_tminus_step**0.5
		sigmas_tminus_step = (1-abars_tminus_step)**0.5

		latent_tminus_step = alphas_tminus_step*x0 + sigmas_tminus_step*eps_t

		return latent_tminus_step

	def get_rand_t_for(self, x: Float[T, "ZN ..."]) -> Int[T, "ZN"]:
		return torch.randint(0, self.noise_scheduler.t_max, (x.size(0),), device=x.device)
