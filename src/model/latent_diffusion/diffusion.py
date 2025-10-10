import torch
import torch.nn as nn
from utils.model_utils.latent_diffusion.diffusion_utils import NodeDenoiser, CosineScheduler

class Diffusion(nn.Module):
	def __init__(self, d_model=256, d_sc_latent=16, d_bb_latent=256, layers=12, heads=8, t_max=1000, parameterization="eps"):
		super().__init__()

		self.d_sc_latent = d_sc_latent
		self.register_buffer("t_wavenumbers", 10000**-(2*torch.arange(d_model//2)/d_model))
		self.noise_scheduler = CosineScheduler(t_max)
		self.parameterization = parameterization

		self.latent_proj = MLP(d_in=d_sc_latent+d_bb_latent, d_out=d_model, d_hidden=d_model, num_hidden=1, act="silu")
		self.denoiser = NodeDenoiser(d_model=d_model, layers=layers, heads=heads)
		self.pred_proj = nn.Linear(d_model, d_sc_latent, bias=False)
	
	def forward(self, sc_latent, bb_latent, t, seq, cu_seqlens, max_seqlen):
		
		return self.denoise(sc_latent, bb_latent, t, seq, cu_seqlens, max_seqlen)

	def denoise(self, sc_latent, bb_latent, t, seq, cu_seqlens, max_seqlen):
		
		latent = self.fuse_latents(sc_latent, bb_latent)
		condition = self.featurize_condition(t, seq)
		latent = self.denoiser(latent, condition, cu_seqlens, max_seqlen)
		pred = self.pred_proj(latent)

		return pred

	def noise(self, latent, t):
		abars = self.noise_scheduler.get_abars(t).unsqueeze(-1)
		noise = torch.randn_like(latent)
		noised_latent = (abars**0.5)*latent + ((1 - abars)**0.5)*noise

		match self.parameterization:
			case "eps":
				trgt = noise
			case "x0":
				trgt = latent
			case "vpred":
				trgt = (abars**0.5)*noise - ((1 - abars)**0.5)*latent

		return noised_latent, trgt

	def generate(self, bb_latent, seq, cu_seqlens, max_seqlen, step_size=1):

		# prep and start from white noise latents
		ZN = bb_latent.size(0)
		device = bb_latent.device
		sc_latent = torch.randn([ZN, self.d_latent], device=device)

		# initialize t
		t = torch.tensor([self.noise_scheduler.t_max-1], device=device)

		while t.item() >= 0:
			
			# predict noise
			pred = self.denoise(sc_latent, bb_latent, seq, t, cu_seqlens, max_seqlen)

			# remove noise
			sc_latent = self.nudge(sc_latent, pred, t, step_size=step_size)

			# next timestep
			t = t - step_size

		return sc_latent

	def nudge(self, latent, pred, t, step_size=1): 
		'''
		uses DDIM
		'''

		abars = self.noise_scheduler.get_abars(t).reshape(-1, 1)
		alphas = abars**0.5
		sigmas = (1-abars)**0.5

		if self.parameterization=="eps":
			eps_t = pred
			x0 = (latent - eps_t*sigmas)/alphas
		elif self.parameterization=="x0":
			x0 = pred
			eps_t = (latent - x0*alphas)/sigmas
		elif self.parameterization=="vpred":
			x0 = (alphas*latent - sigmas*v_pred) /  (alphas**2 + sigmas**2)
			eps_t = (sigmas*latent + alphas*v_pred) /  (alphas**2 + sigmas**2)

		abars_tminus_step = self.noise_scheduler.get_abars(max(0,t-step_size)).reshape(-1, 1)
		alphas_tminus_step = abars_tminus_step**0.5
		sigmas_tminus_step = (1-abars_tminus_step)**0.5

		latent_tminus_step = alphas_tminus_step*x0 + sigmas_tminus_step*eps_t

		return latent_tminus_step

	def get_rand_t_for(self, x):
		return torch.randint(0, self.noise_scheduler.t_max, (x.size(0),), device=x.device)

	def featurize_condition(self, t, seq):
		phase = self.t_wavenumbers.unsqueeze(0) * t.unsqueeze(1)
		sines = torch.sin(phase)
		cosines = torch.cos(phase)
		t = torch.stack([sines, cosines], dim=-1).reshape(t.size(0), -1)
		seq = self.seq_emb(seq)
		condition = self.conditioner(torch.cat([t, seq], dim=-1))
		return condition

	def fuse_latents(self, sc_latent, bb_latent):
		return self.latent_proj(torch.cat([sc_latent, bb_latent], dim=-1))