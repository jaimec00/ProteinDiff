import torch
import torch.nn as nn
from utils.model_utils.latent_diffusion.diffusion_utils import NodeDenoiser, BetaScheduler, CosineScheduler

class Diffusion(nn.Module):
	def __init__(self, d_model=256, top_k=32, d_latent=16, layers=12, t_max=1000):
		super().__init__()

		self.d_latent = d_latent
		self.register_buffer("t_wavenumbers", 10000**-(2*torch.arange(d_model//2)/d_model))
		self.noise_scheduler = CosineScheduler(t_max)

		self.latent_proj = nn.Linear(d_latent, d_model)
		self.denoiser = NodeDenoiser(d_model=d_model, top_k=top_k, layers=layers)
		self.v_proj = nn.Linear(d_model, d_latent, bias=False)
	
	def forward(self, coords_bb, latent, t, valid_mask):
		
		Z, N, d_latent, Vx, Vy, Vz = latent.shape
		latent = latent.reshape(Z, N, d_latent)
		edges, nbrs, nbr_mask = self.denoiser.get_constants(coords_bb, valid_mask)

		v_pred = self.denoise(latent, t, edges, nbrs, nbr_mask).reshape(Z, N, d_latent, Vx, Vy, Vz)
		
		return v_pred

	def denoise(self, latent, t, edges, nbrs, nbr_mask):
		
		latent = self.latent_proj(latent)
		t = self.featurize_t(t)

		latent = self.denoiser(latent, t, edges, nbrs, nbr_mask)

		v_pred = self.v_proj(latent)

		return v_pred

	def noise(self, latent, t):
		abars = self.noise_scheduler.get_abars(t).view(-1,1,1,1,1,1)
		noise = torch.randn_like(latent)
		noised_latent = (abars**0.5)*latent + ((1 - abars)**0.5)*noise

		v = (abars**0.5)*noise - ((1 - abars)**0.5)*latent

		return noised_latent, v

	def generate(self, coords_bb, valid_mask):

		# prep and start from white noise latents
		Z, N = coords_bb.shape[:2]
		latent = torch.randn([Z, N, self.d_latent], device=coords_bb.device)

		# get nbrs and edges
		edges, nbrs, nbr_mask = self.denoiser.get_constants(coords_bb, valid_mask)

		# initialize t
		t = torch.tensor([self.noise_scheduler.t_max-1], device=coords_bb.device)

		while t.item() >= 0:
			
			# predict noise
			v_pred = self.denoise(latent, t, edges, nbrs, nbr_mask)

			# remove noise
			latent = self.nudge(latent, v_pred, t)

			# next timestep
			t = t - 1

		latent = latent.reshape(Z, N, self.d_latent, 1,1,1)

		return latent

	def nudge(self, latent, v_pred, t): # need to adjust this if i choose to go with v pred

		abars = self.noise_scheduler.get_abars(t).view(-1, 1, 1)
		betas = self.noise_scheduler.get_betas(t).view(-1, 1, 1)

		noise_pred = (abars**0.5)*v_pred + ((1 - abars)**0.5)*latent

		latent_tminus1 = (abars**-0.5)*(latent - betas*((1-abars)**-0.5)*noise_pred) + (betas**0.5)*torch.randn_like(latent)

		return latent_tminus1

	def get_rand_t_for(self, x):
		return torch.randint(0, self.noise_scheduler.t_max, (x.size(0),), device=x.device)

	def featurize_t(self, t):
		phase = self.t_wavenumbers.unsqueeze(0) * t.unsqueeze(1)
		sines = torch.sin(phase)
		cosines = torch.cos(phase)
		t = torch.stack([sines, cosines], dim=2).reshape(t.size(0), 1, -1)
		return t

