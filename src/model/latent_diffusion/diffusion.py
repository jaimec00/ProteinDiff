import torch
import torch.nn as nn
from utils.model_utils.latent_diffusion.diffusion_utils import NodeDenoiser, CosineScheduler

class Diffusion(nn.Module):
	def __init__(self, d_model=256, d_latent=16, layers=12, t_max=1000, top_k=16, parameterization="eps", min_rbf=2.0, max_rbf=22.0, num_rbf=16):
		super().__init__()

		self.d_latent = d_latent
		self.register_buffer("t_wavenumbers", 10000**-(2*torch.arange(d_model//2)/d_model))
		self.noise_scheduler = CosineScheduler(t_max)
		self.parameterization = parameterization

		self.latent_proj = nn.Linear(d_latent, d_model)
		self.denoiser = NodeDenoiser(d_model=d_model, layers=layers, top_k=top_k, min_rbf=min_rbf, max_rbf=max_rbf, num_rbf=num_rbf)
		self.pred_proj = nn.Linear(d_model, d_latent, bias=False)
	
	def forward(self, latent, t, C_backbone, frames, valid_mask):
		
		Z, N, d_latent, Vx, Vy, Vz = latent.shape
		latent = latent.reshape(Z, N, d_latent)
		edges, nbrs, nbr_mask = self.denoiser.get_constants(C_backbone, frames, valid_mask)

		pred = self.denoise(latent, t, edges, nbrs, nbr_mask).reshape(Z, N, d_latent, Vx, Vy, Vz)
		
		return pred

	def denoise(self, latent, t, edges, nbrs, nbr_mask):
		
		latent = self.latent_proj(latent)
		t = self.featurize_t(t)

		latent = self.denoiser(latent, t, edges, nbrs, nbr_mask)

		pred = self.pred_proj(latent)

		return pred

	def noise(self, latent, t):
		abars = self.noise_scheduler.get_abars(t).view(-1,1,1,1,1,1)
		noise = torch.randn_like(latent)
		noised_latent = (abars**0.5)*latent + ((1 - abars)**0.5)*noise

		if self.parameterization=="eps":
			trgt = noise
		elif self.parameterization=="x0":
			trgt = latent
		elif self.parameterization=="vpred":
			trgt = (abars**0.5)*noise - ((1 - abars)**0.5)*latent

		return noised_latent, trgt

	def generate(self, C_backbone, frames, valid_mask):

		# prep and start from white noise latents
		Z, N = valid_mask.shape
		latent = torch.randn([Z, N, self.d_latent], device=coords_bb.device)
		edges, nbrs, nbr_mask = self.denoiser.get_constants(C_backbone, frames, valid_mask)

		# initialize t
		t = torch.tensor([self.noise_scheduler.t_max-1], device=coords_bb.device)

		while t.item() >= 0:
			
			# predict noise
			pred = self.denoise(latent, t, edges, nbrs, nbr_mask)

			# remove noise
			latent = self.nudge(latent, pred, t)

			# next timestep
			t = t - 1

		latent = latent.reshape(Z, N, self.d_latent, 1,1,1)

		return latent

	def nudge(self, latent, pred, t): 
		'''
		uses DDIM
		'''

		abars = self.noise_scheduler.get_abars(t).view(-1, 1, 1)
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

		abars_tminus1 = self.noise_scheduler.get_abars(t-1).view(-1, 1, 1)
		alphas_tminus1 = abars_tminus1**0.5
		sigmas_tminus1 = (1-abars_tminus1)**0.5

		latent_tminus1 = alphas_tminus1*x0 + sigmas_tminus1*eps_t

		return latent_tminus1

	def get_rand_t_for(self, x):
		return torch.randint(0, self.noise_scheduler.t_max, (x.size(0),), device=x.device)

	def featurize_t(self, t):
		phase = self.t_wavenumbers.unsqueeze(0) * t.unsqueeze(1)
		sines = torch.sin(phase)
		cosines = torch.cos(phase)
		t = torch.stack([sines, cosines], dim=2).reshape(t.size(0), 1, -1)
		return t

