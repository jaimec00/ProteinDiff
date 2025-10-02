import torch
import torch.nn as nn
from utils.model_utils.latent_diffusion.diffusion_utils import NodeDenoiser, CosineScheduler

class Diffusion(nn.Module):
	def __init__(self, d_model=256, d_sc_latent=16, d_bb_latent=256, layers=12, t_max=1000, top_k=16, parameterization="eps", min_rbf=2.0, max_rbf=22.0, num_rbf=16):
		super().__init__()

		self.d_sc_latent = d_sc_latent
		self.register_buffer("t_wavenumbers", 10000**-(2*torch.arange(d_model//2)/d_model))
		self.noise_scheduler = CosineScheduler(t_max)
		self.parameterization = parameterization

		self.latent_proj = MLP(d_in=d_sc_latent+d_bb_latent, d_out=d_model, d_hidden=d_model, num_hidden=1, act="silu")
		self.denoiser = NodeDenoiser(d_model=d_model, layers=layers, heads=heads)
		self.pred_proj = nn.Linear(d_model, d_sc_latent, bias=False)
	
	def forward(self, sc_latent, bb_latent, t, seq, valid_mask):
		
		Z, N, d_sc_latent, Vx, Vy, Vz = sc_latent.shape
		sc_latent = latent.reshape(Z, N, d_sc_latent)

		pred = self.denoise(sc_latent, bb_latent).reshape(Z, N, d_sc_latent, Vx, Vy, Vz)
		
		return pred

	def denoise(self, sc_latent, bb_latent, t, seq, valid_mask):
		
		latent = self.fuse_latents(sc_latent, bb_latent)
		condition = self.featurize_condition(t, seq)
		latent = self.denoiser(latent, condition, valid_mask)
		pred = self.pred_proj(latent)

		return pred

	def noise(self, latent, t):
		abars = self.noise_scheduler.get_abars(t).view(-1,1,1,1,1,1)
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

	def generate(self, bb_latent, seq, valid_mask):

		# prep and start from white noise latents
		Z, N = valid_mask.shape
		sc_latent = torch.randn([Z, N, self.d_latent], device=bb_latent.device)

		# initialize t
		t = torch.tensor([self.noise_scheduler.t_max-1], device=bb_latent.device)

		while t.item() >= 0:
			
			# predict noise
			pred = self.denoise(sc_latent, bb_latent, seq, t, valid_mask)

			# remove noise
			sc_latent = self.nudge(sc_latent, pred, t)

			# next timestep
			t = t - 1

		sc_latent = sc_latent.reshape(Z, N, self.d_latent, 1,1,1)

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

	def featurize_condition(self, t, seq):
		phase = self.t_wavenumbers.unsqueeze(0) * t.unsqueeze(1)
		sines = torch.sin(phase)
		cosines = torch.cos(phase)
		t = torch.stack([sines, cosines], dim=2).reshape(t.size(0), 1, -1)
		seq = self.seq_emb(seq)
		condition = self.conditioner(torch.cat([t, seq], dim=-1))
		return condition

	def fuse_latents(self, sc_latent, bb_latent):
		return self.latent_proj(torch.cat([sc_latent, bb_latent], dim=-1))