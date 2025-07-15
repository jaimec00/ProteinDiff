import torch
import torch.nn as nn
from utils.model_utils.latent_diffusion.diffusion_utils import StructureEncoder, LatentEncoder, GraphUpdater, DiT

class Diffusion(nn.Module):
	def __init__(self, d_model=128, d_latent=4, struct_enc_layers=3, transformer_layers=3, heads=4, t_max=1000, voxel_dims=(16,16,16)):
		super().__init__()

		self.Vx, self.Vy, self.Vz = voxel_dims
		self.d_latent = d_latent
		
		self.struct_encoder = StructureEncoder(d_model, struct_enc_layers)
		self.latent_encoder = LatentEncoder(d_latent, d_model)

		self.graph_updater = nn.ModuleList([GraphUpdater(d_model) for layer in range(transformer_layers)])
		self.transformers = nn.ModuleList([DiT(d_model, heads) for layer in range(transformer_layers)])

		self.beta_scheduler = BetaScheduler(t_max)
		self.register_buffer("t_wavenumbers", 10000**(2*torch.arange(d_model//2)/d_model))
		self.noise_proj = nn.Conv3d(d_model, d_latent, 1, padding="same", bias=False)
	
	def forward(self, latent, coords_bb, t, nbrs, nbr_mask):

		# prepare inputs
		latent, t = self.prep_inputs(latent, t)

		# get nodes and edges
		nodes, edges = self.structure_encoder(coords_bb, nbrs, nbr_mask)

		# increase latent feature dim, also reshapes so feature dim at the end
		latent = self.latent_encoder(latent)

		# predict noise
		noise_pred = self.denoise(latent, t, nodes, edges, nbrs, nbr_mask)

		return noise_pred

	def prep_inputs(self, latent, t):
		
		Z, N, d_latent, Vx, Vy, Vz = latent.shape
		latent = latent.view(Z*N, d_latent, Vx, Vy, Vz)

		# featurize timestep embeddings and reshape
		t = self.featurize_t(t)
		t = t.unsqueeze(1).expand(Z, N).view(Z*N)

		return latent, t

	def noise(self, latent, t):
		abars = self.beta_scheduler.get_abars(t)
		noise = torch.randn_like(latent)
		noised_latent = (abars**0.5)*latent + (1 - (abars**0.5))*noise
		return noised_latent, noise

	def denoise(self, latent, t, nodes, edges, nbrs, nbr_mask):

		Z, N, _ = nodes.shape

		for transformer in self.transformers:

			# condition nodes on latent state and pass messages
			nodes, edges = self.graph_updater(latent, nodes, edges, nbrs, nbr_msk)

			# run the transformer layer
			latent = transformer(latent, nodes, t)

		# project to predict noise
		noise_pred = self.noise_proj(latent)

		noise_pred = noise_pred.reshape(Z, N, self.d_latent, self.Vx, self.Vy, self.Vz)

		return noise_pred

	def generate(self, coords_bb, nbrs, nbr_mask):

		Z, N = coords.shape[:2]
		latent = torch.randn([Z, N, self.d_latent, self.Vx, self.Vy, self.Vz], device=coords.device)

		# get nodes and edges
		nodes, edges = self.structure_encoder(coords_bb, nbrs, nbr_mask)

		for t in range(self.beta_scheduler.t_max-1. -1, -1):
			
			# prepare inputs
			t = torch.tensor([t], device=latent.device)
			latent_t, et = self.prep_inputs(latent, t)

			# increase latent feature dim
			latent_t = self.latent_encoder(latent_t)

			# predict noise
			noise_pred = self.denoise(latent_t, et, nodes, edges, nbrs, nbr_mask)

			# remove noise
			latent = self.nudge(latent, noise_pred, t)

		return latent

	def nudge(self, latent, noise_pred, t):
		abars = self.beta_scheduler.get_abars(t).view(-1, 1, 1, 1, 1, 1)
		betas = self.beta_scheduler.get_betas(t).view(-1, 1, 1, 1, 1, 1)

		latent_tminus1 = (abars**-0.5)*(latent - betas*((1-abars)**-0.5)*noise_pred) + (betas**0.5)*torch.randn_like(latent)

		return latent_tminus1

	def get_rand_t_for(self, x):
		return torch.randint(0, self.beta_scheduler.t_max, x.size(0), device=x.device)

	def featurize_t(self, t):
		phase = self.t_wavenumbers.unsqueeze(0) * t.unsqueeze(1)
		sines = torch.sin(phase)
		cosines = torch.cos(phase)
		t = torch.stack([sines, cosines], dim=2).view(t.size(0), -1)
		return t

class BetaScheduler(nn.Module):
	def __init__(self, t_max, beta_min=1e-4, beta_max=2e-2):
		super().__init__()
		self.t_max = t_max
		self.beta_min = beta_min 
		self.beta_max = beta_max 
		self.abars = torch.cumprod(1 - torch.linspace(self.beta_min, self.beta_max, self.t_max), dim=0)

	def get_abars(self, t):
		abars = torch.gather(self.abars.unsqueeze(1).expand(-1, t.size(0)), 0, t.unsqueeze(0).expand(self.abars.size(0), -1))
		return abars

	def get_betas(self, t):
		betas = self.beta_min + (t/self.t_max)*(self.beta_max - self.beta_min)
		return betas