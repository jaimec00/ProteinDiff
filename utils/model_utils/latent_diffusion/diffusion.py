import torch
import torch.nn as nn
from utils.model_utils.latent_diffusion.diffusion_utils import StructureEncoder, LatentEncoder, NodeUpdater, DiT

class Diffusion(nn.Module):
    def __init__(self, d_model=128, d_latent=4, struct_enc_layers=3, transformer_layers=3, heads=4, t_max=1000):
        super().__init__()

        self.struct_encoder = StructureEncoder(d_model, struct_enc_layers)
        self.latent_encoder = LatentEncoder(d_latent, d_model)

        self.node_updater = nn.ModuleList([NodeUpdater(d_model) for layer in transformer_layers])
        self.transformers = nn.ModuleList([DiT(d_model, heads) for layer in transformer_layers])

        self.beta_scheduler = BetaScheduler(t_max)
        self.register_buffer("t_wavenumbers", torch.linspace(0, d_model/2))
        self.noise_proj = nn.Linear(d_model, d_latent)
    
    def forward(self, latent, coords_bb, t, nbrs, nbr_mask):

        Z, N, d_latent, Vx, Vy, Vz = latent.shape
        latent = latent.view(Z*N, d_latent, Vx, Vy, Vz)

        # get nodes and edges
        nodes, edges = self.structure_encoder(coords_bb, nbrs, nbr_mask)

        # increase latent feature dim, also reshapes so feature dim at the end
        latent = self.latent_encoder(latent)

        # featurize timestep embeddings
        t = self.featurize_t(t)

        for transformer in self.transformers:

            # condition nodes on latent state and pass messages
            nodes = self.node_updater(latent, nodes)

            # run the transformer layer
            latent = transformer(latent, nodes, t)

        # project to predict noise
        noise_pred = self.noise_proj(latent)

        # reshape 

        return noise_pred

    def noise(self, latent, t):
        pass

    def generate(self, coords_bb, nbrs, nbr_mask):
        pass

    def get_rand_t_like(self, x)
        return torch.randint(0, self.beta_scheduler.t_max, x.size(0), device=x.device)
        

    def featurize_t(self, t):
        pass


class BetaScheduler(nn.Module):
    def __init__(self, t_max, beta_min=):
        super().__init__()
        self.t_max = t_max
