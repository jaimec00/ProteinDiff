import torch
import torch.nn as nn
from utils.model_utils.base_modules import adaLN, FiLM, MPNN, DiT
from utlis.model_utils.latent_diffusion.struct_enc import StructureEncoder

class Diffusion(nn.Module):
    def __init__(self, d_model, layers, t_max):
        super().__init__()

        self.struct_encoder = StructureEncoder()
        self.latent_encoder = None
        self.node_updater = None
        self.beta_scheduler = BetaScheduler(t_max)
        self.transformers = nn.ModuleList([Dit() for layer in layers])
        self.noise_proj = nn.Conv3d()
    
    def forward(self, latent, coords_bb, t, nbrs, nbr_mask):
        nodes, edges = self.structure_encoder(coords_bb, nbrs, nbr_mask)
        latent = self.latent_encoder(latent)
        t = self.featurize_t(t)

        for transformer in self.transformers:
            nodes = self.node_updater(latent, nodes)
            latent = transformer(latent, nodes, t)

        noise_pred = self.noise_conv(latent)

        return noise_pred

    def noise(self, latent, t):
        pass

    def generate(self, coords_bb, nbrs, nbr_mask):
        pass

    def get_rand_t_like(self, x)
        t = torch.randint(0, self.beta_scheduler.t_max, x.size(0), device=x.device).view(-1, 1, 1)
        return t

    def featurize_t(self, t):
        pass

class DiT(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()

    def forward(self):
        pass

class BetaScheduler(nn.Module):
    def __init__(self, t_max):
        super().__init__()
        self.t_max = t_max
