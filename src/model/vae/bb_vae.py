'''
this is the backbone encoder
plan is to pass the frames, bb coords, 
'''

import torch
import torch.nn as nn

from model.utils.base_modules import MLP, MPNN
from model.utils.kernels.attn import FlashAttn

class BackBoneVae(nn.Module):
    def __init__(self, d_model=256, top_k=16):
        super().__init__()
        self.enc = BackBoneEncoder()
        self.dec = BackBoneDecoder()

class BackBoneEncoder(nn.Module):
    def __init__(self, d_model=256, top_k=16, layers=3):
        super().__init__()
        self.edge_enc = EdgeEncoder(d_model, top_k)
        self.start_nodes = nn.Parameter(torch.randn((d_model,)))
        self.mpnns = nn.ModuleList([MPNN(d_model) for _ in range(layers)])
		self.latent_proj = nn.Linear(d_model, 2*d_model, bias=False)

    def forward(self):

        edges, nbrs, nbr_mask = self.edge_enc(coords_bb, frames, seq_pos, chain_pos, valid_mask)
        nodes = self.start_nodes.reshape(1,1,-1)

		for mpnn in self.mpnns:
			nodes, edges = mpnn(nodes, edges, nbrs, nbr_mask)

		z_mu, z_logvar = torch.chunk(self.latent_proj(nodes), dim=-1, chunks=2)
		z = torch.exp(0.5*z_logvar)*torch.randn_like(z_mu) + z_mu

		return z_mu, z_logvar, z


class BackBoneDecoder(nn.Module):

