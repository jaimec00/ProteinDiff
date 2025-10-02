'''
this is the backbone encoder
plan is to pass the frames, bb coords, 
'''

import torch
import torch.nn as nn

from model.utils.base_modules import MLP
from model.vae.vae_utils import MPNN, Transformer, PairwiseProjHead
from static.constants import canonical_aas

class BackBoneVae(nn.Module):
    def __init__(self, d_model=256, top_k=16, enc_layers=3, dec_layers=3, heads=8):
        super().__init__()
        self.enc = BackBoneEncoder(d_model=d_model, top_k=top_k, layers=enc_layers)
        self.dec = BackBoneDecoder(d_model=d_model, heads=heads, layers=dec_layers)

    def forward(self, coords_bb, frames, seq_pos, chain_pos, valid_mask):
        z, z_mu, z_logvar = self.enc(coords_bb, frames, seq_pos, chain_pos, valid_mask)
        distogram, anglogram, seq_pred = self.dec(z, valid_mask)

        return z, z_mu, z_logvar, distogram, anglogram, seq_pred

class BackBoneEncoder(nn.Module):
    def __init__(self, d_model=256, top_k=16, layers=3):
        super().__init__()
        self.edge_enc = EdgeEncoder(d_model, top_k)
        self.start_nodes = nn.Parameter(torch.randn((d_model,)))
        self.mpnns = nn.ModuleList([MPNN(d_model, update_edges=i<(layers-1)) for i in range(layers)])
		self.latent_proj = nn.Linear(d_model, 2*d_model, bias=False)

    def forward(self, coords_bb, frmaes, seq_pos, chain_pos, valid_mask):

        edges, nbrs, nbr_mask = self.edge_enc(coords_bb, frames, seq_pos, chain_pos, valid_mask)
        nodes = self.start_nodes.reshape(1,1,-1)

		for mpnn in self.mpnns:
			nodes, edges = mpnn(nodes, edges, nbrs, nbr_mask)

		z_mu, z_logvar = torch.chunk(self.latent_proj(nodes), dim=-1, chunks=2)
		z = torch.exp(0.5*z_logvar)*torch.randn_like(z_mu) + z_mu

		return z, z_mu, z_logvar

class BackBoneDecoder(nn.Module):
    def __init__(self, d_model, heads, layers):
        super().__init__()

        # uses flash attention 3
        self.transformers = nn.ModuleList([Transformer(d_model, heads) for _ in range(layers)])
        self.pw_classification_head = PairwiseProjHead(d_model, 128, 64, 16)
        self.classifier = nn.Linear(d_model, len(canonical_aas))

    def forward(self, x, valid_mask):
        '''
        passes the structure latents through N transformer layers
        use these logits to compute a LxLx4x4x64 distogram, and LxLx3x16 anglogram
        '''

        for transformer in self.transformers:
            x = transformer(x, valid_mask)

        distogram, anglogram = self.pw_classification_head(x)
        seq_pred = self.classifier(x)

        return distogram, anglogram, seq_pred
