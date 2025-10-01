'''
this is the backbone encoder
plan is to pass the frames, bb coords, 
'''

import torch
import torch.nn as nn

from model.utils.base_modules import MLP, MPNN, Transformer

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

    def forward(self, coords_bb, frmaes, seq_pos, chain_pos, valid_mask):

        edges, nbrs, nbr_mask = self.edge_enc(coords_bb, frames, seq_pos, chain_pos, valid_mask)
        nodes = self.start_nodes.reshape(1,1,-1)

		for mpnn in self.mpnns:
			nodes, edges = mpnn(nodes, edges, nbrs, nbr_mask)

		z_mu, z_logvar = torch.chunk(self.latent_proj(nodes), dim=-1, chunks=2)
		z = torch.exp(0.5*z_logvar)*torch.randn_like(z_mu) + z_mu

		return z_mu, z_logvar, z


class BackBoneDecoder(nn.Module):
    def __init__(self, d_model, heads, layers):
        super().__init__()

        # uses flash attention 3
        self.transformers = nn.ModuleList([Transformer(d_model, heads) for _ in range(layers)])
        self.pw_classification_head = PairwiseProjHead(d_model, 128, 64, 16)

    def forward(self, x, valid_mask):
        '''
        passes the structure latents through N transformer layers
        use these logits to compute a LxLx4x4x64 distogram, and LxLx3x16 anglogram
        '''

        for transformer in self.transformers:
            x = transformer(x, valid_mask)

        distogram, anglogram = self.pw_classification_head(x)

        return distogram, anglogram

class PairwiseProjHead(nn.Module):
    def __init__(self, d_model, d_down, dist_bins, angle_bins):
        super().__init__()
        self.downsample = nn.Linear(d_model, d_down)
        self.bin = MLP(d_model=2*d_down, d_hidden=dist_bins+angle_bins, d_out=dist_bins+angle_bins, num_hidden=1, act="silu")
        self._dist_bins = dist_bins
        self._angle_bins = angle_bins

    def forward(self, x):

        q, k = torch.chunk(self.downsample(x), chunks=2, dim=-1) # Z x N x D//2
        q_i, k_j = q.unsqueeze(2), k.unsqueeze(1)
        prod, diff = q_i*k_j, k_j-q_i
        pw = torch.cat([prod, diff], dim=-1) # Z x N x N x D
        binned = self.bin(pw)
        torch.chunk(binned, chunks=[self._dist_bins, self._angle_bins], dim=-1)
