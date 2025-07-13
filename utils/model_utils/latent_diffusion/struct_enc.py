import torch
import torch.nn as nn
from utils.model_utils.base_modules improt MLP, MPNN
from data.constants import alphabet

class StructureEncoder(nn.Module):
    def __init__(self, d_model=128, layers=3, dropout=0.0):
        super().__init__()

        # might try to intiialize context nodes to seq instead of zeros as conditioning, but not for now
        # seq_emb = nn.Embedding(len(alphabet), d_model)
        min_rbf, max_rbf, num_rbf = 2.0,22.0,16

        self.register_buffer("V_start", torch.zeros(d_model))
        self.register_buffer("rbf_centers", torch.linspace(min_rbf, max_rbf, num_rbf))
        self.spread = (max_rbf - min_rbf) / num_rbf

        self.edge_norm = nn.LayerNorm(d_model)
        self.edge_proj = nn.Linear(4*4*16, d_model)

        self.encs = nn.ModuleList([MPNN(d_model, update_edges=True, dropout=dropout) for _ in layers])

    def forward(C_backbone, L, nbrs, nbr_mask):

        Z, N, _ = C_backbone.shape

        V = self.V_start.view(1,1,-1).expand(Z, N, -1)
        E = self.get_edges(C_backbone, nbrs)

        for enc in self.encs:
            V, E = enc(V, E, nbrs, nbr_mask)

        return V, E

    def get_edges(C_backbone, nbrs):

        Z, N, S = C_backbone.shape
        _, _, K = nbrs.shape

        C_nbrs = torch.gather(C_backbone.unsqueeze(2).expand(Z, N, K, S), 1, nbrs.unsqueeze(3).expand(Z, N, K, S)) # Z,N,K,S

        dists = torch.sqrt(torch.sum((C_backbone.unsqueeze(2) - C_nbrs)**2), dim=3) # Z,N,1,S - Z,N,K,S --> Z,N,K

        rbf_numerator = (dists.view(Z, N, K, 1) - self.rbf_centers.view(1,1,1,-1))**2

        rbf = torch.exp(-rbf_numerator / (self.spread**2))

        edges = self.edge_proj(self.edge_norm(rbf))

        return edges
