import torch
import torch.nn as nn

from dataclasses import dataclass, field
from einops import rearrange

from proteus.model.base import Base
from proteus.model.vae.vae_utils import (
    DownsampleModel, DownsampleModelCfg, 
    LatentProjectionHead, LatentProjectionHeadCfg,
)
from proteus.model.mpnn.mpnn import MPNNModel, MPNNModelCfg
from proteus.model.transformer.transformer import TransformerModel, TransformerModelCfg
from proteus.types import Float, Int, Bool, T, Tuple
from proteus.utils.tensor import unpad, repad

@dataclass
class EncoderCfg:
    downsample: DownsampleModelCfg = field(default_factory = DownsampleModelCfg)
    mpnn: MPNNModelCfg = field(default_factory = MPNNModelCfg)
    transformer: TransformerModelCfg = field(default_factory = TransformerModelCfg)
    latent_projection_head: LatentProjectionHeadCfg = field(default_factory = LatentProjectionHeadCfg)

class Encoder(Base):
    def __init__(self, cfg: EncoderCfg) -> None:
        super().__init__()
        self.downsample: DownsampleModel = DownsampleModel(cfg.downsample)
        self.mpnn: MPNNModel = MPNNModel(cfg.mpnn)
        self.transformer: TransformerModel = TransformerModel(cfg.transformer)
        self.latent_projection_head: LatentProjectionHead = LatentProjectionHead(cfg.latent_projection_head)


    def forward(
        self,
        divergence: Float[T, "B L 1 Vx Vy Vz"],
        bb_coords: Float[T, "B L 4 3"],
        frames: Float[T, "B L 3 3"],
        seq_idx: Int[T, "B L"],
        chain_idx: Int[T, "B L"],
        pad_mask: Bool[T, "B L"],
    ) -> Tuple[Float[T, "B L d_latent"], Float[T, "B L d_latent"], Float[T, "B L d_latent"]]:

        # All modules now accept B,L format with pad_mask
        x = self.downsample(divergence, pad_mask)
        x = self.mpnn(bb_coords, frames, seq_idx, chain_idx, pad_mask, x)
        x = self.transformer(x, pad_mask)

        # Latent projection head - unpack for processing
        [x_u], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)

        latent_u, mu_u, logvar_u = self.latent_projection_head(x_u)

        [latent, mu, logvar] = repad(
            latent_u, mu_u, logvar_u,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )

        return latent, mu, logvar
