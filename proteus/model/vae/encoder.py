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
from proteus.types import Float, Int, Bool, T

@dataclass
class EncoderCfg:
    downsample: DownsampleModelCfg = field(default_factory = DownsampleModelCfg)
    mpnn: MPNNModelCfg = field(default_factory = MPNNModelCfg)
    transformer: TransformerModelCfg = field(default_factory = TransformerModelCfg)
    latent_projection_head: LatentProjectionHeadCfg = field(default_factory = LatentProjectionHeadCfg)

class Encoder(Base):
    def __init__(self, cfg: EncoderCfg):
        super().__init__()
        self.downsample = DownsampleModel(cfg.downsample)
        self.mpnn = MPNNModel(cfg.mpnn)
        self.transformer = TransformerModel(cfg.transformer)
        self.latent_projection_head = LatentProjectionHead(cfg.latent_projection_head)


    def forward(
        self,
        divergence: Float[T, "ZN 1 Vx Vy Vz"],
        bb_coords: Float[T, "ZN 4 3"],
        frames: Float[T, "ZN 3 3"],
        seq_idx: Int[T, "ZN"],
        chain_idx: Int[T, "ZN"],
        sample_idx: Int[T, "ZN"],
        cu_seqlens: Int[T, "Z+1"],
        max_seqlen: int,
    ) -> tuple[Float[T, "ZN d_latent"], Float[T, "ZN d_latent"], Float[T, "ZN d_latent"]]:

        x = self.downsample(divergence)
        x = self.mpnn(bb_coords, frames, seq_idx, chain_idx, cu_seqlens, x)
        x = self.transformer(x, cu_seqlens, max_seqlen)
        latent, mu, logvar = self.latent_projection_head(x)
        
        return latent, mu, logvar
