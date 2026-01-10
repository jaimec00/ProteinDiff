import torch
import torch.nn as nn

from dataclasses import dataclass, field
from einops import rearrange

from proteindiff.model import Base
from proteindiff.model.vae.vae_utils import ResNetModel, ResNetModelCfg
from proteindiff.model.mpnn import MPNNModel, MPNNModelCfg
from proteindiff.model.transformer import TransformerModel, TransformerModelCfg
from proteindiff.typing import Float, Int, Bool, T

@dataclass
class EncoderCfg:
    resnet: ResNetModelCfg = field(default_factory = ResNetModelCfg)
    mpnn: MPNNModelCfg = field(default_factory = MPNNModelCfg)
    transformer: TransformerModelCfg = field(default_factory = TransformerModelCfg)
    latent_projection: LatentProjectionCfg = field(default_factory = LatentProjectionCfg)

class Encoder(Base):
    def __init__(self, cfg: EncoderCfg):
        super().__init__()
        self.resnet = ResNetModel(cfg.resnet)
        self.mpnn = MPNNModel(cfg.mpnn)
        self.transformer = TransformerModel(cfg.transformer)
        self.latent_projection = LatentProjection(cfg.latent_projection)


    def forward(
        self,
        divergence: Float[T, "ZN 1 Vx Vy Vz"],
        bb_coords: Float[T, "ZN 4 3"],
        frames: Float[T, "ZN 3 3"],
        seq_idx: Int[T, "ZN"],
        chain_idx: Int[T, "ZN"],
        sample_idx: Int[T, "ZN"],
        cu_seqlens: Int[T, "Z"],
        max_seqlen: int,
    ) -> Float[T, "ZN d_model"]:

        x = self.resnet(divergence)
        x = rearrange(x, "ZN d_model 1 1 1 -> ZN (d_model 1 1 1)")
        x = self.mpnn(bb_coords, frames, seq_idx, chain_idx, sample_idx, x)
        x = self.transformer(x, cu_seqlens, max_seqlen)
        latent, mean, logvar = self.sample_latent(x)

        return latent, mean, logvar
