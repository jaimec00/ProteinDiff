import torch
import torch.nn as nn

from dataclasses import dataclass, field
from einops import rearrange

from proteindiff.model.base import Base
from proteindiff.model.vae.vae_utils import (
    UpsampleModel, UpsampleModelCfg,
    SeqProjectionHead, SeqProjectionHeadCfg,
    StructProjectionHead, StructProjectionHeadCfg,
)
from proteindiff.model.transformer.transformer import TransformerModel, TransformerModelCfg
from proteindiff.types import Float, Int, Bool, T, Tuple

@dataclass
class DecoderCfg:
    transformer: TransformerModelCfg = field(default_factory = TransformerModelCfg)
    divergence_projection_head: UpsampleModelCfg = field(default_factory = UpsampleModelCfg)
    seq_projection_head: SeqProjectionHeadCfg = field(default_factory = SeqProjectionHeadCfg)
    struct_projection_head: StructProjectionHeadCfg = field(default_factory = StructProjectionHeadCfg)


class Decoder(Base):
    def __init__(self, cfg: DecoderCfg):
        super().__init__()
        self.transformer = TransformerModel(cfg.transformer)
        self.divergence_projection_head = UpsampleModel(cfg.divergence_projection_head)
        self.seq_projection_head = SeqProjectionHead(cfg.seq_projection_head)
        self.struct_projection_head = StructProjectionHead(cfg.struct_projection_head)


    def forward(
        self,
        x: Float[T, "ZN d_latent"],
        cu_seqlens: Int[T, "Z+1"],
        max_seqlen: int,
    ) -> Tuple[
        Float[T, "ZN 1 Vx Vy Vz"],
        Float[T, "ZN n_aa"],
        Float[T, "ZN ZN d_dist"],
        Float[T, "ZN ZN d_angle"],
        Float[T, "ZN 3"],
        Float[T, "ZN 3"],
        Float[T, "ZN 3"],
        Float[T, "ZN 7"],
        Float[T, "ZN 7"],
        Float[T, "ZN ZN d_plddt"],
        Float[T, "ZN ZN d_pae"],
    ]:

        x = self.transformer(x, cu_seqlens, max_seqlen)
        divergence_pred = self.divergence_projection_head(x)
        seq_pred = self.seq_projection_head(x)
        distogram, anglogram, t, x, y, sin, cos, plddt, pae = self.struct_projection_head(x)

        return (
            divergence_pred, 
            seq_pred, 
            distogram, 
            anglogram, 
            t, x, y, 
            sin, cos,
			plddt, pae, 
        )