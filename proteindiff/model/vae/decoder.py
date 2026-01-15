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
    d_latent: int = 16
    d_model: int = 64
    transformer: TransformerModelCfg = field(default_factory = TransformerModelCfg)
    divergence_projection_head: UpsampleModelCfg = field(default_factory = UpsampleModelCfg)
    seq_projection_head: SeqProjectionHeadCfg = field(default_factory = SeqProjectionHeadCfg)
    struct_projection_head: StructProjectionHeadCfg = field(default_factory = StructProjectionHeadCfg)


class Decoder(Base):
    def __init__(self, cfg: DecoderCfg):
        super().__init__()
        self.up_proj = nn.Linear(cfg.d_latent, cfg.d_model)
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
        Float[T, "ZN d_model"],
    ]:

        x = self.up_proj(x)
        x = self.transformer(x, cu_seqlens, max_seqlen)
        divergence_pred = self.divergence_projection_head(x)
        seq_pred = self.seq_projection_head(x)
        
        # these are passed to the struct_projection head later to directly compute loss
        # avoids materializing large tensors for pairwise predictions via fused kernels
        struct_logits = x

        return (
            divergence_pred, 
            seq_pred, 
            struct_logits,
            self.struct_projection_head,
        )