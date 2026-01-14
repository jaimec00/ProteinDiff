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

        x = self.up_proj(x)
        x = self.transformer(x, cu_seqlens, max_seqlen)
        divergence_pred = self.divergence_projection_head(x)
        seq_pred = self.seq_projection_head(x)

        '''
        thinking of having struct proj head's forward to simply return the callabl to implement the loss
        or maybe have it return the weights? no, i dont want the logic of how to run this outside of the module
        so basically my problem is im preparing for the future once the inputs are quite large
        i am using a zn tensor for the inputs, and 4 of these output are ZN x ZN. that is a lot of wasted mem,
        since if we had Z,N we would get Z,N,N. our upside is we dont have padding, so i think i can at least fix that issue
        will simply also take the sample idx as well and tmp pad the input, get a Z,N,N tensor, unpad (dims 1 and 2), to get a ZNN
        tensor
        the other problem was that I think it would be a good idea to make a 2d cel kernel in triton, where the logits and the coords are the input
        and we dont materialize a ZNN tensor. the more i think of it a triton kernel that directly computes the scaler loss is the way to go

        so the input would be 
        ZN,d_model logits, 
        ZN,3 for CaCb vecs
        Z+1 cu_seqlens

        we compute the labels and the losses on the fly
        '''

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