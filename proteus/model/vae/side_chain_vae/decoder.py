import torch
import torch.nn as nn

from dataclasses import dataclass, field
from einops import rearrange

from proteus.model.base import Base
from proteus.model.vae.vae_utils import (
    UpsampleModel, UpsampleModelCfg,
    SeqProjectionHead, SeqProjectionHeadCfg,
    StructProjectionHead, StructProjectionHeadCfg,
)
from proteus.model.transformer.transformer import TransformerModel, TransformerModelCfg
from proteus.types import Float, Int, Bool, T, Tuple, Dict, Any
from proteus.utils.tensor import unpad, repad

@dataclass
class DecoderCfg:
    d_latent: int = 16
    d_model: int = 64
    transformer: TransformerModelCfg = field(default_factory = TransformerModelCfg)
    divergence_projection_head: UpsampleModelCfg = field(default_factory = UpsampleModelCfg)
    seq_projection_head: SeqProjectionHeadCfg = field(default_factory = SeqProjectionHeadCfg)
    struct_projection_head: StructProjectionHeadCfg = field(default_factory = StructProjectionHeadCfg)


class Decoder(Base):
    def __init__(self, cfg: DecoderCfg) -> None:
        super().__init__()
        self.up_proj: nn.Linear = nn.Linear(cfg.d_latent, cfg.d_model)
        self.transformer: TransformerModel = TransformerModel(cfg.transformer)
        self.divergence_projection_head: UpsampleModel = UpsampleModel(cfg.divergence_projection_head)
        self.seq_projection_head: SeqProjectionHead = SeqProjectionHead(cfg.seq_projection_head)
        self.struct_projection_head: StructProjectionHead = StructProjectionHead(cfg.struct_projection_head)

    def forward(
        self,
        latent: Float[T, "B L d_latent"],
        pad_mask: Bool[T, "B L"],
    ) -> Dict[str, Any]:

        # Unpack for linear projection
        [latent_u], cu_seqlens, max_seqlen = unpad(latent, pad_mask=pad_mask)
        x_u = self.up_proj(latent_u)
        [x] = repad(x_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Transformer accepts B,L
        x = self.transformer(x, pad_mask)

        # Divergence head returns B,L
        divergence = self.divergence_projection_head(x, pad_mask)

        # Sequence head - unpack for processing
        [x_u], cu, ml = unpad(x, pad_mask=pad_mask)
        seq_logits_u = self.seq_projection_head(x_u)
        [seq_logits] = repad(seq_logits_u, cu_seqlens=cu, max_seqlen=ml)

        # Structure heads work on unpacked BL
        struct_outputs = {
            "struct_logits": x_u,  # BL,d_model (for losses)
            "struct_projection_head": self.struct_projection_head,
            "cu_seqlens": cu_seqlens,
        }

        return {
            "divergence": divergence,  # B,L,1,Vx,Vy,Vz
            "seq_logits": seq_logits,  # B,L,n_aa
            **struct_outputs
        }