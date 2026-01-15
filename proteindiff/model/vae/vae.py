
import torch
import torch.nn as nn

from dataclasses import dataclass, field

from proteindiff.model.base import Base
from proteindiff.model.vae.encoder import Encoder, EncoderCfg 
from proteindiff.model.vae.decoder import Decoder, DecoderCfg
from proteindiff.types import Float, Int, T, Tuple

@dataclass
class VAEModelCfg:
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    decoder: DecoderCfg = field(default_factory=DecoderCfg)

class VAEModel(Base):
    def __init__(self, cfg: VAEModelCfg):
        super().__init__()
        self.encoder = Encoder(cfg.encoder)
        self.decoder = Decoder(cfg.decoder)

    def forward(
        self,
		divergence: Float[T, "ZN 1 Vx Vy Vz"],
        coords_bb: Float[T, "ZN 14 3"],
		frames: Float[T, "ZN 3 3"],
		seq_idx: Int[T, "ZN"],
		chain_idx: Int[T, "ZN"],
        sample_idx: Int[T, "ZN"],
		cu_seqlens: Int[T, "Z+1"],
		max_seqlen: int,
    ) -> Tuple[
        Float[T, "ZN d_latent"],
        Float[T, "ZN d_latent"],
        Float[T, "ZN d_latent"],
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

        latent, mu, logvar = self.encoder(
            divergence,
            coords_bb,
            frames,
            seq_idx,
            chain_idx,
            sample_idx,
            cu_seqlens,
            max_seqlen
        )
        (
            divergence_pred, 
            seq_pred, 
            struct_logits,
            struct_head,
        ) = self.decoder(latent, cu_seqlens, max_seqlen)

        return (
            latent, mu, logvar,
            divergence_pred, 
            seq_pred, 
            struct_logits,
            struct_head,
        )
