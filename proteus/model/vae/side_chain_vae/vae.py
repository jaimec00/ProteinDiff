
import torch
import torch.nn as nn

from dataclasses import dataclass, field

from proteus.model.base import Base
from proteus.model.vae.encoder import Encoder, EncoderCfg 
from proteus.model.vae.decoder import Decoder, DecoderCfg
from proteus.types import Float, Int, Bool, T, Tuple, Dict, Any

@dataclass
class VAEModelCfg:
    encoder: EncoderCfg = field(default_factory=EncoderCfg)
    decoder: DecoderCfg = field(default_factory=DecoderCfg)

class VAEModel(Base):
    def __init__(self, cfg: VAEModelCfg) -> None:
        super().__init__()
        self.encoder: Encoder = Encoder(cfg.encoder)
        self.decoder: Decoder = Decoder(cfg.decoder)

    def forward(
        self,
        divergence: Float[T, "B L 1 Vx Vy Vz"],
        coords_bb: Float[T, "B L 4 3"],
        frames: Float[T, "B L 3 3"],
        seq_idx: Int[T, "B L"],
        chain_idx: Int[T, "B L"],
        pad_mask: Bool[T, "B L"],
    ) -> Dict[str, Any]:

        latent, mu, logvar = self.encoder(
            divergence,
            pad_mask
        )

        decoder_outputs = self.decoder(latent, pad_mask)

        return {
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
            **decoder_outputs
        }
