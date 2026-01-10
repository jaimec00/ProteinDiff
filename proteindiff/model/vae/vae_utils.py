
import torch
import torch.nn as nn
T = torch.Tensor

from dataclasses import dataclass, field
from jaxtyping import Float
from typing import List

from proteindiff.model.utils import MLP
from proteindiff.model import Base

@dataclass
class ResNetBlockCfg:
    d_in: int = 128
    d_out: int = 128
    kernel_size: int = 2

class ResNetBlock(Base):
    def __init__(self, cfg: ResNetBlockCfg):
        super().__init__()

        d_in = cfg.d_in
        self.pre_conv = None
        if cfg.d_in != cfg.d_out:
            self.pre_conv = nn.Sequential(
                nn.Conv3d(cfg.d_in, cfg.d_out, cfg.kernel_size, stride=1, padding="same", bias=False),
                nn.GroupNorm(max(cfg.d_out//16, 1), cfg.d_out),
                nn.SiLU()
            )
            d_in = cfg.d_out

        self.conv = nn.Sequential(
            nn.Conv3d(d_in, cfg.d_out, cfg.kernel_size, stride=1, padding="same", bias=False),
            nn.GroupNorm(max(cfg.d_out//16, 1), cfg.d_out),
            nn.SiLU()
        )


    def forward(self, x: Float[T, "ZN C Vx Vy Vz"]) -> Float[T, "ZN C Vx Vy Vz"]:
        x1 = self.pre_conv(x) if self.pre_conv else x
        return x1 + self.conv(x1)

@dataclass
class ResNetModelCfg:
    blocks: List[ResNetBlockCfg] = field(default_factory = lambda: [ResNetBlockCfg()])

class ResNetModel(Base):
    def __init__(self, cfg: ResNetModelCfg):
        super().__init__()

        self.blocks = nn.Sequential(*[
            ResNetBlock(block)
            for block in cfg.blocks
        ])

    def forward(self, x: Float[T, "ZN C Vx Vy Vz"]) -> Float[T, "ZN C Vx Vy Vz"]:
        return self.blocks(x)