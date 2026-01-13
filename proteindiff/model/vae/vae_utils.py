
import torch
import torch.nn as nn

from dataclasses import dataclass, field
from einops import rearrange
import math

from proteindiff.model.model_utils.mlp import MLP, MLPCfg, ProjectionHead, ProjectionHeadCfg
from proteindiff.model.base import Base
from proteindiff.static.constants import canonical_aas
from proteindiff.types import List, Float, Int, T, Tuple

@dataclass
class ResNetBlockCfg:
    d_in: int = 256
    d_out: int = 256
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
class DownConvBlockCfg:
    d_in: int = 256
    d_out: int = 256
    kernel_size: int = 2
    downsample_factor: int = 2

class DownConvBlock(Base):
    def __init__(self, cfg: DownConvBlockCfg):
        super().__init__()

        # Infer padding from kernel_size and downsample_factor
        padding = self._infer_padding(cfg.kernel_size, cfg.downsample_factor)

        self.conv = nn.Sequential(
            nn.Conv3d(cfg.d_in, cfg.d_out, cfg.kernel_size, stride=cfg.downsample_factor, padding=padding, bias=False),
            nn.GroupNorm(max(cfg.d_out//16, 1), cfg.d_out),
            nn.SiLU()
        )

    @staticmethod
    def _infer_padding(kernel_size: int, stride: int) -> int:
        """Infer padding for clean downsampling. Raises ValueError if not possible."""
        # For stride=kernel_size, padding=0 works cleanly
        if kernel_size == stride:
            return 0

        # For odd kernel with stride, try padding=(k-1)//2
        # This works when: output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
        # For clean downsampling: output_size = input_size // stride
        if kernel_size % 2 == 1:
            padding = (kernel_size - 1) // 2
            # Verify this gives clean downsampling
            # For any input size divisible by stride: (n + 2*p - k) / stride + 1 = n/stride
            # n + 2*p - k = n - stride
            # 2*p = stride - k
            if 2 * padding == stride - kernel_size:
                return padding

        # Common case: kernel_size=3, stride=2 -> padding=1
        if kernel_size == 3 and stride == 2:
            return 1

        raise ValueError(
            f"Cannot infer clean padding for kernel_size={kernel_size} and stride={stride}. "
            f"Use kernel_size=stride for padding=0, or kernel_size=3 with stride=2."
        )

    def forward(self, x: Float[T, "ZN C Vx Vy Vz"]) -> Float[T, "ZN Cout Vx//d Vy//d Vz//d"]:
        return self.conv(x)


@dataclass
class UpConvBlockCfg:
    d_in: int = 256
    d_out: int = 256
    kernel_size: int = 2
    upsample_factor: int = 2

class UpConvBlock(Base):
    def __init__(self, cfg: UpConvBlockCfg):
        super().__init__()

        # Infer padding for transposed conv upsampling
        padding = self._infer_padding(cfg.kernel_size, cfg.upsample_factor)

        self.conv = nn.Sequential(
            nn.ConvTranspose3d(cfg.d_in, cfg.d_out, cfg.kernel_size, stride=cfg.upsample_factor, padding=padding, bias=False),
            nn.GroupNorm(max(cfg.d_out//16, 1), cfg.d_out),
            nn.SiLU()
        )

    @staticmethod
    def _infer_padding(kernel_size: int, stride: int) -> int:
        """Infer padding for clean upsampling. Raises ValueError if not possible."""
        # For stride=kernel_size, padding=0 works cleanly
        if kernel_size == stride:
            return 0

        # For transposed conv: output_size = (input_size - 1) * stride - 2*padding + kernel_size
        # For clean upsampling: output_size = input_size * stride
        # input_size * stride = (input_size - 1) * stride - 2*padding + kernel_size
        # input_size * stride = input_size * stride - stride - 2*padding + kernel_size
        # 0 = -stride - 2*padding + kernel_size
        # 2*padding = kernel_size - stride
        padding = (kernel_size - stride) // 2
        if 2 * padding == kernel_size - stride:
            return padding

        raise ValueError(
            f"Cannot infer clean padding for kernel_size={kernel_size} and stride={stride}. "
            f"Use kernel_size=stride for padding=0, or ensure (kernel_size - stride) is even."
        )

    def forward(self, x: Float[T, "ZN C Vx Vy Vz"]) -> Float[T, "ZN Cout Vx*u Vy*u Vz*u"]:
        return self.conv(x)

@dataclass
class DownsampleModelCfg:
    d_in: int = 256
    d_hidden: int = 256
    d_out: int = 256
    kernel_size: int = 2
    starting_dim: int = 16
    resnets_per_downconv: int = 3

class DownsampleModel(Base):
    def __init__(self, cfg: DownsampleModelCfg):
        super().__init__()

        num_downsamples = math.log(cfg.starting_dim, 2)
        assert int(num_downsamples) == num_downsamples
        num_downsamples = int(num_downsamples)

        blocks = []
        first_resnet_cfg = ResNetBlockCfg(d_in=cfg.d_in, d_out=cfg.d_hidden, kernel_size=cfg.kernel_size)
        down_conv_block_cfg = DownConvBlockCfg(d_in=cfg.d_hidden, d_out=cfg.d_hidden, kernel_size=cfg.kernel_size, downsample_factor=2)
        mid_resnet_cfg = ResNetBlockCfg(d_in=cfg.d_hidden, d_out=cfg.d_hidden, kernel_size=cfg.kernel_size)
        last_resnet_cfg = ResNetBlockCfg(d_in=cfg.d_hidden, d_out=cfg.d_out, kernel_size=cfg.kernel_size)

        blocks.append(ResNetBlock(first_resnet_cfg))

        for downsample in range(num_downsamples):
            blocks.append(DownConvBlock(down_conv_block_cfg))

            for resnet in range(cfg.resnets_per_downconv):
                blocks.append(ResNetBlock(mid_resnet_cfg))

        blocks.append(ResNetBlock(last_resnet_cfg))

        self.down_blocks = nn.Sequential(*blocks)

    def forward(self, x: Float[T, "ZN C Vx Vy Vz"]) -> Float[T, "ZN d_model"]:
        return rearrange(self.blocks(x), "ZN C one one one -> ZN (C one one one)")

@dataclass
class UpsampleModelCfg:
    d_in: int = 256
    d_hidden: int = 256
    d_out: int = 256
    kernel_size: int = 2
    final_dim: int = 16
    resnets_per_upconv: int = 3

class UpsampleModel(Base):
    def __init__(self, cfg: UpsampleModelCfg):
        super().__init__()

        num_upsamples = math.log(cfg.final_dim, 2)
        assert int(num_upsamples) == num_upsamples
        num_upsamples = int(num_upsamples)

        blocks = []
        first_resnet_cfg = ResNetBlockCfg(d_in=cfg.d_in, d_out=cfg.d_hidden, kernel_size=cfg.kernel_size)
        up_conv_block_cfg = UpConvBlockCfg(d_in=cfg.d_hidden, d_out=cfg.d_hidden, kernel_size=cfg.kernel_size, upsample_factor=2)
        mid_resnet_cfg = ResNetBlockCfg(d_in=cfg.d_hidden, d_out=cfg.d_hidden, kernel_size=cfg.kernel_size)
        last_resnet_cfg = ResNetBlockCfg(d_in=cfg.d_hidden, d_out=cfg.d_out, kernel_size=cfg.kernel_size)

        blocks.append(ResNetBlock(first_resnet_cfg))

        for upsample in range(num_upsamples):
            blocks.append(UpConvBlock(up_conv_block_cfg))
        
            for resnet in range(cfg.resnets_per_upconv):
                blocks.append(ResNetBlock(mid_resnet_cfg))

        blocks.append(ResNetBlock(last_resnet_cfg))

        self.up_blocks = nn.Sequential(*blocks)

    def forward(self, x: Float[T, "ZN C"]) -> Float[T, "ZN C Vx Vy Vz"]:
        x = self.blocks(rearrange(x, "ZN C -> ZN C one one one", one=1))
        return x


@dataclass
class LatentProjectionHeadCfg:
    d_model: int = 256
    d_latent: int = 16

class LatentProjectionHead(ProjectionHead):
    def __init__(self, cfg: LatentProjectionHeadCfg):
        projection_cfg = ProjectionHeadCfg(d_in=cfg.d_model, d_out=2*cfg.d_latent)
        super().__init__(projection_cfg)

    def forward(self, x: Float[T, "ZN d_model"]) -> Tuple[Float[T, "ZN d_latent"], Float[T, "ZN d_latent"], Float[T, "ZN d_latent"]]:
        mu_logvar = super()(x)
        mu, logvar = torch.chunk(mu_logvar, chunks=2, dim=-1)
        latent = mu + torch.randn_like(mu)*torch.exp(-0.5*logvar)
        
        return latent, mu, logvar

@dataclass
class SeqProjectionHeadCfg:
    d_model: int = 256

class SeqProjectionHead(ProjectionHead):
    def __init__(self, cfg: SeqProjectionHeadCfg):
        projection_cfg = ProjectionHeadCfg(
            d_in=cfg.d_model, 
            d_out=len(canonical_aas),
        )
        super().__init__(projection_cfg)

@dataclass
class PairwiseProjectionHeadCfg:
    d_model: int = 256
    d_down: int = 64
    num_bins: int = 64

class PairwiseProjectionHead(Base):
    def __init__(self, cfg: PairwiseProjectionHeadCfg):
        super().__init__()
        self.Wqk = nn.Linear(cfg.d_model, 2*cfg.d_down)
        projection_cfg = ProjectionHeadCfg(d_in=cfg.d_down, d_out=cfg.num_bins)
        self.proj = ProjectionHead(projection_cfg)

    def forward(self, x: Float[T, "ZN d_model"]) -> Float[T, "ZN ZN d_bins"]:
        qk = self.Wqk(x)
        q, k = torch.chunk(qk, chunks=2, dim=-1)
        q, k = q.unsqueeze(0), k.unsqueeze(1)
        prod, diff = q*k, q-k
        prod_diff = torch.cat([prod, diff], dim=-1)
        return self.proj(prod_diff)


@dataclass
class StructProjectionHeadCfg:
    d_model: int = 256
    dist_proj: PairwiseProjectionHeadCfg = field(default_factory = PairwiseProjectionHeadCfg)
    angle_proj: PairwiseProjectionHeadCfg = field(default_factory = PairwiseProjectionHeadCfg)
    plddt_proj: PairwiseProjectionHeadCfg = field(default_factory = PairwiseProjectionHeadCfg)
    pae_proj: PairwiseProjectionHeadCfg = field(default_factory = PairwiseProjectionHeadCfg)

class StructProjectionHead(Base):
    def __init__(self, cfg: StructProjectionHeadCfg):
        super().__init__()
        self.dist_proj = PairwiseProjectionHead(cfg.dist_proj)
        self.angle_proj = PairwiseProjectionHead(cfg.angle_proj)
        self.frame_proj = ProjectionHead(ProjectionHeadCfg(d_in=cfg.d_model, d_out=3*3))
        self.torsion_proj = ProjectionHead(ProjectionHeadCfg(d_in=cfg.d_model, d_out=7*2))
        self.plddt_proj = PairwiseProjectionHead(cfg.plddt_proj)
        self.pae_proj = PairwiseProjectionHead(cfg.pae_proj)

    def forward(self, x: Float[T, "ZN d_model"]
    ) -> Tuple[
        Float[T, "ZN ZNN d_dist"], 
        Float[T, "ZN ZNN d_angle"],
        Float[T, "ZN 3"],
        Float[T, "ZN 3"],
        Float[T, "ZN 3"],
        Float[T, "ZN 7"],
        Float[T, "ZN 7"],
        Float[T, "ZN ZNN d_plddt"],
        Float[T, "ZN ZNN d_pae"],
    ]:
        distogram = self.dist_proj(x)
        anglogram = self.angle_proj(x)

        txy = self.frame_proj(x)
        t, x, y = torch.chunk(txy, chunks=3, dim=-1)
        
        sincos = self.torsion_proj(x)
        sin, cos = torch.chunk(sincos, chunks=2, dim=-1)
        
        plddt = self.plddt_proj(x)
        pae = self.pae_proj(x)

        return distogram, anglogram, t, x, y, sin, cos, plddt, pae
