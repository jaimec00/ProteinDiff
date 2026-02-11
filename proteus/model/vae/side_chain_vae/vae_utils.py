
import torch
import torch.nn as nn

from dataclasses import dataclass, field
from einops import rearrange
import math

from proteus.model.model_utils.mlp import MLP, MLPCfg, ProjectionHead, ProjectionHeadCfg
from proteus.model.base import Base
from proteus.static.constants import (
    canonical_aas,
    restype_rigid_group_default_frame,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    chi_angles_mask,
)
from proteus.types import Bool, List, Float, Int, T, Tuple, Callable, Optional
from proteus.utils.tensor import unpad, repad


@dataclass
class ResNetBlockCfg:
    d_in: int = 256
    d_out: int = 256
    kernel_size: int = 3  # Use odd kernel size to avoid copy with padding="same"

class ResNetBlock(Base):
    def __init__(self, cfg: ResNetBlockCfg) -> None:
        super().__init__()

        # Use explicit padding for odd kernels: padding = kernel_size // 2
        # This avoids the copy that padding="same" creates with even kernels
        padding: int = cfg.kernel_size // 2

        d_in: int = cfg.d_in
        self.pre_conv: Optional[nn.Sequential] = None
        if cfg.d_in != cfg.d_out:
            self.pre_conv = nn.Sequential(
                nn.Conv3d(cfg.d_in, cfg.d_out, cfg.kernel_size, stride=1, padding=padding, bias=False),
                nn.GroupNorm(max(cfg.d_out//16, 1), cfg.d_out),
                nn.SiLU()
            )
            d_in = cfg.d_out

        self.conv: nn.Sequential = nn.Sequential(
            nn.Conv3d(d_in, cfg.d_out, cfg.kernel_size, stride=1, padding=padding, bias=False),
            nn.GroupNorm(max(cfg.d_out//16, 1), cfg.d_out),
            nn.SiLU()
        )


    def forward(self, x: Float[T, "BL C Vx Vy Vz"]) -> Float[T, "BL C Vx Vy Vz"]:
        x1 = self.pre_conv(x) if self.pre_conv else x
        return x1 + self.conv(x1)

@dataclass
class DownConvBlockCfg:
    d_in: int = 256
    d_out: int = 256
    kernel_size: int = 2
    downsample_factor: int = 2

class DownConvBlock(Base):
    def __init__(self, cfg: DownConvBlockCfg) -> None:
        super().__init__()

        # Infer padding from kernel_size and downsample_factor
        padding: int = self._infer_padding(cfg.kernel_size, cfg.downsample_factor)

        self.conv: nn.Sequential = nn.Sequential(
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

    def forward(self, x: Float[T, "BL C Vx Vy Vz"]) -> Float[T, "BL Cout Vx//d Vy//d Vz//d"]:
        return self.conv(x)


@dataclass
class UpConvBlockCfg:
    d_in: int = 256
    d_out: int = 256
    kernel_size: int = 2
    upsample_factor: int = 2

class UpConvBlock(Base):
    def __init__(self, cfg: UpConvBlockCfg) -> None:
        super().__init__()

        # Infer padding for transposed conv upsampling
        padding: int = self._infer_padding(cfg.kernel_size, cfg.upsample_factor)

        self.conv: nn.Sequential = nn.Sequential(
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

    def forward(self, x: Float[T, "BL C Vx Vy Vz"]) -> Float[T, "BL Cout Vx*u Vy*u Vz*u"]:
        return self.conv(x)

@dataclass
class DownsampleModelCfg:
    d_out: int = 256
    resnet_kernel_size: int = 3  # Odd kernel for ResNetBlocks (preserves spatial dims)
    downsample_kernel_size: int = 2  # kernel=stride for DownConvBlocks
    starting_dim: int = 16
    resnets_per_downconv: int = 3

class DownsampleModel(Base):
    def __init__(self, cfg: DownsampleModelCfg) -> None:
        super().__init__()

        num_downsamples: float = math.log(cfg.starting_dim, 2)
        assert int(num_downsamples) == num_downsamples
        num_downsamples_int: int = int(num_downsamples)

        # 32 64 128 256
        d_model_list: List[int] = [cfg.d_out // 2**i for i in range(num_downsamples_int, -1, -1)]
        blocks: List[nn.Module] = []

        # initial block
        blocks.append(
            ResNetBlock(
                ResNetBlockCfg(
                    d_in=1,
                    d_out=d_model_list[0],
                    kernel_size=cfg.resnet_kernel_size
                )
            )
        )

        for downsample in range(num_downsamples_int):

            # up conv
            blocks.append(
                DownConvBlock(
                    DownConvBlockCfg(
                        d_in=d_model_list[downsample], 
                        d_out=d_model_list[downsample+1], 
                        kernel_size=cfg.downsample_kernel_size, 
                        downsample_factor=cfg.downsample_kernel_size
                    )
                )
            )

            # resnets
            for resnet in range(cfg.resnets_per_downconv):
                blocks.append(
                    ResNetBlock(
                        ResNetBlockCfg(
                            d_in=d_model_list[downsample+1],
                            d_out=d_model_list[downsample+1],
                            kernel_size=cfg.resnet_kernel_size
                        )
                    )
                )

        # post resnet
        blocks.append(
            ResNetBlock(
                ResNetBlockCfg(
                    d_in=d_model_list[-1],
                    d_out=d_model_list[-1],
                    kernel_size=cfg.resnet_kernel_size
                )
            )
        )

        self.down_blocks: nn.Sequential = nn.Sequential(*blocks)

    def forward(
        self,
        x: Float[T, "B L C Vx Vy Vz"],
        pad_mask: Bool[T, "B L"]
    ) -> Float[T, "B L d_model"]:
        # Unpad B,L → BL for CNN processing
        [x_unpacked], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)

        # Process with CNNs (existing logic)
        out_unpacked = self.down_blocks(x_unpacked).reshape(x_unpacked.size(0), -1)

        # Repad BL → B,L
        [out_padded] = repad(out_unpacked, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        return out_padded

@dataclass
class UpsampleModelCfg:
    d_in: int = 256
    resnet_kernel_size: int = 3  # Odd kernel for ResNetBlocks (preserves spatial dims)
    upsample_kernel_size: int = 2  # kernel=stride for UpConvBlocks
    final_dim: int = 16
    resnets_per_upconv: int = 3

class UpsampleModel(Base):
    def __init__(self, cfg: UpsampleModelCfg) -> None:
        super().__init__()

        num_upsamples: float = math.log(cfg.final_dim, 2)
        assert int(num_upsamples) == num_upsamples
        num_upsamples_int: int = int(num_upsamples)

        # 256 128 64 32
        # 1   2   4  8
        d_model_list: List[int] = [cfg.d_in // 2**i for i in range(num_upsamples_int+1)]
        blocks: List[nn.Module] = []

        # initial block
        blocks.append(
            ResNetBlock(
                ResNetBlockCfg(
                    d_in=d_model_list[0],
                    d_out=d_model_list[0],
                    kernel_size=cfg.resnet_kernel_size
                )
            )
        )

        for upsample in range(num_upsamples_int):

            # up conv
            blocks.append(
                UpConvBlock(
                    UpConvBlockCfg(
                        d_in=d_model_list[upsample], 
                        d_out=d_model_list[upsample+1], 
                        kernel_size=cfg.upsample_kernel_size, 
                        upsample_factor=cfg.upsample_kernel_size
                    )
                )
            )

            # resnets
            for resnet in range(cfg.resnets_per_upconv):
                blocks.append(
                    ResNetBlock(
                        ResNetBlockCfg(
                            d_in=d_model_list[upsample+1],
                            d_out=d_model_list[upsample+1],
                            kernel_size=cfg.resnet_kernel_size
                        )
                    )
                )

        # post resnet
        blocks.append(
            ResNetBlock(
                ResNetBlockCfg(
                    d_in=d_model_list[-1],
                    d_out=1,
                    kernel_size=cfg.resnet_kernel_size
                )
            )
        )

        self.up_blocks: nn.Sequential = nn.Sequential(*blocks)

    def forward(
        self,
        x: Float[T, "B L d_latent"],
        pad_mask: Bool[T, "B L"]
    ) -> Float[T, "B L 1 Vx Vy Vz"]:
        # Unpad B,L → BL for CNN processing
        [x_unpacked], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)

        # Existing CNN logic
        x_unpacked = x_unpacked.reshape(*x_unpacked.shape, 1, 1, 1)
        out_unpacked = self.up_blocks(x_unpacked)

        # Repad BL → B,L
        [out_padded] = repad(out_unpacked, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        return out_padded


@dataclass
class LatentProjectionHeadCfg:
    d_model: int = 256
    d_latent: int = 16

class LatentProjectionHead(ProjectionHead):
    def __init__(self, cfg: LatentProjectionHeadCfg) -> None:
        projection_cfg: ProjectionHeadCfg = ProjectionHeadCfg(d_in=cfg.d_model, d_out=2*cfg.d_latent)
        super().__init__(projection_cfg)

    def forward(self, x: Float[T, "BL d_model"]) -> Tuple[Float[T, "BL d_latent"], Float[T, "BL d_latent"], Float[T, "BL d_latent"]]:
        mu_logvar = super().forward(x) # TODO: change this as wont run preforward hooks once implement fsdp (or maybe it will since it inherits ProjectionHead weights and called forward on the child?)
        mu, logvar = torch.chunk(mu_logvar, chunks=2, dim=-1)
        latent = mu + torch.randn_like(mu)*torch.exp(-0.5*logvar)
        return latent, mu, logvar


