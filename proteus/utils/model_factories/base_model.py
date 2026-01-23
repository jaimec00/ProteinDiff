import torch

from dataclasses import dataclass, field

from proteus.utils.cfg_utils.model_cfg_utils import SimpleProteusCfg
from proteus.model.Proteus import ProteusCfg
from proteus.model.tokenizer.tokenizer import TokenizerCfg
from proteus.model.vae.vae import VAEModelCfg
from proteus.model.vae.encoder import EncoderCfg
from proteus.model.vae.decoder import DecoderCfg
from proteus.model.vae.vae_utils import (
    DownsampleModelCfg, ResNetBlockCfg, DownConvBlockCfg,
    UpsampleModelCfg, UpConvBlockCfg,
    LatentProjectionHeadCfg, SeqProjectionHeadCfg,
    StructProjectionHeadCfg, PairwiseProjectionHeadCfg
)
from proteus.model.mpnn.mpnn import MPNNModelCfg, MPNNBlockCfg, EdgeEncoderCfg
from proteus.model.model_utils.mlp import MPNNMLPCfg, FFNCfg, ProjectionHeadCfg
from proteus.model.transformer.transformer import TransformerModelCfg, TransformerBlockCfg
from proteus.model.transformer.attention import MHACfg
from proteus.model.diffusion.diffusion import DiffusionModelCfg, Parameterization
from proteus.model.diffusion.diffusion_utils import ConditionerCfg, DiTModelCfg, DiTBlockCfg

def build_model_cfg_from_simple_cfg(cfg: SimpleProteusCfg) -> ProteusCfg:
    return ProteusCfg(
        tokenizer=build_tokenizer_cfg(cfg.voxel_dim, cfg.cell_dim),
        vae=build_vae_cfg(cfg.d_model, cfg.d_latent, cfg.voxel_dim, cfg.n_mpnn_blocks, cfg.n_transformer_blocks, cfg.n_heads),
        diffusion=build_diffusion_cfg(cfg.d_model, cfg.d_latent, cfg.t_max, cfg.n_mpnn_blocks, cfg.n_transformer_blocks, cfg.n_heads, cfg.n_dit_blocks)
    )

def build_tokenizer_cfg(voxel_dim: int, cell_dim: float):
    return TokenizerCfg(voxel_dim=voxel_dim, cell_dim=cell_dim)

def build_vae_cfg(d_model: int, d_latent: int, voxel_dim: int, n_mpnn_blocks: int, n_transformer_blocks: int, n_heads: int):
    return VAEModelCfg(
        encoder=build_encoder_cfg(d_model, d_latent, voxel_dim, n_mpnn_blocks, n_transformer_blocks, n_heads),
        decoder=build_decoder_cfg(d_model, d_latent, voxel_dim, n_transformer_blocks, n_heads)
    )

def build_encoder_cfg(d_model: int, d_latent: int, voxel_dim: int, n_mpnn_blocks: int, n_transformer_blocks: int, n_heads: int):
    return EncoderCfg(
        downsample=build_downsample_cfg(d_model, voxel_dim),
        mpnn=build_mpnn_cfg(d_model, n_mpnn_blocks),
        transformer=build_transformer_cfg(d_model, n_transformer_blocks, n_heads),
        latent_projection_head=build_latent_proj_cfg(d_model, d_latent)
    )

def build_decoder_cfg(d_model: int, d_latent: int, voxel_dim: int, n_transformer_blocks: int, n_heads: int):
    return DecoderCfg(
        d_model = d_model,
        d_latent=d_latent,
        transformer=build_transformer_cfg(d_model, n_transformer_blocks, n_heads),
        divergence_projection_head=build_upsample_cfg(d_model, voxel_dim),
        seq_projection_head=build_seq_proj_cfg(d_model),
        struct_projection_head=build_struct_proj_cfg(d_model)
    )

def build_downsample_cfg(d_model: int, voxel_dim: int):
    return DownsampleModelCfg(
        d_out=d_model,
        starting_dim=voxel_dim,
        resnets_per_downconv=3,
    )

def build_upsample_cfg(d_model: int, voxel_dim: int):
    return UpsampleModelCfg(
        d_in=d_model,
        final_dim=voxel_dim,
        resnets_per_upconv=3,
    )

def build_mpnn_cfg(d_model: int, n_mpnn_blocks: int):
    return MPNNModelCfg(
        edge_encoder=EdgeEncoderCfg(
            d_model=d_model,
            edge_mlp=MPNNMLPCfg(d_model=d_model)
        ),
        mpnn_block=MPNNBlockCfg(
            d_model=d_model,
            node_mlp=MPNNMLPCfg(d_model=d_model),
            ffn_mlp=FFNCfg(d_model=d_model),
            edge_mlp=MPNNMLPCfg(d_model=d_model)
        ),
        layers=n_mpnn_blocks
    )

def build_transformer_cfg(d_model: int, n_transformer_blocks: int, n_heads: int):
    return TransformerModelCfg(
        transformer_block=TransformerBlockCfg(
            d_model=d_model,
            attn=MHACfg(d_model=d_model, heads=n_heads),
            ffn=FFNCfg(d_model=d_model)
        ),
        layers=n_transformer_blocks
    )

def build_latent_proj_cfg(d_model: int, d_latent: int):
    return LatentProjectionHeadCfg(d_model=d_model, d_latent=d_latent)

def build_seq_proj_cfg(d_model: int):
    return SeqProjectionHeadCfg(d_model=d_model)

def build_struct_proj_cfg(d_model: int):
    d_down = 64
    return StructProjectionHeadCfg(
        d_model=d_model,
        dist_proj=PairwiseProjectionHeadCfg(d_model=d_model, d_down=d_down, num_bins=64, qk_proj=ProjectionHeadCfg(d_in=d_model, d_out=d_down)),
        angle_proj=PairwiseProjectionHeadCfg(d_model=d_model, d_down=d_down, num_bins=16, num_outputs=6, qk_proj=ProjectionHeadCfg(d_in=d_model, d_out=d_down)),
        plddt_proj=PairwiseProjectionHeadCfg(d_model=d_model, d_down=d_down, num_bins=64, qk_proj=ProjectionHeadCfg(d_in=d_model, d_out=d_down)),
        pae_proj=PairwiseProjectionHeadCfg(d_model=d_model, d_down=d_down, num_bins=64, qk_proj=ProjectionHeadCfg(d_in=d_model, d_out=d_down))
    )

def build_diffusion_cfg(d_model: int, d_latent: int, t_max: int, n_mpnn_blocks: int, n_transformer_blocks: int, n_heads: int, n_dit_blocks: int):
    return DiffusionModelCfg(
        d_model=d_model,
        d_latent=d_latent,
        t_max=t_max,
        parameterization=Parameterization.DEFAULT,
        conditioner=build_conditioner_cfg(d_model, n_mpnn_blocks, n_transformer_blocks, n_heads),
        denoiser=build_dit_cfg(d_model, n_heads, n_dit_blocks)
    )

def build_conditioner_cfg(d_model: int, n_mpnn_blocks: int, n_transformer_blocks: int, n_heads: int):
    return ConditionerCfg(
        d_model=d_model,
        d_conditioning=d_model,
        conditioning_mpnn=build_mpnn_cfg(d_model, n_mpnn_blocks),
        conditioning_transformer=build_transformer_cfg(d_model, n_transformer_blocks, n_heads)
    )

def build_dit_cfg(d_model: int, n_heads: int, n_dit_blocks: int):
    return DiTModelCfg(
        dit_block=DiTBlockCfg(
                d_model=d_model,
                attn=MHACfg(d_model=d_model, heads=n_heads),
                ffn=FFNCfg(d_model=d_model)
        ),
        layers=n_dit_blocks
    )