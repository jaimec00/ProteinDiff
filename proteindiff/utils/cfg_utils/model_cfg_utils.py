import torch

from dataclasses import dataclass, field

from proteindiff.model.ProteinDiff import ProteinDiffCfg
from proteindiff.model.tokenizer.tokenizer import TokenizerCfg
from proteindiff.model.vae.vae import VAEModelCfg
from proteindiff.model.vae.encoder import EncoderCfg
from proteindiff.model.vae.decoder import DecoderCfg
from proteindiff.model.vae.vae_utils import (
    DownsampleModelCfg, ResNetBlockCfg, DownConvBlockCfg,
    UpsampleModelCfg, UpConvBlockCfg,
    LatentProjectionHeadCfg, SeqProjectionHeadCfg,
    StructProjectionHeadCfg, PairwiseProjectionHeadCfg
)
from proteindiff.model.mpnn.mpnn import MPNNModelCfg, MPNNBlockCfg, EdgeEncoderCfg
from proteindiff.model.model_utils.mlp import MPNNMLPCfg, FFNCfg
from proteindiff.model.transformer.transformer import TransformerModelCfg, TransformerBlockCfg
from proteindiff.model.transformer.attention import MHACfg
from proteindiff.model.diffusion.diffusion import DiffusionModelCfg, Parameterization
from proteindiff.model.diffusion.diffusion_utils import ConditionerCfg, DiTModelCfg, DiTBlockCfg


@dataclass
class SimpleProteinDiffCfg:
    d_model: int = 256
    d_latent: int = 16
    voxel_dim: int = 8
    cell_dim: float = 1.0
    t_max: int = 1000
    n_mpnn_blocks: int = 3
    n_transformer_blocks: int = 6
    n_dit_blocks: int = 12
    n_heads: int = 8

def build_model_cfg_from_simple_cfg(cfg: SimpleProteinDiffCfg) -> ProteinDiffCfg:
    return ProteinDiffCfg(
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
        d_in=1,
        d_hidden=d_model,
        d_out=d_model,
        starting_dim=voxel_dim,
        resnets_per_downconv=3,
    )

def build_upsample_cfg(d_model: int, voxel_dim: int):
    return UpsampleModelCfg(
        d_in=d_model,
        d_hidden=d_model*2,
        d_out=1,
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
    return StructProjectionHeadCfg(
        d_model=d_model,
        dist_proj=PairwiseProjectionHeadCfg(d_model=d_model),
        angle_proj=PairwiseProjectionHeadCfg(d_model=d_model),
        plddt_proj=PairwiseProjectionHeadCfg(d_model=d_model),
        pae_proj=PairwiseProjectionHeadCfg(d_model=d_model)
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