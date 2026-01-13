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
        tokenizer=build_tokenizer_cfg(cfg),
        vae=build_vae_cfg(cfg),
        diffusion=build_diffusion_cfg(cfg)
    )

def build_tokenizer_cfg(cfg: SimpleProteinDiffCfg):
    return TokenizerCfg(
        voxel_dim=cfg.voxel_dim,
        cell_dim=cfg.cell_dim
    )

def build_vae_cfg(cfg: SimpleProteinDiffCfg):
    return VAEModelCfg(
        encoder=build_encoder_cfg(cfg),
        decoder=build_decoder_cfg(cfg)
    )

def build_encoder_cfg(cfg: SimpleProteinDiffCfg):
    return EncoderCfg(
        downsample=build_downsample_cfg(cfg),
        mpnn=build_mpnn_cfg(cfg),
        transformer=build_transformer_cfg(cfg),
        latent_projection_head=build_latent_proj_cfg(cfg)
    )

def build_decoder_cfg(cfg: SimpleProteinDiffCfg):
    return DecoderCfg(
        transformer=build_transformer_cfg(cfg),
        divergence_projection_head=build_upsample_cfg(cfg),
        seq_projection_head=build_seq_proj_cfg(cfg),
        struct_projection_head=build_struct_proj_cfg(cfg)
    )

def build_downsample_cfg(cfg: SimpleProteinDiffCfg):
    return DownsampleModelCfg(
        d_in=1,
        d_hidden=cfg.d_model,
        d_out=cfg.d_model,
        starting_dim=cfg.voxel_dim,
        resnets_per_downconv=3,
    )

def build_upsample_cfg(cfg: SimpleProteinDiffCfg):
    return UpsampleModelCfg(
        d_in=cfg.d_model,
        d_hidden=cfg.d_model*2,
        d_out=1,
        final_dim=cfg.voxel_dim,
        resnets_per_upconv=3,
    )

def build_mpnn_cfg(cfg: SimpleProteinDiffCfg):
    return MPNNModelCfg(
        edge_encoder=EdgeEncoderCfg(
            d_model=cfg.d_model,
            edge_mlp=MPNNMLPCfg(d_model=cfg.d_model)
        ),
        mpnn_block=MPNNBlockCfg(
            d_model=cfg.d_model,
            node_mlp=MPNNMLPCfg(d_model=cfg.d_model),
            ffn_mlp=FFNCfg(d_model=cfg.d_model),
            edge_mlp=MPNNMLPCfg(d_model=cfg.d_model)
        ),
        layers=cfg.n_mpnn_blocks
    )

def build_transformer_cfg(cfg: SimpleProteinDiffCfg):
    return TransformerModelCfg(
        transformer_block=TransformerBlockCfg(
            d_model=cfg.d_model,
            attn=MHACfg(d_model=cfg.d_model, heads=cfg.n_heads),
            ffn=FFNCfg(d_model=cfg.d_model)
        ),
        layers=4
    )

def build_latent_proj_cfg(cfg: SimpleProteinDiffCfg):
    return LatentProjectionHeadCfg(d_model=cfg.d_model, d_latent=cfg.d_latent)

def build_seq_proj_cfg(cfg: SimpleProteinDiffCfg):
    return SeqProjectionHeadCfg(d_model=cfg.d_model)

def build_struct_proj_cfg(cfg: SimpleProteinDiffCfg):
    return StructProjectionHeadCfg(
        d_model=cfg.d_model,
        dist_proj=PairwiseProjectionHeadCfg(d_model=cfg.d_model),
        angle_proj=PairwiseProjectionHeadCfg(d_model=cfg.d_model),
        plddt_proj=PairwiseProjectionHeadCfg(d_model=cfg.d_model),
        pae_proj=PairwiseProjectionHeadCfg(d_model=cfg.d_model)
    )

def build_diffusion_cfg(cfg: SimpleProteinDiffCfg):
    return DiffusionModelCfg(
        d_model=cfg.d_model,
        d_latent=cfg.d_latent,
        t_max=cfg.t_max,
        parameterization=Parameterization.DEFAULT,
        conditioner=build_conditioner_cfg(cfg),
        denoiser=build_dit_cfg(cfg)
    )

def build_conditioner_cfg(cfg: SimpleProteinDiffCfg):
    return ConditionerCfg(
        d_model=cfg.d_model,
        d_conditioning=cfg.d_model,
        conditioning_mpnn=build_mpnn_cfg(cfg),
        conditioning_transformer=build_transformer_cfg(cfg)
    )

def build_dit_cfg(cfg: SimpleProteinDiffCfg):
    return DiTModelCfg(
        dit_block=DiTBlockCfg(
                d_model=cfg.d_model,
                attn=MHACfg(d_model=cfg.d_model, heads=cfg.n_heads),
                ffn=FFNCfg(d_model=cfg.d_model)
        ),
        layers=cfg.n_dit_blocks
    )