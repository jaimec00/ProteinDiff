from dataclasses import dataclass, field


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
    model_factory: str = "proteindiff.utils.model_factories.base_model.build_model_cfg_from_simple_cfg"
