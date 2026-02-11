
import torch
import torch.nn as nn

from dataclasses import dataclass, field
from omegaconf import II

from proteus.model.base import Base
from proteus.types import Float, Bool, Int, T
from proteus.model.model_utils.mlp import MLP, MLPCfg
from proteus.model.transformer.transformer import TransformerModel, TransformerModelCfg

# simplified for now, no MoQ
@dataclass
class PairAggregatorCfg:
    d_model: int = II("model.d_model")
    d_pair: int = II("model.d_pair")

class PairAggregator(Base):
    '''
    wrapping it so i can checkpoint it
    '''
    def __init__(self, cfg: PairAggregatorCfg) -> None:
        super().__init__()
        router_cfg = MLPCfg(d_in=cfg.d_pair, d_out=cfg.d_model, d_hidden=cfg.d_model, hidden_layers=2)
        self.router = MLP(router_cfg)
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(
        self, 
        pairs: Float[T, "BLL D"], 
        pair_reduce_idxs: Int[T, "BLL"], 
        reduction_buffer: Float[T, "BL"]
    ) -> Float[T, "BL D"]:

        pairs =  self.router(pairs)
        BLL, D = pairs.shape
        BL = reduction_buffer.shape[0]
        singles = reduction_buffer.unsqueeze(-1).expand(BL, D).contiguous().clone()
        return self.ln(singles.scatter_add_(0, pair_reduce_idxs.unsqueeze(-1).expand(BLL, D), pairs))


