
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from proteus.training import TrainingRun, TrainingRunCfg
from proteus.conf.register_configs import register_configs
from proteus.model.composed.inverse_folding.bert_pairformer import BertPairformerCfg
from proteus.model.tokenizer import StructureTokenizerCfg
from proteus.model.transformer import (
    PairformerModelCfg, PairformerBlockCfg, PairMHACfg, PairAggregatorCfg,
    TransformerBlockCfg, TransformerModelCfg, MHACfg,
)
from proteus.data.construct_registry import ConstructFunctionNames
from proteus.types import Any

defaults = [
    "_self_",
    {"data": "small_seq"},
    {"logger": "default"},
    {"losses": "cel_loss"},
    {"optim": "adamw"},
    {"scheduler": "sqrt"},
    {"profiler": "no_profile"},
    {"training_params": "default"},
]

@dataclass
class BertPairformerPretrainCfg(TrainingRunCfg):
    defaults: list = field(default_factory=lambda: defaults)
    construct_function: str = ConstructFunctionNames.PAIRFORMER
    model: Any = MISSING

# everything is interpolated, can be more specific if you like
D_MODEL = 512
D_PAIR = 128
HEADS = 16
HEADS_PAIR = 4
TRANSFORMER_LAYERS = 32
PAIRFORMER_LAYERS = 8

@dataclass
class SimpleModel(BertPairformerPretrainCfg):
    model: BertPairformerCfg = field(default_factory=lambda: BertPairformerCfg(
        d_model=D_MODEL,
        d_pair=D_PAIR,
        tokenizer=StructureTokenizerCfg(),
        pairformer=PairformerModelCfg(
            transformer_block=PairformerBlockCfg(attn=PairMHACfg(heads=HEADS_PAIR)),
            layers=PAIRFORMER_LAYERS,
        ),
        pair_aggregator=PairAggregatorCfg(),
        transformer=TransformerModelCfg(
            transformer_block=TransformerBlockCfg(
                attn=MHACfg(heads=HEADS),
            ),
            layers=TRANSFORMER_LAYERS,
        )
    ))

register_configs()
cs = ConfigStore.instance()
cs.store(name="simple", node=SimpleModel)

@hydra.main(version_base=None, config_name="simple")
def main(cfg):
    TrainingRun(cfg)

if __name__ == "__main__":
    main()