
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from proteus.training import TrainingRun, TrainingRunCfg
from proteus.conf.register_configs import register_configs
from proteus.model.composed.inverse_folding.proteusAI import proteusAICfg
from proteus.model.tokenizer import WaveFunctionTokenizerCfg
from proteus.utils.masking import MaskerCfg
from proteus.model.transformer import (
    TransformerBlockCfg, 
    TransformerModelCfg, 
    MHACfg,
)
from proteus.data.construct_registry import ConstructFunctionNames
from proteus.types import Any

defaults = [
    "_self_",
    {"data": "medium_seq"},
    {"logger": "default"},
    {"losses": "cel_loss"},
    {"optim": "adamw"},
    {"scheduler": "sqrt"},
    {"profiler": "no_profile"},
    {"training_params": "default"},
]

@dataclass
class proteusAIPretrainCfg(TrainingRunCfg):
    defaults: list = field(default_factory=lambda: defaults)
    construct_function: str = ConstructFunctionNames.PROTEUS
    model: Any = MISSING

# everything is interpolated, can be more specific if you like
D_MODEL = 512
D_WF = 128
HEADS = 16
TRANSFORMER_LAYERS = 32
MASK_RATE = 0.15

@dataclass
class SimpleModel(proteusAIPretrainCfg):
    model: proteusAICfg = field(default_factory=lambda: proteusAICfg(
        d_model=D_MODEL,
        tokenizer=WaveFunctionTokenizerCfg(d_wf=D_WF),
        masker=MaskerCfg(mask_rate=MASK_RATE),
        transformer=TransformerModelCfg(
            transformer_block=TransformerBlockCfg(
                attn=MHACfg(heads=HEADS)
            ),
            layers = TRANSFORMER_LAYERS,
        ),
    ))

register_configs()
cs = ConfigStore.instance()
cs.store(name="simple", node=SimpleModel)

@hydra.main(version_base=None, config_name="simple")
def main(cfg):
    TrainingRun(cfg)

if __name__ == "__main__":
    main()