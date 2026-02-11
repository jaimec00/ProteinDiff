
# training params

from proteus.training.training_run import TrainingParamsCfg
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

@dataclass
class DefaultTrainParams(TrainingParamsCfg):
  max_steps: int = 100_000
  val_interval: int = 500
  accumulation_steps: int = 8
  grad_clip_norm: float = 5.0
  compile_model: bool = False
  load_from_checkpoint: str = ""
  checkpoint_interval: int = 5_000
  gc_interval: int = 500

@dataclass
class DebugTrainParams(TrainingParamsCfg):
  max_steps: int = 10
  val_interval: int = 5
  accumulation_steps: int = 1
  grad_clip_norm: float = 0.0
  compile_model: bool = False
  load_from_checkpoint: str = ""
  checkpoint_interval: int = 500
  gc_interval: int = 500


def register_training_params():
  cs = ConfigStore.instance()
  cs.store("default", DefaultTrainParams, group="training_params")
  cs.store("debug", DebugTrainParams, group="training_params")
