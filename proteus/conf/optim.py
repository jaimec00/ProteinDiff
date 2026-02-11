
from proteus.training.optim import OptimCfg
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

class AdamW(OptimCfg):
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    weight_decay: float = 0.01

class Adam(OptimCfg):
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    weight_decay: float = 0.0

def register_optim():
    cs = ConfigStore.instance()
    cs.store("adamw", AdamW, group="optim")
    cs.store("adam", Adam, group="optim")
