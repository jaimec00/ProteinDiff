
from dataclasses import dataclass

@dataclass
class OptimCfg:
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-2
