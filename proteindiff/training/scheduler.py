from dataclasses import dataclass

@dataclass
class SchedulerCfg:
    d_model: int
    lr_type: str = "static"
    lr_step: float = 5e-3
