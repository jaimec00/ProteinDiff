from proteus.training.scheduler import SchedulerCfg
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import II

@dataclass
class SqrtScheduler(SchedulerCfg):
    d_model: int = II("model.d_model")
    lr_type: str = "sqrt"
    lr_step: float = 3.0e-5
    warmup_steps: int = 6000

@dataclass
class StaticScheduler(SchedulerCfg):
    d_model: int = II("model.d_model")
    lr_type: str = "static"
    lr_step: float = 2.0e-5

def register_scheduler():
    cs = ConfigStore.instance()
    cs.store("sqrt", SqrtScheduler, group="scheduler")
    cs.store("static", StaticScheduler, group="scheduler")