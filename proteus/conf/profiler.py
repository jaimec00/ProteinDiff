# profiler config
from proteus.utils.profiling import ProfilerCfg
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class NoProfile(ProfilerCfg):
    enable: bool = False

@dataclass
class YesProfile(ProfilerCfg):
    enable: bool = False
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 1
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = True

def register_profiler():
    cs = ConfigStore.instance()
    cs.store("no_profile", NoProfile, group="profiler")
    cs.store("yes_profile", YesProfile, group="profiler")