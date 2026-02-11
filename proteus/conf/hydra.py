from dataclasses import dataclass, field
from hydra.conf import RunDir, SweepDir, HydraConf
from hydra.core.config_store import ConfigStore

@dataclass
class Hydra(HydraConf):
  run: RunDir = field(default_factory=lambda: RunDir("/home/ubuntu/proteinDiffVirgina/experiments/run/${now:%Y-%m-%d}/${now:%H-%M-%S}"))
  sweep: SweepDir = field(default_factory=lambda: SweepDir(
    dir = "/home/ubuntu/proteinDiffVirgina/experiments/sweep/${now:%Y-%m-%d}/${now:%H-%M-%S}",
    subdir = "${hydra.override_dir}"
  ))

  
def register_hydra():
  cs = ConfigStore.instance()
  cs.store("config", Hydra, group="hydra")