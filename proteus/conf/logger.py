from proteus.training.logger import LoggerCfg
from hydra.core.config_store import ConfigStore

class DefaultLogger(LoggerCfg):
    experiment_name: str = "debug"
    overwrite: bool = True
    log_system_metrics: bool = True
    system_metrics_sample_interval: int = 10
    system_metrics_log_interval: int = 1
    log_interval: int = 10

def register_logger():
    cs = ConfigStore.instance()
    cs.store("default", DefaultLogger, group="logger")
