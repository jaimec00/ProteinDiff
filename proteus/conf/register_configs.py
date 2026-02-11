from proteus.conf.data import register_data
from proteus.conf.logger import register_logger
from proteus.conf.optim import register_optim
from proteus.conf.profiler import register_profiler
from proteus.conf.scheduler import register_scheduler
from proteus.conf.losses import register_losses
from proteus.conf.training_params import register_training_params
from proteus.conf.hydra import register_hydra

def register_configs():
    register_data()
    register_logger()
    register_optim()
    register_profiler()
    register_scheduler()
    register_losses()
    register_training_params()
    register_hydra()