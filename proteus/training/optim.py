
from dataclasses import dataclass
from torch.nn import Module
from torch.optim import AdamW

# TODO: add other options for optimizers

@dataclass
class OptimCfg:
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-2

def setup_optim(cfg: OptimCfg, model: Module):
    optim = AdamW(
        model.parameters(),
        lr=1.0, # will use LambaLR instead
        betas=(cfg.beta1, cfg.beta2), 
        eps=cfg.eps, 
        weight_decay=cfg.weight_decay
    )
    optim.zero_grad()
    return optim