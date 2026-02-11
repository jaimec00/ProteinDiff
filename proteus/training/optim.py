
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
    decay = []
    no_decay = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:  # biases, LayerNorm γ/β
            no_decay.append(param)
        else:
            decay.append(param)

    optim = AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=1.0, # will use LambaLR instead
        betas=(cfg.beta1, cfg.beta2), 
        eps=cfg.eps,
    )
    optim.zero_grad()
    return optim