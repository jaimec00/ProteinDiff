import torch
import torch.nn as nn
from collections import OrderedDict

from utils.model_utils.preprocesser import PreProcesser
from utils.model_utils.vae import VAE
from utils.model_utils.latent_diffusion.diffusion import Diffusion
from utils.model_utils.classifier import Classifier

class ProteinDiff(nn.Module):
    def __init__(self,  d_model=128, d_latent=4, top_k=16, 
                        voxel_dims=(16,16,16), cell_dim=0.75,
                        ):
        super().__init__()
        '''
        this is basically just a wrapper to hold all of the individual models together.
        training run handles how to use them efficiently. forward method is just for inference
        '''

        self.prep = PreProcesser(voxel_dims=voxel_dims, cell_dim=cell_dim)
        self.vae = VAE()
        self.diffusion = Diffusion()
        self.classifier = Classifier()



    def forward(self, C, L, atom_mask=None, valid_mask=None, temp=1e-6):
        '''
        forward is just for inference, might do inpainting later, or simply conditioning on seq also by initializing nodes to seq embedding,
        but for now the diff model starts from white noise. TODO
        '''
        
        pass



class EMA:
    """
    Keeps an exponential moving average of model parameters.
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, 
                 warmup_steps: int = 1000):
        """
        Args:
            model: the torch.nn.Module to track
            decay: the target EMA decay rate (close to 1)
            warmup_steps: number of steps over which to ramp decay from 0 to target
        """
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0

        # Create shadow parameters
        self.shadow: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, param in model.named_parameters():
            if param.requires_grad:
                # clone and detach to keep as fixed tensor
                self.shadow[name] = param.data.clone().detach()
        # For swapping
        self.backup: OrderedDict[str, torch.Tensor] = OrderedDict()

    def _get_decay(self) -> float:
        """
        If using warmup, linearly increase decay from zero to self.decay.
        """
        if self.num_updates < self.warmup_steps:
            return self.decay * (self.num_updates / self.warmup_steps)
        return self.decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """
        Update shadow weights after a training step.
        Call this right after optimizer.step().
        """
        self.num_updates += 1
        decay = self._get_decay()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            shadow_param = self.shadow[name]
            # EMA update: shadow = decay * shadow + (1 - decay) * param
            shadow_param.mul_(decay).add_(param.data, alpha=(1.0 - decay))

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        """
        Copy shadow (EMA) weights into the model.
        Use before evaluation/sampling.
        """
        # Backup current parameters
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: torch.nn.Module):
        """
        Restore the model’s original parameters (undo apply_to()).
        """
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()