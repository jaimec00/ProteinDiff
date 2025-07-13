import torch
import torch.nn as nn

from utils.model_utils.preprocesser import PreProcesser
from utils.model_utils.vae import VAE
from utils.model_utils.latent_diffusion.diffusion import Diffusion
from utils.model_utils.classifier import Classifier

class ProteinDiff(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        this is basically just a wrapper to hold all of the individual models together.
        training run handles how to 
        '''

        self.prep = PreProcesser()
        self.vae = VAE()
        self.diffusion = Diffusion()
        self.classifier = Classifier()

    def forward(self, C, L, atom_mask=None, kp_mask=None, temp=1e-6):
        '''
        forward is just for inference
        '''
        
        pass

