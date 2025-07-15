import torch
import torch.nn as nn

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
