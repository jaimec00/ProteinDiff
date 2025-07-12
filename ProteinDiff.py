import torch
import torch.nn as nn

from utils.model_utils.base_modules import MLP
from utils.model_utils.preprocesser import PreProcesser
from utils.model_utils.vae import VAEEncoder, VAEDecoder
from utils.model_utils.diffusion import Diffusion
from utils.model_utils.classifier import Classifier

class ProteinDiff(nn.Module):
    def __init__(self):
        super().__init__()

        self.prep = PreProcesser()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()
        self.diffusion = Diffusion()
        self.classifier = Classifier()

    def forward(self, C, L=None, t=None, atom_mask=None, kp_mask=None, diffusion=False, inference=False, temp=1e-6):
        
        if inference:
            with torch.no_grad():
                return self.inference(C, kp_mask, temp)

        elif diffusion:
            with torch.no_grad():
                C_backbone, fields, nbrs, nbr_mask = self.prep(C,L,atom_mask,kp_mask)
                Z, Z_mu, Z_logvar = self.encoder(fields, kp_mask)
                noise, Z_noised = self.diffusion.noise(Z, t)
    
            noise_pred = self.diffusion(Z_noised, C_backbone, nbrs, nbr_mask, t)

            return noise, noise_pred

        else:
            C_backbone, fields, nbrs, nbr_mask = self.prep(C,L,atom_mask,kp_mask)
            Z, Z_mu, Z_logvar = self.encoder(fields, kp_mask)
            field_pred = self.decoder(Z, kp_mask)
            seq_pred = self.classifier(field_pred.detach())

            return Z_mu, Z_logvar, fields, field_pred, seq_pred


    def inference(self, C, kp_mask=None, temp=1e-6):
        '''
        for now starting from full noise, but if this works i will look into inpainting
        '''

        pass
