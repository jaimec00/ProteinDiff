import torch
import torch.nn as nn

from utils.model_utils.base_modules import MLP
from utils.model_utils.featurization import Featurizer
from utils.model_utils.vae import VAEEncoder, VAEDecoder
from utils.model_utils.diffusion import Diffusion
from utils.model_utils.classifier import Classifier

class ProteinDiff(nn.Module):
    def __init__(self):
        super().__init__()

        self.featurizer = Featurizer()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()
        self.diffusion = Diffusion()
        self.classifier = Classifier()

    def forward(self, C, L=None, chain_idxs=None, mask=None):
        pass