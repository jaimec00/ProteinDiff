import torch
import torch.nn as nn
from data.constants import canonical_aas

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # start at 16x16x16x3
        # increase latent to 32
        self.classifier = nn.Sequential(
                                        # increase channels, keep spatial res at 16x16x16
                                        nn.Conv3d(3, 16, 3, stride=1, padding='same', bias=False),
                                        nn.GroupNorm(2, 16),
                                        nn.SiLU(),

                                        # downsample 8x8x8
                                        nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False),
                                        nn.GroupNorm(4, 32),
                                        nn.SiLU(),

                                        # downsample 4x4x4
                                        nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
                                        nn.GroupNorm(8, 64),
                                        nn.SiLU(),

                                        # downsample 2x2x2
                                        nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
                                        nn.GroupNorm(16, 128),
                                        nn.SiLU(),
                                        
                                        # downsample 1x1x1
                                        nn.Conv3d(128, 128, 3, stride=2, padding=1, bias=False),
                                        nn.GroupNorm(32, 256),

                                    ) 

        self.classify = nn.Linear(128, len(canonical_aas))


    def forward(self, fields):

        Z, N, Cin, Vx, Vy, Vz = fields.shape
        fields = self.classifier(fields.view(Z*N, Cin, Vx, Vy, Vz))

        # output is Z*N, Cout, 1,1,1 so reshape to Z,N,Cout
        fields = fields.view(Z, N, -1)

        # project to amino acids, Z,N,20
        aas = self.classify(fields)

        return aas