
import torch
import torch.nn as nn

class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

    def forward(self, fields, kp_mask):
        '''
        fields (torch.Tensor): full voxels of each residue, of shape Z,N,Vx,Vy,Vz,3
        '''

        conv