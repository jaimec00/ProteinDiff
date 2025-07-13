
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc = VAEEncoder()
        self.dec = VAEDecoder()

    def forward(self, fields):
        latent, latent_mu, latent_logvar = self.enc(fields)
        fields_pred = self.dec(latent)

        return latent, latent_mu, latent_logvar, fields_pred

class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()


        self.encoder = nn.Sequential(   
                                        # increase channels, keep spatial res
                                        nn.Conv3d(3, 32, 3, stride=1, padding='same', bias=False),
                                        nn.GroupNorm(4, 32),
                                        nn.SiLU(),

                                        # downsample
                                        nn.Conv3d(32, 64, 3, stride=2, padding=1, bias=False),
                                        nn.GroupNorm(8, 64),
                                        nn.SiLU(),

                                        # downsample
                                        nn.Conv3d(64, 128, 3, stride=2, padding=1, bias=False),
                                        nn.GroupNorm(16, 128),
                                        nn.SiLU(),

                                        # project to latent params
                                        nn.Conv3d(128, 8, 1, stride=1, padding="same", bias=False)
                                    )

    def forward(self, fields):
        '''
        fields (torch.Tensor): full voxels of each residue, of shape Z,N,3,Vx,Vy,Vz
        there is no cross talk, each residue is operated on independantly
        '''

        Z, N, Cin, Vx, Vy, Vz = fields.shape 

        # reshape to be compatible w/ torch convolutions, no cross talk, so simply flattent the Z,N part,
        fields = fields.view(Z*N, Cin, Vx, Vy, Vz)

        # get latent params
        latent_params = self.encoder(fields)
        
        # reshape to Z,N,2*Cout,4x4x4
        _, two_C_out, zx, zy, zz = latent_params.shape
        latent_params = latent_params.view(Z, N, two_C_out, zx, zy, zz)

        # split into mu and logvar
        z_mu, z_logvar = torch.chunk(latent_params, dim=2)

        # sample a latent
        z = z_mu + torch.randn_like(z_logvar)*torch.exp(0.5*z_logvar)

        return z, z_mu, z_logvar


class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()


        self.decoder = nn.Sequential(   
                                        # increase channels, keep spatial res at 4x4x4
                                        nn.Conv3d(4, 128, 3, stride=1, padding='same', bias=False),
                                        nn.GroupNorm(16, 128),
                                        nn.SiLU(),

                                        # upsample to 8x8x8
                                        nn.ConvTranspose3d(128, 64, 3, stride=2, padding=1, output_padding=1, bias=False),
                                        nn.GroupNorm(8, 64),
                                        nn.SiLU(),

                                        # upsample 16x16x16
                                        nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1, bias=False),
                                        nn.GroupNorm(4, 32),
                                        nn.SiLU(),

                                        # reconstruct the final voxel
                                        nn.Conv3d(32, 3, 1, stride=1, padding="same", bias=False)
                                    )

    def forward(self, latent):
        '''
        latent (torch.Tensor): latent voxels of each residue, of shape Z,N,4,4,4,4
        there is no cross talk, each residue is operated on independantly
        '''

        Z, N, Cin, Vx, Vy, Vz = latent.shape 

        # reshape to be compatible w/ torch convolutions, no cross talk, so simply flattent the Z,N part,
        latent = latent.view(Z*N, Cin, Vx, Vy, Vz)

        # reconstruct the fields and reshape
        fields = self.decoder(latent)
        fields = fields.view(Z, N, -1, Vx, Vy, Vz)

        # norm them to unit vectors
        fields_norm = torch.linalg.vector_norm(fields, dim=2, keepdim=True)
        fields = fields / fields_norm.masked_fill(fields_norm==0, 1)
        
        return fields

