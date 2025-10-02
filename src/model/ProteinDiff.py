import torch
import torch.nn as nn
import math
from typing import Tuple

from model.utils.preprocesser import PreProcesser
from model.vae.sc_vae import SideChainVAE
from model.vae.bb_vae import BackBoneVAE
from model.latent_diffusion.diffusion import Diffusion
from static.constants import aa_2_lbl

class ProteinDiff(nn.Module):
	def __init__(self,	d_model: int=256, d_diffusion: int=256, d_sc_latent: int=16, d_bb_latent: int=16, 
						top_k: int=16, voxel_dims: int=16, cell_dim: float=0.75,
						sc_vae_layers: int=1, bb_vae_layers: int=3, diff_layers: int=3, diff_parameterization: str="eps"
						) -> None:
		super().__init__()
		'''
		this is basically just a wrapper to hold all of the individual models together.
		training run handles how to use them efficiently. 
		'''

		# just to make it easier for now when doing up/down sampling convs.
		assert math.log(voxel_dims, 2).is_integer()

		self.prep = PreProcesser(voxel_dims=(voxel_dims,)*3, cell_dim=cell_dim)
		self.sc_vae = SideChainVAE(voxel_dim=voxel_dims, d_model=d_model, d_latent=d_latent, resnet_layers=vae_layers)
		self.bb_vae = BackBoneVAE(d_model=d_model, top_k=top_k, enc_layers=enc_layers, dec_layers=dec_layers)
		self.diffusion = Diffusion(d_model=d_diffusion, d_latent=d_latent, layers=diff_layers, parameterization=diff_parameterization, t_max=1000)

	def forward(self, 	coords: torch.Tensor, labels: torch.Tensor, 
						seq_pos: torch.Tensor=None, chain_pos: torch.Tensor=None, 
						atom_mask: torch.Tensor=None, valid_mask: torch.Tensor=None, no_seq_mask: torch.Tensor=None
						run_type: str="inference"
						) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | :

		match run_type:
			case "inference": return self._run_inference(coords, labels, seq_pos, chain_pos, valid_mask)
			case "sc_vae": return self._run_sc_vae(coords, atom_mask, valid_mask)
			case "bb_vae": return self._run_bb_vae(coords, seq_pos, chain_pos, valid_mask)
			case "diffusion": return self._run_diffusion(coords, labels, seq_pos, chain_pos, atom_mask, valid_mask, no_seq_mask)
			case "-": raise ValueError(f"invalid run_type: {run_type}. must be one of ['inference', 'sc_vae', 'bb_vae', 'diffusion']")

	def _run_inference(self, 	coords: torch.Tensor, labels: torch.Tensor, 
								seq_pos: torch.Tensor, chain_pos: torch.Tensor, 
								valid_mask: torch.Tensor, no_seq_mask: torch.Tensor
								) -> torch.Tensor:

		# extract bb coords (including VIRTUAL Cb) and local frames
		coords_bb, frames = self._get_bb_and_frames(coords)

		# compute structure latents
		latent_bb, _, _ = self.bb_vae.enc(coords_bb, frames, seq_pos, chain_pos, valid_mask)

		# diffusion from pure noise, 
		labels = labels.masked_fill(no_seq_mask, aa_2_lbl("<mask>"))
		latent = self.diffusion.generate(latent_bb, labels, valid_mask)

		# decode the latent to a voxel
		voxel = self.sc_vae.dec(latent)

		# predict sequence from decoded voxels
		seq = self.sc_vae.classifier(voxel)

		return seq
	
	def _run_sc_vae(self, 	coords: torch.Tensor, labels: torch.Tensor, valid_mask: torch.Tensor, atom_mask: torch.Tensor
							) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

		# get the voxels
		_, voxels, _ = self.prep(coords, labels, atom_mask, valid_mask)

		# pass through div vae (exclude the sampled latent)
		_, latent_mu, latent_logvar, decoded_voxels, seq = self.sc_vae(voxels)

		return latent_mu, latent_logvar, decoded_voxels, voxels, seq

	def _run_bb_vae(self, 	coords: torch.Tensor, seq_pos: torch.Tensor, chain_pos: torch.Tensor, valid_mask: torch.Tensor
							) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

		# get bb and frames
		coords_bb, frames = self._get_bb_and_frames(coords)

		# run through the bb vae
		_, latent_mu, latent_logvar, distogram, anglogram, seq_pred = self.bb_vae(coords_bb, frames, seq_pos, chain_pos, valid_mask)

		return latent_mu, latent_logvar, distogram, anglogram, seq_pred

	def _run_diffusion(self, 	coords: torch.Tensor, labels: torch.Tensor, 
								seq_pos: torch.Tensor, chain_pos: torch.Tensor, 
								valid_mask: torch.Tensor, no_seq_mask: torch.Tensor
								) -> torch.Tensor:

		coords_bb, voxels, frames = self.prep(coords, labels, atom_mask, valid_mask)
		sc_latent, _, _ = self.sc_vae.enc(voxels)
		bb_latent, _, _ = self.bb_vae.enc(coords_bb, frames, seq_pos, chain_pos, valid_mask)
		t = self.diffusion.get_rand_t_for(sc_latent)
		sc_latent_noised, trgt = self.diffusion.noise(sc_latent, t)
		seq = labels.masked_fill(no_seq_mask, aa_2_lbl("<mask>"))
		pred = self.diffusion(sc_latent, bb_latent, t, seq, valid_mask)

		return pred, trgt

	def _get_bb_and_frames(self, coords):
		coords_bb = self.prep.get_backbone(coords)
		_, frames = self.prep.compute_frames(coords_bb)
		return coords_bb, frames