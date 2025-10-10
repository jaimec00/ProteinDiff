'''
title: 			ProteinDiff.py
author: 		jaime cardenas
description:	main script of the model, holds sub modules for preprocessing, side chain vae, backbone vae, and latent diffusion
				forward has a run_type kwarg to run the appropriate forward pass depending on the training stage (or inference)
'''


import torch
import torch.nn as nn
import math
from typing import Tuple, List

from model.utils.preprocesser import PreProcesser
from model.vae.sc_vae import SideChainVAE
from model.vae.bb_vae import BackBoneVAE
from model.latent_diffusion.diffusion import Diffusion
from static.constants import aa_2_lbl

class ProteinDiff(nn.Module):
	def __init__(self,	d_model: int=256, d_diffusion: int=256, d_sc_latent: int=16, d_bb_latent: int=256,
						top_k: int=16, voxel_dim: int=16, cell_dim: float=0.75,
						sc_enc_layers: int=1, sc_dec_layers: int=1, sc_class_layers: int=1, 
						bb_enc_layers: int=1, bb_dec_layers: int=1, bb_dec_heads: int=8,
						diff_layers: int=1, diff_heads: int=8, diff_parameterization: str="eps", t_max: int=1000
						) -> None:
		super().__init__()
		'''
		this is basically just a wrapper to hold all of the individual models together.
		'''

		# just to make it easier for now when doing up/down sampling convs.
		assert math.log(voxel_dim, 2).is_integer()

		self.prep = PreProcesser(voxel_dim=voxel_dim, cell_dim=cell_dim)
		self.sc_vae = SideChainVAE(voxel_dim=voxel_dim, d_model=d_model, d_latent=d_sc_latent, resnet_enc_layers=sc_enc_layers, resnet_dec_layers=sc_dec_layers, resnet_class_layers=sc_enc_layers)
		self.bb_vae = BackBoneVAE(d_model=d_model, d_latent=d_bb_latent, top_k=top_k, enc_layers=bb_enc_layers, dec_layers=bb_dec_layers, dec_heads=bb_dec_heads)
		self.diffusion = Diffusion(d_model=d_diffusion, d_sc_latent=d_sc_latent, d_bb_latent=d_bb_latent, layers=diff_layers, heads=diff_heads, parameterization=diff_parameterization, t_max=t_max)

	def forward(self, 	coords: torch.Tensor, labels: torch.Tensor, 
						seq_pos: torch.Tensor=None, chain_pos: torch.Tensor=None, 
						atom_mask: torch.Tensor=None, 
						cu_seqlens: torch.Tensor=None, max_seqlen: int=None,
						no_seq_mask: torch.Tensor=None, sample_idx: torch.Tensor=None,
						run_type: str="inference"
						) -> torch.Tensor | Tuple[torch.Tensor]:
		'''
		chooses the appropriate forward pass to run depending on the run_type
		expects the tensors (except cu_seqlens) to be of shape (batch*seq, ...). where all tokens are 
		used for computation. this avoids padding and allows us to use flash attention
		with variable sequence lengths

		args:
			coords: 		atomic coordinates (ZN, 14, 3)
			seq:			amino acid labels (ZN,)
			seq_pos:		sequence indexes (ZN,)
			chain_pos:		chain indexes (ZN,)
			atom_mask:		mask for atoms with no coordinates (ZN, 14)
			no_seq_mask:	mask indicating which positions should not contain sequence info for latent diffusion conditioning (ZN,)
			sample_idx:		the sample idx of each sequence. used to process each sample independantly (ZN,)
			cu_seqlens: 	cumulative sequence lengths for each sample, used for flash attention (Z+1,)
			max_seqlen:		maximum sequence length of the samples, used for flash attention (int)
			run_type: 		indicates the type of forward pass to run, one of 
							["sc_vae", "bb_vae", "diffusion", "inference"]
		'''

		match run_type:
			case "sc_vae": return self._run_sc_vae(coords, labels, atom_mask)
			case "bb_vae": return self._run_bb_vae(coords, seq_pos, chain_pos, cu_seqlens, max_seqlen, sample_idx)
			case "diffusion": return self._run_diffusion(coords, labels, atom_mask, seq_pos, chain_pos, cu_seqlens, max_seqlen, sample_idx, no_seq_mask)
			case "inference": return self._run_inference(coords, labels, seq_pos, chain_pos, cu_seqlens, max_seqlen, sample_idx, no_seq_mask)
			case "-": raise ValueError(f"invalid run_type: {run_type}. must be one of ['inference', 'sc_vae', 'bb_vae', 'diffusion']")
	
	def _run_sc_vae(self, coords: torch.Tensor, labels: torch.Tensor, atom_mask: torch.Tensor) -> Tuple[torch.Tensor]:

		# get the voxels
		_, voxels, _ = self.prep(coords, labels, atom_mask)

		# pass through sc vae (exclude the sampled latent)
		_, latent_mu, latent_logvar, decoded_voxels, seq = self.sc_vae(voxels)

		return latent_mu, latent_logvar, decoded_voxels, voxels, seq

	def _run_bb_vae(self, 	coords: torch.Tensor, seq_pos: torch.Tensor, chain_pos: torch.Tensor, 
							cu_seqlens: torch.Tensor, max_seqlen: int, sample_idx: torch.Tensor
							) -> Tuple[torch.Tensor]:

		# get bb and frames
		coords_bb, frames = self._get_bb_and_frames(coords)

		# run through the bb vae
		_, latent_mu, latent_logvar, distogram, anglogram, seq_pred = self.bb_vae(coords_bb, frames, seq_pos, chain_pos, cu_seqlens, max_seqlen, sample_idx)

		return latent_mu, latent_logvar, distogram, anglogram, seq_pred

	def _run_diffusion(self, 	coords: torch.Tensor, labels: torch.Tensor, atom_mask: torch.Tensor,
								seq_pos: torch.Tensor, chain_pos: torch.Tensor, 
								cu_seqlens: torch.Tensor, max_seqlen: torch.Tensor, 
								sample_idx: torch.Tensor, no_seq_mask: torch.Tensor
								) -> Tuple[torch.Tensor]:

		# get backbone coords
		coords_bb, voxels, frames = self.prep(coords, labels, atom_mask)

		# get side chain latent and reshape
		sc_latent, _, _ = self.sc_vae.enc(voxels)
		sc_latent = sc_latent.reshape(sc_latent.size(0), -1)

		# get backbone latent
		bb_latent, _, _ = self.bb_vae.enc(coords_bb, frames, seq_pos, chain_pos, sample_idx)

		# sample a timestep
		t = self.diffusion.get_rand_t_for(sc_latent)

		# noise the side chain latent and get target (depends on what diffusion_parameterization was)
		sc_latent_noised, trgt = self.diffusion.noise(sc_latent, t)

		# use no seq mask to remove context for conditioning
		seq = labels.masked_fill(no_seq_mask, aa_2_lbl("<mask>"))

		# denoise
		pred = self.diffusion(sc_latent, bb_latent, t, seq, cu_seqlens, max_seqlen)

		return pred, trgt

	def _run_inference(self, 	coords: torch.Tensor, labels: torch.Tensor, 
								seq_pos: torch.Tensor, chain_pos: torch.Tensor, 
								cu_seqlens: torch.Tensor, max_seqlen: int, 
								sample_idx: torch.Tensor, no_seq_mask: torch.Tensor
								) -> torch.Tensor:

		# extract bb coords (including VIRTUAL Cb) and local frames
		coords_bb, frames = self._get_bb_and_frames(coords)

		# compute backbone latents
		bb_latent, _, _ = self.bb_vae.enc(coords_bb, frames, seq_pos, chain_pos, sample_idx)

		# apply no seq mask to determine context diffusion gets
		seq = labels.masked_fill(no_seq_mask, aa_2_lbl("<mask>"))

		# diffusion from pure noise
		sc_latent = self.diffusion.generate(bb_latent, seq, cu_seqlens, max_seqlen)

		# reshape to add Vx,Vy,Vz
		sc_latent = sc_latent.reshape(*sc_latent.shape, 1, 1, 1)

		# decode the latent to a voxel
		voxel = self.sc_vae.dec(sc_latent)

		# predict sequence from decoded voxels
		seq = self.sc_vae.classifier(voxel)

		return seq

	def _get_bb_and_frames(self, coords: torch.Tensor) -> Tuple[torch.Tensor]:
		coords_bb = self.prep.get_backbone(coords)
		_, frames = self.prep.compute_frames(coords_bb)
		return coords_bb, frames