from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math

from proteindiff.static.constants import canonical_aas
from typing import Dict, List

from dataclasses import dataclass, field

# ----------------------------------------------------------------------------------------------------------------------
# losses 

@dataclass
class LossFnCfg:
	seq_lbl_smooth: float=0.0, 
	dist_lbl_smooth: float=0.0, 
	angle_lbl_smooth: float=0.0, 
	kl_div_weight: float = 0.0,
	div_mse_weight: float = 0.0,
	seq_weight: float = 0.0,
	distogram_weight: float = 0.0,
	anglogram_weight: float = 0.0,
	dist_weight: float = 0.0,
	angle_weight: float = 0.0,
	plddt_weight: float = 0.0,
	pae_weight: float = 0.0,
	dist_range: tuple[float, float] = (2.0, 22.0)

@dataclass
class TrainingRunLossesCfg:
	train_type: str
	loss_fn: LossFnCfg = field(default_factory = LossFnCfg)

class TrainingRunLosses:

	def __init__(self, cfg: TrainingRunLossesCfg) -> None:

		self.loss_fn = LossFn(cfg.loss_fn)
		self.train = LossHolder(cfg.train_type)
		self.val = LossHolder(cfg.train_type)
		self.test = LossHolder(cfg.train_type)
		self.tmp = LossHolder(cfg.train_type)

	def clear_tmp_losses(self) -> None:
		self.tmp.clear_losses()

	def set_inference_losses(self, train_type: str) -> None:
		# TODO: implement
		pass

	def to_numpy(self) -> None:
		self.train.to_numpy()
		self.val.to_numpy()
		self.test.to_numpy()

class LossHolder:
	'''
	class to store losses
	'''
	def __init__(self, train_type: str) -> None: 

		if train_type=="vae":
			self.losses = {	
				"full_loss": [],
				"kl_div": [],
				"div_mse": [],
				"seq_cel": [],		
				"seq_accuracy": [],
				"seq_probs": [],
				"distogram_cel": [],					
				"anglogram_cel": [],
				"dist_loss": [], 
				"angle_loss": [], 
				"plddt_loss": [], 
				"pae_loss": [],
			}
		elif train_type=="diffusion":
			self.losses = {"diffusion_mse": []}
		
		# to scale losses for logging, does not affect backprop
		self.valid_toks = 0 # valid tokens to compute avg per token

	def get_avg(self) -> None:
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''
		losses = {loss_type: sum(loss.item() if isinstance(loss, torch.Tensor) else loss for loss in loss_list) / max(1,self.valid_toks.item() if isinstance(self.valid_toks, torch.Tensor) else self.valid_toks) for loss_type, loss_list in self.losses.items()}
		return losses

	def add_losses(self, losses: Dict[str, List[torch.Tensor | float]], valid_toks: int=1) -> None:
		for loss_type, loss in losses.items():
			self.losses[loss_type].append(loss)
		self.valid_toks += valid_toks

	def extend_losses(self, other: Losses) -> None:
		if isinstance(self.valid_toks, torch.Tensor):
			other.to(self.valid_toks.device)
		for loss_type, losses in other.losses.items():
			self.losses[loss_type].extend(losses)
		self.valid_toks += other.valid_toks

	def to(self, device: str) -> None:
		self.losses = {loss_type: [loss.to(device) if isinstance(loss, torch.Tensor) else loss for loss in losses] for loss_type, losses in self.losses.items()}
		self.valid_toks = self.valid_toks.to(device) if isinstance(self.valid_toks, torch.Tensor) else self.valid_toks

	def clear_losses(self) -> None:
		self.losses = {loss_type: [] for loss_type in self.losses.keys()}
		self.valid_toks = 0

	def to_numpy(self) -> None:
		'''utility when plotting losses w/ matplotlib'''
		self.losses = {loss_type: [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else loss for loss in losses] for loss_type, losses in self.losses.items()}

	def get_last_loss(self) -> float | torch.Tensor:
		return self.losses[list(self.losses.keys())[0]][-1]

	def __len__(self) -> int:
		return len(self.losses[list(self.losses.keys())[0]])

# ----------------------------------------------------------------------------------------------------------------------
# loss functions


class LossFn:

	def __init__(self, cfg: LossFnCfg):
		self.seq_cel = CrossEntropyLoss(reduction="sum", ignore_index=-1, label_smoothing=cfg.seq_lbl_smooth)
		self.dist_cel = CrossEntropyLoss(reduction="none", ignore_index=-1, label_smoothing=cfg.dist_lbl_smooth)
		self.angle_cel = CrossEntropyLoss(reduction="none", ignore_index=-1, label_smoothing=cfg.angle_lbl_smooth)
		
		# weights
		self.kl_div_weight = cfg.kl_div_weight
		self.div_mse_weight = cfg.div_mse_weight
		self.seq_weight = cfg.seq_weight
		self.distogram_weight = cfg.distogram_weight
		self.anglogram_weight = cfg.anglogram_weight
		self.dist_weight = cfg.dist_weight
		self.angle_weight = cfg.angle_weight
		self.plddt_weight = cfg.plddt_weight
		self.pae_weight = cfg.pae_weight

		self.min_dist, self.max_dist = cfg.dist_range

	def vae(
		self, 
		latent_mean, 
		latent_logvar, 
		voxels_pred, 
		voxels_true, 
		seq_pred, 
		seq_true, 
		distogram_pred,
		anglogram_pred,
		coords_true, coords_bb_true, frames_true,
		t, x, y,
		sin, cos,
		plddt, pae,
		sample_idx,

	) -> Dict[str, torch.Tensor]:

		# latent loss
		kl_div = self.kl_div(latent_mean, latent_logvar)

		# reconstruction loss
		div_mse = self.mse(voxels_pred, voxels_true)

		# inverse folding loss (+ others)
		seq_cel, matches, probs = self.seq_loss(seq_pred, seq_true)

		# distogram and anglogram loss
		distogram_cel, anglogram_cel = distogram_pred.sum()*0, anglogram_pred.sum()*0
		# self.coarse_struct_loss(distogram_pred, anglogram_pred, coords_bb_true, sample_idx)

		# full struct loss
		dist_loss, angle_loss, plddt_loss, pae_loss = self.fine_struct_loss(
			coords_true, 
			coords_bb_true, 
			frames_true, 
			t, 
			x, 
			y, 
			sin, 
			cos, 
			plddt, 
			pae,
			sample_idx,
		)

		full_loss = (
			self.kl_div_weight*kl_div 
			+ self.div_mse_weight*div_mse
			+ self.seq_weight*seq_cel
			+ self.distogram_weight*distogram_cel
			+ self.anglogram_weight*anglogram_cel
			+ self.dist_weight*dist_loss
			+ self.angle_weight*angle_loss
			+ self.plddt_weight*plddt_loss
			+ self.pae_weight*pae_loss
		)

		losses = {	
			"full_loss": full_loss,
			"kl_div": kl_div,
			"div_mse": div_mse,
			"seq_cel": seq_cel,		
			"seq_accuracy": matches,
			"seq_probs": probs,
			"distogram_cel": distogram_cel,					
			"anglogram_cel": anglogram_cel,
			"dist_loss": dist_loss, 
			"angle_loss": angle_loss, 
			"plddt_loss": plddt_loss, 
			"pae_loss": pae_loss,
		}

		return losses

	def fine_struct_loss(
		self,
		coords_true, 
		coords_bb_true, 
		frames_true, 
		t, 
		x, 
		y, 
		sin, 
		cos, 
		plddt, 
		pae,
		sample_idx,
	):
	
		# TODO: implement this
		return t.sum()*0, x.sum()*0, y.sum()*0, sin.sum()*0


	def diffusion(self, pred: torch.Tensor, trgt: torch.Tensor) -> torch.Tensor:
		mse = self.mse(pred, trgt)
		losses = {"diffusion_mse": mse}
		return losses

	def kl_div(self, z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
		kl_div = -0.5*(1 + z_logvar - z_mu**2 - torch.exp(z_logvar))
		return kl_div.sum()

	def mse(self, pred: torch.Tensor, trgt: torch.Tensor) -> torch.Tensor:
		squared_err = ((pred - trgt)**2)
		mse = squared_err.sum() / torch.tensor(pred.shape[1:], device=pred.device).prod()
		return mse

	def seq_loss(self, seq_pred: torch.Tensor, seq_true: torch.Tensor) -> torch.Tensor:
		cel = self.seq_cel(seq_pred, seq_true)
		matches = self.compute_matches(seq_pred, seq_true)
		probs = self.compute_probs(seq_pred, seq_true)
		return cel, matches, probs

	def compute_matches(self, seq_pred: torch.Tensor, seq_true: torch.Tensor) -> torch.Tensor:
		'''greedy selection'''
		return (torch.argmax(seq_pred, dim=-1) == seq_true).sum() # 1, 

	def compute_probs(self, seq_pred: torch.Tensor, seq_true: torch.Tensor) -> torch.Tensor:
		probs = torch.softmax(seq_pred, dim=-1)
		probs_sum = (torch.gather(probs, -1, seq_true.unsqueeze(-1))).sum()
		return probs_sum

	def coarse_struct_loss(self, distogram: torch.Tensor, anglogram: torch.Tensor, coords_bb: torch.Tensor, sample_idx: torch.Tensor) -> Tuple[torch.Tensor]:
		
		# get Cb position and CaCb unit vec
		Cb, CaCb = self.get_Cb_and_CaCb(coords_bb)

		# distogram loss
		dist_loss = self.distogram_cel(distogram, Cb)

		# anglogram loss
		angle_loss = self.anglogram_cel(anglogram, CaCb)

		return dist_loss, angle_loss

	def get_Cb_and_CaCb(self, coords_bb: torch.Tensor) -> Tuple[torch.Tensor]:

		# extract Ca and Cb coords
		Ca, Cb = coords_bb[:, 1, :], coords_bb[:, 3, :]

		# compute Ca->Cb unit vec
		CaCb = F.normalize(Cb-Ca, p=2, dim=-1, eps=1e-12)

		return Cb, CaCb


	def distogram_cel(self, distogram, Cb):
		dist_bins = distogram.size(-1)
		distogram_true = self.get_dist_lbl(Cb, dist_bins)
		dist_loss = self.struct_reduce_cel(distogram, distogram_true, mode="distogram")
		return dist_loss

	def anglogram_cel(self, anglogram, CaCb):
		angle_bins = anglogram.size(-1)
		anglogram_true = self.get_angle_lbl(CaCb, angle_bins)
		angle_loss = self.struct_reduce_cel(anglogram, anglogram_true, mode="anglogram")
		return angle_loss

	def get_dist_lbl(self, Cb: torch.Tensor, bins: int) -> torch.Tensor:

		# get distogram bins
		dist_bins = torch.linspace(self.min_dist, self.max_dist, bins, device=Cb.device) # bins,

		# compute true distances
		true_dists = torch.linalg.vector_norm(Cb.unsqueeze(1) - Cb.unsqueeze(0), dim=-1) # ZN, ZN

		return self.get_lbl(true_dists, dist_bins)

	def get_angle_lbl(self, CaCb: torch.Tensor, bins: int) -> torch.Tensor:

		# get anglogram bins
		angle_bins = torch.linspace(-1, 1, bins, device=CaCb.device) # bins,

		# compute true distances
		true_angles = torch.linalg.vecdot(CaCb.unsqueeze(1), CaCb.unsqueeze(0), dim=-1) # ZN, ZN

		return self.get_lbl(true_angles, angle_bins)

	def get_lbl(self, true, bins):

		# get the labels
		true_lbl = torch.argmin((true.unsqueeze(-1) - bins.reshape(1,1,-1)).abs(), dim=-1) # ZN, ZN

		return true_lbl

	def struct_reduce_cel(self, pred: torch.Tensor, true: torch.Tensor, mode="distogram"):

		# choose criteria
		match mode:
			case "distogram": criteria = self.distogram_cel
			case "anglogram": criteria = self.anglogram_cel
			case "-": raise ValueError(f"mode must be one of [dist, angle], not {mode}")

		# compute cel, no reduction
		cel = criteria(pred.reshape(-1, pred.size(-1)), true.reshape(-1)).reshape(pred.shape[:-1]) # ZN,ZN

		# get average for each token
		# TODO: sum for now, but want avg of each token, need logic for ZN tensors
		cel = cel.sum(dim=-1) 

		# sum the averages
		return cel.sum()

