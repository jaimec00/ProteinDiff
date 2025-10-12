import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math
from static.constants import canonical_aas
from typing import Dict
from model.utils.preprocesser import PreProcesser

# ----------------------------------------------------------------------------------------------------------------------
# losses 

class TrainingRunLosses:

	def __init__(self, train_type: str, 
						sc_lbl_smooth: float=0.0, sc_beta: float=1e-6,
						bb_dist_lbl_smooth: float=0.0, bb_angle_lbl_smooth: float=0.0, bb_seq_lbl_smooth: float=0.0,
						bb_beta: float=1e-6, bb_min_dist: float=2.0, bb_max_dist: float=22.0 
						) -> None:

		self.loss_function = LossFunction(	sc_lbl_smooth, sc_beta, 
											bb_dist_lbl_smooth, bb_angle_lbl_smooth, bb_seq_lbl_smooth, 
											bb_beta, bb_min_dist, bb_max_dist
										)

		self.train = Losses(train_type)
		self.val = Losses(train_type)
		self.test = Losses(train_type)
		self.tmp = Losses(train_type)

	def clear_tmp_losses(self) -> None:
		self.tmp.clear_losses()

	def set_inference_losses(self, train_type: str) -> None:

		# inference mode only applicable after train vae and diffusion
		if train_type=="diffusion": # run full inference
			self.tmp.losses = {	"Cross Entropy Loss": [],
								"Top 1 Accuracy": [],
								"Top 3 Accuracy": [],
								"Top 5 Accuracy": [],
								"True AA Predicted Probability": []
							}
			self.test.losses = {	"Cross Entropy Loss": [],
									"Top 1 Accuracy": [],
									"Top 3 Accuracy": [],
									"Top 5 Accuracy": [],
									"True AA Predicted Probability": []
								}
		else:
			self.clear_tmp_losses() # (sc/bb)_vae evaluation is the same as training

	def to_numpy(self) -> None:
		self.train.to_numpy()
		self.val.to_numpy()
		self.test.to_numpy()

class Losses:
	'''
	class to store losses
	'''
	def __init__(self, train_type: str) -> None: 

		if train_type=="sc_vae":
			self.losses = {	
							"Full Loss": [],
							"KL Divergence": [],
							"Mean Squared Error": [],
							"Cross Entropy Loss": [],
							"Accuracy": [],
							"True AA Predicted Probability": []
						}
		elif train_type=="bb_vae":
			self.losses = {	
							"Full Loss": [],

							"KL Divergence": [],
							
							"Distogram Cross Entropy Loss": [],
							"Distogram Accuracy": [],
							"True DistBin Predicted Probability": []
							
							"Anglogram Cross Entropy Loss": [],
							"Anglogram Accuracy": [],
							"True AngleBin Predicted Probability": []
							
							"Sequence Cross Entropy Loss": [],
							"Sequence Accuracy": [],
							"True AA Predicted Probability": []
						}
		elif train_type=="diffusion":
			self.losses = {"Mean Squared Error": []}
		
		# to scale losses for logging, does not affect backprop
		self.valid_toks = 0 # valid tokens to compute avg per token

	def get_avg(self) -> None:
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''
		losses = {loss_type: sum(loss.item() if isinstance(loss, torch.Tensor) else loss for loss in loss_list) / (self.valid_toks.item() if isinstance(self.valid_toks, torch.Tensor) else self.valid_toks) for loss_type, loss_list in self.losses.items()}
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

class LossFunction:

	def __init__(self, sc_lbl_smooth: float=0.0, sc_beta: float=1e-6,
						bb_dist_lbl_smooth: float=0.0, bb_angle_lbl_smooth: float=0.0, bb_seq_lbl_smooth: float=0.0,
						bb_beta: float=1e-6, bb_min_dist: float=2.0, bb_max_dist: float=22.0 
				):
		self._sc_seq_cel = CrossEntropyLoss(reduction="sum", ignore_index=-1, label_smoothing=sc_lbl_smooth)
		self._bb_seq_cel = CrossEntropyLoss(reduction="sum", ignore_index=-1, label_smoothing=bb_seq_lbl_smooth)
		self._bb_dist_cel = CrossEntropyLoss(reduction="none", ignore_index=-1, label_smoothing=bb_dist_lbl_smooth)
		self._bb_angle_cel = CrossEntropyLoss(reduction="none", ignore_index=-1, label_smoothing=bb_angle_lbl_smooth)
		self._sc_beta = sc_beta
		self._bb_beta = bb_beta
		
		self._bb_min_dist = bb_min_dist
		self._bb_max_dist = bb_max_dist

	def sc_vae(self, latent_mean: torch.Tensor, latent_logvar: torch.Tensor, 
					voxels_pred: torch.Tensor, voxels_true: torch.Tensor, 
					seq_pred: torch.Tensor, seq_true: torch.Tensor, 
					mask: torch.Tensor) -> Dict[str, torch.Tensor]:

		kl_div = self._kl_div(latent_mean, latent_logvar, mask)
		mse = self._mse(voxels_pred, voxels_true, mask)
		cel = self._seq_cel(seq_pred, seq_true, mask, mode="sc")
		matches = self._compute_matches(seq_pred, seq_true, mask)
		probs = self._compute_probs(seq_pred, seq_true, mask)

		full_loss = self._sc_beta*kl_div + mse + cel

		losses = {	"Full Loss": full_loss,
					"KL Divergence": kl_div,
					"Mean Squared Error": mse,
					"Cross Entropy Loss": cel,					
					"Accuracy": matches,
					"True AA Predicted Probability": probs
				}

		return losses

	def bb_vae(self, latent_mean: torch.Tensor, latent_logvar: torch.Tensor, 
					distogram: torch.Tensor, anglogram: torch.Tensor, coords: torch.Tensor,
					seq_pred: torch.Tensor, seq_true: torch.Tensor, 
					mask: torch.Tensor) -> Dict[str, torch.Tensor]:

		kl_div = self._kl_div(latent_mean, latent_logvar, mask)
		dist_loss, angle_loss = self._struct_cel(distogram, anglogram, coords, mask)
		seq_loss = self._seq_cel(seq_pred, seq_true, mask, mode="bb")
		matches = self._compute_matches(seq_pred, seq_true, mask)
		probs = self._compute_probs(seq_pred, seq_true, mask)

		full_loss = self._bb_beta*kl_div + dist_loss + angle_loss + seq_loss

		losses = {	"Full Loss": full_loss,
					"KL Divergence": kl_div,
					"Distogram Cross Entropy Loss": dist_loss,					
					"Anglogram Cross Entropy Loss": angle_loss,					
					"Sequence Cross Entropy Loss": seq_loss,					
					"Accuracy": matches,
					"True AA Predicted Probability": probs
				}

		return losses

	def diffusion(self, pred: torch.Tensor, trgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		mse = self._mse(pred, trgt, mask)
		losses = {"Mean Squared Error": mse}

		return losses

	def inference(self, seq_pred: torch.Tensor, seq_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		cel = self.cel(seq_pred, seq_true, mask)
		matches = self.compute_matches(seq_pred, seq_true, mask)
		probs = self.compute_probs(seq_pred, seq_true, mask)

		losses = {	"Cross Entropy Loss": cel,					
					"Accuracy": matches,
					"True AA Predicted Probability": probs
				}

		return losses

	def _broadcast_to(self, mask: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
		'''reshapes a ZN mask so that it can be broadcast to other shape'''
		return mask.reshape(mask.size(0), *((1,)*(other.dim()-1)))

	def _kl_div(self, z_mu: torch.Tensor, z_logvar: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		kl_div = -0.5*(1 + z_logvar - z_mu**2 - torch.exp(z_logvar))*self._broadcast_to(mask, z_logvar)
		return kl_div.sum()

	def _mse(self, pred: torch.Tensor, trgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		squared_err = ((pred - trgt)**2) * self._broadcast_to(mask, pred)
		mse = squared_err.sum() / torch.tensor(pred.shape[1:], device=pred.device).prod()
		return mse

	def _seq_cel(self, seq_pred: torch.Tensor, seq_true: torch.Tensor, mask: torch.Tensor, mode: str="sc") -> torch.Tensor:
		match mode:
			case "sc": criteria = self._sc_seq_cel
			case "bb": criteria = self._bb_seq_cel
			case "-": raise ValueError(f"mode must be one of [sc, bb], not {mode}")
		seq_true = seq_true.masked_fill(~mask, -1)
		cel = criteria(seq_pred, seq_true)
		return cel

	def _struct_cel(self, distogram: torch.Tensor, anglogram: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor]:
		
		# get Cb position and CaCb unit vec
		Cb, CaCb = self._get_Cb_and_CaCb(coords)

		# get ZN,ZN pairwise mask
		pw_mask = self._get_pw_mask(mask)

		# distogram loss
		dist_loss = self._dist_cel(distogram, Cb, pw_mask)

		# anglogram loss
		angle_loss = self._angle_cel(anglogram, CaCb, pw_mask)

		return dist_loss, angle_loss

	def _get_Cb_and_CaCb(self, coords: torch.Tensor) -> Tuple[torch.Tensor]:

		# get backbone
		coords_bb = PreProcesser.get_backbone(coords)

		# extract Ca and Cb coords
		Ca, Cb = coords_bb[:, 1, :], coords_bb[:, 3, :]

		# compute Ca->Cb unit vec
		CaCb = F.normalize(Cb-Ca, p=2, dim=-1, eps=1e-12)

		return Cb, CaCb

	def _get_pw_mask(self, mask: torch.Tensor) -> torch.Tensor:
		return mask.unsqueeze(0) & mask.unsqueeze(1)

	def _dist_cel(self, distogram, Cb, mask):
		dist_bins = distogram.size(-1)
		distogram_true = self._get_dist_lbl(Cb, dist_bins, mask)
		dist_loss = self._struct_reduce_cel(distogram, distogram_true, mask, mode="dist")
		return dist_loss

	def _angle_cel(self, anglogram, CaCb, mask):
		angle_bins = anglogram.size(-1)
		anglogram_true = self._get_angle_lbl(CaCb, angle_bins, mask)
		angle_loss = self._struct_reduce_cel(anglogram, anglogram_true, mask, mode="angle")
		return angle_loss

	def _get_dist_lbl(self, Cb: torch.Tensor, bins: int, mask: torch.Tensor) -> torch.Tensor:

		# get distogram bins
		dist_bins = torch.linspace(self._bb_min_dist, self._bb_max_dist, bins, device=Cb.device) # bins,

		# compute true distances
		true_dists = torch.linalg.vector_norm(Cb.unsqueeze(1) - Cb.unsqueeze(0), dim=-1) # ZN, ZN

		return self._get_lbl(true_dists, dist_bins, mask)

	def _get_angle_lbl(self, CaCb: torch.Tensor, bins: int, mask: torch.Tensor) -> torch.Tensor:

		# get anglogram bins
		angle_bins = torch.linspace(-1, 1, bins, device=CaCb.device) # bins,

		# compute true distances
		true_angles = torch.linalg.vecdot(CaCb.unsqueeze(1), CbCb.unsqueeze(0), dim=-1) # ZN, ZN

		return self._get_lbl(true_angles, angle_bins, mask)

	def _get_lbl(self, true, bins, mask):

		# get the labels
		true_lbl = torch.argmin((true.unsqueeze(-1) - bins.reshape(1,1,-1)).abs(), dim=-1) # ZN, ZN

		# ignore masked positions
		true_lbl.masked_fill_(~mask, -1)

		return true_lbl

	def _struct_reduce_cel(self, pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor, mode="dist"):

		# choose criteria
		match mode:
			case "dist": criteria = self._bb_dist_cel
			case "angle": criteria = self._bb_angle_cel
			case "-": raise ValueError(f"mode must be one of [dist, angle], not {mode}")

		# compute cel, no reduction
		cel = criteria(pred.reshape(-1, pred.size(-1)), true_lbl.reshape(-1)).reshape(pred.shape[:-1]) # ZN,ZN

		# get average for each token
		cel = cel.sum(dim=-1) / valid.sum(dim=-1).clamp(min=1) # ZN,

		# sum the averages
		return cel.sum()

	def _compute_matches(self, seq_pred: torch.Tensor, seq_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		'''greedy selection'''
		return ((torch.argmax(seq_pred, dim=-1).reshape(-1) == seq_true) & mask).sum() # 1, 

	def _compute_probs(self, seq_pred: torch.Tensor, seq_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		probs = torch.softmax(seq_pred, dim=-1)
		probs_sum = (mask.unsqueeze(-1)*torch.gather(probs, 2, (seq_true*mask).unsqueeze(2))).sum()
		return probs_sum