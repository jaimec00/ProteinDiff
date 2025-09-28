import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import numpy as np
import math
from static.constants import canonical_aas

# ----------------------------------------------------------------------------------------------------------------------
# losses 

class TrainingRunLosses():

	def __init__(self, train_type, label_smoothing=0.0, beta=1e-3):

		self.loss_function = LossFunction(label_smoothing, beta)

		self.train = Losses(train_type)
		self.val = Losses(train_type)
		self.test = Losses(train_type)
		self.tmp = Losses(train_type)

	def clear_tmp_losses(self):
		self.tmp.clear_losses()

	def set_inference_losses(self, train_type):

		# inference mode only applicable after train vae and diffusion
		if train_type=="vae":
			self.clear_tmp_losses() # vae evaluation is the same as training
		elif train_type=="diffusion": # run full inference
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

	def to_numpy(self):
		self.train.to_numpy()
		self.val.to_numpy()
		self.test.to_numpy()

class Losses():
	'''
	class to store losses
	'''
	def __init__(self, train_type): 

		if train_type=="diffusion":
			self.losses = {"Mean Squared Error": []}
		elif train_type=="vae":
			self.losses = {	
							"Full Loss": [],
							"Mean Squared Error": [],
							"KL Divergence": [],
							"Cross Entropy Loss": [],
							"Top 1 Accuracy": [],
							"Top 3 Accuracy": [],
							"Top 5 Accuracy": [],
							"True AA Predicted Probability": []
						}

		# to scale losses for logging, does not affect backprop
		self.valid_toks = 0 # valid tokens to compute avg per token

	def get_avg(self):
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''
		losses = {loss_type: sum(loss.item() if isinstance(loss, torch.Tensor) else loss for loss in loss_list) / (self.valid_toks.item() if isinstance(self.valid_toks, torch.Tensor) else self.valid_toks) for loss_type, loss_list in self.losses.items()}
		return losses

	def add_losses(self, losses, valid_toks=1):
		for loss_type, loss in losses.items():
			self.losses[loss_type].append(loss)
		self.valid_toks += valid_toks

	def extend_losses(self, other):
		if isinstance(self.valid_toks, torch.Tensor):
			other.to(self.valid_toks.device)
		for loss_type, losses in other.losses.items():
			self.losses[loss_type].extend(losses)
		self.valid_toks += other.valid_toks

	def to(self, device):
		self.losses = {loss_type: [loss.to(device) if isinstance(loss, torch.Tensor) else loss for loss in losses] for loss_type, losses in self.losses.items()}
		self.valid_toks = self.valid_toks.to(device) if isinstance(self.valid_toks, torch.Tensor) else self.valid_toks

	def clear_losses(self):
		self.losses = {loss_type: [] for loss_type in self.losses.keys()}
		self.valid_toks = 0

	def to_numpy(self):
		'''utility when plotting losses w/ matplotlib'''
		self.losses = {loss_type: [loss.detach().to("cpu").numpy() if isinstance(loss, torch.Tensor) else loss for loss in losses] for loss_type, losses in self.losses.items()}

	def get_last_loss(self):
		return self.losses[list(self.losses.keys())[0]][-1]

	def __len__(self):
		return len(self.losses[list(self.losses.keys())[0]])

# ----------------------------------------------------------------------------------------------------------------------
# loss functions 

class LossFunction():

	def __init__(self, label_smoothing=0.0, beta=1e-3):
		self.cel_raw = CrossEntropyLoss(reduction="sum", ignore_index=-1, label_smoothing=label_smoothing)
		self.beta = beta

	def kl_div(self, latent_mean, latent_logvar, mask):

		Z, N, S, Vx, Vy, Vz = latent_mean.shape

		kl_div = -0.5*(1 + latent_logvar - latent_mean**2 - torch.exp(latent_logvar))*mask.view(Z,N,1,1,1,1)

		return kl_div.sum()

	def mse(self, pred, trgt, mask):
		Z, N, S, Vx, Vy, Vz = pred.shape
		squared_err = ((pred - trgt)**2) * mask.view(Z,N,1,1,1,1)
		mse = squared_err.sum() / (S*Vx*Vy*Vz)
		return mse

	def cel(self, seq_pred, seq_true, mask):
		Z, N, AA = seq_pred.shape
		seq_true = seq_true.masked_fill(~mask, -1)
		cel = self.cel_raw(seq_pred.view(-1, AA), seq_true.view(-1))
		return cel

	def compute_matches(self, seq_pred, seq_true, mask):
		'''
		greedy selection, computed seq sim here for simplicity, will do it with other losses later 
		also computes top3 and top5 accuracy
		'''
		
		# dont need softmax for topk analysis
		top1 = torch.argmax(seq_pred, dim=2).view(-1) # Z*N,
		top3 = torch.topk(seq_pred, 3, 2, largest=True, sorted=False).indices.view(-1, 3) # Z*N x 3
		top5 = torch.topk(seq_pred, 5, 2, largest=True, sorted=False).indices.view(-1, 5) # Z*N x 5

		true_flat = seq_true.view(-1) # Z x N --> Z*N,
		mask_flat = mask.view(-1)

		matches1 = ((top1 == true_flat) & mask_flat).sum() # 1, 
		matches3 = ((top3 == true_flat[:, None]).any(dim=1) & mask_flat).sum() # 1, 
		matches5 = ((top5 == true_flat[:, None]).any(dim=1) & mask_flat).sum() # 1, 

		return matches1, matches3, matches5

	def compute_probs(self, seq_pred, seq_true, mask):
		probs = torch.softmax(seq_pred, dim=2)
		probs_sum = (mask.unsqueeze(2)*torch.gather(probs, 2, (seq_true*mask).unsqueeze(2))).sum()
		return probs_sum

	def vae(self, latent_mean, latent_logvar, fields_pred, fields_true, seq_pred, seq_true, mask):

		kl_div = self.kl_div(latent_mean, latent_logvar, mask)
		mse = self.mse(fields_pred, fields_true, mask)
		cel = self.cel(seq_pred, seq_true, mask)
		matches1, matches3, matches5 = self.compute_matches(seq_pred, seq_true, mask)
		probs = self.compute_probs(seq_pred, seq_true, mask)

		full_loss = self.beta*kl_div + mse + cel

		losses = {	"Full Loss": full_loss,
					"KL Divergence": kl_div,
					"Mean Squared Error": mse,
					"Cross Entropy Loss": cel,					
					"Top 1 Accuracy": matches1,
					"Top 3 Accuracy": matches3,
					"Top 5 Accuracy": matches5,
					"True AA Predicted Probability": probs
				}

		return losses


	def diff(self, pred, trgt, mask):
		mse = self.mse(pred, trgt, mask)
		losses = {"Mean Squared Error": mse}

		return losses

	def inference(self, seq_pred, seq_true, mask):
		cel = self.cel(seq_pred, seq_true, mask)
		matches1, matches3, matches5 = self.compute_matches(seq_pred, seq_true, mask)
		probs = self.compute_probs(seq_pred, seq_true, mask)

		losses = {	"Cross Entropy Loss": cel,					
					"Top 1 Accuracy": matches1,
					"Top 3 Accuracy": matches3,
					"Top 5 Accuracy": matches5,
					"True AA Predicted Probability": probs
				}

		return losses


		