from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math

from proteus.static.constants import canonical_aas
from typing import Dict, List, Tuple, Any, Union
from proteus.types import Float, Int, Bool, T

from dataclasses import dataclass, field
from omegaconf import MISSING
from collections import defaultdict

# ----------------------------------------------------------------------------------------------------------------------
# losses 

@dataclass
class LossFnCfg:
	weights: Dict[str, float] = MISSING
	loss_fns: Dict[str, str] = MISSING
	inputs: Dict[str, List[Any]] = MISSING

class TrainingRunLosses:

	def __init__(self, loss_fn_cfg: LossFnCfg) -> None:

		self.loss_fn: LossFn = LossFn(loss_fn_cfg)
		self.train: LossHolder = LossHolder()
		self.val: LossHolder = LossHolder()
		self.test: LossHolder = LossHolder()
		self.tmp: LossHolder = LossHolder()

class LossHolder:
	'''
	class to store losses
	'''
	def __init__(self) -> None:

		self.losses = defaultdict(list)
		
		# to scale losses for logging, does not affect backprop
		self.valid_toks: Union[int, Int[T, "1"]] = 0 # valid tokens to compute avg per token

	def get_avg(self) -> Dict[str, float]:
		'''this method is just for logging purposes, does not rescale loss used in bwd pass'''
		avg_losses = {}
		for loss_type in self.losses.keys():
			avg_losses[loss_type] = torch.stack(self.losses[loss_type]).sum() / (self.valid_toks.clamp(min=1) if isinstance(self.valid_toks, torch.Tensor) else max(1, self.valid_toks))
		avg_losses = {loss_type: loss.item() for loss_type, loss in avg_losses.items()}
		return avg_losses

	def add_losses(self, losses: Dict[str, Union[torch.Tensor, float]], valid_toks: Int[T, "1"] | int = 1) -> None:
		for loss_type, loss in losses.items():
			self.losses[loss_type].append(loss)
		self.valid_toks += valid_toks

	def extend_losses(self, other: 'LossHolder') -> None:
		if isinstance(self.valid_toks, torch.Tensor):
			other.to(self.valid_toks.device)
		for loss_type, losses in other.losses.items():
			self.losses[loss_type].extend(losses)
		self.valid_toks += other.valid_toks

	def to(self, device: str) -> None:
		self.losses = {loss_type: [loss.to(device) for loss in losses] for loss_type, losses in self.losses.items()}
		self.valid_toks = self.valid_toks.to(device) if isinstance(self.valid_toks, torch.Tensor) else self.valid_toks

	def clear(self) -> None:
		self.losses.clear()
		self.valid_toks = 0

	def get_last_loss(self) -> Union[float, torch.Tensor]:
		return self.losses["full_loss"][-1]

	def get_last_losses(self, scale: float = 1) -> Dict[str, float]:
		# .item() causes cpu gpu sync, slows down, but this is only done on logging steps 
		return {k: (losses[-1]*scale).item() for k, losses in self.losses.items()}

	def __len__(self) -> int:
		return len(self.losses[list(self.losses.keys())[0]])


class LossFn(nn.Module):
	def __init__(self, cfg: LossFnCfg):
		super().__init__()
		self.weights = cfg.weights
		self.loss_fns = cfg.loss_fns
		self.inputs = cfg.inputs

		for loss_fn in self.loss_fns.values():
			assert hasattr(self, loss_fn), f"LossFn has no {loss_fn} method"

	def forward(self, outputs: Dict[str, Any]):

		losses = {"full_loss": 0.0}

		for loss_fn_str in self.loss_fns.keys():

			loss_fn = getattr(self, self.loss_fns[loss_fn_str])
			args, kwargs = self._get_args_kwargs(loss_fn_str, outputs)
			loss = loss_fn(*args, **kwargs)

			weight = self.weights[loss_fn_str]
			if weight:
				losses["full_loss"] += loss*weight

			losses[loss_fn_str] = loss

		return losses

	def _get_args_kwargs(self, loss_fn_str, outputs):
		'''
		the convention here is that args come from the output, kwargs are "hard coded"
		'''
		arg_inputs, kwargs = self.inputs[loss_fn_str]
		args = [outputs[arg] for arg in arg_inputs]
		return args, kwargs
		
	# loss functions
	def kl_div(self, mu, logvar):
		...

	def mse(self, pred, trgt):
		...

	def cel(self, logits, labels, mask):
		labels = labels.masked_fill(~mask, -1)
		cel = F.cross_entropy(logits, labels, ignore_index=-1, reduction="sum")

		return cel

	def focal_loss(self, logits, labels, mask, alpha=1.0, gamma=2.0):
		labels = labels.masked_fill(~mask, -1)
		cel = F.cross_entropy(logits, labels, ignore_index=-1, reduction="none")
		p_t = torch.exp(-cel)  # p_t = exp(log(p_t))
		focal_weight = (1 - p_t) ** gamma
		focal_loss = alpha * focal_weight * cel

		return focal_loss.sum()

	def matches(self, logits, labels, mask):
		labels = labels.masked_fill(~mask, -1)
		matches = (torch.argmax(logits, dim=-1) == labels) * mask
		return matches.sum()

	def probs(self, logits, labels, mask):
		labels = labels.masked_fill(~mask, 0)
		probs = torch.gather(torch.softmax(logits, dim=-1), -1, labels.unsqueeze(-1)).squeeze(-1)*mask
		return probs.sum()