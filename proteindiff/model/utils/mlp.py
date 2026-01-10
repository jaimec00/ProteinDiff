import torch.nn.functional as F
import torch.nn.init as init
from dataclasses import dataclass, field
from enum import StrEnum

import torch
import torch.nn as nn

from proteindiff.model import Base


class ActivationFn(StrEnum):
	GELU = "gelu"
	SILU = "silu"
	RELU = "relu"
	SIGMOID = "sigmoid"

@dataclass
class MLPCfg:
	d_in: int = 512
	d_out: int = 512
	d_hidden: int = 1024
	hidden_layers: int = 0
	dropout: float = 0.0
	act: ActivationFn = ActivationFn.GELU
	zeros: bool = False

class MLP(Base):
	'''
	base mlp class for use by other modules. uses gelu
	'''

	def __init__(self, cfg: MLPCfg):
		super().__init__()

		self.in_proj = nn.Linear(cfg.d_in, cfg.d_hidden)
		self.hidden_proj = nn.ModuleList([nn.Linear(cfg.d_hidden, cfg.d_hidden) for _ in range(cfg.hidden_layers)])
		self.out_proj = nn.Linear(cfg.d_hidden, cfg.d_out)

		self.in_dropout = nn.Dropout(cfg.dropout)
		self.hidden_dropout = nn.ModuleList([nn.Dropout(cfg.dropout) for _ in range(cfg.hidden_layers)])

		if cfg.act == ActivationFn.GELU:
			self.act = F.gelu
		elif cfg.act == ActivationFn.SILU:
			self.act = F.silu
		elif cfg.act == ActivationFn.RELU:
			self.act = F.relu
		elif cfg.act == ActivationFn.SIGMOID:
			self.act = F.sigmoid
		else:
			raise ValueError(f"Invalid Activation: {cfg.act}")

		self.init_linears(zeros=cfg.zeros)

	def init_linears(self, zeros=False):

		init_xavier(self.in_proj)  # Xavier for the first layer

		for layer in self.hidden_proj:
			init_kaiming(layer)  # Kaiming for hidden layers

		if zeros:
			init_zeros(self.out_proj) 
		else:
			init_xavier(self.out_proj)  # Xavier for output layer

	def forward(self, x):
		x = self.in_dropout(self.act(self.in_proj(x)))
		for hidden, dropout in zip(self.hidden_proj, self.hidden_dropout):
			x = dropout(self.act(hidden(x)))
		x = self.out_proj(x) # no activation or dropout on output

		return x

@dataclass
class MPNNMLPCfg:
	d_model: int = 512
	hidden_layers: int = 2
	dropout: float = 0.0
	act: ActivationFn = ActivationFn.GELU
	zeros: bool = False

class MPNNMLP(MLP):
	def __init__(self, cfg: MPNNMLPCfg):
		mlp_cfg = MLPCfg(
			d_in=3*cfg.d_model, 
			d_out=cfg.d_model, 
			d_hidden=cfg.d_model,
			hidden_layers=cfg.hidden_layers,
			dropout=cfg.dropout,
			act=cfg.act,
			zeros=cfg.zeros,
		)
		super().__init__(mlp_cfg)
	
@dataclass
class FFNCfg:
	d_model: int = 512
	expansion_factor: int = 4
	dropout: float = 0.0
	act: ActivationFn = ActivationFn.GELU
	zeros: bool = False

class FFN(MLP):
	def __init__(self, cfg: FFNCfg):
		mlp_cfg = MLPCfg(
			d_in=cfg.d_model, 
			d_out=cfg.d_model, 
			d_hidden=cfg.expansion_factor*cfg.d_model,
			hidden_layers=0,
			dropout=cfg.dropout,
			act=cfg.act,
			zeros=cfg.zeros,
		)
		super().__init__(mlp_cfg)


# initializations for linear layers
def init_orthogonal(m):
	if isinstance(m, nn.Linear):
		init.orthogonal_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_kaiming(m):
	if isinstance(m, nn.Linear):
		init.kaiming_uniform_(m.weight, nonlinearity=ActivationFn.RELU)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_xavier(m):
	if isinstance(m, nn.Linear):
		init.xavier_uniform_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_zeros(m):
	if isinstance(m, nn.Linear):
		init.zeros_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)