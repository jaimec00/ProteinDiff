
from dataclasses import dataclass, field
import torch.nn.init as init
from enum import StrEnum
from omegaconf import II

import torch.nn.functional as F
import torch
import torch.nn as nn

from proteus.model.base import Base
from proteus.types import T, Callable
from proteus.static.constants import canonical_aas


class ActivationFn(StrEnum):
	GELU = "gelu"
	SILU = "silu"
	RELU = "relu"
	SIGMOID = "sigmoid"

@dataclass
class MLPCfg:
	d_in: int = II("model.d_model")
	d_out: int = II(".d_in")
	d_hidden: int = II(".d_out")
	hidden_layers: int = 0
	dropout: float = 0.0
	act: ActivationFn = ActivationFn.GELU
	zeros: bool = False

class MLP(Base):
	'''
	base mlp class for use by other modules. uses gelu
	'''

	def __init__(self, cfg: MLPCfg) -> None:
		super().__init__()

		self.in_proj: nn.Linear = nn.Linear(cfg.d_in, cfg.d_hidden)
		self.hidden_proj: nn.ModuleList = nn.ModuleList([nn.Linear(cfg.d_hidden, cfg.d_hidden) for _ in range(cfg.hidden_layers)])
		self.out_proj: nn.Linear = nn.Linear(cfg.d_hidden, cfg.d_out)

		self.in_dropout: nn.Dropout = nn.Dropout(cfg.dropout)
		self.hidden_dropout: nn.ModuleList = nn.ModuleList([nn.Dropout(cfg.dropout) for _ in range(cfg.hidden_layers)])
		self.act: Callable

		cfg.act = cfg.act.lower()

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

	def init_linears(self, zeros: bool = False) -> None:

		init_xavier(self.in_proj)  # Xavier for the first layer

		for layer in self.hidden_proj:
			init_kaiming(layer)  # Kaiming for hidden layers

		if zeros:
			init_zeros(self.out_proj) 
		else:
			init_xavier(self.out_proj)  # Xavier for output layer

	def forward(self, x: T) -> T:
		x = self.in_dropout(self.act(self.in_proj(x)))
		for hidden, dropout in zip(self.hidden_proj, self.hidden_dropout):
			x = dropout(self.act(hidden(x)))
		x = self.out_proj(x) # no activation or dropout on output

		return x
	
@dataclass
class FFNCfg:
	d_model: int = II("model.d_model")
	expansion_factor: int = 4
	dropout: float = 0.0
	act: ActivationFn = ActivationFn.GELU
	zeros: bool = True

class FFN(MLP):
	def __init__(self, cfg: FFNCfg) -> None:
		mlp_cfg: MLPCfg = MLPCfg(
			d_in=cfg.d_model,
			d_out=cfg.d_model,
			d_hidden=cfg.expansion_factor*cfg.d_model,
			hidden_layers=0,
			dropout=cfg.dropout,
			act=cfg.act,
			zeros=cfg.zeros,
		)
		super().__init__(mlp_cfg)

# no defaults because this is a base class for the below
@dataclass
class ProjectionHeadCfg:
    d_in: int
    d_out: int

class ProjectionHead(MLP):
    def __init__(self, cfg: ProjectionHeadCfg) -> None:
        mlp_cfg: MLPCfg = MLPCfg(
            d_in = cfg.d_in,
            d_out = cfg.d_out,
            d_hidden = max(cfg.d_in, cfg.d_out),
            hidden_layers=0,
            dropout = 0.0,
            act = ActivationFn.GELU,
        )
        super().__init__(mlp_cfg)

@dataclass
class SeqProjectionHeadCfg:
    d_model: int = II("model.d_model")

class SeqProjectionHead(ProjectionHead):
    def __init__(self, cfg: SeqProjectionHeadCfg) -> None:
        projection_cfg: ProjectionHeadCfg = ProjectionHeadCfg(
            d_in=cfg.d_model,
            d_out=len(canonical_aas),
        )
        super().__init__(projection_cfg)
		
@dataclass
class EdgeMLPCfg:
    d_model: int = II("model.d_pair")

class EdgeMLP(MLP):
    def __init__(self, cfg: EdgeMLPCfg) -> None:
        mlp_cfg: MLPCfg = MLPCfg(
            d_in=cfg.d_model,
            d_out=cfg.d_model,
			d_hidden=cfg.d_model,
			hidden_layers=2,
        )
        super().__init__(mlp_cfg)

# initializations for linear layers
def init_orthogonal(m: nn.Module) -> None:
	if isinstance(m, nn.Linear):
		init.orthogonal_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_kaiming(m: nn.Module) -> None:
	if isinstance(m, nn.Linear):
		init.kaiming_uniform_(m.weight, nonlinearity=ActivationFn.RELU)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_xavier(m: nn.Module) -> None:
	if isinstance(m, nn.Linear):
		init.xavier_uniform_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)
def init_zeros(m: nn.Module) -> None:
	if isinstance(m, nn.Linear):
		init.zeros_(m.weight)
		if m.bias is not None:
			init.zeros_(m.bias)