# ----------------------------------------------------------------------------------------------------------------------
'''
author: 		jaime cardenas
title:  		io_utils.py
description:	utility classes for input/output operations during training 
'''
# ----------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
import logging
import torch
import math
import sys
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings
import shutil 
from dataclasses import dataclass
from hydra.core.hydra_config import HydraConfig

# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class LoggerCfg:
	out_path: str = ""
	experiment_name: str = "debug"
	overwrite: bool = False

class Logger():

	def __init__(self, cfg: LoggerCfg):

		self.out_path = Path(HydraConfig.get().runtime.output_dir)
		self.plot_path = self.out_path / "plots"
		self.log = logging.getLogger(__name__)

	def log_trainingrun(self):
		'''basically just prints the config file w/ a little more info'''

		# TODO: implement this once the config format is locked down
		log = 	textwrap.dedent(f'''
			nothing rn
		''')

		self.log.info(log)

	def log_epoch(self, epoch, step, current_lr):

		self.log.info(textwrap.dedent(f'''
		
			{'-'*80}
			Epoch {epoch}, Step {step:,}: 
			{'-'*80}
			
			Current Learning Rate: {current_lr}
		''')
		)

	def log_losses(self, losses_dict):

		for loss_type, loss in losses_dict.items():
			self.log.info(f"{mode} {loss_type} per token: {str(loss)}")	
		self.log.info("")
	

	def plot_training(self, losses):

		# convert to numpy arrays
		losses.to_numpy()

		# make the output directory
		self.plot_path.mkdir(exist_ok=True)

		# specify number of epochs
		epochs = np.arange(len(losses.train))

		# extract the keys and iterate
		loss_types = losses.val.losses.keys()
		for loss_type in loss_types:
			plt.plot(epochs, losses.train.losses[loss_type], marker='o', color='red', label="Training")
			plt.plot(epochs, losses.val.losses[loss_type], marker='o', color='blue', label="Validation")
			plt.title(f'{loss_type} vs. Epochs')
			plt.xlabel('Epochs')
			plt.ylabel(loss_type)
			plt.legend()
			plt.grid(True)
			loss_path = self.plot_path / Path(f"{'_'.join(loss_type.lower().split(' '))}.png")
			plt.savefig(loss_path)
			self.log.info(f"Plot of {loss_type} vs. Epochs saved to {loss_path}")
			plt.figure()

# ----------------------------------------------------------------------------------------------------------------------