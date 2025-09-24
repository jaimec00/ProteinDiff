

from __future__ import annotations

from torch.utils.data import IterableDataset, DataLoader
import torch

from typing import Generator
from multiprocessing import Value as shared_val
from pathlib import Path
import pandas as pd

from train.data.data_utils import Assembly, PDBCache, BatchBuilder, DataBatch, Sampler

class DataHolder:

	'''
	hold DataLoader Objects, one each for train, test and val
	multi-chain (experimental structures): https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz 
	'''

	def __init__(self, 	data_path: str, 
						num_train: int=-1, num_val: int=-1, num_test: int=-1, 
						batch_tokens: int=16384, min_seq_size: int=16, max_seq_size: int=16384,
						max_resolution: float=3.5, homo_thresh: float=0.70, asymmetric_units_only: bool=False,
						num_workers: int=8, prefetch_factor: int=2, rng_seed: int=42, buffer_size: int=32
					) -> None:

		# define data path and path to pdbs
		data_path = Path(data_path)
		pdb_path = data_path / Path("pdb")
		train_info, val_info, test_info = self._get_splits(data_path)

		# to increment the epoch
		self._epoch = shared_val("I", 0) # uint32

		def init_loader(df: pd.DataFrame, samples: int=-1) -> DataLoader:
			'''helper to init a different loader for train val and test'''

			data = Data(pdb_path, df, 
						num_clusters=samples, batch_tokens=batch_tokens, min_seq_size=min_seq_size, max_seq_size=max_seq_size, 
						homo_thresh=homo_thresh, asymmetric_units_only=asymmetric_units_only, buffer_size=buffer_size, seed=rng_seed, epoch=self._epoch
						)

			loader = DataLoader(data, batch_size=None, num_workers=num_workers, collate_fn=lambda x: x[0],
								prefetch_factor=prefetch_factor if num_workers else None, persistent_workers=num_workers>0										
								)

			return loader

		# initialize the loaders
		self.train = init_loader(train_info, num_train)
		self.val = init_loader(val_info, num_val)
		self.test = init_loader(test_info, num_test)

	def _get_splits(self, data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

		# get the csv info, filter out anything above max res
		clusters = pd.read_csv(data_path / Path("list.csv"), header=0, usecols=["CHAINID", "RESOLUTION", "CLUSTER"], engine="pyarrow") 
		clusters = clusters.loc[clusters.RESOLUTION <= max_resolution, :]
		clusters.pop("RESOLUTION") # only need this to filter by res

		# get the val and test clusters
		with open(data_path / Path("valid_clusters.txt"), "r") as v:
			val_clusters = [i for i in v.read().split("\n") if i]
		with open(data_path / Path("test_clusters.txt"), "r") as t:
			test_clusters = [i for i in t.read().split("\n") if i]

		# split the csv accordingly
		train_info = clusters.loc[~clusters.CLUSTER.isin(val_clusters+test_clusters), :]
		val_info = clusters.loc[clusters.CLUSTER.isin(val_clusters), :]
		test_info = clusters.loc[clusters.CLUSTER.isin(test_clusters), :]

		return train_info, val_info, test_info

	def increment_epoch(self) -> None:
		with self._epoch.get_lock():
			self._epoch.value += 1

class Data(IterableDataset):
	def __init__(self, 	data_path: Path, clusters_df: pd.DataFrame, 
						num_clusters: int=-1, batch_tokens: int=16384,
						min_seq_size: int=16, max_seq_size: int=16384, 
						homo_thresh: float=0.70, asymmetric_units_only: bool=False, 
						buffer_size: int=32, seed: int=42, epoch=None
					) -> None:

		super().__init__()

		# keep a cache of pdbs
		self._pdb_cache = PDBCache(	data_path, 
									min_seq_size=min_seq_size, max_seq_size=max_seq_size, 
									homo_thresh=homo_thresh, asymmetric_units_only=asymmetric_units_only
									)

		# for deterministic and UNIQUE sampling
		self._sampler = Sampler(clusters_df, num_clusters, seed, epoch)

		# define sizes
		self._batch_tokens = batch_tokens
		self._buffer_size = buffer_size

	def _get_asmb(self, row: pd.Series) -> Assembly:

		# get pdb and chain name
		pdb, chain = row.CHAINID.split("_")

		# get the data corresponding to this pdb
		pdb_data = self._pdb_cache.get_pdb(pdb)

		# sample an assembly containing this chain
		asmb = pdb_data.sample_asmb(chain)

		return asmb

	def __iter__(self) -> Generator[DataBatch]:

		# sample rows from the df
		sampled_rows = self._sampler.sample_rows()

		# init the batch builder
		batch_builder = BatchBuilder(self._batch_tokens, self._buffer_size)

		# iterate through the sampled chains
		for _, row in sampled_rows.iterrows():

			# add the sample, only yields if batch is ready
			yield from batch_builder.add_sample(self._get_asmb(row))

		# drain the buffer and yield last batches
		yield from batch_builder.drain_buffer()

	def __len__(self) -> int:
		return len(self._sampler)