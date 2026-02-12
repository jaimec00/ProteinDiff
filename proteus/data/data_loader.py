

from __future__ import annotations

from torch.utils.data import IterableDataset, DataLoader
import torch

from typing import Generator, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from proteus.data.data_utils import Assembly, PDBCache, BatchBuilder, DataBatch, Sampler, PDBCacheCfg, SamplerCfg, BatchBuilderCfg

@dataclass
class DataHolderCfg:
	data_path: Path
	num_train: int = -1
	num_val: int = -1
	num_test: int = -1
	batch_tokens: int = 16384
	min_seq_size: int = 16
	max_seq_size: int = 16384
	max_resolution: float = 3.5
	homo_thresh: float = 0.70
	asymmetric_units_only: bool = False
	num_workers: int = 8
	prefetch_factor: int = 2
	rng_seed: int = 42
	buffer_size: int = 32

@dataclass
class DataCfg:
	data_path: Path
	clusters_df: pd.DataFrame
	num_clusters: int = -1
	batch_tokens: int = 16384
	min_seq_size: int = 16
	max_seq_size: int = 16384
	homo_thresh: float = 0.70
	asymmetric_units_only: bool = False
	buffer_size: int = 32
	seed: int = 42
	epoch: Any = None


class DataHolder:

	'''
	hold DataLoader Objects, one each for train, test and val
	multi-chain (experimental structures): https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz
	'''

	def __init__(self, config: DataHolderCfg) -> None:

		# define data path and path to pdbs
		data_path = Path(config.data_path)
		pdb_path = data_path / "pdb"
		train_info, val_info, test_info = self._get_splits(data_path, config.max_resolution)

		def init_loader(df: pd.DataFrame, samples: int=-1) -> DataLoader:
			'''helper to init a different loader for train val and test'''

			data = Data(DataCfg(
				data_path=pdb_path,
				clusters_df=df,
				num_clusters=samples,
				batch_tokens=config.batch_tokens,
				min_seq_size=config.min_seq_size,
				max_seq_size=config.max_seq_size,
				homo_thresh=config.homo_thresh,
				asymmetric_units_only=config.asymmetric_units_only,
				buffer_size=config.buffer_size,
				seed=config.rng_seed,
			))

			loader = DataLoader(
				data, 
				batch_size=None, 
				num_workers=config.num_workers, 
				collate_fn=lambda x: x,
				prefetch_factor=config.prefetch_factor if config.num_workers else None, 
				persistent_workers=config.num_workers>0
			)

			return loader

		# initialize the loaders
		self.train = init_loader(train_info, config.num_train)
		self.val = init_loader(val_info, config.num_val)
		self.test = init_loader(test_info, config.num_test)

	def _get_splits(self, data_path: Path, max_resolution: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

		# get the csv info, filter out anything above max res
		clusters = pd.read_csv(data_path / Path("list.csv"), header=0, usecols=["CHAINID", "RESOLUTION", "CLUSTER"], engine="python")
		clusters = clusters.loc[clusters.RESOLUTION <= max_resolution, :]
		clusters.pop("RESOLUTION") # only need this to filter by res

		# get the val and test clusters
		with open(data_path / Path("valid_clusters.txt"), "r") as v:
			val_clusters = [int(i) for i in v.read().split("\n") if i]
		with open(data_path / Path("test_clusters.txt"), "r") as t:
			test_clusters = [int(i) for i in t.read().split("\n") if i]

		# split the csv accordingly
		train_info = clusters.loc[~clusters.CLUSTER.isin(val_clusters+test_clusters), :]
		val_info = clusters.loc[clusters.CLUSTER.isin(val_clusters), :]
		test_info = clusters.loc[clusters.CLUSTER.isin(test_clusters), :]

		return train_info, val_info, test_info

class Data(IterableDataset):
	def __init__(self, config: DataCfg) -> None:

		super().__init__()

		# keep a cache of pdbs
		self._pdb_cache = PDBCache(PDBCacheCfg(
			pdb_path=config.data_path,
			min_seq_size=config.min_seq_size,
			max_seq_size=config.max_seq_size,
			homo_thresh=config.homo_thresh,
			asymmetric_units_only=config.asymmetric_units_only
		))

		# for deterministic and UNIQUE sampling
		self._sampler = Sampler(config.clusters_df, SamplerCfg(
			num_clusters=config.num_clusters,
			seed=config.seed
		))

		# for batch building
		self._batch_builder_config = BatchBuilderCfg(
			batch_tokens=config.batch_tokens,
			buffer_size=config.buffer_size
		)

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
		batch_builder = BatchBuilder(self._batch_builder_config)

		# iterate through the sampled chains
		for _, row in sampled_rows.iterrows():

			# add the sample, only yields if batch is ready
			sample = self._get_asmb(row)
			if sample:
				yield from batch_builder.add_sample(sample)

		# drain the buffer and yield last batches
		yield from batch_builder.drain_buffer()


	def __len__(self) -> int:
		return len(self._sampler)