from __future__ import annotations

from torch.utils.data import get_worker_info
from torch.nn.utils.rnn import pad_sequence
import torch

from typing import List, Dict, Tuple, Generator, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from bisect import insort, bisect_right
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import random

from proteus.static.constants import aa_2_lbl, seq_2_lbls
from proteus.types import A, T, Float, Int, Bool
from proteus.data.construct_registry import ConstructRegistry


@dataclass
class SamplerCfg:
	num_clusters: int = -1
	seed: int = 42

@dataclass
class BatchBuilderCfg:
	batch_tokens: int = 16384
	buffer_size: int = 32

@dataclass
class PDBCacheCfg:
	pdb_path: Path
	min_seq_size: int = 16
	max_seq_size: int = 16384
	homo_thresh: float = 0.70
	asymmetric_units_only: bool = False

@dataclass
class PDBDataCfg:
	pdb: str
	pdb_path: Path
	min_seq_size: int = 16
	max_seq_size: int = 16384
	homo_thresh: float = 0.70
	asymmetric_units_only: bool = False

class Sampler:
	def __init__(self, clusters_df: pd.DataFrame, cfg: SamplerCfg, epoch: int = 0) -> None:
		self._base_seed: int = cfg.seed
		self._epoch: int = epoch
		self._big_prime: int = 1_000_003

		# init the df w/ cluster info
		self._clusters_df: pd.DataFrame = clusters_df
		self._num_clusters: int = min(cfg.num_clusters if cfg.num_clusters!=-1 else float("inf"), len(clusters_df.CLUSTER.drop_duplicates()))

	def _get_rand_state(self, rng: np.random.Generator) -> int:
		return int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

	def _get_rng(self) -> np.random.Generator:
		self._epoch += 1
		return np.random.default_rng((self._base_seed + self._epoch*self._big_prime) % 2**32)

	def _partition_pdbs(self, pdb: str, num_workers: int) -> int:
		h = hashlib.blake2b(pdb.encode('utf-8'), digest_size=8, key=b'arbitrary_string_for_determinism').digest()
		return int.from_bytes(h, 'big') % num_workers

	def sample_rows(self) -> pd.DataFrame:

		# get worker info to partition the samples
		worker_info = get_worker_info()
		if worker_info is None: # single process
			wid, num_workers = 0, 1
		else: # multi process
			wid, num_workers = worker_info.id, worker_info.num_workers

		# sample rows using deterministic rng
		rng = self._get_rng()
		sampled_rows = (	self._clusters_df
							.groupby("CLUSTER")
							.sample(n=1, random_state=self._get_rand_state(rng)) # first sample gets one chain from each cluster
							.sample(frac=1, random_state=self._get_rand_state(rng)) # second is to randomly shuffle chains
							.iloc[:self._num_clusters, :] # only get num clusters
						)

		# each worker only uses its assigned pdbs, via the partition function. ensures no duplicate caches
		worker_mask = sampled_rows.CHAINID.map(lambda p: self._partition_pdbs(p.split("_")[0], num_workers) == wid)

		return sampled_rows[worker_mask]

	def __len__(self) -> int:
		return self._num_clusters

class BatchBuilder:
	def __init__(self, cfg: BatchBuilderCfg) -> None:

		# init buffer, batch, and token count
		self._buffer: List[Assembly] = []
		self._cur_batch: List[Assembly] = []
		self._cur_tokens: int = 0
		self._buffer_size: int = cfg.buffer_size
		self._batch_tokens: int = cfg.batch_tokens

	def add_sample(self, sample: Assembly) -> Generator[DataBatch, None, None]:
		
		self._add_buffer(sample)

		if self._buffer_full():
			if self._batch_full():
				data_batch = DataBatch(self._cur_batch)
				if not data_batch.is_empty:
					yield data_batch
				self._clear_batch()
			self._add_batch()

	def drain_buffer(self) -> Generator[DataBatch, None, None]:

		# all assemblies have been batched or in buffer, empty the buffer
		while self._buffer:
			if self._batch_full():
				data_batch = DataBatch(self._cur_batch)
				if not data_batch.is_empty:
					yield data_batch
				self._clear_batch()
			self._add_batch()

		# if not empty, yield the last batch
		if self._cur_batch:
			yield DataBatch(self._cur_batch)

	def _add_buffer(self, asmb: Assembly) -> None:
		insort(self._buffer, asmb, key=len)

	def _add_batch(self) -> None:
		remaining = self._batch_tokens - self._cur_tokens
		# find the largest assembly that fits in remaining capacity
		idx = bisect_right(self._buffer, remaining, key=len) - 1
		assert idx >= 0, f"_add_batch called but nothing fits: {remaining=}, smallest={len(self._buffer[0])}"
		sampled_asmb = self._buffer.pop(idx)
		self._cur_batch.append(sampled_asmb)
		self._cur_tokens += len(sampled_asmb)

	def _clear_batch(self) -> None:
		self._cur_batch.clear()
		self._cur_tokens = 0			
	
	def _buffer_full(self) -> bool:
		return len(self._buffer)>=self._buffer_size

	def _batch_full(self) -> bool:
		return (self._cur_tokens + (len(self._buffer[0]) if self._buffer else 0) > self._batch_tokens) and self._cur_tokens > 0


class DataBatch:

	@torch.no_grad()
	def __init__(self, batch_list: List[Assembly]) -> None:
		
		if not batch_list:
			raise ValueError()

		needs_pair_cuseqlens = ConstructRegistry.needs_pair_cuseqlens()

		batch_dict = defaultdict(list)
		seq_lens = []
		pair_seq_lens = []
		pair_reduce_idxs = []
		tot_tokens = 0

		for idx, asmb in enumerate(batch_list):
			for key, value in asmb.construct().items():
				tokens = value.size(0)
				if not tokens:
					break
				if key == "labels":
					seq_lens.append(tokens)
					if needs_pair_cuseqlens:
						pair_seq_lens.extend([tokens]*tokens)
						pair_reduce_idxs.extend([tot_tokens + i for i in range(tokens) for _ in range(tokens)])
					tot_tokens += tokens
				batch_dict[key].append(value)

		self.is_empty = not seq_lens
		if self.is_empty:
			return 

		seq_lens = torch.tensor(seq_lens, dtype=torch.int, device="cpu")
		pair_seq_lens = torch.tensor(pair_seq_lens, dtype=torch.int, device="cpu")

		self.max_seqlen = seq_lens.max().item()
		self.samples = seq_lens.size(0)

		self.cu_seqlens = torch.nn.functional.pad(seq_lens.cumsum(dim=0), pad=(1,0), mode="constant", value=0).int()
		self.pair_cu_seqlens = torch.nn.functional.pad(pair_seq_lens.cumsum(dim=0), pad=(1,0), mode="constant", value=0).int()
		self.pair_reduce_idxs = torch.tensor(pair_reduce_idxs, dtype=torch.long, device="cpu")
		
		assert "labels" in batch_dict
		assert "loss_mask" in batch_dict
		self._tensor_names = list(batch_dict.keys()) + ["cu_seqlens"] 
		if needs_pair_cuseqlens:
			self._tensor_names += ["pair_cu_seqlens", "pair_reduce_idxs"]

		for tensor_name, tensor_list in batch_dict.items():
			tensor = torch.cat(tensor_list, dim=0)
			setattr(self, tensor_name, tensor)
	
	@property
	def loss_tokens(self):
		return self.loss_mask.sum().item()
		
	@property
	def tokens(self):
		return self.loss_mask.size(0)
		
	def move_to(self, device: torch.device) -> None:
		if not hasattr(self, "_tensor_names"):
			raise ValueError(f"no tensors!! the data batch is probably empty: {self.is_empty=}")
		for tensor_name in self._tensor_names:
			tensor = getattr(self, tensor_name)
			if isinstance(tensor, T):
				setattr(self, tensor_name, tensor.to(device))

	def __len__(self) -> int:
		return self.tokens

class PDBCache:
	def __init__(self, cfg: PDBCacheCfg) -> None:
		self._cache: Dict[str, PDBData] = {} # {pdb: pdb_data}
		self._cfg: PDBCacheCfg = cfg

	def _add_pdb(self, pdb: str) -> None:
		self._cache[pdb] = PDBData(PDBDataCfg(
			pdb=pdb,
			pdb_path=self._cfg.pdb_path,
			min_seq_size=self._cfg.min_seq_size,
			max_seq_size=self._cfg.max_seq_size,
			homo_thresh=self._cfg.homo_thresh,
			asymmetric_units_only=self._cfg.asymmetric_units_only
		))

	def get_pdb(self, pdb: str) -> PDBData:
		if pdb not in self._cache:
			self._add_pdb(pdb)
		return self._cache[pdb]

class PDBData:
	def __init__(self, cfg: PDBDataCfg) -> None:

		# load the metadata
		self._base_path: Path = cfg.pdb_path / Path(cfg.pdb[1:3])
		self._pdb: str = cfg.pdb
		metadata: Dict[str, Any] = torch.load(self._base_path / Path(self._pdb + ".pt"), weights_only=True, map_location="cpu")

		# remove any keys not used (most is just pdb metadata), convert to np if possible
		removed_keys: set = {"method", "date", "resolution", "id", "asmb_details", "asmb_method", "asmb_ids"}
		self._metadata: Dict[str, Any] = {key: (metadata[key].numpy() if isinstance(metadata[key], T) else metadata[key]) for key in metadata.keys() if key not in removed_keys}

		# change this to a dict instead of list
		self._metadata["chains"] = {c: i for i, c in enumerate(self._metadata["chains"])}

		# other stuff
		self._chain_cache: Dict[str, Optional[Dict[str, Any]]] = {} # {chain: chain_data}
		self._min_seq_size: int = cfg.min_seq_size
		self._max_seq_size: int = cfg.max_seq_size
		self._homo_thresh: float = cfg.homo_thresh
		self._asymmetric_units_only: bool = cfg.asymmetric_units_only

	def sample_asmb(self, chain: str) -> Assembly:
		
		# sample an asmb that contains this chain
		asmb_id = random.choice(self._get_chain(chain)["asmb_ids"])

		# get the other chains in this assembly
		if asmb_id == -1: # -1 means just itself
			asmb_chains = [chain]
		else:
			asmb_chains = self._metadata["asmb_chains"][asmb_id].split(",")

		# shuffle the asmb chains to vary their order, only matters when we need to crop
		# make sure the target chain is always first though, since we would rather not crop that one
		asmb_chains = [chain] + random.sample([c for c in asmb_chains if c!=chain], k=len(asmb_chains)-1)

		# init lists
		labels = []
		coords = []
		atom_mask = []
		chain_info = [] # list of (chain_idx, size)
		trgt_chain_idx = self._metadata["chains"][chain] # get the chain idx of the target chain

		# construct tensors
		for asmb_chain in asmb_chains:

			# get the data for this chain
			asmb_chain_data = self._get_chain(asmb_chain)
			if asmb_chain_data is None:
				continue
			
			# extract tensors			
			labels.append(asmb_chain_data["seq"]) # vectorizes the conversion from str -> labels
			coords.append(asmb_chain_data["xyz"])
			atom_mask.append(asmb_chain_data["mask"])
			chain_info.append((self._metadata["chains"][asmb_chain], asmb_chain_data["mask"].shape[0]))

		# cat
		labels = np.concatenate(labels, axis=0)
		coords = np.concatenate(coords, axis=0)
		atom_mask = np.concatenate(atom_mask, axis=0)

		# mask for homo chain
		homo_chains = np.arange(len(self._metadata["chains"]))[self._metadata["tm"][trgt_chain_idx, :, 1]>=self._homo_thresh]

		# get the corresponding xform
		asmb_xform = (
			np.expand_dims(np.eye(4), 0) 
			if self._asymmetric_units_only 
			or asmb_id == -1 
			else self._metadata[f"asmb_xform{asmb_id}"]
		)

		# init the assembly, also applies the xform and takes care of cropping based on max size
		asmb = Assembly(coords, labels, atom_mask,
						chain_info, trgt_chain_idx, homo_chains,
						asmb_xform, self._max_seq_size
					)

		return asmb

	def _get_chain(self, chain: str) -> Optional[Dict[str, Any]]:

		# add chain to cache if not in there
		if chain not in self._chain_cache:
			self._add_chain(chain)

		# get the data
		return self._chain_cache[chain]

	def _add_chain(self, chain: str) -> None:

		# load the chain data
		chain_path = self._base_path / Path(self._pdb + f"_{chain}.pt")
		if not chain_path.exists():
			self._chain_cache[chain] = None
			return
		chain_data = torch.load(chain_path, weights_only=True, map_location="cpu")

		# remove unnecessary keys
		used_keys = {"seq", "xyz", "mask"}
		chain_data = {key: chain_data[key].numpy() if isinstance(chain_data[key], T) else chain_data[key] for key in chain_data.keys() if key in used_keys}
		chain_data["seq"] = seq_2_lbls(chain_data["seq"])# convert to labels
		chain_data["xyz"][np.isnan(chain_data["xyz"])] = 0.0 # replace nans with 0

		# crop to max seq size
		chain_data["seq"] = chain_data["seq"][:self._max_seq_size] 
		chain_data["xyz"] = chain_data["xyz"][:self._max_seq_size, :, :] 
		chain_data["mask"] = chain_data["mask"][:self._max_seq_size, :] 

		# keep a list of the biounits this chain is a part of
		chain_data["asmb_ids"] = []
		for asmb_id, asmb in enumerate(self._metadata["asmb_chains"]):
			if chain in asmb.split(","):
				chain_data["asmb_ids"].append(asmb_id)
		
		if not chain_data["asmb_ids"]:
			chain_data["asmb_ids"] = [-1] # signifies the only chain is itself

		# add to the cache
		self._chain_cache[chain] = chain_data

class Assembly:
	def __init__(self, 	coords: Float[A, "L 14 3"], labels: Int[A, "L"], atom_mask: Bool[A, "L 14"],
						chain_info: List[Tuple[int, int]], trgt_chain: int, homo_chains: Int[A, "H"],
						asmb_xform: Float[A, "N 4 4"], max_seq_size: int
					) -> None:

		self.coords = coords
		self.labels = labels
		self.atom_mask = atom_mask

		self._chain_info = chain_info
		self._trgt_chain = trgt_chain
		self._homo_chains = homo_chains

		self.asmb_xform = asmb_xform  

		self._crop(max_seq_size)

	@torch.no_grad()
	def construct(self) -> None:

		coords = torch.from_numpy(self.coords).float()
		labels = torch.from_numpy(self.labels).long()

		# compute seq idxs and chain idxs, note that need to crop in case labels was cropped earlier
		seq_idx = torch.cat([torch.arange(size) for _, size in self._chain_info], dim=0)[:labels.size(0)].long()
		chain_idx = torch.cat([torch.full((size,), idx) for idx, size in self._chain_info], dim=0)[:labels.size(0)].long()

		# make the masks to remove invalid (no coords)
		atom_mask = torch.from_numpy(self.atom_mask).bool()
		valid_mask = atom_mask[..., :3].all(dim=-1)  # L

		# clean up invalid,
		coords = coords[valid_mask]
		labels = labels[valid_mask]
		seq_idx = seq_idx[valid_mask]
		chain_idx = chain_idx[valid_mask]

		# perform the xform on coords and adjust the other tensors accordingly
		asmb_xform = torch.from_numpy(self.asmb_xform).float()

		# check how many copies you can make based on max size param. 
		N, A, S = coords.shape
		num_copies = asmb_xform.size(0) # this means we prefer making less copies over cropping chains

		R = asmb_xform[:, :3, :3] # num_copies x 3 x 3
		T = asmb_xform[:, :3, 3] # num_copies x 3

		# adjust sizes based on the number of copies made
		coords = (torch.einsum("bij,raj->brai", R, coords) + T.view(num_copies, 1,1,3)).reshape(N*num_copies, A, S)
		labels = labels.repeat(num_copies)
		seq_idx = seq_idx.repeat(num_copies)
		chain_idx = chain_idx.repeat(num_copies)

		trgt_mask = chain_idx == self._trgt_chain
		homo_mask = torch.isin(chain_idx, torch.from_numpy(self._homo_chains))
		caa_mask = labels != aa_2_lbl("X")  # L

		return ConstructRegistry.construct(
			coords, labels, seq_idx, chain_idx, trgt_mask, homo_mask, caa_mask, atom_mask
		)

		
	@torch.no_grad()
	def _crop(self, max_seq_size: int) -> None:

		# check how many copies you can make based on max size param. 
		N, A, S = self.coords.shape
		num_copies = min(max_seq_size//N, self.asmb_xform.shape[0]) # this means we prefer making less copies over cropping chains

		if num_copies == 0: # this means N>max_size, so need to crop N
			self.coords = self.coords[:max_seq_size, :, :]
			self.labels = self.labels[:max_seq_size]
			self.atom_mask = self.atom_mask[:max_seq_size, :]
			self.asmb_xform = np.expand_dims(np.eye(4), 0)
		else:
			self.asmb_xform = self.asmb_xform[:num_copies, :, :]

	def __len__(self) -> int:
		return self.labels.shape[0]*self.asmb_xform.shape[0]

