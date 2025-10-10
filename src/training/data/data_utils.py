from __future__ import annotations

from torch.utils.data import get_worker_info
from torch.nn.utils.rnn import pad_sequence
import torch

from typing import List, Dict, Tuple, Generator
from bisect import insort
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib
import random

from static.constants import aa_2_lbl, seq_2_lbls

class Sampler:
	def __init__(self, clusters_df: pd.DataFrame, num_clusters: int, seed: int, epoch: int) -> None:
		self._base_seed = seed
		self._epoch = epoch
		self._big_prime = 1_000_003

		# init the df w/ cluster info
		self._clusters_df = clusters_df
		self._num_clusters = min(num_clusters if num_clusters!=-1 else float("inf"), len(clusters_df.CLUSTER.drop_duplicates()))

	def _get_rand_state(self, rng: np.random.Generator) -> int:
		return int(rng.integers(0, 2**32 - 1, dtype=np.uint32))

	def _get_rng(self) -> np.random.Generator:
		return np.default_rng((self._base_seed + self._epoch.value*self._big_prime) % 2**32)

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
	def __init__(self, batch_tokens: int, buffer_size: int=32) -> None:

		# init buffer, batch, and token count
		self._buffer = []
		self._cur_batch = []
		self._cur_tokens = 0
		self._buffer_size = buffer_size
		self._batch_tokens = batch_tokens

	def add_sample(self, sample: Assembly) -> Generator[DataBatch]:
		
		self._add_buffer(sample)

		if self._buffer_full():
			if self._batch_full():
				yield DataBatch(self._cur_batch)
				self._clear_batch()
			self._add_batch()

	def drain_buffer(self) -> Generator[DataBatch]:

		# all assemblies have been batched or in buffer, empty the buffer
		while self._buffer:
			if self._batch_full():
				yield DataBatch(self._cur_batch)
				self._clear_batch()
			self._add_batch()

		# if not empty, yield the last batch
		if self._cur_batch:
			yield DataBatch(self._cur_batch)

	def _add_buffer(self, asmb: Assembly) -> None:
		insort(self._buffer, asmb, key=len)

	def _add_batch(self) -> None:
		sampled_asmb = self._buffer.pop()
		self._cur_batch.append(sampled_asmb)
		self._cur_tokens += len(sampled_asmb)

	def _clear_batch(self) -> None:
		self._cur_batch.clear()
		self._cur_tokens = 0			
	
	def _buffer_full(self) -> bool:
		return len(self._buffer)>=self._buffer_size

	def _batch_full(self) -> bool:
		return self._cur_tokens+(len(self._buffer[-1]) if self._buffer else 0)>=self._batch_tokens


class DataBatch:

	@torch.no_grad()
	def __init__(self, batch_list: List[Assembly]) -> None:
		coords = []
		labels = []

		seq_pos = []
		chain_pos = []
		sample_idx = []

		atom_mask = []
		trgt_mask = []
		homo_mask = []

		for idx, asmb in batch_list:
			
			asmb.construct() # materialize the full tensors (including AU copies)

			coords.append(asmb.coords)
			labels.append(asmb.labels)

			seq_pos.append(asmb.seq_pos)
			chain_pos.append(asmb.chain_pos)
			sample_idx.append(torch.full(asmb.labels.shape, idx))

			atom_mask.append(asmb.atom_mask)
			trgt_mask.append(asmb.trgt_mask)
			homo_mask.append(asmb.homo_mask)

		# no padding, just keep track of sample idxs
		self.coords = torch.cat(coords, dim=0) # ZN x 14 x 3
		self.labels = torch.cat(labels, dim=0) # ZN

		self.seq_pos = torch.cat(seq_pos, dim=0)# ZN
		self.chain_pos = torch.cat(chain_pos, dim=0)# ZN
		self.sample_idx = torch.cat(sample_idx, dim=0) # ZN
		
		self.atom_mask = torch.cat(atom_mask, dim=0)# ZN x 14
		self.trgt_mask = torch.cat(trgt_mask, dim=0)# ZN
		self.homo_mask = torch.cat(homo_mask, dim=0)# ZN
		
		# other useful masks
		self.coords_mask = self.atom_mask[:, :3].all(dim=-1) # means not missing any bb coords, ZN
		self.caa_mask = self.labels!=aa_2_lbl("X") # non canonical amino acids, ZN

class PDBCache:
	def __init__(self, 	pdb_path: Path, 
						min_seq_size: int=16, max_seq_size: int=16384, 
						homo_thresh: float=0.70, asymmetric_units_only: bool=False
				) -> None:
		self._cache = {} # {pdb: pdb_data}
		self._pdb_path = pdb_path
		self._min_seq_size = min_seq_size
		self._max_seq_size = max_seq_size
		self._homo_thresh = homo_thresh
		self._asymmetric_units_only = asymmetric_units_only

	def _add_pdb(self, pdb: str) -> None:
		self._cache[pdb] = PDBData(	pdb, self._pdb_path, 
									min_seq_size=self._min_seq_size, max_seq_size=self._max_seq_size, 
									homo_thresh=self._homo_thresh, asymmetric_units_only=self._asymmetric_units_only
								)

	def get_pdb(self, pdb: str) -> PDBData:
		if pdb not in self._cache:
			self._add_pdb(pdb)
		return self._cache[pdb]

class PDBData:
	def __init__(self, 	pdb: str, pdb_path: Path, 
						min_seq_size: int=16, max_seq_size: int=16384, 
						homo_thresh: float=0.70, asymmetric_units_only: bool=False
					) -> None:

		# load the metadata
		self._base_path = pdb_path / Path(pdb[1:3])
		self._pdb = pdb
		metadata = torch.load(self._base_path / Path(self._pdb + ".pt"), weights_only=True, map_location="cpu")

		# remove any keys not used (most is just pdb metadata), convert to np if possible
		removed_keys = {"method", "date", "resolution", "id", "asmb_details", "asmb_method", "asmb_ids"}
		self._metadata = {key: (metadata[key].numpy() if isinstance(metadata[key], torch.Tensor) else metadata[key]) for key in metadata.keys() if key not in removed_keys}
		
		# change this to a dict instead of list
		self._metadata["chains"] = {c: i for i, c in enumerate(self._metadata["chains"])}

		# other stuff
		self._chain_cache = {} # {chain: chain_data}
		self._min_seq_size = min_seq_size
		self._max_seq_size = max_seq_size
		self._homo_thresh = homo_thresh
		self._asymmetric_units_only = asymmetric_units_only

	def sample_asmb(self, chain: str) -> Assembly:
		
		# sample an asmb that contains this chain
		asmb_id = random.choice(self._get_chain(chain)["asmb_ids"])

		# get the other chains in this assembly
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
		asmb_xform = np.expand_dims(np.eye(4), 0) if self._asymmetric_units_only else self._metadata[f"asmb_xform{asmb_id}"]

		# init the assembly, also applies the xform and takes care of cropping based on max size
		asmb = Assembly(coords, labels, atom_mask,
						chain_info, trgt_chain_idx, homo_chains,
						asmb_xform, self._max_seq_size
					)

		return asmb

	def _get_chain(self, chain: str) -> Dict[str, np.ndarray | List[int]]:

		# add chain to cache if not in there
		if chain not in self._chain_cache:
			self._add_chain(chain)

		# get the data
		return self._chain_cache[chain]

	def _add_chain(self, chain: str) -> None:

		# load the chain data
		chain_path = self._base_path / Path(self._pdb + f"_{chain}.pt")
		chain_data = torch.load(chain_path, weights_only=True, map_location="cpu")

		# remove unnecessary keys
		used_keys = {"seq", "xyz", "mask"}
		chain_data = {key: chain_data[key].numpy() if isinstance(chain_data[key], torch.Tensor) else chain_data[key] for key in chain_data.keys() if key in used_keys}
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

		# add to the cache
		self._chain_cache[chain] = chain_data

class Assembly:
	def __init__(self, 	coords: np.ndarray, labels: np.ndarray, atom_mask: np.ndarray,
						chain_info: List[Tuple[int, int]], trgt_chain: int, homo_chains: np.ndarray,
						asmb_xform: np.ndarray, max_seq_size: int
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

		self.coords = torch.from_numpy(self.coords) 
		self.labels = torch.from_numpy(self.labels) 
		self.atom_mask = torch.from_numpy(self.atom_mask) 
		self.asmb_xform = torch.from_numpy(self.asmb_xform) 

		# compute seq idxs and chain idxs, note that need to crop in case labels was cropped earlier
		self.seq_pos = torch.cat([torch.arange(size) for _, size in self._chain_info], dim=0)[:self.labels.size(0)]
		self.chain_pos = torch.cat([torch.full((size,), idx) for idx, size in self._chain_info], dim=0)[:self.labels.size(0)]

		# make the mask for trgt chain and for homomers
		self.trgt_mask = self.chain_pos == self._trgt_chain
		self.homo_mask = torch.isin(self.chain_pos, self._homo_chains)

		# perform the xform
		self._xform()

	@torch.no_grad()
	def _xform(self) -> None:

		# perform the xform on coords and adjust the other tensors accordingly

		# check how many copies you can make based on max size param. 
		N, A, S = self.coords.shape
		num_copies = self.asmb_xform.size(0) # this means we prefer making less copies over cropping chains

		R = self.asmb_xform[:, :3, :3] # num_copies x 3 x 3
		T = self.asmb_xform[:, :3, 3] # num_copies x 3

		# adjust sizes based on the number of copies made
		self.coords = (torch.einsum("bij,raj->brai", R, self.coords) + T.view(num_copies, 1,1,3)).reshape(N*num_copies, A, S)
		self.labels = self.labels.repeat(num_copies)
		self.seq_pos = self.seq_pos.repeat(num_copies)
		self.chain_pos = self.chain_pos.repeat(num_copies)
		self.atom_mask = self.atom_mask.repeat([num_copies, 1])
		self.trgt_mask = self.trgt_mask.repeat(num_copies)
		self.homo_mask = self.homo_mask.repeat(num_copies)
		
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