
import enum
import torch
from proteus.types import T, Float, Int, Bool
from proteus.utils.struct_utils import get_CA_raw_and_CB_unit, get_backbone, compute_frames

CONSTRUCT_FUNCTION = None

class ConstructFunctionNames(enum.StrEnum):
	PROTEUS = "proteus"
	PAIRFORMER = "pairformer"

class ConstructRegisry:

	@staticmethod
	def set_construct_function(construct_function: str):
		global CONSTRUCT_FUNCTION
		ConstructRegisry._assert_not_set()
		assert construct_function in ConstructFunctionNames
		CONSTRUCT_FUNCTION = construct_function

	@property
	@staticmethod
	def needs_pair_cuseqlens():
		ConstructRegisry._assert_set()
		return CONSTRUCT_FUNCTION in [ConstructFunctionNames.PAIRFORMER]

	@staticmethod
	def construct(*args):
		ConstructRegisry._assert_set()
		return getattr(ConstructFunctions, CONSTRUCT_FUNCTION)(*args)

	@staticmethod
	def _assert_set():
		assert CONSTRUCT_FUNCTION is not None, f"never called ConstructRegistry.set_construct_function, CONSTRUCT_FUNCTION=None"

	@staticmethod
	def _assert_not_set():
		assert CONSTRUCT_FUNCTION is None, f"already called ConstructRegistry.set_construct_function, {CONSTRUCT_FUNCTION=}"


class ConstructFunctions:

	@staticmethod
	@torch.no_grad()
	def proteus(
		coords: Float[T, "L 14 3"], 
		labels: Int[T, "L"], 
		seq_idx: Int[T, "L"], 
		chain_idx: Int[T, "L"], 
		trgt_mask: Bool[T, "L"], 
		homo_mask: Bool[T, "L"],
		caa_mask: Bool[T, "L"],
		atom_mask: Bool[T, "L"]
	):

		coords_ca, coords_cb_unit = get_CA_raw_and_CB_unit(coords)
		seq_mask = homo_mask & ~trgt_mask
		loss_mask = caa_mask & trgt_mask

		return {
			"coords_ca": coords_ca,
			"coords_cb_unit": coords_cb_unit,
			"labels": labels,
			"seq_mask": seq_mask,
			"loss_mask": loss_mask,
		}


	@staticmethod
	@torch.no_grad()
	def pairformer(
		coords: Float[T, "L"], 
		labels: Int[T, "L"], 
		seq_idx: Int[T, "L"], 
		chain_idx: Int[T, "L"], 
		trgt_mask: Bool[T, "L"], 
		homo_mask: Bool[T, "L"],
		caa_mask: Bool[T, "L"],
		atom_mask: Bool[T, "L"]
	):

		loss_mask = caa_mask & trgt_mask

		# get backbone and frames
		coords_bb = get_backbone(coords)
		_, frames = compute_frames(coords_bb)

		# pairwise stuff
		coords_bb_dist = torch.sum((coords_bb[:, None, :, None, :] - coords_bb[None, :, None, :, :]).pow_(2), dim=-1).sqrt_()
		diff_chain = chain_idx[:, None] != chain_idx[None, :]
		rel_seq_idx = seq_idx[:, None] - seq_idx[None, :]
		rel_frames = torch.matmul(frames[:, None, :, :].transpose(-2,-1), frames[None, :, :, :])

		# flatten pairwise stuff
		L = coords_bb.size(0)
		coords_bb_dist = coords_bb_dist.reshape(L*L, 4, 4)
		diff_chain = diff_chain.reshape(L*L)
		rel_seq_idx = rel_seq_idx.reshape(L*L)
		rel_frames = rel_frames.reshape(L*L, 3, 3)

		# a buffer to reduce the pairs into singles
		reduction_buffer = torch.zeros_like(labels, dtype=torch.float)

		return {
			"coords_bb_dist": coords_bb_dist,
			"diff_chain": diff_chain,
			"rel_seq_idx": rel_seq_idx,
			"rel_frames": rel_frames,
			"reduction_buffer": reduction_buffer,
			"labels": labels,
			"seq_mask": seq_mask,
			"loss_mask": loss_mask,
		}
