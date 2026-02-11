import torch
import numpy as np
from . import _C as wf_embedding_static_aa_kernel

def wf_embedding_static_aa(coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, cu_seqlens, dropout_p=0.0):
	return _wf_embedding_static_aa.apply(coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, cu_seqlens, dropout_p)

class _wf_embedding_static_aa(torch.autograd.Function):

	@staticmethod
	def forward(ctx, coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, cu_seqlens, dropout_p):

		# get tensor sizes
		BL = coordsA.shape[0]
		num_wn, num_aa = aa_magnitudes.shape
		d_model = 2 * num_wn

		assert num_aa < 32, "num_aa should be less than WarpSize (32)"

		# gather dtype for recasting output
		o_dtype = coordsA.dtype

		# convert dtypes and make contiguous, transpose to [3, BL] for coalesced access
		coordsA = coordsA.T.to(torch.float32).contiguous()
		coordsB = coordsB.T.to(torch.float32).contiguous()

		# wavenumbers
		wavenumbers = wavenumbers.to(torch.float32).contiguous()

		# aas
		aa_labels = aa_labels.to(torch.int16).contiguous()
		aa_magnitudes = aa_magnitudes.to(torch.float16).contiguous()

		# cu_seqlens
		cu_seqlens = cu_seqlens.to(torch.int32).contiguous()

		# instantiate output tensor (no d_aa needed — aa_magnitudes are static)
		out = torch.zeros(BL, d_model, dtype=torch.float32, device=coordsA.device).contiguous()

		rng = np.random.default_rng()
		rng_seed = rng.integers(0, (2**32) - 1, dtype=np.uint32)

		# call the kernel
		wf_embedding_static_aa_kernel.forward(
			coordsA, coordsB,
			aa_labels, aa_magnitudes,
			wavenumbers, cu_seqlens,
			out,
			dropout_p, rng_seed
		)

		return out.to(o_dtype)

	@staticmethod
	def backward(ctx, dO):
		return None, None, None, None, None, None, None
