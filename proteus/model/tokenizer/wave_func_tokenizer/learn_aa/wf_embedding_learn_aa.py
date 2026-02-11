import torch
import numpy as np
from . import _C as wf_embedding_learn_aa_kernel

def wf_embedding_learn_aa(coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, cu_seqlens, dropout_p=0.0):
	return _wf_embedding_learn_aa.apply(coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, cu_seqlens, dropout_p)

class _wf_embedding_learn_aa(torch.autograd.Function):

	@staticmethod
	def forward(ctx, coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, cu_seqlens, dropout_p):

		# get tensor sizes
		BL = coordsA.shape[0]
		num_wn, num_aa = aa_magnitudes.shape
		d_model = 2 * num_wn

		assert num_aa < 32, "num_aa should be less than WarpSize (32)"

		# gather dtypes for recasting output
		o_dtype = coordsA.dtype
		d_aa_dtype = aa_magnitudes.dtype

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

		# instantiate output tensors
		out = torch.zeros(BL, d_model, dtype=torch.float32, device=coordsA.device).contiguous()
		d_aa = torch.zeros(BL, d_model, num_aa, dtype=torch.float32, device=aa_magnitudes.device).contiguous()

		rng = np.random.default_rng()
		rng_seed = rng.integers(0, (2**32) - 1, dtype=np.uint32)

		# call the kernel
		wf_embedding_learn_aa_kernel.forward(
			coordsA, coordsB,
			aa_labels, aa_magnitudes,
			wavenumbers, cu_seqlens,
			out, d_aa,
			dropout_p, rng_seed
		)

		# save for backward, recast to input dtype
		ctx.save_for_backward(d_aa.to(d_aa_dtype))

		return out.to(o_dtype)

	@staticmethod
	def backward(ctx, dO):

		# load saved tensors from bwd
		d_aa, = ctx.saved_tensors

		# mult w/ dO and sum BL dim; BL x D x 1 * BL x D x A --> D x A
		d_aa = (dO.unsqueeze(2) * d_aa).sum(dim=0)
		d_aa = d_aa[::2, :] + d_aa[1::2, :]  # sum real and imag parts; K x A

		return None, None, None, d_aa, None, None, None
