import torch
from proteus.utils.test_utils import calculate_error


# --- reference implementation ---

def wf_embedding_torch(coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, cu_seqlens, dropout_p=0.0):
	"""packed wf embedding reference — pad, compute batched, unpad"""

	BL = coordsA.shape[0]
	B = cu_seqlens.shape[0] - 1
	max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
	num_wn = wavenumbers.size(0)
	_, num_classes = aa_magnitudes.shape
	d_model = num_wn * 2
	device = coordsA.device

	# pad packed tensors into [B, max_seqlen, ...] batched format
	coordsA_pad = torch.zeros(B, max_seqlen, 3, dtype=coordsA.dtype, device=device)
	coordsB_pad = torch.zeros(B, max_seqlen, 3, dtype=coordsB.dtype, device=device)
	aa_labels_pad = torch.zeros(B, max_seqlen, dtype=aa_labels.dtype, device=device)
	pad_mask = torch.ones(B, max_seqlen, dtype=torch.bool, device=device)  # True = padded

	for i in range(B):
		start = cu_seqlens[i].item()
		end = cu_seqlens[i + 1].item()
		seqlen = end - start
		coordsA_pad[i, :seqlen] = coordsA[start:end]
		coordsB_pad[i, :seqlen] = coordsB[start:end]
		aa_labels_pad[i, :seqlen] = aa_labels[start:end]
		pad_mask[i, :seqlen] = False

	# pairwise displacement vectors
	R = coordsA_pad[:, :, None, :] - coordsA_pad[:, None, :, :]  # B x N x N x 3

	# unit vectors and distances
	R_norm = torch.linalg.vector_norm(R, dim=3, keepdim=True)  # B x N x N x 1
	R_unit = R / torch.where(R_norm == 0, float("inf"), R_norm)  # B x N x N x 3
	distsA = R_norm  # B x N x N x 1

	# anisotropic modulation: scale beta carbons by per-AA magnitudes
	aa_onehot = torch.nn.functional.one_hot(aa_labels_pad.long(), num_classes=num_classes)  # B x N x A
	coordsB_magnitudes = (aa_onehot[:, :, None, :] * aa_magnitudes[None, None, :, :]).sum(dim=3)  # B x N x K
	coordsB_scaled = coordsB_pad[:, :, None, :] * coordsB_magnitudes[:, :, :, None]  # B x N x K x 3

	# projected beta distances along pairwise unit vectors
	distsB = torch.sum(R_unit[:, :, :, None, :] * coordsB_scaled[:, None, :, :, :], dim=4)  # B x N x N x K
	distsAB = distsA - distsB  # B x N x N x K

	# phase = distance * wavenumber
	phases = distsAB * wavenumbers[None, None, None, :]  # B x N x N x K

	# combine padding mask with self-interaction
	combined_mask = pad_mask[:, :, None, None] | pad_mask[:, None, :, None] | (distsA == 0)  # B x N x N x 1

	# 1/|R| magnitude with masked positions set to zero
	magnitudes = 1 / torch.where(combined_mask, float("inf"), distsA)  # B x N x N x 1

	# real and imaginary components
	real = magnitudes * torch.cos(phases)  # B x N x N x K
	imag = magnitudes * torch.sin(phases)  # B x N x N x K

	# superpose along source dimension
	real_superposition = real.sum(dim=2)  # B x N x K
	imag_superposition = imag.sum(dim=2)  # B x N x K

	# interleave real/imag into feature vector
	features_pad = torch.stack([real_superposition, imag_superposition], dim=-1).view(B, max_seqlen, d_model)  # B x N x d_model

	# unpad back to [BL, d_model]
	features = torch.zeros(BL, d_model, dtype=features_pad.dtype, device=device)
	for i in range(B):
		start = cu_seqlens[i].item()
		end = cu_seqlens[i + 1].item()
		seqlen = end - start
		features[start:end] = features_pad[i, :seqlen]

	return features


# --- test data factory ---

def make_test_data(seq_lens, d_model, num_classes, seed=42, learnable_aa=False):
	"""factory for test inputs. learnable_aa=True sets requires_grad on aa_magnitudes."""
	device = torch.device('cuda')
	torch.manual_seed(seed)

	B = len(seq_lens)
	BL = sum(seq_lens)
	num_wn = d_model // 2
	max_wl = 20.0

	# cu_seqlens
	cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
	for i, sl in enumerate(seq_lens):
		cu_seqlens[i + 1] = cu_seqlens[i] + sl

	# packed coords
	coordsA = max_wl * torch.randn(BL, 3, dtype=torch.float32, device=device)
	coordsB = torch.randn(BL, 3, dtype=torch.float32, device=device)
	coordsB = coordsB / torch.linalg.vector_norm(coordsB, dim=1, keepdim=True)

	# aa labels and magnitudes
	aa_labels = torch.randint(0, num_classes, (BL,), dtype=torch.int32, device=device)
	aa_magnitudes = torch.rand(num_wn, num_classes, dtype=torch.float32, device=device,
	                           requires_grad=learnable_aa)

	# wavenumbers
	wavenumbers = torch.randn(num_wn, dtype=torch.float32, device=device,
	                           requires_grad=learnable_aa)

	return coordsA, coordsB, aa_labels, aa_magnitudes, wavenumbers, cu_seqlens, 0.0


# --- shared test configs ---

CORRECTNESS_CONFIGS = {
	'degenerate':      dict(seq_lens=[1, 2, 3],       d_model=128, num_classes=20),
	'sub_warp':        dict(seq_lens=[7, 16, 24, 31], d_model=128, num_classes=20),
	'warp_boundary':   dict(seq_lens=[31, 32, 33],    d_model=128, num_classes=20),
	'block_boundary':  dict(seq_lens=[511, 512, 513], d_model=128, num_classes=20),
	'long':            dict(seq_lens=[1024, 2048],     d_model=128, num_classes=20),
	'single_class':    dict(seq_lens=[128, 256],       d_model=128, num_classes=1),
	'two_classes':     dict(seq_lens=[128, 256],       d_model=128, num_classes=2),
	'max_classes':     dict(seq_lens=[128, 256],       d_model=128, num_classes=31),
	'small_dmodel':    dict(seq_lens=[256, 256],       d_model=64,  num_classes=20),
	'large_dmodel':    dict(seq_lens=[256, 256],       d_model=256, num_classes=20),
	'many_short':      dict(seq_lens=[32]*32,          d_model=128, num_classes=20),
}

STRESS_CONFIGS = {
	'stress_8x1024': dict(seq_lens=[1024]*8,  d_model=128, num_classes=20),
	'stress_4x2048': dict(seq_lens=[2048]*4,  d_model=128, num_classes=20),
}
