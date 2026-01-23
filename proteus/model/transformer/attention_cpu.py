import torch
import torch.nn.functional as F

from proteus.types import Float, Int, T

def torch_attn_varlen_qkvpacked_func(
	qkv: Float[T, "ZN 3 H Dk"],
	cu_seqlens: Int[T, "Z+1"],
	max_seqlen: int,
	dropout_p: float = 0.0,
	softmax_scale: float | None = None,
	deterministic: bool = False
) -> Float[T, "ZN H Dk"]:
	"""
	Pure PyTorch implementation of variable-length attention with QKV packing (vectorized).
	Matches flash_attn_varlen_qkvpacked_func API and output format.
	Just a placeholder for cpu testing

	Args:
		qkv: (ZN, 3, H, Dk) packed QKV tensors where ZN is total tokens across batch
		cu_seqlens: (B+1,) cumulative sequence lengths [0, L1, L1+L2, ..., ZN]
		max_seqlen: Maximum sequence length in batch
		dropout_p: Dropout probability
		softmax_scale: Scaling factor for attention (default: 1/sqrt(Dk))
		deterministic: Whether to use deterministic implementation

	Returns:
		out: (ZN, H, Dk) attention output
	"""
	ZN, _, H, Dk = qkv.shape
	device = qkv.device
	dtype = qkv.dtype

	if softmax_scale is None:
		softmax_scale = Dk ** -0.5

	# Unpack QKV: (ZN, 3, H, Dk) -> 3x(ZN, H, Dk)
	Q = qkv[:, 0, :, :]  # (ZN, H, Dk)
	K = qkv[:, 1, :, :]  # (ZN, H, Dk)
	V = qkv[:, 2, :, :]  # (ZN, H, Dk)

	batch_size = len(cu_seqlens) - 1

	# Create sample_idx: maps each token to its sequence index
	sample_idx = torch.zeros(ZN, dtype=torch.long, device=device)
	for i in range(batch_size):
		start_idx = cu_seqlens[i].item()
		end_idx = cu_seqlens[i + 1].item()
		sample_idx[start_idx:end_idx] = i

	# Compute attention scores: (ZN, H, Dk) @ (ZN, H, Dk).T = (ZN, H, ZN) for each head
	# Reshape for bmm: (ZN, H, Dk) -> (H, ZN, Dk), then compute (H, ZN, Dk) @ (H, Dk, ZN) = (H, ZN, ZN)
	Q_T = Q.transpose(0, 1)  # (H, ZN, Dk)
	K_T = K.transpose(0, 1).transpose(1, 2)  # (H, Dk, ZN)
	scores = torch.bmm(Q_T, K_T) * softmax_scale  # (H, ZN, ZN)
	scores = scores.transpose(0, 1)  # (ZN, H, ZN)

	# Create attention mask: tokens can only attend within their sequence
	# mask[i, j] = True if tokens i and j are in the same sequence
	mask = sample_idx.unsqueeze(1) == sample_idx.unsqueeze(0)  # (ZN, ZN)
	mask = mask.unsqueeze(1)  # (ZN, 1, ZN) for broadcasting with (ZN, H, ZN)

	# Apply mask: set invalid positions to -inf
	scores = scores.masked_fill(~mask, float('-inf'))

	# Apply softmax
	attn_weights = F.softmax(scores, dim=-1)  # (ZN, H, ZN)
	attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle NaN from softmax of all -inf

	# Apply dropout if needed
	if dropout_p > 0.0:
		attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

	# Apply attention to values: (H, ZN, ZN) @ (H, ZN, Dk) = (H, ZN, Dk)
	attn_weights_T = attn_weights.transpose(0, 1)  # (H, ZN, ZN)
	V_T = V.transpose(0, 1)  # (H, ZN, Dk)
	out = torch.bmm(attn_weights_T, V_T)  # (H, ZN, Dk)
	out = out.transpose(0, 1)  # (ZN, H, Dk)

	return out
