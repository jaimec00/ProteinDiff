import torch
import torch.nn.functional as F

from proteus.types import Float, Int, T, Optional

def torch_attn_varlen_qkvpacked_func(
	qkv: Float[T, "BL 3 H Dk"],
	cu_seqlens: Int[T, "B+1"],
	max_seqlen: int,
	dropout_p: float = 0.0,
	softmax_scale: Optional[float] = None,
	deterministic: bool = False
) -> Float[T, "BL H Dk"]:
	"""
	Pure PyTorch implementation of variable-length attention with QKV packing (vectorized).
	Matches flash_attn_varlen_qkvpacked_func API and output format.
	Just a placeholder for cpu testing

	Args:
		qkv: (BL, 3, H, Dk) packed QKV tensors where BL is total tokens across batch
		cu_seqlens: (B+1,) cumulative sequence lengths [0, L1, L1+L2, ..., BL]
		max_seqlen: Maximum sequence length in batch
		dropout_p: Dropout probability
		softmax_scale: Scaling factor for attention (default: 1/sqrt(Dk))
		deterministic: Whether to use deterministic implementation

	Returns:
		out: (BL, H, Dk) attention output
	"""
	BL, _, H, Dk = qkv.shape
	device = qkv.device
	dtype = qkv.dtype

	if softmax_scale is None:
		softmax_scale = Dk ** -0.5

	# Unpack QKV: (BL, 3, H, Dk) -> 3x(BL, H, Dk)
	Q = qkv[:, 0, :, :]  # (BL, H, Dk)
	K = qkv[:, 1, :, :]  # (BL, H, Dk)
	V = qkv[:, 2, :, :]  # (BL, H, Dk)

	batch_size = len(cu_seqlens) - 1

	# Create sample_idx: maps each token to its sequence index
	sample_idx = torch.zeros(BL, dtype=torch.long, device=device)
	for i in range(batch_size):
		start_idx = cu_seqlens[i].item()
		end_idx = cu_seqlens[i + 1].item()
		sample_idx[start_idx:end_idx] = i

	# Compute attention scores: (BL, H, Dk) @ (BL, H, Dk).T = (BL, H, BL) for each head
	# Reshape for bmm: (BL, H, Dk) -> (H, BL, Dk), then compute (H, BL, Dk) @ (H, Dk, BL) = (H, BL, BL)
	Q_T = Q.transpose(0, 1)  # (H, BL, Dk)
	K_T = K.transpose(0, 1).transpose(1, 2)  # (H, Dk, BL)
	scores = torch.bmm(Q_T, K_T) * softmax_scale  # (H, BL, BL)
	scores = scores.transpose(0, 1)  # (BL, H, BL)

	# Create attention mask: tokens can only attend within their sequence
	# mask[i, j] = True if tokens i and j are in the same sequence
	mask = sample_idx.unsqueeze(1) == sample_idx.unsqueeze(0)  # (BL, BL)
	mask = mask.unsqueeze(1)  # (BL, 1, BL) for broadcasting with (BL, H, BL)

	# Apply mask: set invalid positions to -inf
	scores = scores.masked_fill(~mask, float('-inf'))

	# Apply softmax
	attn_weights = F.softmax(scores, dim=-1)  # (BL, H, BL)
	attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # Handle NaN from softmax of all -inf

	# Apply dropout if needed
	if dropout_p > 0.0:
		attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

	# Apply attention to values: (H, BL, BL) @ (H, BL, Dk) = (H, BL, Dk)
	attn_weights_T = attn_weights.transpose(0, 1)  # (H, BL, BL)
	V_T = V.transpose(0, 1)  # (H, BL, Dk)
	out = torch.bmm(attn_weights_T, V_T)  # (H, BL, Dk)
	out = out.transpose(0, 1)  # (BL, H, Dk)

	return out
