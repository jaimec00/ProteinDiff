"""Tensor utilities for converting between packed (BL) and padded (B,L) formats."""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from proteus.types import Bool, Int


def unpad(
    *tensors: Tensor,
    pad_mask: Bool[Tensor, "B L"],
) -> Tuple[List[Tensor], Int[Tensor, "B+1"], int]:
    """
    Remove padding from B,L tensors to create BL format.

    Args:
        *tensors: Variable number of (B, L, ...) tensors to unpack
        pad_mask: Boolean mask (B, L) where True=valid, False=padded

    Returns:
        unpacked_tensors: List of (BL, ...) tensors with padding removed
        cu_seqlens: Cumulative sequence lengths (B+1,) as int32
        max_seqlen: Maximum sequence length as int

    Example:
        >>> x = torch.randn(2, 5, 128)  # B=2, L=5, d=128
        >>> mask = torch.tensor([[True, True, True, False, False],
        ...                      [True, True, False, False, False]])
        >>> [x_unpacked], cu_seqlens, max_seqlen = unpad(x, pad_mask=mask)
        >>> x_unpacked.shape  # (5, 128) - concatenated valid residues
        >>> cu_seqlens  # tensor([0, 3, 5], dtype=int32)
        >>> max_seqlen  # 3
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided to unpad")

    # Validate tensor shapes match pad_mask
    B, L = pad_mask.shape
    for i, tensor in enumerate(tensors):
        if tensor.shape[:2] != (B, L):
            raise ValueError(
                f"Tensor {i} shape {tensor.shape[:2]} doesn't match pad_mask shape {(B, L)}"
            )

    # Compute sequence lengths and cumulative lengths
    seqlens = pad_mask.sum(dim=1)  # (B,)
    cu_seqlens = F.pad(seqlens.cumsum(dim=0), (1, 0), value=0).to(torch.int32)
    max_seqlen = seqlens.max().item()  # Maximum actual sequence length in batch

    # Create flat boolean mask for indexing
    flat_mask = pad_mask.flatten()  # (B*L,)

    unpacked_tensors = []
    for tensor in tensors:
        trailing_shape = tensor.shape[2:]

        # Flatten first two dimensions and apply mask
        tensor_flat = tensor.reshape(B * L, *trailing_shape)
        tensor_unpacked = tensor_flat[flat_mask]

        unpacked_tensors.append(tensor_unpacked)

    return unpacked_tensors, cu_seqlens, max_seqlen


def repad(
    *tensors: Tensor,
    cu_seqlens: Int[Tensor, "B+1"],
    max_seqlen: int,
) -> List[Tensor]:
    """
    Restore padding to BL tensors to create B,L format.

    Args:
        *tensors: Variable number of (BL, ...) tensors to pad
        cu_seqlens: Cumulative sequence lengths (B+1,) as int32
        max_seqlen: Maximum sequence length for padding

    Returns:
        padded_tensors: List of (B, L, ...) tensors with padding restored

    Example:
        >>> x_unpacked = torch.randn(5, 128)  # BL=5, d=128
        >>> cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        >>> max_seqlen = 3
        >>> [x_padded] = repad(x_unpacked, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        >>> x_padded.shape  # (2, 3, 128) - B=2, L=3
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided to repad")

    # Compute batch size from cu_seqlens
    B = len(cu_seqlens) - 1
    device = cu_seqlens.device
    expected_total_len = cu_seqlens[-1].item()

    # Validate tensor lengths match cu_seqlens
    for i, tensor in enumerate(tensors):
        if tensor.shape[0] != expected_total_len:
            raise ValueError(
                f"Tensor {i} length {tensor.shape[0]} doesn't match cu_seqlens total {expected_total_len}"
            )

    padded_tensors = []
    for tensor in tensors:
        trailing_shape = tensor.shape[1:]
        dtype = tensor.dtype

        # Allocate output tensor with zeros (padding value)
        output = torch.zeros(B, max_seqlen, *trailing_shape, device=device, dtype=dtype)

        # Fill in valid positions for each sample in batch
        for i in range(B):
            start_idx = cu_seqlens[i].item()
            end_idx = cu_seqlens[i + 1].item()
            seq_len = end_idx - start_idx

            if seq_len > 0:
                output[i, :seq_len] = tensor[start_idx:end_idx]

        padded_tensors.append(output)

    return padded_tensors
