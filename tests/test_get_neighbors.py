import torch
import pytest
import sys
import os

# add parent dir to path to avoid triggering full package import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proteindiff.model.mpnn.get_neighbors import get_neighbors


def get_neighbors_ref(
    Ca: torch.Tensor,  # (ZN, 3)
    top_k: int,
    cu_seqlens: torch.Tensor,  # (Z+1,)
) -> torch.Tensor:  # (ZN, K)
    """Reference implementation for get_neighbors."""
    ZN = Ca.shape[0]
    Z = cu_seqlens.shape[0] - 1
    device = Ca.device

    # build sample_idx from cu_seqlens
    sample_idx = torch.zeros(ZN, dtype=torch.int32, device=device)
    for s in range(Z):
        start, end = cu_seqlens[s].item(), cu_seqlens[s + 1].item()
        sample_idx[start:end] = s

    # compute pairwise squared distances
    diff = Ca.unsqueeze(0) - Ca.unsqueeze(1)  # ZN x ZN x 3
    dists_sq = torch.sum(diff * diff, dim=-1)  # ZN x ZN

    # mask out positions from other samples
    dists_sq.masked_fill_(sample_idx.unsqueeze(0) != sample_idx.unsqueeze(1), float("inf"))

    # get topk per row (clamp K to max seq len)
    k = min(top_k, ZN)
    _, nbrs = dists_sq.topk(k, dim=1, largest=False)

    return nbrs


@pytest.mark.parametrize("ZN,Z,top_k", [
    (16, 1, 8),      # single sequence
    (32, 2, 8),      # two sequences
    (64, 4, 16),     # multiple sequences
    (100, 5, 10),    # uneven sequences
])
def test_get_neighbors_matches_ref(ZN, Z, top_k):
    device = torch.device("cuda")

    # create random cu_seqlens ensuring each seq has at least top_k tokens
    min_seq_len = top_k + 1
    total_min = min_seq_len * Z
    if ZN < total_min:
        ZN = total_min

    # evenly spaced boundaries to ensure each seq >= top_k
    seq_len = ZN // Z
    boundaries = [seq_len * i for i in range(1, Z)]
    cu_seqlens = torch.tensor([0] + boundaries + [ZN], dtype=torch.int32, device=device)

    # random coords
    Ca = torch.randn(ZN, 3, device=device, dtype=torch.float32)

    # run ref
    nbrs_ref = get_neighbors_ref(Ca, top_k, cu_seqlens)

    # run kernel
    nbrs, nbr_mask = get_neighbors(Ca, top_k, cu_seqlens)

    # check shapes
    assert nbrs.shape == nbrs_ref.shape, f"shape mismatch: {nbrs.shape} vs {nbrs_ref.shape}"

    # check values match
    assert torch.equal(nbrs, nbrs_ref), f"values mismatch"

    # mask: False for self (col 0), True for non-self neighbors
    self_idx = torch.arange(ZN, device=device, dtype=nbrs.dtype).unsqueeze(1)
    expected_mask = nbrs != self_idx
    assert torch.equal(nbr_mask, expected_mask), "mask mismatch"


@pytest.mark.parametrize("top_k", [4, 8, 16, 32])
def test_get_neighbors_single_seq(top_k):
    """Test with single sequence - simpler case."""
    device = torch.device("cuda")
    ZN = 64

    cu_seqlens = torch.tensor([0, ZN], dtype=torch.int32, device=device)
    Ca = torch.randn(ZN, 3, device=device, dtype=torch.float32)

    nbrs_ref = get_neighbors_ref(Ca, top_k, cu_seqlens)
    nbrs, nbr_mask = get_neighbors(Ca, top_k, cu_seqlens)

    assert torch.equal(nbrs, nbrs_ref)
    # col 0 is self (mask=False), rest are non-self (mask=True)
    assert not nbr_mask[:, 0].any(), "col 0 should be self (mask=False)"
    assert nbr_mask[:, 1:].all(), "cols 1+ should be non-self (mask=True)"


def test_get_neighbors_self_is_nearest():
    """Each token's nearest neighbor should be itself (distance 0)."""
    device = torch.device("cuda")
    ZN, top_k = 32, 8

    cu_seqlens = torch.tensor([0, ZN], dtype=torch.int32, device=device)
    Ca = torch.randn(ZN, 3, device=device, dtype=torch.float32)

    nbrs, _ = get_neighbors(Ca, top_k, cu_seqlens)

    # first column should be self-index
    self_idx = torch.arange(ZN, device=device)
    assert torch.equal(nbrs[:, 0], self_idx), "self should be nearest neighbor"


def test_get_neighbors_no_cross_sample():
    """Neighbors should not cross sample boundaries."""
    device = torch.device("cuda")
    ZN, Z, top_k = 64, 4, 16

    # equal size sequences
    seq_len = ZN // Z
    cu_seqlens = torch.tensor([i * seq_len for i in range(Z + 1)], dtype=torch.int32, device=device)

    Ca = torch.randn(ZN, 3, device=device, dtype=torch.float32)

    nbrs, _ = get_neighbors(Ca, top_k, cu_seqlens)

    # build sample_idx
    sample_idx = torch.zeros(ZN, dtype=torch.int32, device=device)
    for s in range(Z):
        start, end = cu_seqlens[s].item(), cu_seqlens[s + 1].item()
        sample_idx[start:end] = s

    # check no neighbor crosses sample boundary
    for i in range(ZN):
        sample_i = sample_idx[i].item()
        for j in range(top_k):
            nbr_idx = nbrs[i, j].item()
            sample_nbr = sample_idx[nbr_idx].item()
            assert sample_i == sample_nbr, f"token {i} (sample {sample_i}) has neighbor {nbr_idx} (sample {sample_nbr})"


def test_get_neighbors_small_seq():
    """Test when sequence length < K. Padding should be self-index, mask should be False."""
    device = torch.device("cuda")
    ZN, Z, top_k = 20, 2, 16  # seq_len=10 < K=16

    cu_seqlens = torch.tensor([0, 10, 20], dtype=torch.int32, device=device)
    Ca = torch.randn(ZN, 3, device=device, dtype=torch.float32)

    nbrs, nbr_mask = get_neighbors(Ca, top_k, cu_seqlens)

    # build sample_idx
    sample_idx = torch.zeros(ZN, dtype=torch.int32, device=device)
    sample_idx[10:] = 1

    # for each row:
    # - col 0: self (mask=False)
    # - cols 1 to seq_len-1: non-self same-seq neighbors (mask=True)
    # - cols seq_len to K-1: padding with self (mask=False)
    seq_len = 10
    for i in range(ZN):
        for k in range(top_k):
            nbr_idx = nbrs[i, k].item()
            is_non_self = nbr_mask[i, k].item()
            if k == 0:
                # self at position 0
                assert nbr_idx == i, f"row {i}, col 0: expected self"
                assert not is_non_self, f"row {i}, col 0: mask should be False (self)"
            elif k < seq_len:
                # valid non-self neighbor from same sequence
                assert sample_idx[nbr_idx] == sample_idx[i], f"row {i}, col {k}: cross-seq neighbor"
                assert nbr_idx != i, f"row {i}, col {k}: should not be self"
                assert is_non_self, f"row {i}, col {k}: mask should be True (non-self)"
            else:
                # padding - self-index and mask False
                assert nbr_idx == i, f"row {i}, col {k}: expected self-pad, got {nbr_idx}"
                assert not is_non_self, f"row {i}, col {k}: mask should be False (padding)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
