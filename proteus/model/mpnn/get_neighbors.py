import torch
from unittest.mock import MagicMock

from proteus.types import Float, Int, Bool, T, Tuple

if torch.cuda.is_available():
    import triton
    import triton.language as tl
else:
    triton = MagicMock()
    tl = MagicMock()


@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_B": b}, num_warps=w)
        for b in [32, 64, 128]
        for w in [4, 8, 16]
    ],
    key=["K"],
    restore_value=["nbrs_ptr"],
)
@triton.jit
def get_neighbors_kernel(
    # inputs
    Ca_ptr, cu_seqlens_ptr,
    # outputs
    nbrs_ptr,
    # shapes
    BL, B, K: tl.constexpr,
    BLOCK_K: tl.constexpr,  # power of 2 >= K
    # strides
    stride_ca_bl, stride_ca_3,
    stride_nbrs_bl, stride_nbrs_k,
    BLOCK_B: tl.constexpr
):
    """Get K nearest neighbors for each token within its sequence.

    Each block handles one token i.
    Masked positions are filled with self-index (pid).
    """
    pid = tl.program_id(0)  # token index i

    # load cu_seqlens and find sequence bounds
    offs_b = tl.arange(0, BLOCK_B)
    cu_seqlens = tl.load(cu_seqlens_ptr + offs_b, mask=offs_b < B + 1, other=BL)

    # find sequence bounds: start is largest cu_seqlens <= pid, end is smallest cu_seqlens > pid
    is_start = (cu_seqlens <= pid)
    is_end = (cu_seqlens > pid)
    seq_start = tl.max(tl.where(is_start, cu_seqlens, 0))
    seq_end = tl.min(tl.where(is_end, cu_seqlens, BL))

    # load coords[i]
    xi = tl.load(Ca_ptr + pid * stride_ca_bl + 0 * stride_ca_3)
    yi = tl.load(Ca_ptr + pid * stride_ca_bl + 1 * stride_ca_3)
    zi = tl.load(Ca_ptr + pid * stride_ca_bl + 2 * stride_ca_3)

    # initialize top-k with inf distances and self-index
    # use BLOCK_K (power of 2) for triton, but only use first K
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    top_dists = tl.full([BLOCK_K], float("inf"), dtype=tl.float32)
    top_idxs = tl.full([BLOCK_K], pid, dtype=tl.int64)

    # loop over all j in sequence
    for j in range(seq_start, seq_end):
        # load coords[j]
        xj = tl.load(Ca_ptr + j * stride_ca_bl + 0 * stride_ca_3)
        yj = tl.load(Ca_ptr + j * stride_ca_bl + 1 * stride_ca_3)
        zj = tl.load(Ca_ptr + j * stride_ca_bl + 2 * stride_ca_3)

        # compute squared distance
        dx, dy, dz = xi - xj, yi - yj, zi - zj
        dist_sq = dx * dx + dy * dy + dz * dz

        # find position to insert (if any)
        # only consider first K positions (k_mask), rest are inf
        max_dist = tl.max(tl.where(k_mask, top_dists, -float("inf")))

        if dist_sq < max_dist:
            # find the position of max and replace it
            is_max = (top_dists == max_dist) & k_mask
            # get first index where is_max is true
            max_pos = tl.argmax(is_max.to(tl.int32), axis=0)

            # replace at max_pos
            top_dists = tl.where(offs_k == max_pos, dist_sq, top_dists)
            top_idxs = tl.where(offs_k == max_pos, j, top_idxs)

    # sort top_idxs by top_dists (simple bubble sort for small K)
    # we need to output sorted by distance
    for i_sort in range(K):
        for j_sort in range(K - 1 - i_sort):
            dist_a = tl.sum(tl.where(offs_k == j_sort, top_dists, 0.0))
            dist_b = tl.sum(tl.where(offs_k == j_sort + 1, top_dists, 0.0))
            idx_a = tl.sum(tl.where(offs_k == j_sort, top_idxs, 0))
            idx_b = tl.sum(tl.where(offs_k == j_sort + 1, top_idxs, 0))

            should_swap = dist_a > dist_b

            # swap if needed
            new_dist_a = tl.where(should_swap, dist_b, dist_a)
            new_dist_b = tl.where(should_swap, dist_a, dist_b)
            new_idx_a = tl.where(should_swap, idx_b, idx_a)
            new_idx_b = tl.where(should_swap, idx_a, idx_b)

            top_dists = tl.where(offs_k == j_sort, new_dist_a, top_dists)
            top_dists = tl.where(offs_k == j_sort + 1, new_dist_b, top_dists)
            top_idxs = tl.where(offs_k == j_sort, new_idx_a, top_idxs)
            top_idxs = tl.where(offs_k == j_sort + 1, new_idx_b, top_idxs)

    # store results (only first K)
    # masked positions already have self-index (pid) from initialization
    nbrs_out_ptr = nbrs_ptr + pid * stride_nbrs_bl + offs_k * stride_nbrs_k
    tl.store(nbrs_out_ptr, top_idxs, mask=k_mask)

@torch.no_grad()
def get_neighbors(
    Ca: Float[T, "BL 3"],
    top_k: int,
    cu_seqlens: Int[T, "B+1"],
) -> Tuple[Int[T, "BL K"], Bool[T, "BL K"]]:
    """Get K nearest neighbors for each token within its sequence.

    Returns:
        nbrs: indices of K nearest neighbors (BL, K)
        nbr_mask: True for valid neighbors, False for padding (BL, K)
    """
    BL = Ca.shape[0]
    B = cu_seqlens.shape[0] - 1
    K = top_k

    # assertions
    assert Ca.shape == (BL, 3), f"Ca shape {Ca.shape} != ({BL}, 3)"
    assert Ca.is_cuda and cu_seqlens.is_cuda

    # ensure contiguous and correct dtypes
    Ca = Ca.to(torch.float32).contiguous()
    cu_seqlens = cu_seqlens.to(torch.int32).contiguous()

    # allocate output
    nbrs = torch.zeros(BL, K, device=Ca.device, dtype=torch.int64)

    # BLOCK_K must be power of 2 for triton
    BLOCK_K = triton.next_power_of_2(K)

    # launch kernel
    grid = (BL,)
    get_neighbors_kernel[grid](
        Ca, cu_seqlens,
        nbrs,
        BL, B, K, BLOCK_K,
        Ca.stride(0), Ca.stride(1),
        nbrs.stride(0), nbrs.stride(1),
    )

    # compute mask: True where neighbor != self, False where neighbor == self (padding or self-loop)
    nbr_mask = nbrs != torch.arange(BL, device=nbrs.device, dtype=nbrs.dtype).unsqueeze(1)

    return nbrs, nbr_mask
