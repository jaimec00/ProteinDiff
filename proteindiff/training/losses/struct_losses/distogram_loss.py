import triton
import triton.language as tl
import torch

from proteindiff.types import Float, Int, T


@triton.autotune(
    configs=[
        triton.Config(kwargs={}, num_warps=w)
        for w in [4, 8, 16]
    ],
    key=["BLOCK_D"],
    restore_value=["per_token_loss_ptr", "d_logits_ptr", "d_fc1_ptr", "d_fc2_ptr", "d_coords_ptr"],
)
@triton.jit
def distogram_loss_fwd_bwd(
    # inputs
    logits_ptr, fc1_ptr, fc2_ptr, coords_ptr, cu_seqlens_ptr,
    # outputs
    per_token_loss_ptr, d_logits_ptr, d_fc1_ptr, d_fc2_ptr, d_coords_ptr,
    # shapes
    ZN, Z, d_in, d_hidden, d_out,
    # distance bin params
    min_dist, max_dist,
    # strides - logits (ZN, 2, d_in)
    stride_logits_zn, stride_logits_2, stride_logits_d,
    # strides - fc1 (d_hidden, 2*d_in)
    stride_fc1_hidden, stride_fc1_in,
    # strides - fc2 (d_out, d_hidden)
    stride_fc2_out, stride_fc2_hidden,
    # strides - coords (ZN, 3)
    stride_coords_zn, stride_coords_3,
    # strides - d_logits (ZN, 2, d_in)
    stride_dlogits_zn, stride_dlogits_2, stride_dlogits_d,
    # strides - d_fc1 (d_hidden, 2*d_in)
    stride_dfc1_hidden, stride_dfc1_in,
    # strides - d_fc2 (d_out, d_hidden)
    stride_dfc2_out, stride_dfc2_hidden,
    # strides - d_coords (ZN, 3)
    stride_dcoords_zn, stride_dcoords_3,
    # block sizes
    BLOCK_D: tl.constexpr,
):
    """Fused distogram loss forward and backward.

    Each block handles one token i and loops over all j in same sequence.
    Computes loss and gradients without materializing full pairwise matrix.
    """
    pid = tl.program_id(0)  # token index i

    # find sequence bounds via linear search
    seq_start = 0
    seq_end = 0
    for s in range(Z):
        start_s = tl.load(cu_seqlens_ptr + s)
        end_s = tl.load(cu_seqlens_ptr + s + 1)
        if start_s <= pid and pid < end_s:
            seq_start = start_s
            seq_end = end_s

    bin_width = (max_dist - min_dist) / d_out

    # compute n_pairs ahead of time for normalization (includes self i==j)
    seq_len = seq_end - seq_start
    inv_n_pairs = 1.0 / tl.maximum(seq_len, 1.0)

    # offsets
    offs = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_D)  # for hidden dim
    offs_o = tl.arange(0, BLOCK_D)  # for output dim

    # load fc1 (d_hidden, 2*d_in) - first and second halves
    fc1_mask = (offs_h[:, None] < d_hidden) & (offs[None, :] < d_in)
    fc1_first = tl.load(
        fc1_ptr + offs_h[:, None] * stride_fc1_hidden + offs[None, :] * stride_fc1_in,
        mask=fc1_mask, other=0.0
    )
    fc1_second = tl.load(
        fc1_ptr + offs_h[:, None] * stride_fc1_hidden + (d_in + offs[None, :]) * stride_fc1_in,
        mask=fc1_mask, other=0.0
    )

    # load fc2 (d_out, d_hidden)
    fc2_mask = (offs_o[:, None] < d_out) & (offs_h[None, :] < d_hidden)
    fc2 = tl.load(
        fc2_ptr + offs_o[:, None] * stride_fc2_out + offs_h[None, :] * stride_fc2_hidden,
        mask=fc2_mask, other=0.0
    )

    # load q[i], k[i]
    qi_ptr = logits_ptr + pid * stride_logits_zn + 0 * stride_logits_2 + offs * stride_logits_d
    ki_ptr = logits_ptr + pid * stride_logits_zn + 1 * stride_logits_2 + offs * stride_logits_d
    qi = tl.load(qi_ptr, mask=offs < d_in, other=0.0)
    ki = tl.load(ki_ptr, mask=offs < d_in, other=0.0)

    # load coords[i]
    xi = tl.load(coords_ptr + pid * stride_coords_zn + 0)
    yi = tl.load(coords_ptr + pid * stride_coords_zn + 1)
    zi = tl.load(coords_ptr + pid * stride_coords_zn + 2)

    # accumulators
    loss_acc = 0.0
    d_qi_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_fc1_first_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    d_fc1_second_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    d_fc2_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    n_pairs_f = 0.0

    # loop over j in sequence (includes self i==j)
    for j in range(seq_start, seq_end):
        valid = True

        # load q[j], k[j]
        qj_ptr = logits_ptr + j * stride_logits_zn + 0 * stride_logits_2 + offs * stride_logits_d
        kj_ptr = logits_ptr + j * stride_logits_zn + 1 * stride_logits_2 + offs * stride_logits_d
        qj = tl.load(qj_ptr, mask=offs < d_in, other=0.0)
        kj = tl.load(kj_ptr, mask=offs < d_in, other=0.0)

        # load coords[j]
        xj = tl.load(coords_ptr + j * stride_coords_zn + 0)
        yj = tl.load(coords_ptr + j * stride_coords_zn + 1)
        zj = tl.load(coords_ptr + j * stride_coords_zn + 2)

        # compute distance
        dx, dy, dz = xi - xj, yi - yj, zi - zj
        dist = tl.sqrt(dx * dx + dy * dy + dz * dz + 1e-8)

        # bin index (target)
        bin_idx_f = (dist - min_dist) / bin_width
        bin_idx = tl.minimum(tl.maximum(bin_idx_f, 0.0), d_out - 1.0).to(tl.int32)

        # FFN input: cat(q[i]*k[j], q[i]-k[j])
        qk_mul = qi * kj
        qk_sub = qi - kj

        # FFN forward using tl.dot for matrix-vector products
        # hidden = relu(fc1_first @ qk_mul + fc1_second @ qk_sub)
        d_mask = offs < d_in
        h_mask = offs_h < d_hidden
        o_mask = offs_o < d_out

        # mask inputs before dot
        qk_mul_masked = tl.where(d_mask, qk_mul, 0.0)
        qk_sub_masked = tl.where(d_mask, qk_sub, 0.0)

        # fc1_first @ qk_mul + fc1_second @ qk_sub using tl.dot
        # reshape vectors to 2D: (BLOCK_D,) -> (BLOCK_D, 1), then sum axis=1 to squeeze
        hidden_pre = (tl.sum(tl.dot(fc1_first, qk_mul_masked[:, None]), axis=1) +
                      tl.sum(tl.dot(fc1_second, qk_sub_masked[:, None]), axis=1))
        hidden_pre = tl.where(h_mask, hidden_pre, 0.0)
        hidden = tl.maximum(hidden_pre, 0.0)  # relu
        relu_mask = (hidden_pre > 0) & h_mask

        # out_logits = fc2 @ hidden using tl.dot
        out_logits = tl.sum(tl.dot(fc2, hidden[:, None]), axis=1)
        out_logits = tl.where(o_mask, out_logits, 0.0)

        # log_softmax for numerical stability (avoid underflow in exp)
        out_max = tl.max(tl.where(o_mask, out_logits, -1e9))
        out_shifted = out_logits - out_max
        out_exp = tl.exp(out_shifted) * o_mask
        out_sum = tl.sum(out_exp) + 1e-8
        log_sum_exp = tl.log(out_sum)
        log_probs = out_shifted - log_sum_exp  # log_softmax

        # softmax probs needed for gradient
        probs = out_exp / out_sum

        # cross entropy: -log_probs[bin_idx]
        log_prob_target = tl.sum(tl.where(offs_o == bin_idx, log_probs, 0.0))
        loss_contrib = -log_prob_target

        # === backward ===
        
        # d_out_logits = (probs - one_hot(bin_idx))
        d_out_logits = (probs - tl.where(offs_o == bin_idx, 1.0, 0.0))
        d_out_logits_masked = tl.where(o_mask, d_out_logits, 0.0)

        # grad_h = fc2.T @ d_out_logits using tl.dot with transpose
        # fc2 is (d_out, d_hidden), fc2.T is (d_hidden, d_out)
        fc2_t = tl.trans(fc2)
        grad_h = tl.sum(tl.dot(fc2_t, d_out_logits_masked[:, None]), axis=1)
        grad_h = tl.where(h_mask, grad_h, 0.0)

        # d_fc2 += outer(d_out_logits, hidden)
        d_fc2_local = d_out_logits[:, None] * hidden[None, :]

        # relu backward
        grad_h = tl.where(relu_mask, grad_h, 0.0)

        # d_fc1 += outer(grad_h, qk_mul/qk_sub)
        d_fc1_first_local = grad_h[:, None] * qk_mul[None, :]
        d_fc1_second_local = grad_h[:, None] * qk_sub[None, :]

        # d_qk_mul = fc1_first.T @ grad_h, d_qk_sub = fc1_second.T @ grad_h using tl.dot
        grad_h_masked = tl.where(h_mask, grad_h, 0.0)
        fc1_first_t = tl.trans(fc1_first)
        fc1_second_t = tl.trans(fc1_second)
        d_qk_mul = tl.sum(tl.dot(fc1_first_t, grad_h_masked[:, None]), axis=1)
        d_qk_sub = tl.sum(tl.dot(fc1_second_t, grad_h_masked[:, None]), axis=1)
        d_qk_mul = tl.where(d_mask, d_qk_mul, 0.0)
        d_qk_sub = tl.where(d_mask, d_qk_sub, 0.0)

        # d_qi = d_qk_mul * kj + d_qk_sub, d_kj = d_qk_mul * qi - d_qk_sub
        d_qi_local = d_qk_mul * kj + d_qk_sub
        d_kj_local = d_qk_mul * qi - d_qk_sub

        # accumulate (masked by valid)
        loss_acc += tl.where(valid, loss_contrib, 0.0)
        n_pairs_f += tl.where(valid, 1.0, 0.0)
        d_qi_acc += tl.where(valid, d_qi_local, 0.0)
        d_fc1_first_acc += tl.where(valid, d_fc1_first_local, 0.0)
        d_fc1_second_acc += tl.where(valid, d_fc1_second_local, 0.0)
        d_fc2_acc += tl.where(valid, d_fc2_local, 0.0)

        # atomic add d_kj to d_logits[j, 1, :] (normalized)
        d_kj_ptr = d_logits_ptr + j * stride_dlogits_zn + 1 * stride_dlogits_2 + offs * stride_dlogits_d
        d_kj_norm = tl.where(valid, d_kj_local * inv_n_pairs, 0.0)
        tl.atomic_add(d_kj_ptr, d_kj_norm, mask=d_mask)

    # normalize and write (avoid conditional to prevent Triton compiler issue)
    inv_n = 1.0 / tl.maximum(n_pairs_f, 1.0)
    loss_acc = loss_acc * inv_n
    d_qi_acc = d_qi_acc * inv_n
    d_fc1_first_acc = d_fc1_first_acc * inv_n
    d_fc1_second_acc = d_fc1_second_acc * inv_n
    d_fc2_acc = d_fc2_acc * inv_n

    # store per-token loss (no atomic needed)
    tl.store(per_token_loss_ptr + pid, loss_acc)

    # atomic add d_qi to d_logits[i, 0, :]
    d_mask = offs < d_in
    d_qi_ptr = d_logits_ptr + pid * stride_dlogits_zn + 0 * stride_dlogits_2 + offs * stride_dlogits_d
    tl.atomic_add(d_qi_ptr, d_qi_acc, mask=d_mask)

    # atomic add d_fc1 (first half: cols 0..d_in-1, second half: cols d_in..2*d_in-1)
    fc1_grad_mask = (offs_h[:, None] < d_hidden) & (offs[None, :] < d_in)
    d_fc1_first_ptr = d_fc1_ptr + offs_h[:, None] * stride_dfc1_hidden + offs[None, :] * stride_dfc1_in
    tl.atomic_add(d_fc1_first_ptr, d_fc1_first_acc, mask=fc1_grad_mask)
    d_fc1_second_ptr = d_fc1_ptr + offs_h[:, None] * stride_dfc1_hidden + (d_in + offs[None, :]) * stride_dfc1_in
    tl.atomic_add(d_fc1_second_ptr, d_fc1_second_acc, mask=fc1_grad_mask)

    # atomic add d_fc2
    fc2_grad_mask = (offs_o[:, None] < d_out) & (offs_h[None, :] < d_hidden)
    d_fc2_ptr_out = d_fc2_ptr + offs_o[:, None] * stride_dfc2_out + offs_h[None, :] * stride_dfc2_hidden
    tl.atomic_add(d_fc2_ptr_out, d_fc2_acc, mask=fc2_grad_mask)


def distogram_loss(
    logits: Float[T, "ZN 2 d_in"],
    fc1: Float[T, "d_hidden 2*d_in"],
    fc2: Float[T, "d_out d_hidden"],
    coords: Float[T, "ZN 3"],
    cu_seqlens: Int[T, "Z+1"],
    min_dist: float = 2.0,
    max_dist: float = 22.0,
) -> Float[T, "1"]:
    return DistogramLoss.apply(logits, fc1, fc2, coords, cu_seqlens, min_dist, max_dist)


class DistogramLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        logits: Float[T, "ZN 2 d_in"],
        fc1: Float[T, "d_hidden 2*d_in"],
        fc2: Float[T, "d_out d_hidden"],
        coords: Float[T, "ZN 3"],
        cu_seqlens: Int[T, "Z+1"],
        min_dist: float,
        max_dist: float,
    ) -> Float[T, "1"]:
        # shapes
        ZN, _, d_in = logits.shape
        d_hidden, _ = fc1.shape
        d_out, _ = fc2.shape
        Z = cu_seqlens.shape[0] - 1

        # assertions
        assert coords.shape == (ZN, 3), f"coords {coords.shape} != ({ZN}, 3)"
        assert logits.shape == (ZN, 2, d_in), f"logits {logits.shape} != ({ZN}, 2, {d_in})"
        assert fc1.shape == (d_hidden, 2 * d_in), f"fc1 {fc1.shape} != ({d_hidden}, {2 * d_in})"
        assert fc2.shape == (d_out, d_hidden), f"fc2 {fc2.shape} != ({d_out}, {d_hidden})"
        assert cu_seqlens.shape == (Z + 1,), f"cu_seqlens {cu_seqlens.shape} != ({Z + 1},)"
        assert coords.is_cuda and logits.is_cuda and fc1.is_cuda and fc2.is_cuda and cu_seqlens.is_cuda

        # get orig dtypes
        coords_dtype = coords.dtype
        logits_dtype = logits.dtype
        fc1_dtype = fc1.dtype
        fc2_dtype = fc2.dtype

        # all computation in fp32
        coords = coords.to(torch.float32).contiguous()
        logits = logits.to(torch.float32).contiguous()
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
        fc1 = fc1.to(torch.float32).contiguous()
        fc2 = fc2.to(torch.float32).contiguous()

        # allocate outputs (use fp32 for accuracy)
        per_token_loss = torch.zeros(ZN, device=coords.device, dtype=torch.float32)
        d_coords = torch.zeros(ZN, 3, device=coords.device, dtype=torch.float32)
        d_logits = torch.zeros(ZN, 2, d_in, device=coords.device, dtype=torch.float32)
        d_fc1 = torch.zeros(d_hidden, 2 * d_in, device=coords.device, dtype=torch.float32)
        d_fc2 = torch.zeros(d_out, d_hidden, device=coords.device, dtype=torch.float32)

        # block size - use max of all dims, rounded to power of 2
        # tl.dot requires K >= 16, so ensure BLOCK_D >= 16
        BLOCK_D = max(16, triton.next_power_of_2(max(d_in, d_hidden, d_out)))
        grid = (ZN,)

        # launch kernel
        distogram_loss_fwd_bwd[grid](
            # inputs
            logits, fc1, fc2, coords, cu_seqlens,
            # outputs
            per_token_loss, d_logits, d_fc1, d_fc2, d_coords,
            # shapes
            ZN, Z, d_in, d_hidden, d_out,
            # distance bin params
            min_dist, max_dist,
            # strides - logits
            logits.stride(0), logits.stride(1), logits.stride(2),
            # strides - fc1
            fc1.stride(0), fc1.stride(1),
            # strides - fc2
            fc2.stride(0), fc2.stride(1),
            # strides - coords
            coords.stride(0), coords.stride(1),
            # strides - d_logits
            d_logits.stride(0), d_logits.stride(1), d_logits.stride(2),
            # strides - d_fc1
            d_fc1.stride(0), d_fc1.stride(1),
            # strides - d_fc2
            d_fc2.stride(0), d_fc2.stride(1),
            # strides - d_coords
            d_coords.stride(0), d_coords.stride(1),
            # block size
            BLOCK_D,
        )

        # save grads for backward
        ctx.save_for_backward(d_coords, d_logits, d_fc1, d_fc2)
        ctx.dtypes = (coords_dtype, logits_dtype, fc1_dtype, fc2_dtype)

        return per_token_loss.sum()

    @staticmethod
    def backward(ctx, grad_output):
        d_coords, d_logits, d_fc1, d_fc2 = ctx.saved_tensors
        coords_dtype, logits_dtype, fc1_dtype, fc2_dtype = ctx.dtypes

        # scale by upstream grad and cast back to orig dtypes
        d_coords = (d_coords * grad_output).to(coords_dtype)
        d_logits = (d_logits * grad_output).to(logits_dtype)
        d_fc1 = (d_fc1 * grad_output).to(fc1_dtype)
        d_fc2 = (d_fc2 * grad_output).to(fc2_dtype)

        # None for cu_seqlens, min_dist, max_dist
        return d_logits, d_fc1, d_fc2, d_coords, None, None, None
