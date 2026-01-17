import triton
import triton.language as tl
import torch

from proteindiff.types import Float, Int, T

# 6 non-redundant dot product pairs: (vec_i_idx, vec_j_idx)
# unit_vecs = [CaN, CaCb, CaC] indexed as [0, 1, 2]
# DOT_PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


@triton.autotune(
    configs=[
        triton.Config(kwargs={}, num_warps=w)
        for w in [4, 8, 16]
    ],
    key=["BLOCK_D", "BLOCK_BINS"],
    restore_value=["per_token_loss_ptr", "d_logits_ptr", "d_fc1_ptr", "d_fc2_ptr", "d_unit_vecs_ptr"],
)
@triton.jit
def anglogram_loss_fwd_bwd(
    # inputs
    logits_ptr, fc1_ptr, fc2_ptr, unit_vecs_ptr, cu_seqlens_ptr,
    # outputs
    per_token_loss_ptr, d_logits_ptr, d_fc1_ptr, d_fc2_ptr, d_unit_vecs_ptr,
    # shapes
    ZN, Z, d_in, d_hidden, num_bins,
    # strides - logits (ZN, 2, d_in)
    stride_logits_zn, stride_logits_2, stride_logits_d,
    # strides - fc1 (d_hidden, 2*d_in)
    stride_fc1_hidden, stride_fc1_in,
    # strides - fc2 (6*num_bins, d_hidden)
    stride_fc2_out, stride_fc2_hidden,
    # strides - unit_vecs (ZN, 3, 3)
    stride_vecs_zn, stride_vecs_v, stride_vecs_xyz,
    # strides - d_logits (ZN, 2, d_in)
    stride_dlogits_zn, stride_dlogits_2, stride_dlogits_d,
    # strides - d_fc1 (d_hidden, 2*d_in)
    stride_dfc1_hidden, stride_dfc1_in,
    # strides - d_fc2 (6*num_bins, d_hidden)
    stride_dfc2_out, stride_dfc2_hidden,
    # strides - d_unit_vecs (ZN, 3, 3)
    stride_dvecs_zn, stride_dvecs_v, stride_dvecs_xyz,
    # block sizes
    BLOCK_D: tl.constexpr,
    BLOCK_BINS: tl.constexpr,
):
    """Fused anglogram loss forward and backward with 6 non-redundant dot products.

    Each block handles one token i and loops over all j in same sequence.
    Computes 6 dot products between [CaN, CaCb, CaC] vectors:
      (0,0): CaN_i · CaN_j
      (0,1): CaN_i · CaCb_j
      (0,2): CaN_i · CaC_j
      (1,1): CaCb_i · CaCb_j
      (1,2): CaCb_i · CaC_j
      (2,2): CaC_i · CaC_j
    All computation in fp32.
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

    # fixed range for dot product: [-1, 1]
    min_val = -1.0
    max_val = 1.0
    bin_width = (max_val - min_val) / num_bins

    # compute n_pairs from sequence length (includes self i==j)
    seq_len_f = (seq_end - seq_start).to(tl.float32)
    inv_n_pairs = 1.0 / tl.maximum(seq_len_f, 1.0)

    # offsets
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, BLOCK_D)
    offs_b = tl.arange(0, BLOCK_BINS)

    # load fc1 (d_hidden, 2*d_in) - first and second halves (fp32)
    fc1_mask = (offs_h[:, None] < d_hidden) & (offs_d[None, :] < d_in)
    fc1_first = tl.load(
        fc1_ptr + offs_h[:, None] * stride_fc1_hidden + offs_d[None, :] * stride_fc1_in,
        mask=fc1_mask, other=0.0
    ).to(tl.float32)
    fc1_second = tl.load(
        fc1_ptr + offs_h[:, None] * stride_fc1_hidden + (d_in + offs_d[None, :]) * stride_fc1_in,
        mask=fc1_mask, other=0.0
    ).to(tl.float32)

    # load fc2 (6*num_bins, d_hidden) - load all 6 slices (fp32)
    # fc2_p[p] is (num_bins, d_hidden) for dot product p
    fc2_mask = (offs_b[:, None] < num_bins) & (offs_h[None, :] < d_hidden)
    fc2_0 = tl.load(
        fc2_ptr + (0 * num_bins + offs_b[:, None]) * stride_fc2_out + offs_h[None, :] * stride_fc2_hidden,
        mask=fc2_mask, other=0.0
    ).to(tl.float32)
    fc2_1 = tl.load(
        fc2_ptr + (1 * num_bins + offs_b[:, None]) * stride_fc2_out + offs_h[None, :] * stride_fc2_hidden,
        mask=fc2_mask, other=0.0
    ).to(tl.float32)
    fc2_2 = tl.load(
        fc2_ptr + (2 * num_bins + offs_b[:, None]) * stride_fc2_out + offs_h[None, :] * stride_fc2_hidden,
        mask=fc2_mask, other=0.0
    ).to(tl.float32)
    fc2_3 = tl.load(
        fc2_ptr + (3 * num_bins + offs_b[:, None]) * stride_fc2_out + offs_h[None, :] * stride_fc2_hidden,
        mask=fc2_mask, other=0.0
    ).to(tl.float32)
    fc2_4 = tl.load(
        fc2_ptr + (4 * num_bins + offs_b[:, None]) * stride_fc2_out + offs_h[None, :] * stride_fc2_hidden,
        mask=fc2_mask, other=0.0
    ).to(tl.float32)
    fc2_5 = tl.load(
        fc2_ptr + (5 * num_bins + offs_b[:, None]) * stride_fc2_out + offs_h[None, :] * stride_fc2_hidden,
        mask=fc2_mask, other=0.0
    ).to(tl.float32)

    # load q[i], k[i] (fp32)
    qi_ptr = logits_ptr + pid * stride_logits_zn + 0 * stride_logits_2 + offs_d * stride_logits_d
    ki_ptr = logits_ptr + pid * stride_logits_zn + 1 * stride_logits_2 + offs_d * stride_logits_d
    qi = tl.load(qi_ptr, mask=offs_d < d_in, other=0.0).to(tl.float32)
    ki = tl.load(ki_ptr, mask=offs_d < d_in, other=0.0).to(tl.float32)

    # load unit_vecs[i] - 3 vectors: CaN, CaCb, CaC (fp32)
    # unit_vecs: (ZN, 3, 3) where dim1=[CaN,CaCb,CaC], dim2=xyz
    vi_0_x = tl.load(unit_vecs_ptr + pid * stride_vecs_zn + 0 * stride_vecs_v + 0).to(tl.float32)
    vi_0_y = tl.load(unit_vecs_ptr + pid * stride_vecs_zn + 0 * stride_vecs_v + 1).to(tl.float32)
    vi_0_z = tl.load(unit_vecs_ptr + pid * stride_vecs_zn + 0 * stride_vecs_v + 2).to(tl.float32)
    vi_1_x = tl.load(unit_vecs_ptr + pid * stride_vecs_zn + 1 * stride_vecs_v + 0).to(tl.float32)
    vi_1_y = tl.load(unit_vecs_ptr + pid * stride_vecs_zn + 1 * stride_vecs_v + 1).to(tl.float32)
    vi_1_z = tl.load(unit_vecs_ptr + pid * stride_vecs_zn + 1 * stride_vecs_v + 2).to(tl.float32)
    vi_2_x = tl.load(unit_vecs_ptr + pid * stride_vecs_zn + 2 * stride_vecs_v + 0).to(tl.float32)
    vi_2_y = tl.load(unit_vecs_ptr + pid * stride_vecs_zn + 2 * stride_vecs_v + 1).to(tl.float32)
    vi_2_z = tl.load(unit_vecs_ptr + pid * stride_vecs_zn + 2 * stride_vecs_v + 2).to(tl.float32)

    # accumulators (all fp32)
    loss_acc = 0.0
    d_qi_acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_fc1_first_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    d_fc1_second_acc = tl.zeros([BLOCK_D, BLOCK_D], dtype=tl.float32)
    # fc2 grads for each of 6 outputs
    d_fc2_0_acc = tl.zeros([BLOCK_BINS, BLOCK_D], dtype=tl.float32)
    d_fc2_1_acc = tl.zeros([BLOCK_BINS, BLOCK_D], dtype=tl.float32)
    d_fc2_2_acc = tl.zeros([BLOCK_BINS, BLOCK_D], dtype=tl.float32)
    d_fc2_3_acc = tl.zeros([BLOCK_BINS, BLOCK_D], dtype=tl.float32)
    d_fc2_4_acc = tl.zeros([BLOCK_BINS, BLOCK_D], dtype=tl.float32)
    d_fc2_5_acc = tl.zeros([BLOCK_BINS, BLOCK_D], dtype=tl.float32)

    # masks
    d_mask = offs_d < d_in
    h_mask = offs_h < d_hidden
    b_mask = offs_b < num_bins

    # loop over j in sequence (includes self i==j)
    for j in range(seq_start, seq_end):
        valid = True

        # load q[j], k[j] (fp32)
        qj_ptr = logits_ptr + j * stride_logits_zn + 0 * stride_logits_2 + offs_d * stride_logits_d
        kj_ptr = logits_ptr + j * stride_logits_zn + 1 * stride_logits_2 + offs_d * stride_logits_d
        qj = tl.load(qj_ptr, mask=d_mask, other=0.0).to(tl.float32)
        kj = tl.load(kj_ptr, mask=d_mask, other=0.0).to(tl.float32)

        # load unit_vecs[j] - 3 vectors (fp32)
        vj_0_x = tl.load(unit_vecs_ptr + j * stride_vecs_zn + 0 * stride_vecs_v + 0).to(tl.float32)
        vj_0_y = tl.load(unit_vecs_ptr + j * stride_vecs_zn + 0 * stride_vecs_v + 1).to(tl.float32)
        vj_0_z = tl.load(unit_vecs_ptr + j * stride_vecs_zn + 0 * stride_vecs_v + 2).to(tl.float32)
        vj_1_x = tl.load(unit_vecs_ptr + j * stride_vecs_zn + 1 * stride_vecs_v + 0).to(tl.float32)
        vj_1_y = tl.load(unit_vecs_ptr + j * stride_vecs_zn + 1 * stride_vecs_v + 1).to(tl.float32)
        vj_1_z = tl.load(unit_vecs_ptr + j * stride_vecs_zn + 1 * stride_vecs_v + 2).to(tl.float32)
        vj_2_x = tl.load(unit_vecs_ptr + j * stride_vecs_zn + 2 * stride_vecs_v + 0).to(tl.float32)
        vj_2_y = tl.load(unit_vecs_ptr + j * stride_vecs_zn + 2 * stride_vecs_v + 1).to(tl.float32)
        vj_2_z = tl.load(unit_vecs_ptr + j * stride_vecs_zn + 2 * stride_vecs_v + 2).to(tl.float32)

        # compute 6 non-redundant dot products
        dot_0 = vi_0_x * vj_0_x + vi_0_y * vj_0_y + vi_0_z * vj_0_z  # (0,0) CaN·CaN
        dot_1 = vi_0_x * vj_1_x + vi_0_y * vj_1_y + vi_0_z * vj_1_z  # (0,1) CaN·CaCb
        dot_2 = vi_0_x * vj_2_x + vi_0_y * vj_2_y + vi_0_z * vj_2_z  # (0,2) CaN·CaC
        dot_3 = vi_1_x * vj_1_x + vi_1_y * vj_1_y + vi_1_z * vj_1_z  # (1,1) CaCb·CaCb
        dot_4 = vi_1_x * vj_2_x + vi_1_y * vj_2_y + vi_1_z * vj_2_z  # (1,2) CaCb·CaC
        dot_5 = vi_2_x * vj_2_x + vi_2_y * vj_2_y + vi_2_z * vj_2_z  # (2,2) CaC·CaC

        # bin indices (targets) for each of 6 dot products
        bin_0 = tl.minimum(tl.maximum((dot_0 - min_val) / bin_width, 0.0), num_bins - 1.0).to(tl.int32)
        bin_1 = tl.minimum(tl.maximum((dot_1 - min_val) / bin_width, 0.0), num_bins - 1.0).to(tl.int32)
        bin_2 = tl.minimum(tl.maximum((dot_2 - min_val) / bin_width, 0.0), num_bins - 1.0).to(tl.int32)
        bin_3 = tl.minimum(tl.maximum((dot_3 - min_val) / bin_width, 0.0), num_bins - 1.0).to(tl.int32)
        bin_4 = tl.minimum(tl.maximum((dot_4 - min_val) / bin_width, 0.0), num_bins - 1.0).to(tl.int32)
        bin_5 = tl.minimum(tl.maximum((dot_5 - min_val) / bin_width, 0.0), num_bins - 1.0).to(tl.int32)

        # FFN input: cat(q[i]*k[j], q[i]-k[j])
        qk_mul = qi * kj
        qk_sub = qi - kj

        # FFN forward (all fp32, using element-wise ops for matrix-vector products)
        # hidden = relu(qk_mul @ fc1_first.T + qk_sub @ fc1_second.T)
        # where fc1_first.T[i, h] = fc1_first[h, i]
        qk_mul_masked = tl.where(d_mask, qk_mul, 0.0)
        qk_sub_masked = tl.where(d_mask, qk_sub, 0.0)

        # hidden[h] = sum_i(qk_mul[i] * fc1_first[h, i]) + sum_i(qk_sub[i] * fc1_second[h, i])
        #           = sum_i(fc1_first[h, i] * qk_mul[i]) + ...
        # Using element-wise: fc1_first * qk_mul[None, :] then sum over axis=1
        hidden_pre = (tl.sum(fc1_first * qk_mul_masked[None, :], axis=1) +
                      tl.sum(fc1_second * qk_sub_masked[None, :], axis=1))
        hidden_pre = tl.where(h_mask, hidden_pre, 0.0)
        hidden = tl.maximum(hidden_pre, 0.0)  # relu
        relu_mask = (hidden_pre > 0) & h_mask
        hidden_masked = tl.where(h_mask, hidden, 0.0)

        # compute logits for each of 6 outputs: out_p[b] = sum_h(hidden[h] * fc2_p[b, h])
        # fc2_p is (num_bins, d_hidden), hidden is (d_hidden,)
        out_0 = tl.sum(fc2_0 * hidden_masked[None, :], axis=1)
        out_1 = tl.sum(fc2_1 * hidden_masked[None, :], axis=1)
        out_2 = tl.sum(fc2_2 * hidden_masked[None, :], axis=1)
        out_3 = tl.sum(fc2_3 * hidden_masked[None, :], axis=1)
        out_4 = tl.sum(fc2_4 * hidden_masked[None, :], axis=1)
        out_5 = tl.sum(fc2_5 * hidden_masked[None, :], axis=1)

        out_0 = tl.where(b_mask, out_0, 0.0)
        out_1 = tl.where(b_mask, out_1, 0.0)
        out_2 = tl.where(b_mask, out_2, 0.0)
        out_3 = tl.where(b_mask, out_3, 0.0)
        out_4 = tl.where(b_mask, out_4, 0.0)
        out_5 = tl.where(b_mask, out_5, 0.0)

        # compute softmax and cross entropy for each output, accumulate loss and grads
        loss_contrib = 0.0
        d_hidden_acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        # --- output 0 ---
        out_max = tl.max(tl.where(b_mask, out_0, -1e9))
        out_shifted = out_0 - out_max
        out_exp = tl.exp(out_shifted) * b_mask
        out_sum = tl.sum(out_exp) + 1e-8
        log_sum_exp = tl.log(out_sum)
        log_probs = out_shifted - log_sum_exp
        probs = out_exp / out_sum
        log_prob_target = tl.sum(tl.where(offs_b == bin_0, log_probs, 0.0))
        loss_contrib += -log_prob_target
        d_out_0 = probs - tl.where(offs_b == bin_0, 1.0, 0.0)
        d_out_0 = tl.where(b_mask, d_out_0, 0.0)
        # d_hidden[h] = sum_b(d_out[b] * fc2[b, h]) using element-wise
        d_hidden_acc += tl.sum(fc2_0 * d_out_0[:, None], axis=0)
        d_fc2_0_acc += d_out_0[:, None] * hidden_masked[None, :]

        # --- output 1 ---
        out_max = tl.max(tl.where(b_mask, out_1, -1e9))
        out_shifted = out_1 - out_max
        out_exp = tl.exp(out_shifted) * b_mask
        out_sum = tl.sum(out_exp) + 1e-8
        log_sum_exp = tl.log(out_sum)
        log_probs = out_shifted - log_sum_exp
        probs = out_exp / out_sum
        log_prob_target = tl.sum(tl.where(offs_b == bin_1, log_probs, 0.0))
        loss_contrib += -log_prob_target
        d_out_1 = probs - tl.where(offs_b == bin_1, 1.0, 0.0)
        d_out_1 = tl.where(b_mask, d_out_1, 0.0)
        d_hidden_acc += tl.sum(fc2_1 * d_out_1[:, None], axis=0)
        d_fc2_1_acc += d_out_1[:, None] * hidden_masked[None, :]

        # --- output 2 ---
        out_max = tl.max(tl.where(b_mask, out_2, -1e9))
        out_shifted = out_2 - out_max
        out_exp = tl.exp(out_shifted) * b_mask
        out_sum = tl.sum(out_exp) + 1e-8
        log_sum_exp = tl.log(out_sum)
        log_probs = out_shifted - log_sum_exp
        probs = out_exp / out_sum
        log_prob_target = tl.sum(tl.where(offs_b == bin_2, log_probs, 0.0))
        loss_contrib += -log_prob_target
        d_out_2 = probs - tl.where(offs_b == bin_2, 1.0, 0.0)
        d_out_2 = tl.where(b_mask, d_out_2, 0.0)
        d_hidden_acc += tl.sum(fc2_2 * d_out_2[:, None], axis=0)
        d_fc2_2_acc += d_out_2[:, None] * hidden_masked[None, :]

        # --- output 3 ---
        out_max = tl.max(tl.where(b_mask, out_3, -1e9))
        out_shifted = out_3 - out_max
        out_exp = tl.exp(out_shifted) * b_mask
        out_sum = tl.sum(out_exp) + 1e-8
        log_sum_exp = tl.log(out_sum)
        log_probs = out_shifted - log_sum_exp
        probs = out_exp / out_sum
        log_prob_target = tl.sum(tl.where(offs_b == bin_3, log_probs, 0.0))
        loss_contrib += -log_prob_target
        d_out_3 = probs - tl.where(offs_b == bin_3, 1.0, 0.0)
        d_out_3 = tl.where(b_mask, d_out_3, 0.0)
        d_hidden_acc += tl.sum(fc2_3 * d_out_3[:, None], axis=0)
        d_fc2_3_acc += d_out_3[:, None] * hidden_masked[None, :]

        # --- output 4 ---
        out_max = tl.max(tl.where(b_mask, out_4, -1e9))
        out_shifted = out_4 - out_max
        out_exp = tl.exp(out_shifted) * b_mask
        out_sum = tl.sum(out_exp) + 1e-8
        log_sum_exp = tl.log(out_sum)
        log_probs = out_shifted - log_sum_exp
        probs = out_exp / out_sum
        log_prob_target = tl.sum(tl.where(offs_b == bin_4, log_probs, 0.0))
        loss_contrib += -log_prob_target
        d_out_4 = probs - tl.where(offs_b == bin_4, 1.0, 0.0)
        d_out_4 = tl.where(b_mask, d_out_4, 0.0)
        d_hidden_acc += tl.sum(fc2_4 * d_out_4[:, None], axis=0)
        d_fc2_4_acc += d_out_4[:, None] * hidden_masked[None, :]

        # --- output 5 ---
        out_max = tl.max(tl.where(b_mask, out_5, -1e9))
        out_shifted = out_5 - out_max
        out_exp = tl.exp(out_shifted) * b_mask
        out_sum = tl.sum(out_exp) + 1e-8
        log_sum_exp = tl.log(out_sum)
        log_probs = out_shifted - log_sum_exp
        probs = out_exp / out_sum
        log_prob_target = tl.sum(tl.where(offs_b == bin_5, log_probs, 0.0))
        loss_contrib += -log_prob_target
        d_out_5 = probs - tl.where(offs_b == bin_5, 1.0, 0.0)
        d_out_5 = tl.where(b_mask, d_out_5, 0.0)
        d_hidden_acc += tl.sum(fc2_5 * d_out_5[:, None], axis=0)
        d_fc2_5_acc += d_out_5[:, None] * hidden_masked[None, :]

        # relu backward
        d_hidden_acc = tl.where(h_mask, d_hidden_acc, 0.0)
        grad_h = tl.where(relu_mask, d_hidden_acc, 0.0)

        # fc1 grads: d_fc1[h, i] = grad_h[h] * qk[i]
        d_fc1_first_local = grad_h[:, None] * qk_mul[None, :]
        d_fc1_second_local = grad_h[:, None] * qk_sub[None, :]

        # d_qk[i] = sum_h(grad_h[h] * fc1[h, i]) using element-wise
        grad_h_masked = tl.where(h_mask, grad_h, 0.0)
        d_qk_mul = tl.sum(fc1_first * grad_h_masked[:, None], axis=0)
        d_qk_sub = tl.sum(fc1_second * grad_h_masked[:, None], axis=0)
        d_qk_mul = tl.where(d_mask, d_qk_mul, 0.0)
        d_qk_sub = tl.where(d_mask, d_qk_sub, 0.0)

        # d_qi = d_qk_mul * kj + d_qk_sub, d_kj = d_qk_mul * qi - d_qk_sub
        d_qi_local = d_qk_mul * kj + d_qk_sub
        d_kj_local = d_qk_mul * qi - d_qk_sub

        # accumulate
        loss_acc += tl.where(valid, loss_contrib, 0.0)
        d_qi_acc += tl.where(valid, d_qi_local, 0.0)
        d_fc1_first_acc += tl.where(valid, d_fc1_first_local, 0.0)
        d_fc1_second_acc += tl.where(valid, d_fc1_second_local, 0.0)

        # atomic add d_kj to d_logits[j, 1, :] (normalized by inv_n_pairs)
        d_kj_ptr = d_logits_ptr + j * stride_dlogits_zn + 1 * stride_dlogits_2 + offs_d * stride_dlogits_d
        d_kj_norm = tl.where(valid, d_kj_local * inv_n_pairs, 0.0)
        tl.atomic_add(d_kj_ptr, d_kj_norm, mask=d_mask)

    # normalize and write (use inv_n_pairs computed from cu_seqlens)
    loss_acc = loss_acc * inv_n_pairs
    d_qi_acc = d_qi_acc * inv_n_pairs
    d_fc1_first_acc = d_fc1_first_acc * inv_n_pairs
    d_fc1_second_acc = d_fc1_second_acc * inv_n_pairs
    d_fc2_0_acc = d_fc2_0_acc * inv_n_pairs
    d_fc2_1_acc = d_fc2_1_acc * inv_n_pairs
    d_fc2_2_acc = d_fc2_2_acc * inv_n_pairs
    d_fc2_3_acc = d_fc2_3_acc * inv_n_pairs
    d_fc2_4_acc = d_fc2_4_acc * inv_n_pairs
    d_fc2_5_acc = d_fc2_5_acc * inv_n_pairs

    # store per-token loss
    tl.store(per_token_loss_ptr + pid, loss_acc)

    # atomic add d_qi to d_logits[i, 0, :]
    d_qi_ptr = d_logits_ptr + pid * stride_dlogits_zn + 0 * stride_dlogits_2 + offs_d * stride_dlogits_d
    tl.atomic_add(d_qi_ptr, d_qi_acc, mask=d_mask)

    # atomic add d_fc1 (first half: cols 0..d_in-1, second half: cols d_in..2*d_in-1)
    fc1_grad_mask = (offs_h[:, None] < d_hidden) & (offs_d[None, :] < d_in)
    d_fc1_first_ptr = d_fc1_ptr + offs_h[:, None] * stride_dfc1_hidden + offs_d[None, :] * stride_dfc1_in
    tl.atomic_add(d_fc1_first_ptr, d_fc1_first_acc, mask=fc1_grad_mask)
    d_fc1_second_ptr = d_fc1_ptr + offs_h[:, None] * stride_dfc1_hidden + (d_in + offs_d[None, :]) * stride_dfc1_in
    tl.atomic_add(d_fc1_second_ptr, d_fc1_second_acc, mask=fc1_grad_mask)

    # atomic add d_fc2 for each of 6 outputs
    fc2_grad_mask = (offs_b[:, None] < num_bins) & (offs_h[None, :] < d_hidden)
    d_fc2_0_ptr = d_fc2_ptr + (0 * num_bins + offs_b[:, None]) * stride_dfc2_out + offs_h[None, :] * stride_dfc2_hidden
    tl.atomic_add(d_fc2_0_ptr, d_fc2_0_acc, mask=fc2_grad_mask)
    d_fc2_1_ptr = d_fc2_ptr + (1 * num_bins + offs_b[:, None]) * stride_dfc2_out + offs_h[None, :] * stride_dfc2_hidden
    tl.atomic_add(d_fc2_1_ptr, d_fc2_1_acc, mask=fc2_grad_mask)
    d_fc2_2_ptr = d_fc2_ptr + (2 * num_bins + offs_b[:, None]) * stride_dfc2_out + offs_h[None, :] * stride_dfc2_hidden
    tl.atomic_add(d_fc2_2_ptr, d_fc2_2_acc, mask=fc2_grad_mask)
    d_fc2_3_ptr = d_fc2_ptr + (3 * num_bins + offs_b[:, None]) * stride_dfc2_out + offs_h[None, :] * stride_dfc2_hidden
    tl.atomic_add(d_fc2_3_ptr, d_fc2_3_acc, mask=fc2_grad_mask)
    d_fc2_4_ptr = d_fc2_ptr + (4 * num_bins + offs_b[:, None]) * stride_dfc2_out + offs_h[None, :] * stride_dfc2_hidden
    tl.atomic_add(d_fc2_4_ptr, d_fc2_4_acc, mask=fc2_grad_mask)
    d_fc2_5_ptr = d_fc2_ptr + (5 * num_bins + offs_b[:, None]) * stride_dfc2_out + offs_h[None, :] * stride_dfc2_hidden
    tl.atomic_add(d_fc2_5_ptr, d_fc2_5_acc, mask=fc2_grad_mask)


def anglogram_loss(
    logits: Float[T, "ZN 2 d_in"],
    fc1: Float[T, "d_hidden 2*d_in"],
    fc2: Float[T, "6*num_bins d_hidden"],
    unit_vecs: Float[T, "ZN 3 3"],
    cu_seqlens: Int[T, "Z+1"],
) -> Float[T, "1"]:
    return AnglogramLoss.apply(logits, fc1, fc2, unit_vecs, cu_seqlens)


class AnglogramLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        logits: Float[T, "ZN 2 d_in"],
        fc1: Float[T, "d_hidden 2*d_in"],
        fc2: Float[T, "6*num_bins d_hidden"],
        unit_vecs: Float[T, "ZN 3 3"],
        cu_seqlens: Int[T, "Z+1"],
    ) -> Float[T, "1"]:
        # shapes
        ZN, _, d_in = logits.shape
        d_hidden, _ = fc1.shape
        d_out, _ = fc2.shape
        num_bins = d_out // 6
        Z = cu_seqlens.shape[0] - 1

        # assertions
        assert unit_vecs.shape == (ZN, 3, 3), f"unit_vecs {unit_vecs.shape} != ({ZN}, 3, 3)"
        assert logits.shape == (ZN, 2, d_in), f"logits {logits.shape} != ({ZN}, 2, {d_in})"
        assert fc1.shape == (d_hidden, 2 * d_in), f"fc1 {fc1.shape} != ({d_hidden}, {2 * d_in})"
        assert fc2.shape == (d_out, d_hidden), f"fc2 {fc2.shape} != ({d_out}, {d_hidden})"
        assert d_out % 6 == 0, f"fc2 d_out={d_out} must be divisible by 6"
        assert cu_seqlens.shape == (Z + 1,), f"cu_seqlens {cu_seqlens.shape} != ({Z + 1},)"
        assert unit_vecs.is_cuda and logits.is_cuda and fc1.is_cuda and fc2.is_cuda and cu_seqlens.is_cuda

        # get orig dtypes
        unit_vecs_dtype = unit_vecs.dtype
        logits_dtype = logits.dtype
        fc1_dtype = fc1.dtype
        fc2_dtype = fc2.dtype

        # all computation in fp32
        unit_vecs = unit_vecs.to(torch.float32).contiguous()
        logits = logits.to(torch.float32).contiguous()
        cu_seqlens = cu_seqlens.to(torch.int32).contiguous()
        fc1 = fc1.to(torch.float32).contiguous()
        fc2 = fc2.to(torch.float32).contiguous()

        # allocate outputs (fp32)
        per_token_loss = torch.zeros(ZN, device=unit_vecs.device, dtype=torch.float32)
        d_unit_vecs = torch.zeros(ZN, 3, 3, device=unit_vecs.device, dtype=torch.float32)
        d_logits = torch.zeros(ZN, 2, d_in, device=unit_vecs.device, dtype=torch.float32)
        d_fc1 = torch.zeros(d_hidden, 2 * d_in, device=unit_vecs.device, dtype=torch.float32)
        d_fc2 = torch.zeros(d_out, d_hidden, device=unit_vecs.device, dtype=torch.float32)

        # block sizes - tl.dot requires K >= 16
        BLOCK_D = max(16, triton.next_power_of_2(max(d_in, d_hidden)))
        BLOCK_BINS = max(16, triton.next_power_of_2(num_bins))
        grid = (ZN,)

        # launch kernel
        anglogram_loss_fwd_bwd[grid](
            # inputs
            logits, fc1, fc2, unit_vecs, cu_seqlens,
            # outputs
            per_token_loss, d_logits, d_fc1, d_fc2, d_unit_vecs,
            # shapes
            ZN, Z, d_in, d_hidden, num_bins,
            # strides - logits
            logits.stride(0), logits.stride(1), logits.stride(2),
            # strides - fc1
            fc1.stride(0), fc1.stride(1),
            # strides - fc2
            fc2.stride(0), fc2.stride(1),
            # strides - unit_vecs
            unit_vecs.stride(0), unit_vecs.stride(1), unit_vecs.stride(2),
            # strides - d_logits
            d_logits.stride(0), d_logits.stride(1), d_logits.stride(2),
            # strides - d_fc1
            d_fc1.stride(0), d_fc1.stride(1),
            # strides - d_fc2
            d_fc2.stride(0), d_fc2.stride(1),
            # strides - d_unit_vecs
            d_unit_vecs.stride(0), d_unit_vecs.stride(1), d_unit_vecs.stride(2),
            # block sizes
            BLOCK_D, BLOCK_BINS,
        )

        # save grads for backward
        ctx.save_for_backward(d_unit_vecs, d_logits, d_fc1, d_fc2)
        ctx.dtypes = (unit_vecs_dtype, logits_dtype, fc1_dtype, fc2_dtype)

        return per_token_loss.sum()

    @staticmethod
    def backward(ctx, grad_output):
        d_unit_vecs, d_logits, d_fc1, d_fc2 = ctx.saved_tensors
        unit_vecs_dtype, logits_dtype, fc1_dtype, fc2_dtype = ctx.dtypes

        # scale by upstream grad and cast back to orig dtypes
        d_unit_vecs = (d_unit_vecs * grad_output).to(unit_vecs_dtype)
        d_logits = (d_logits * grad_output).to(logits_dtype)
        d_fc1 = (d_fc1 * grad_output).to(fc1_dtype)
        d_fc2 = (d_fc2 * grad_output).to(fc2_dtype)

        # None for cu_seqlens
        return d_logits, d_fc1, d_fc2, d_unit_vecs, None
