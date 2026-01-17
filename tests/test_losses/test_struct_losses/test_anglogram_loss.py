import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from proteindiff.training.losses.struct_losses.anglogram_loss import (
    anglogram_loss,
)

# 6 non-redundant dot product indices: (vec_i_idx, vec_j_idx)
# unit_vecs[i] = [CaN, CaCb, CaC] indexed as [0, 1, 2]
DOT_PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def anglogram_loss_ref(
    logits: torch.Tensor,
    fc1: torch.Tensor,
    fc2: torch.Tensor,
    unit_vecs: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Reference impl for 6 non-redundant dot products.

    unit_vecs: (ZN, 3, 3) - stacks [CaN, CaCb, CaC] per token
    fc2: (6*num_bins, d_hidden) - outputs 6 bin distributions

    Bins dot product of unit vectors from -1 to 1.
    Reduction: mean over pairs per token (dim1), then sum over tokens (dim0).
    """
    ZN, _, d_in = logits.shape
    d_out, d_hidden = fc2.shape
    num_bins = d_out // 6
    Z = cu_seqlens.shape[0] - 1
    min_val, max_val = -1.0, 1.0
    bin_width = (max_val - min_val) / num_bins

    q = logits[:, 0]  # (ZN, d_in)
    k = logits[:, 1]  # (ZN, d_in)

    # build pairwise FFN inputs: qk_mul[i,j] = q[i] * k[j], qk_sub[i,j] = q[i] - k[j]
    qk_mul = q[:, None, :] * k[None, :, :]  # (ZN, ZN, d_in)
    qk_sub = q[:, None, :] - k[None, :, :]  # (ZN, ZN, d_in)
    ffn_in = torch.cat([qk_mul, qk_sub], dim=-1)  # (ZN, ZN, 2*d_in)

    # FFN forward: hidden = relu(ffn_in @ fc1.T), out = hidden @ fc2.T
    hidden = F.relu(ffn_in @ fc1.T)  # (ZN, ZN, d_hidden)
    out_logits = hidden @ fc2.T  # (ZN, ZN, 6*num_bins)

    # reshape to (ZN, ZN, 6, num_bins)
    out_logits = out_logits.view(ZN, ZN, 6, num_bins)

    # compute 6 dot products and bin indices
    # unit_vecs: (ZN, 3, 3) where dim1 is [CaN, CaCb, CaC] and dim2 is xyz
    dots = []
    for p, (vi, vj) in enumerate(DOT_PAIRS):
        # dot product: unit_vecs[i, vi, :] dot unit_vecs[j, vj, :]
        vec_i = unit_vecs[:, vi, :]  # (ZN, 3)
        vec_j = unit_vecs[:, vj, :]  # (ZN, 3)
        dot = (vec_i[:, None, :] * vec_j[None, :, :]).sum(dim=-1)  # (ZN, ZN)
        dots.append(dot)
    dots = torch.stack(dots, dim=-1)  # (ZN, ZN, 6)

    # bin indices
    bin_idx = torch.clamp((dots - min_val) / bin_width, 0, num_bins - 1).long()  # (ZN, ZN, 6)

    # cross entropy loss per pair, summed over 6 dot products
    ce = nn.CrossEntropyLoss(reduction="none")
    loss_matrix = torch.zeros(ZN, ZN, device=unit_vecs.device, dtype=torch.float32)
    for p in range(6):
        loss_p = ce(out_logits[:, :, p, :].reshape(-1, num_bins), bin_idx[:, :, p].reshape(-1))
        loss_matrix = loss_matrix + loss_p.view(ZN, ZN)

    # build mask: valid pairs are (i, j in same sequence), includes self i==j
    mask = torch.zeros(ZN, ZN, device=unit_vecs.device, dtype=torch.bool)
    for s in range(Z):
        start, end = cu_seqlens[s].item(), cu_seqlens[s + 1].item()
        mask[start:end, start:end] = True

    # apply mask
    loss_matrix = loss_matrix * mask.float()

    # count valid pairs per token (for mean)
    n_pairs = mask.sum(dim=1).float().clamp(min=1.0)

    # mean over dim1 (pairs per token), sum over dim0 (tokens)
    per_token_loss = loss_matrix.sum(dim=1) / n_pairs
    total_loss = per_token_loss.sum()

    return total_loss


@pytest.fixture
def device():
    return torch.device("cuda")


def rand_unit_vecs(n, device, dtype=torch.float32):
    """Generate random unit vectors (n, 3)."""
    vecs = torch.randn(n, 3, device=device, dtype=dtype)
    return F.normalize(vecs, dim=-1)


def rand_unit_vecs_3x3(n, device, dtype=torch.float32):
    """Generate random unit vectors (n, 3, 3) - 3 unit vectors per token."""
    vecs = torch.randn(n, 3, 3, device=device, dtype=dtype)
    return F.normalize(vecs, dim=-1)


class TestAnglogramLossSingleSequence:
    """Test with single sequence."""

    def test_forward_fp32(self, device):
        torch.manual_seed(42)
        # disable tf32 for tighter numerical accuracy
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

        try:
            ZN, d_in, d_hidden, num_bins = 8, 32, 32, 32
            d_out = 6 * num_bins

            unit_vecs = rand_unit_vecs_3x3(ZN, device, torch.float32)
            logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float32)
            fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float32)
            fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float32)
            cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

            loss_ref = anglogram_loss_ref(logits, fc1, fc2, unit_vecs, cu_seqlens)
            loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

            assert torch.allclose(loss_ref, loss_tri, atol=1e-3, rtol=1e-3), (
                f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def test_backward_fp32(self, device):
        torch.manual_seed(42)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

        try:
            ZN, d_in, d_hidden, num_bins = 6, 16, 16, 16
            d_out = 6 * num_bins

            unit_vecs = rand_unit_vecs_3x3(ZN, device, torch.float32)
            logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float32)
            fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float32)
            fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float32)
            cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

            # reference
            logits_ref = logits.clone().requires_grad_(True)
            fc1_ref = fc1.clone().requires_grad_(True)
            fc2_ref = fc2.clone().requires_grad_(True)
            loss_ref = anglogram_loss_ref(logits_ref, fc1_ref, fc2_ref, unit_vecs, cu_seqlens)
            loss_ref.backward()

            # triton
            logits_tri = logits.clone().requires_grad_(True)
            fc1_tri = fc1.clone().requires_grad_(True)
            fc2_tri = fc2.clone().requires_grad_(True)
            loss_tri = anglogram_loss(logits_tri, fc1_tri, fc2_tri, unit_vecs, cu_seqlens)
            loss_tri.backward()

            assert torch.allclose(
                logits_ref.grad, logits_tri.grad, atol=1e-3, rtol=1e-3
            ), f"fp32 logits.grad max_diff={(logits_ref.grad - logits_tri.grad).abs().max().item():.6f}"
            assert torch.allclose(
                fc1_ref.grad, fc1_tri.grad, atol=1e-3, rtol=1e-3
            ), f"fp32 fc1.grad max_diff={(fc1_ref.grad - fc1_tri.grad).abs().max().item():.6f}"
            assert torch.allclose(
                fc2_ref.grad, fc2_tri.grad, atol=1e-3, rtol=1e-3
            ), f"fp32 fc2.grad max_diff={(fc2_ref.grad - fc2_tri.grad).abs().max().item():.6f}"
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


class TestAnglogramLossMultiSequence:
    """Test with multiple sequences."""

    def test_forward_two_sequences(self, device):
        torch.manual_seed(123)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

        try:
            seq_lens = [5, 7]
            ZN = sum(seq_lens)
            d_in, d_hidden, num_bins = 24, 24, 24
            d_out = 6 * num_bins

            unit_vecs = rand_unit_vecs_3x3(ZN, device, torch.float32)
            logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float32)
            fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float32)
            fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float32)
            cu_seqlens = torch.tensor([0, seq_lens[0], ZN], device=device, dtype=torch.int32)

            loss_ref = anglogram_loss_ref(logits, fc1, fc2, unit_vecs, cu_seqlens)
            loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

            assert torch.allclose(loss_ref, loss_tri, atol=1e-3, rtol=1e-3), (
                f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def test_backward_three_sequences(self, device):
        torch.manual_seed(456)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

        try:
            seq_lens = [3, 5, 4]
            ZN = sum(seq_lens)
            d_in, d_hidden, num_bins = 16, 16, 16
            d_out = 6 * num_bins

            unit_vecs = rand_unit_vecs_3x3(ZN, device, torch.float32)
            logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float32)
            fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float32)
            fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float32)
            cu_seqlens = torch.tensor(
                [0, seq_lens[0], seq_lens[0] + seq_lens[1], ZN],
                device=device,
                dtype=torch.int32,
            )

            # reference
            logits_ref = logits.clone().requires_grad_(True)
            fc1_ref = fc1.clone().requires_grad_(True)
            fc2_ref = fc2.clone().requires_grad_(True)
            loss_ref = anglogram_loss_ref(logits_ref, fc1_ref, fc2_ref, unit_vecs, cu_seqlens)
            loss_ref.backward()

            # triton
            logits_tri = logits.clone().requires_grad_(True)
            fc1_tri = fc1.clone().requires_grad_(True)
            fc2_tri = fc2.clone().requires_grad_(True)
            loss_tri = anglogram_loss(logits_tri, fc1_tri, fc2_tri, unit_vecs, cu_seqlens)
            loss_tri.backward()

            assert torch.allclose(loss_ref, loss_tri, atol=1e-3, rtol=1e-3)
            assert torch.allclose(logits_ref.grad, logits_tri.grad, atol=1e-3, rtol=1e-3)
            assert torch.allclose(fc1_ref.grad, fc1_tri.grad, atol=1e-3, rtol=1e-3)
            assert torch.allclose(fc2_ref.grad, fc2_tri.grad, atol=1e-3, rtol=1e-3)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32


class TestAnglogramLossEdgeCases:
    """Edge cases."""

    def test_seq_len_2(self, device):
        """Minimal case: each token has exactly 1 neighbor."""
        torch.manual_seed(789)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

        try:
            ZN, d_in, d_hidden, num_bins = 2, 16, 16, 16
            d_out = 6 * num_bins

            unit_vecs = rand_unit_vecs_3x3(ZN, device, torch.float32)
            logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float32)
            fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float32)
            fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float32)
            cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

            loss_ref = anglogram_loss_ref(logits, fc1, fc2, unit_vecs, cu_seqlens)
            loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

            assert torch.allclose(loss_ref, loss_tri, atol=1e-3, rtol=1e-3), (
                f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def test_seq_len_1(self, device):
        """Single token has 1 self-pair."""
        torch.manual_seed(999)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

        try:
            ZN, d_in, d_hidden, num_bins = 1, 16, 16, 16
            d_out = 6 * num_bins

            unit_vecs = rand_unit_vecs_3x3(ZN, device, torch.float32)
            logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float32)
            fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float32)
            fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float32)
            cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

            loss_ref = anglogram_loss_ref(logits, fc1, fc2, unit_vecs, cu_seqlens)
            loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

            assert torch.allclose(loss_ref, loss_tri, atol=1e-3, rtol=1e-3), (
                f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32

    def test_multiple_single_token_sequences(self, device):
        """Multiple sequences each with 1 token (each has self-pair)."""
        torch.manual_seed(111)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

        try:
            ZN, d_in, d_hidden, num_bins = 3, 16, 16, 16
            d_out = 6 * num_bins

            unit_vecs = rand_unit_vecs_3x3(ZN, device, torch.float32)
            logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float32)
            fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float32)
            fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float32)
            cu_seqlens = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)

            loss_ref = anglogram_loss_ref(logits, fc1, fc2, unit_vecs, cu_seqlens)
            loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

            assert torch.allclose(loss_ref, loss_tri, atol=1e-3, rtol=1e-3), (
                f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
