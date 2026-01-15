import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from proteindiff.training.losses.struct_losses.anglogram_loss import (
    anglogram_loss,
)


def anglogram_loss_ref(
    logits: torch.Tensor,
    fc1: torch.Tensor,
    fc2: torch.Tensor,
    unit_vecs: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Reference impl using nn.CrossEntropyLoss(reduction='none').

    Bins dot product of unit vectors from -1 to 1.
    Reduction: mean over pairs per token (dim1), then sum over tokens (dim0).
    """
    ZN, _, d_in = logits.shape
    d_out, d_hidden = fc2.shape
    Z = cu_seqlens.shape[0] - 1
    min_val, max_val = -1.0, 1.0
    bin_width = (max_val - min_val) / d_out

    q = logits[:, 0]  # (ZN, d_in)
    k = logits[:, 1]  # (ZN, d_in)

    # build pairwise FFN inputs: qk_mul[i,j] = q[i] * k[j], qk_sub[i,j] = q[i] - k[j]
    qk_mul = q[:, None, :] * k[None, :, :]  # (ZN, ZN, d_in)
    qk_sub = q[:, None, :] - k[None, :, :]  # (ZN, ZN, d_in)
    ffn_in = torch.cat([qk_mul, qk_sub], dim=-1)  # (ZN, ZN, 2*d_in)

    # FFN forward: hidden = relu(ffn_in @ fc1.T), out = hidden @ fc2.T
    hidden = F.relu(ffn_in @ fc1.T)  # (ZN, ZN, d_hidden)
    out_logits = hidden @ fc2.T  # (ZN, ZN, d_out)

    # pairwise dot products and bin indices
    dot = (unit_vecs[:, None, :] * unit_vecs[None, :, :]).sum(dim=-1)  # (ZN, ZN)
    bin_idx = torch.clamp((dot - min_val) / bin_width, 0, d_out - 1).long()

    # cross entropy loss per pair
    ce = nn.CrossEntropyLoss(reduction="none")
    loss_matrix = ce(out_logits.view(-1, d_out), bin_idx.view(-1)).view(ZN, ZN)

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
    """Generate random unit vectors."""
    vecs = torch.randn(n, 3, device=device, dtype=dtype)
    return F.normalize(vecs, dim=-1)


class TestAnglogramLossSingleSequence:
    """Test with single sequence."""

    def test_forward_fp16(self, device):
        torch.manual_seed(42)
        ZN, d_in, d_hidden, d_out = 8, 32, 32, 32

        unit_vecs = rand_unit_vecs(ZN, device, torch.float32)
        logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float16)
        fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float16)
        fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float16)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        # reference in fp32
        loss_ref = anglogram_loss_ref(
            logits.float(), fc1.float(), fc2.float(), unit_vecs, cu_seqlens
        )
        loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri.float(), atol=0.5, rtol=0.1), (
            f"fp16 loss mismatch: ref={loss_ref.item():.4f}, tri={loss_tri.item():.4f}"
        )

    def test_backward_fp16(self, device):
        torch.manual_seed(42)
        ZN, d_in, d_hidden, d_out = 6, 16, 16, 16

        unit_vecs = rand_unit_vecs(ZN, device, torch.float32)
        logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float16)
        fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float16)
        fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float16)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        # reference in fp32
        unit_vecs_ref = unit_vecs.clone().requires_grad_(True)
        logits_ref = logits.float().requires_grad_(True)
        fc1_ref = fc1.float().requires_grad_(True)
        fc2_ref = fc2.float().requires_grad_(True)
        loss_ref = anglogram_loss_ref(logits_ref, fc1_ref, fc2_ref, unit_vecs_ref, cu_seqlens)
        loss_ref.backward()

        # triton fp16
        unit_vecs_tri = unit_vecs.clone().requires_grad_(True)
        logits_tri = logits.clone().requires_grad_(True)
        fc1_tri = fc1.clone().requires_grad_(True)
        fc2_tri = fc2.clone().requires_grad_(True)
        loss_tri = anglogram_loss(logits_tri, fc1_tri, fc2_tri, unit_vecs_tri, cu_seqlens)
        loss_tri.backward()

        # compare grads with looser tolerance
        assert torch.allclose(
            logits_ref.grad, logits_tri.grad.float(), atol=0.5, rtol=0.1
        ), f"fp16 logits.grad max_diff={(logits_ref.grad - logits_tri.grad.float()).abs().max().item():.4f}"
        assert torch.allclose(
            fc1_ref.grad, fc1_tri.grad.float(), atol=0.5, rtol=0.1
        ), f"fp16 fc1.grad max_diff={(fc1_ref.grad - fc1_tri.grad.float()).abs().max().item():.4f}"
        assert torch.allclose(
            fc2_ref.grad, fc2_tri.grad.float(), atol=0.5, rtol=0.1
        ), f"fp16 fc2.grad max_diff={(fc2_ref.grad - fc2_tri.grad.float()).abs().max().item():.4f}"


class TestAnglogramLossMultiSequence:
    """Test with multiple sequences."""

    def test_forward_two_sequences(self, device):
        torch.manual_seed(123)
        seq_lens = [5, 7]
        ZN = sum(seq_lens)
        d_in, d_hidden, d_out = 24, 24, 24

        unit_vecs = rand_unit_vecs(ZN, device, torch.float32)
        logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float16)
        fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float16)
        fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float16)
        cu_seqlens = torch.tensor([0, seq_lens[0], ZN], device=device, dtype=torch.int32)

        loss_ref = anglogram_loss_ref(
            logits.float(), fc1.float(), fc2.float(), unit_vecs, cu_seqlens
        )
        loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri.float(), atol=0.5, rtol=0.1), (
            f"fp16 loss mismatch: ref={loss_ref.item():.4f}, tri={loss_tri.item():.4f}"
        )

    def test_backward_three_sequences(self, device):
        torch.manual_seed(456)
        seq_lens = [3, 5, 4]
        ZN = sum(seq_lens)
        d_in, d_hidden, d_out = 16, 16, 16

        unit_vecs = rand_unit_vecs(ZN, device, torch.float32)
        logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float16)
        fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float16)
        fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float16)
        cu_seqlens = torch.tensor(
            [0, seq_lens[0], seq_lens[0] + seq_lens[1], ZN],
            device=device,
            dtype=torch.int32,
        )

        # reference in fp32
        unit_vecs_ref = unit_vecs.clone().requires_grad_(True)
        logits_ref = logits.float().requires_grad_(True)
        fc1_ref = fc1.float().requires_grad_(True)
        fc2_ref = fc2.float().requires_grad_(True)
        loss_ref = anglogram_loss_ref(logits_ref, fc1_ref, fc2_ref, unit_vecs_ref, cu_seqlens)
        loss_ref.backward()

        # triton fp16
        unit_vecs_tri = unit_vecs.clone().requires_grad_(True)
        logits_tri = logits.clone().requires_grad_(True)
        fc1_tri = fc1.clone().requires_grad_(True)
        fc2_tri = fc2.clone().requires_grad_(True)
        loss_tri = anglogram_loss(logits_tri, fc1_tri, fc2_tri, unit_vecs_tri, cu_seqlens)
        loss_tri.backward()

        assert torch.allclose(loss_ref, loss_tri.float(), atol=0.5, rtol=0.1)
        assert torch.allclose(logits_ref.grad, logits_tri.grad.float(), atol=0.5, rtol=0.1)
        assert torch.allclose(fc1_ref.grad, fc1_tri.grad.float(), atol=0.5, rtol=0.1)
        assert torch.allclose(fc2_ref.grad, fc2_tri.grad.float(), atol=0.5, rtol=0.1)


class TestAnglogramLossEdgeCases:
    """Edge cases."""

    def test_seq_len_2(self, device):
        """Minimal case: each token has exactly 1 neighbor."""
        torch.manual_seed(789)
        ZN, d_in, d_hidden, d_out = 2, 16, 16, 16

        unit_vecs = rand_unit_vecs(ZN, device, torch.float32)
        logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float16)
        fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float16)
        fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float16)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        loss_ref = anglogram_loss_ref(
            logits.float(), fc1.float(), fc2.float(), unit_vecs, cu_seqlens
        )
        loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri.float(), atol=0.5, rtol=0.1), (
            f"fp16 loss mismatch: ref={loss_ref.item():.4f}, tri={loss_tri.item():.4f}"
        )

    def test_seq_len_1(self, device):
        """Single token has 1 self-pair."""
        torch.manual_seed(999)
        ZN, d_in, d_hidden, d_out = 1, 16, 16, 16

        unit_vecs = rand_unit_vecs(ZN, device, torch.float32)
        logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float16)
        fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float16)
        fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float16)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        loss_ref = anglogram_loss_ref(
            logits.float(), fc1.float(), fc2.float(), unit_vecs, cu_seqlens
        )
        loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri.float(), atol=0.5, rtol=0.1), (
            f"fp16 loss mismatch: ref={loss_ref.item():.4f}, tri={loss_tri.item():.4f}"
        )

    def test_multiple_single_token_sequences(self, device):
        """Multiple sequences each with 1 token (each has self-pair)."""
        torch.manual_seed(111)
        ZN, d_in, d_hidden, d_out = 3, 16, 16, 16

        unit_vecs = rand_unit_vecs(ZN, device, torch.float32)
        logits = torch.randn(ZN, 2, d_in, device=device, dtype=torch.float16)
        fc1 = torch.randn(d_hidden, 2 * d_in, device=device, dtype=torch.float16)
        fc2 = torch.randn(d_out, d_hidden, device=device, dtype=torch.float16)
        cu_seqlens = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)

        loss_ref = anglogram_loss_ref(
            logits.float(), fc1.float(), fc2.float(), unit_vecs, cu_seqlens
        )
        loss_tri = anglogram_loss(logits, fc1, fc2, unit_vecs, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri.float(), atol=0.5, rtol=0.1), (
            f"fp16 loss mismatch: ref={loss_ref.item():.4f}, tri={loss_tri.item():.4f}"
        )
