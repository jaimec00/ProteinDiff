import pytest
import torch

from proteus.training.losses.struct_losses.distance_loss import (
    distance_loss,
    DistanceLoss,
)


def distance_loss_ref(
    pred_coords: torch.Tensor,  # (ZN, num_atoms, 3)
    gt_coords: torch.Tensor,    # (ZN, num_atoms, 3)
    atom_mask: torch.Tensor,    # (ZN, num_atoms) bool - True = valid atom
    cu_seqlens: torch.Tensor,   # (Z+1,)
    clamp_max: float = 25.0,
) -> torch.Tensor:
    """Reference implementation using pure PyTorch.

    For each residue pair (i, j) in same sequence (including i == j):
      For each valid atom pair (a in i, b in j) where atom_mask[i,a] and atom_mask[j,b]:
        pred_d = ||pred_coords[i,a] - pred_coords[j,b]||
        gt_d = ||gt_coords[i,a] - gt_coords[j,b]||
        loss_pair = min((pred_d - gt_d)², clamp_max)

    Normalize: divide by valid_atom_pairs per residue i to get mean loss per atom pair
    """
    ZN, num_atoms, _ = pred_coords.shape
    Z = cu_seqlens.shape[0] - 1

    # Compute all pairwise atom distances
    # pred_coords: (ZN, A, 3) -> (ZN, A, 1, 1, 3)
    # pred_coords: (ZN, A, 3) -> (1, 1, ZN, A, 3)
    # diff: (ZN, A, ZN, A, 3)
    pred_diff = pred_coords[:, :, None, None, :] - pred_coords[None, None, :, :, :]
    gt_diff = gt_coords[:, :, None, None, :] - gt_coords[None, None, :, :, :]

    pred_dist = torch.sqrt((pred_diff ** 2).sum(dim=-1) + 1e-8)  # (ZN, A, ZN, A)
    gt_dist = torch.sqrt((gt_diff ** 2).sum(dim=-1) + 1e-8)

    # Clamped squared error
    sq_err = (pred_dist - gt_dist) ** 2
    clamped_loss = torch.clamp(sq_err, max=clamp_max)  # (ZN, A, ZN, A)

    # Build atom pair mask: (ZN, A, ZN, A)
    # atom_mask[i, a] & atom_mask[j, b] for all i, a, j, b
    atom_pair_mask = atom_mask[:, :, None, None] & atom_mask[None, None, :, :]
    clamped_loss = clamped_loss * atom_pair_mask.float()

    # Sum over atom pairs -> (ZN, ZN)
    loss_matrix = clamped_loss.sum(dim=(1, 3))  # sum over atoms a, b

    # Build mask for same-sequence pairs (including i == j)
    seq_mask = torch.zeros(ZN, ZN, device=pred_coords.device, dtype=torch.bool)
    for s in range(Z):
        start, end = cu_seqlens[s].item(), cu_seqlens[s + 1].item()
        seq_mask[start:end, start:end] = True

    loss_matrix = loss_matrix * seq_mask.float()

    # Count valid atom pairs per (i, j): sum over a, b of atom_pair_mask
    valid_pairs_per_ij = atom_pair_mask.float().sum(dim=(1, 3))  # (ZN, ZN)

    # Sum valid pairs for each residue i (across all j in same sequence)
    valid_pairs_per_i = (valid_pairs_per_ij * seq_mask.float()).sum(dim=1)  # (ZN,)

    # Normalize per residue: divide by valid_atom_pairs to get mean loss per atom pair
    norm_factor = valid_pairs_per_i.clamp(min=1.0)
    per_token_loss = loss_matrix.sum(dim=1) / norm_factor

    return per_token_loss.sum()


@pytest.fixture
def device():
    return torch.device("cuda")


class TestDistanceLossSingleSequence:
    """Test with single sequence."""

    def test_forward_fp32(self, device):
        torch.manual_seed(42)
        ZN, num_atoms = 8, 4

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        loss_ref = distance_loss_ref(pred_coords, gt_coords, atom_mask, cu_seqlens)
        loss_tri = distance_loss(pred_coords, gt_coords, atom_mask, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )

    def test_backward_fp32(self, device):
        torch.manual_seed(42)
        ZN, num_atoms = 6, 3

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        # reference
        pred_ref = pred_coords.clone().requires_grad_(True)
        loss_ref = distance_loss_ref(pred_ref, gt_coords, atom_mask, cu_seqlens)
        loss_ref.backward()

        # triton
        pred_tri = pred_coords.clone().requires_grad_(True)
        loss_tri = distance_loss(pred_tri, gt_coords, atom_mask, cu_seqlens)
        loss_tri.backward()

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )
        assert torch.allclose(pred_ref.grad, pred_tri.grad, atol=1e-3, rtol=1e-3), (
            f"fp32 grad max_diff={(pred_ref.grad - pred_tri.grad).abs().max().item():.6f}"
        )


class TestDistanceLossMultiSequence:
    """Test with multiple sequences."""

    def test_forward_two_sequences(self, device):
        torch.manual_seed(123)
        seq_lens = [5, 7]
        ZN = sum(seq_lens)
        num_atoms = 4

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, seq_lens[0], ZN], device=device, dtype=torch.int32)

        loss_ref = distance_loss_ref(pred_coords, gt_coords, atom_mask, cu_seqlens)
        loss_tri = distance_loss(pred_coords, gt_coords, atom_mask, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )

    def test_backward_three_sequences(self, device):
        torch.manual_seed(456)
        seq_lens = [3, 5, 4]
        ZN = sum(seq_lens)
        num_atoms = 3

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor(
            [0, seq_lens[0], seq_lens[0] + seq_lens[1], ZN],
            device=device,
            dtype=torch.int32,
        )

        # reference
        pred_ref = pred_coords.clone().requires_grad_(True)
        loss_ref = distance_loss_ref(pred_ref, gt_coords, atom_mask, cu_seqlens)
        loss_ref.backward()

        # triton
        pred_tri = pred_coords.clone().requires_grad_(True)
        loss_tri = distance_loss(pred_tri, gt_coords, atom_mask, cu_seqlens)
        loss_tri.backward()

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4)
        assert torch.allclose(pred_ref.grad, pred_tri.grad, atol=1e-3, rtol=1e-3)


class TestDistanceLossEdgeCases:
    """Edge cases."""

    def test_seq_len_2(self, device):
        """Minimal case: 2 tokens with self and neighbor pairs."""
        torch.manual_seed(789)
        ZN, num_atoms = 2, 2

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        loss_ref = distance_loss_ref(pred_coords, gt_coords, atom_mask, cu_seqlens)
        loss_tri = distance_loss(pred_coords, gt_coords, atom_mask, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )

    def test_seq_len_1(self, device):
        """Single token with self-pair only."""
        torch.manual_seed(999)
        ZN, num_atoms = 1, 3

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        loss_ref = distance_loss_ref(pred_coords, gt_coords, atom_mask, cu_seqlens)
        loss_tri = distance_loss(pred_coords, gt_coords, atom_mask, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )

    def test_multiple_single_token_sequences(self, device):
        """Multiple sequences each with 1 token - only self-pairs."""
        torch.manual_seed(111)
        ZN, num_atoms = 3, 2

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)

        loss_ref = distance_loss_ref(pred_coords, gt_coords, atom_mask, cu_seqlens)
        loss_tri = distance_loss(pred_coords, gt_coords, atom_mask, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )

    def test_num_atoms_1(self, device):
        """Single atom per residue - simplest case."""
        torch.manual_seed(222)
        ZN, num_atoms = 5, 1

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        loss_ref = distance_loss_ref(pred_coords, gt_coords, atom_mask, cu_seqlens)
        loss_tri = distance_loss(pred_coords, gt_coords, atom_mask, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"fp32 loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )


class TestDistanceLossGradientCheck:
    """Gradient checks comparing to reference implementation."""

    def test_gradient_vs_reference(self, device):
        """Compare Triton gradients to reference implementation gradients."""
        torch.manual_seed(333)
        ZN, num_atoms = 4, 3

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        # reference
        pred_ref = pred_coords.clone().requires_grad_(True)
        loss_ref = distance_loss_ref(pred_ref, gt_coords, atom_mask, cu_seqlens)
        loss_ref.backward()

        # triton
        pred_tri = pred_coords.clone().requires_grad_(True)
        loss_tri = distance_loss(pred_tri, gt_coords, atom_mask, cu_seqlens)
        loss_tri.backward()

        # compare gradients
        assert torch.allclose(pred_ref.grad, pred_tri.grad, atol=1e-3, rtol=1e-3), (
            f"Gradient mismatch: max_diff={(pred_ref.grad - pred_tri.grad).abs().max().item():.6f}"
        )


class TestDistanceLossWithAtomMask:
    """Tests with partial atom masks."""

    def test_partial_mask_forward(self, device):
        """Test with some atoms masked out."""
        torch.manual_seed(444)
        ZN, num_atoms = 6, 4

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        # Mask out last atom for first 3 residues, first atom for last 3 residues
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        atom_mask[:3, -1] = False
        atom_mask[3:, 0] = False
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        loss_ref = distance_loss_ref(pred_coords, gt_coords, atom_mask, cu_seqlens)
        loss_tri = distance_loss(pred_coords, gt_coords, atom_mask, cu_seqlens)

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"partial mask forward mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )

    def test_partial_mask_backward(self, device):
        """Test gradients with partial mask."""
        torch.manual_seed(555)
        ZN, num_atoms = 5, 3

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        # Mask out middle atom for all residues
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        atom_mask[:, 1] = False
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        # reference
        pred_ref = pred_coords.clone().requires_grad_(True)
        loss_ref = distance_loss_ref(pred_ref, gt_coords, atom_mask, cu_seqlens)
        loss_ref.backward()

        # triton
        pred_tri = pred_coords.clone().requires_grad_(True)
        loss_tri = distance_loss(pred_tri, gt_coords, atom_mask, cu_seqlens)
        loss_tri.backward()

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"partial mask loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )
        assert torch.allclose(pred_ref.grad, pred_tri.grad, atol=1e-3, rtol=1e-3), (
            f"partial mask grad max_diff={(pred_ref.grad - pred_tri.grad).abs().max().item():.6f}"
        )
        # Masked atoms should have zero gradient
        assert torch.all(pred_tri.grad[:, 1, :] == 0), "Masked atoms should have zero gradient"

    def test_varying_mask_per_residue(self, device):
        """Test with different number of valid atoms per residue."""
        torch.manual_seed(666)
        ZN, num_atoms = 4, 5

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        # Different valid atoms per residue: 5, 4, 3, 2
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        atom_mask[1, 4] = False  # residue 1 has 4 atoms
        atom_mask[2, 3:] = False  # residue 2 has 3 atoms
        atom_mask[3, 2:] = False  # residue 3 has 2 atoms
        cu_seqlens = torch.tensor([0, ZN], device=device, dtype=torch.int32)

        # reference
        pred_ref = pred_coords.clone().requires_grad_(True)
        loss_ref = distance_loss_ref(pred_ref, gt_coords, atom_mask, cu_seqlens)
        loss_ref.backward()

        # triton
        pred_tri = pred_coords.clone().requires_grad_(True)
        loss_tri = distance_loss(pred_tri, gt_coords, atom_mask, cu_seqlens)
        loss_tri.backward()

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"varying mask loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )
        assert torch.allclose(pred_ref.grad, pred_tri.grad, atol=1e-3, rtol=1e-3), (
            f"varying mask grad max_diff={(pred_ref.grad - pred_tri.grad).abs().max().item():.6f}"
        )

    def test_multi_sequence_with_mask(self, device):
        """Test multiple sequences with different masks."""
        torch.manual_seed(777)
        seq_lens = [3, 4]
        ZN = sum(seq_lens)
        num_atoms = 4

        pred_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        gt_coords = torch.randn(ZN, num_atoms, 3, device=device, dtype=torch.float32)
        atom_mask = torch.ones(ZN, num_atoms, device=device, dtype=torch.bool)
        # First sequence: only first 2 atoms valid
        atom_mask[:seq_lens[0], 2:] = False
        # Second sequence: only last 3 atoms valid
        atom_mask[seq_lens[0]:, 0] = False
        cu_seqlens = torch.tensor([0, seq_lens[0], ZN], device=device, dtype=torch.int32)

        # reference
        pred_ref = pred_coords.clone().requires_grad_(True)
        loss_ref = distance_loss_ref(pred_ref, gt_coords, atom_mask, cu_seqlens)
        loss_ref.backward()

        # triton
        pred_tri = pred_coords.clone().requires_grad_(True)
        loss_tri = distance_loss(pred_tri, gt_coords, atom_mask, cu_seqlens)
        loss_tri.backward()

        assert torch.allclose(loss_ref, loss_tri, atol=1e-4, rtol=1e-4), (
            f"multi-seq mask loss mismatch: ref={loss_ref.item():.6f}, tri={loss_tri.item():.6f}"
        )
        assert torch.allclose(pred_ref.grad, pred_tri.grad, atol=1e-3, rtol=1e-3), (
            f"multi-seq mask grad max_diff={(pred_ref.grad - pred_tri.grad).abs().max().item():.6f}"
        )
