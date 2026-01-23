"""Unit tests for proteus/utils/struct_utils.py - coordinate reconstruction functions."""

import pytest
import torch
import math

from proteus.utils.struct_utils import (
    gram_schmidt,
    normalize_torsions,
    torsion_to_frames,
    make_se3,
    torsion_angles_to_frames,
    frames_to_atom14_pos,
    coords_from_txy_sincos,
    _get_af2_tensor,
)


class TestGramSchmidt:
    """Tests for gram_schmidt orthonormalization."""

    def test_identity_from_canonical_vectors(self):
        """gram_schmidt([1,0,0], [0,1,0]) should produce identity rotation."""
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[0.0, 1.0, 0.0]])

        R = gram_schmidt(x, y)

        expected = torch.eye(3).unsqueeze(0)
        torch.testing.assert_close(R, expected, atol=1e-6, rtol=1e-6)

    def test_orthonormality(self):
        """Result should be orthonormal (R @ R.T = I)."""
        x = torch.randn(10, 3)
        y = torch.randn(10, 3)

        R = gram_schmidt(x, y)

        # R @ R.T should be identity for each sample
        RRT = torch.matmul(R, R.transpose(-1, -2))
        expected = torch.eye(3).unsqueeze(0).expand(10, -1, -1)
        torch.testing.assert_close(RRT, expected, atol=1e-5, rtol=1e-5)

    def test_determinant_is_one(self):
        """Rotation matrices should have determinant +1."""
        x = torch.randn(10, 3)
        y = torch.randn(10, 3)

        R = gram_schmidt(x, y)
        det = torch.linalg.det(R)

        torch.testing.assert_close(det, torch.ones(10), atol=1e-5, rtol=1e-5)

    def test_first_column_is_normalized_x(self):
        """First column should be normalized x."""
        x = torch.tensor([[3.0, 4.0, 0.0]])  # norm = 5
        y = torch.tensor([[0.0, 1.0, 0.0]])

        R = gram_schmidt(x, y)

        expected_e0 = torch.tensor([[0.6, 0.8, 0.0]])
        torch.testing.assert_close(R[:, :, 0], expected_e0, atol=1e-6, rtol=1e-6)

    def test_batch_processing(self):
        """Should handle batched inputs."""
        batch_size = 32
        x = torch.randn(batch_size, 3)
        y = torch.randn(batch_size, 3)

        R = gram_schmidt(x, y)

        assert R.shape == (batch_size, 3, 3)


class TestNormalizeTorsions:
    """Tests for normalize_torsions function."""

    def test_already_normalized(self):
        """sin^2 + cos^2 = 1 should remain unchanged."""
        angles = torch.tensor([0.0, math.pi / 4, math.pi / 2, math.pi])
        sin = torch.sin(angles).unsqueeze(0)
        cos = torch.cos(angles).unsqueeze(0)

        sin_n, cos_n = normalize_torsions(sin, cos)

        torch.testing.assert_close(sin_n, sin, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(cos_n, cos, atol=1e-6, rtol=1e-6)

    def test_scales_unnormalized(self):
        """Should scale to unit circle."""
        sin = torch.tensor([[0.6, 0.0]])
        cos = torch.tensor([[0.8, 2.0]])

        sin_n, cos_n = normalize_torsions(sin, cos)

        # Check they're on unit circle
        norm = torch.sqrt(sin_n**2 + cos_n**2)
        torch.testing.assert_close(norm, torch.ones_like(norm), atol=1e-6, rtol=1e-6)

    def test_preserves_ratio(self):
        """Should preserve sin/cos ratio (i.e., angle)."""
        sin = torch.tensor([[3.0]])
        cos = torch.tensor([[4.0]])

        sin_n, cos_n = normalize_torsions(sin, cos)

        # ratio should be preserved
        torch.testing.assert_close(sin_n / cos_n, sin / cos, atol=1e-6, rtol=1e-6)


class TestTorsionToFrames:
    """Tests for torsion_to_frames function."""

    def test_zero_angle(self):
        """Zero rotation (sin=0, cos=1) should give identity."""
        sin = torch.zeros(1, 4)
        cos = torch.ones(1, 4)

        R = torsion_to_frames(sin, cos)

        expected = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 4, -1, -1)
        torch.testing.assert_close(R, expected, atol=1e-6, rtol=1e-6)

    def test_90_degree_rotation(self):
        """90 degree rotation around x-axis."""
        sin = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # chi1 = 90 degrees
        cos = torch.tensor([[0.0, 1.0, 1.0, 1.0]])  # others = 0 degrees

        R = torsion_to_frames(sin, cos)

        # Chi1 should be [[1,0,0], [0,0,-1], [0,1,0]]
        expected_chi1 = torch.tensor([[[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]])
        torch.testing.assert_close(R[:, 0], expected_chi1, atol=1e-6, rtol=1e-6)

    def test_shape(self):
        """Should produce correct output shape."""
        sin = torch.randn(10, 4)
        cos = torch.randn(10, 4)

        R = torsion_to_frames(sin, cos)

        assert R.shape == (10, 4, 3, 3)

    def test_orthonormality(self):
        """Rotation matrices should be orthonormal."""
        sin = torch.randn(5, 4)
        cos = torch.randn(5, 4)
        sin, cos = normalize_torsions(sin, cos)

        R = torsion_to_frames(sin, cos)

        # Flatten to (20, 3, 3) for easier testing
        R_flat = R.reshape(-1, 3, 3)
        RRT = torch.matmul(R_flat, R_flat.transpose(-1, -2))
        expected = torch.eye(3).unsqueeze(0).expand(20, -1, -1)
        torch.testing.assert_close(RRT, expected, atol=1e-5, rtol=1e-5)


class TestMakeSE3:
    """Tests for make_se3 function."""

    def test_identity(self):
        """Identity rotation with zero translation."""
        R = torch.eye(3).unsqueeze(0)
        t = torch.zeros(1, 3)

        T = make_se3(R, t)

        expected = torch.eye(4).unsqueeze(0)
        torch.testing.assert_close(T, expected, atol=1e-6, rtol=1e-6)

    def test_translation_only(self):
        """Identity rotation with non-zero translation."""
        R = torch.eye(3).unsqueeze(0)
        t = torch.tensor([[1.0, 2.0, 3.0]])

        T = make_se3(R, t)

        assert T[0, 0, 3] == 1.0
        assert T[0, 1, 3] == 2.0
        assert T[0, 2, 3] == 3.0
        assert T[0, 3, 3] == 1.0

    def test_shape(self):
        """Should produce correct output shape."""
        R = torch.randn(10, 3, 3)
        t = torch.randn(10, 3)

        T = make_se3(R, t)

        assert T.shape == (10, 4, 4)

    def test_bottom_row(self):
        """Bottom row should be [0, 0, 0, 1]."""
        R = torch.randn(5, 3, 3)
        t = torch.randn(5, 3)

        T = make_se3(R, t)

        expected_bottom = torch.tensor([0., 0., 0., 1.])
        for i in range(5):
            torch.testing.assert_close(T[i, 3], expected_bottom, atol=1e-6, rtol=1e-6)


class TestTorsionAnglesToFrames:
    """Tests for torsion_angles_to_frames function."""

    def test_backbone_frame_preserved(self):
        """Group 0 should be exactly the backbone frame."""
        ZN = 5
        backbone_frame = torch.randn(ZN, 4, 4)
        backbone_frame[:, 3, :] = torch.tensor([0., 0., 0., 1.])

        chi_rot = torch.zeros(ZN, 4, 3, 3)
        for i in range(4):
            chi_rot[:, i] = torch.eye(3)

        default_frames = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(ZN, 8, -1, -1).clone()
        chi_mask = torch.ones(ZN, 4)

        all_frames = torsion_angles_to_frames(backbone_frame, chi_rot, default_frames, chi_mask)

        torch.testing.assert_close(all_frames[:, 0], backbone_frame, atol=1e-6, rtol=1e-6)

    def test_shape(self):
        """Should produce correct output shape."""
        ZN = 10
        backbone_frame = torch.eye(4).unsqueeze(0).expand(ZN, -1, -1).clone()
        chi_rot = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(ZN, 4, -1, -1).clone()
        default_frames = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(ZN, 8, -1, -1).clone()
        chi_mask = torch.ones(ZN, 4)

        all_frames = torsion_angles_to_frames(backbone_frame, chi_rot, default_frames, chi_mask)

        assert all_frames.shape == (ZN, 8, 4, 4)

    def test_chi_mask_zeros(self):
        """When chi_mask is 0, should not apply chi rotation."""
        ZN = 1
        backbone_frame = torch.eye(4).unsqueeze(0)

        # Non-identity chi rotation
        chi_rot = torch.zeros(ZN, 4, 3, 3)
        chi_rot[:, :, 0, 0] = 1
        chi_rot[:, :, 1, 1] = 0  # would be different if applied
        chi_rot[:, :, 1, 2] = -1
        chi_rot[:, :, 2, 1] = 1
        chi_rot[:, :, 2, 2] = 0

        default_frames = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(ZN, 8, -1, -1).clone()
        chi_mask = torch.zeros(ZN, 4)  # No chi angles

        all_frames = torsion_angles_to_frames(backbone_frame, chi_rot, default_frames, chi_mask)

        # Chi groups should just be backbone (since default is identity and chi not applied)
        for i in range(4, 8):
            torch.testing.assert_close(
                all_frames[:, i, :3, :3],
                backbone_frame[:, :3, :3],
                atol=1e-5,
                rtol=1e-5
            )


class TestFramesToAtom14Pos:
    """Tests for frames_to_atom14_pos function."""

    def test_identity_frames(self):
        """Identity frames should return reference positions unchanged."""
        ZN = 5
        all_frames = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(ZN, 8, -1, -1).clone()
        rigid_group_positions = torch.randn(ZN, 14, 3)
        atom_to_group = torch.zeros(ZN, 14, dtype=torch.long)  # all in group 0

        atom14_pos = frames_to_atom14_pos(all_frames, rigid_group_positions, atom_to_group)

        torch.testing.assert_close(atom14_pos, rigid_group_positions, atol=1e-5, rtol=1e-5)

    def test_translation_only(self):
        """Translation should shift all atoms."""
        ZN = 1
        all_frames = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(ZN, 8, -1, -1).clone()
        all_frames[:, 0, :3, 3] = torch.tensor([1., 2., 3.])  # translation in group 0

        rigid_group_positions = torch.zeros(ZN, 14, 3)
        atom_to_group = torch.zeros(ZN, 14, dtype=torch.long)  # all in group 0

        atom14_pos = frames_to_atom14_pos(all_frames, rigid_group_positions, atom_to_group)

        expected = torch.tensor([1., 2., 3.]).unsqueeze(0).unsqueeze(0).expand(ZN, 14, -1)
        torch.testing.assert_close(atom14_pos, expected, atol=1e-5, rtol=1e-5)

    def test_different_groups(self):
        """Different atoms should use different group frames."""
        ZN = 1

        # Set up different translations for different groups
        all_frames = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(ZN, 8, -1, -1).clone()
        all_frames[:, 0, :3, 3] = torch.tensor([1., 0., 0.])
        all_frames[:, 3, :3, 3] = torch.tensor([0., 1., 0.])  # psi group

        rigid_group_positions = torch.zeros(ZN, 14, 3)
        atom_to_group = torch.zeros(ZN, 14, dtype=torch.long)
        atom_to_group[0, 3] = 3  # atom 3 in psi group

        atom14_pos = frames_to_atom14_pos(all_frames, rigid_group_positions, atom_to_group)

        assert atom14_pos[0, 0, 0] == 1.0  # atom 0 uses group 0
        assert atom14_pos[0, 3, 1] == 1.0  # atom 3 uses group 3


class TestCoordsFromTxySincos:
    """Integration tests for coords_from_txy_sincos."""

    def test_output_shape(self):
        """Should produce correct output shapes."""
        ZN = 10
        t = torch.randn(ZN, 3)
        x = torch.randn(ZN, 3)
        y = torch.randn(ZN, 3)
        sin = torch.randn(ZN, 4)
        cos = torch.randn(ZN, 4)
        labels = torch.randint(0, 20, (ZN,))

        atom14_pos, atom_mask = coords_from_txy_sincos(t, x, y, sin, cos, labels)

        assert atom14_pos.shape == (ZN, 14, 3)
        assert atom_mask.shape == (ZN, 14)
        assert atom_mask.dtype == torch.bool

    def test_glycine_minimal_atoms(self):
        """Glycine (idx=7) should have minimal atoms (no CB, no sidechain)."""
        ZN = 1
        t = torch.zeros(ZN, 3)
        x = torch.tensor([[1., 0., 0.]])
        y = torch.tensor([[0., 1., 0.]])
        sin = torch.zeros(ZN, 4)
        cos = torch.ones(ZN, 4)
        labels = torch.tensor([7])  # Glycine

        atom14_pos, atom_mask = coords_from_txy_sincos(t, x, y, sin, cos, labels)

        # Glycine has only N, CA, C, O (4 atoms in atom14 format)
        num_atoms = atom_mask.sum().item()
        assert num_atoms == 4, f"Glycine should have 4 atoms, got {num_atoms}"

    def test_alanine_has_cb(self):
        """Alanine (idx=0) should have CB."""
        ZN = 1
        t = torch.zeros(ZN, 3)
        x = torch.tensor([[1., 0., 0.]])
        y = torch.tensor([[0., 1., 0.]])
        sin = torch.zeros(ZN, 4)
        cos = torch.ones(ZN, 4)
        labels = torch.tensor([0])  # Alanine

        atom14_pos, atom_mask = coords_from_txy_sincos(t, x, y, sin, cos, labels)

        # Alanine has N, CA, C, O, CB (5 atoms)
        num_atoms = atom_mask.sum().item()
        assert num_atoms == 5, f"Alanine should have 5 atoms, got {num_atoms}"

    def test_translation_applied(self):
        """CA position should match translation t."""
        ZN = 1
        t = torch.tensor([[10., 20., 30.]])  # CA at (10, 20, 30)
        x = torch.tensor([[1., 0., 0.]])
        y = torch.tensor([[0., 1., 0.]])
        sin = torch.zeros(ZN, 4)
        cos = torch.ones(ZN, 4)
        labels = torch.tensor([0])  # Alanine

        atom14_pos, atom_mask = coords_from_txy_sincos(t, x, y, sin, cos, labels)

        # CA is atom index 1 in atom14 format
        ca_pos = atom14_pos[0, 1]
        torch.testing.assert_close(ca_pos, t[0], atol=1e-4, rtol=1e-4)

    def test_differentiable(self):
        """All operations should be differentiable."""
        ZN = 5
        t = torch.randn(ZN, 3, requires_grad=True)
        x = torch.randn(ZN, 3, requires_grad=True)
        y = torch.randn(ZN, 3, requires_grad=True)
        sin = torch.randn(ZN, 4, requires_grad=True)
        cos = torch.randn(ZN, 4, requires_grad=True)
        labels = torch.randint(0, 20, (ZN,))

        atom14_pos, atom_mask = coords_from_txy_sincos(t, x, y, sin, cos, labels)

        # Compute loss and backprop
        loss = (atom14_pos * atom_mask.unsqueeze(-1).float()).sum()
        loss.backward()

        # Check gradients exist
        assert t.grad is not None
        assert x.grad is not None
        assert y.grad is not None
        assert sin.grad is not None
        assert cos.grad is not None

    def test_rotation_affects_atoms(self):
        """Different rotations should produce different atom positions."""
        ZN = 1
        t = torch.zeros(ZN, 3)
        sin = torch.zeros(ZN, 4)
        cos = torch.ones(ZN, 4)
        labels = torch.tensor([0])  # Alanine

        # Two different orientations
        x1 = torch.tensor([[1., 0., 0.]])
        y1 = torch.tensor([[0., 1., 0.]])

        x2 = torch.tensor([[0., 1., 0.]])
        y2 = torch.tensor([[-1., 0., 0.]])

        pos1, _ = coords_from_txy_sincos(t, x1, y1, sin, cos, labels)
        pos2, _ = coords_from_txy_sincos(t, x2, y2, sin, cos, labels)

        # Positions should be different (except CA which is at origin)
        diff = (pos1 - pos2).abs().sum()
        assert diff > 0.1, "Different rotations should produce different atom positions"


class TestAF2TensorCaching:
    """Tests for AF2 tensor caching mechanism."""

    def test_tensor_shapes(self):
        """AF2 tensors should have correct shapes."""
        device = torch.device('cpu')

        rigid_group_default = _get_af2_tensor('restype_rigid_group_default_frame', device)
        atom14_positions = _get_af2_tensor('restype_atom14_rigid_group_positions', device)
        atom14_to_group = _get_af2_tensor('restype_atom14_to_rigid_group', device, torch.long)
        atom14_mask = _get_af2_tensor('restype_atom14_mask', device)

        assert rigid_group_default.shape == (21, 8, 4, 4)
        assert atom14_positions.shape == (21, 14, 3)
        assert atom14_to_group.shape == (21, 14)
        assert atom14_mask.shape == (21, 14)

    def test_caching_returns_same_tensor(self):
        """Multiple calls should return the same cached tensor."""
        device = torch.device('cpu')

        tensor1 = _get_af2_tensor('restype_atom14_mask', device)
        tensor2 = _get_af2_tensor('restype_atom14_mask', device)

        assert tensor1 is tensor2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
