"""
Sanity tests for model components.
"""
import pytest
import torch
import numpy as np

from proteindiff.model.tokenizer.tokenizer import Tokenizer, TokenizerCfg
from proteindiff.static.constants import aa_2_lbl, canonical_aas


class TestTokenizer:
    """Test suite for the Tokenizer module."""

    @pytest.fixture
    def tokenizer(self, device):
        """Create a tokenizer with default configuration."""
        cfg = TokenizerCfg(voxel_dim=16, cell_dim=1.0)
        return Tokenizer(cfg).to(device)

    @pytest.fixture
    def small_tokenizer(self, device):
        """Create a smaller tokenizer for faster tests."""
        cfg = TokenizerCfg(voxel_dim=8, cell_dim=1.0)
        return Tokenizer(cfg).to(device)

    @pytest.fixture
    def sample_protein(self, device):
        """Create a simple sample protein with valid backbone coordinates."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Create a small protein with 3 residues
        ZN = 3
        A = 14  # All-atom representation

        # Create realistic backbone coordinates using random but structured approach
        # Start at origin and build a chain with realistic bond lengths/angles
        coords = torch.zeros(ZN, A, 3, device=device)

        # Build backbone with approximate bond lengths
        for i in range(ZN):
            # N-CA bond ~1.46A, CA-C bond ~1.52A, C-O bond ~1.23A
            # Place residues ~3.8A apart along extended chain
            base_x = i * 3.8
            coords[i, 0] = torch.tensor([base_x, 0.0, 0.0], device=device) + torch.randn(3, device=device) * 0.1  # N
            coords[i, 1] = torch.tensor([base_x + 1.46, 0.0, 0.0], device=device) + torch.randn(3, device=device) * 0.1  # CA
            coords[i, 2] = torch.tensor([base_x + 2.98, 0.0, 0.0], device=device) + torch.randn(3, device=device) * 0.1  # C
            coords[i, 3] = torch.tensor([base_x + 3.21, 1.23, 0.0], device=device) + torch.randn(3, device=device) * 0.1  # O

        # Create amino acid labels (random selection from canonical amino acids)
        labels = torch.tensor([aa_2_lbl('A'), aa_2_lbl('C'), aa_2_lbl('D')], device=device)

        # Create atom mask (only backbone atoms present for this simple test)
        atom_mask = torch.zeros(ZN, A, device=device)
        atom_mask[:, :4] = 1.0  # Only backbone atoms are present

        return coords, labels, atom_mask

    def test_tokenizer_initialization(self, tokenizer):
        """Test that tokenizer initializes with correct buffers."""
        assert hasattr(tokenizer, 'voxel')
        assert hasattr(tokenizer, 'amber_partial_charges')
        assert tokenizer.res == 1.0

        # Check voxel shape
        assert tokenizer.voxel.shape == (16, 16, 16, 3)

        # Check amber charges shape (22 amino acids including X and <mask>)
        assert tokenizer.amber_partial_charges.shape[0] == 22

    def test_get_backbone(self, sample_protein):
        """Test virtual Cb computation."""
        coords, _, _ = sample_protein

        # Get backbone with virtual Cb
        C_backbone = Tokenizer.get_backbone(coords)

        # Check output shape
        assert C_backbone.shape == (3, 4, 3)

        # Check that N, CA, C are unchanged
        torch.testing.assert_close(C_backbone[:, 0], coords[:, 0])
        torch.testing.assert_close(C_backbone[:, 1], coords[:, 1])
        torch.testing.assert_close(C_backbone[:, 2], coords[:, 2])


    def test_compute_frames(self, small_tokenizer, sample_protein):
        """Test local coordinate frame computation."""
        coords, _, _ = sample_protein
        C_backbone = Tokenizer.get_backbone(coords)

        origins, frames = small_tokenizer.compute_frames(C_backbone)

        # Check output shapes
        assert origins.shape == (3, 3)
        assert frames.shape == (3, 3, 3)

        # Check that frames are orthonormal
        for i in range(3):
            # Extract the three unit vectors
            x = frames[i, 0, :]
            y = frames[i, 1, :]
            z = frames[i, 2, :]

            # Check unit length
            assert abs(torch.linalg.norm(x).item() - 1.0) < 1e-5
            assert abs(torch.linalg.norm(y).item() - 1.0) < 1e-5
            assert abs(torch.linalg.norm(z).item() - 1.0) < 1e-5

            # Check orthogonality
            assert abs(torch.dot(x, y).item()) < 1e-5
            assert abs(torch.dot(y, z).item()) < 1e-5
            assert abs(torch.dot(z, x).item()) < 1e-5

        # Check that origin is at virtual Cb
        torch.testing.assert_close(origins, C_backbone[:, 3], atol=1e-5, rtol=1e-5)

    def test_compute_voxels(self, small_tokenizer, sample_protein):
        """Test voxel grid computation in local frames."""
        coords, _, _ = sample_protein
        C_backbone = Tokenizer.get_backbone(coords)
        origins, frames = small_tokenizer.compute_frames(C_backbone)

        local_voxels = small_tokenizer.compute_voxels(origins, frames)

        # Check output shape
        ZN = 3
        Vx, Vy, Vz = 8, 8, 8
        assert local_voxels.shape == (ZN, Vx, Vy, Vz, 3)

        # Check that center voxel is approximately at origin (virtual Cb)
        center_voxel = local_voxels[:, Vx//2, Vy//4, Vz//2, :]  # Note: y is offset
        # Should be close to origin (within a cell width)
        dist_to_origin = torch.linalg.norm(center_voxel - origins, dim=-1)
        assert torch.all(dist_to_origin < 2.0)

    def test_compute_fields(self, small_tokenizer, sample_protein):
        """Test electric field computation."""
        coords, labels, atom_mask = sample_protein
        C_backbone = Tokenizer.get_backbone(coords)
        origins, frames = small_tokenizer.compute_frames(C_backbone)
        local_voxels = small_tokenizer.compute_voxels(origins, frames)

        fields = small_tokenizer.compute_fields(coords, labels, local_voxels, atom_mask)

        # Check output shape
        ZN = 3
        Vx, Vy, Vz = 8, 8, 8
        assert fields.shape == (ZN, 3, Vx, Vy, Vz)

        # Check that fields are normalized (unit vectors)
        field_norms = torch.linalg.norm(fields.permute(0, 2, 3, 4, 1), dim=-1)
        # Most voxels should have unit norm (some might be zero if no charges nearby)
        nonzero_mask = field_norms > 0.1
        assert torch.allclose(field_norms[nonzero_mask], torch.ones_like(field_norms[nonzero_mask]), atol=1e-5)

        # Check that we have some non-zero fields
        assert torch.any(field_norms > 0.5)

    def test_compute_divergence(self, small_tokenizer, sample_protein):
        """Test divergence computation."""
        coords, labels, atom_mask = sample_protein
        C_backbone = Tokenizer.get_backbone(coords)
        origins, frames = small_tokenizer.compute_frames(C_backbone)
        local_voxels = small_tokenizer.compute_voxels(origins, frames)
        fields = small_tokenizer.compute_fields(coords, labels, local_voxels, atom_mask)

        divergence = small_tokenizer.compute_divergence(fields)

        # Check output shape
        ZN = 3
        Vx, Vy, Vz = 8, 8, 8
        assert divergence.shape == (ZN, 1, Vx, Vy, Vz)

        # Divergence is a scalar field, values should be finite
        assert torch.all(torch.isfinite(divergence))

        # Check that divergence has reasonable magnitude (not all zeros)
        assert torch.abs(divergence).sum() > 0.0

    def test_forward_pass(self, small_tokenizer, sample_protein):
        """Test complete forward pass through tokenizer."""
        coords, labels, atom_mask = sample_protein

        # Run forward pass
        C_backbone, divergence, local_frames = small_tokenizer(coords, labels, atom_mask)

        # Check backbone output
        assert C_backbone.shape == (3, 4, 3)

        # Check divergence output
        assert divergence.shape == (3, 1, 8, 8, 8)
        assert torch.all(torch.isfinite(divergence))

        # Check local frames output
        assert local_frames.shape == (3, 3, 3)

        # Verify frames are orthonormal
        for i in range(3):
            x, y, z = local_frames[i, 0], local_frames[i, 1], local_frames[i, 2]
            assert abs(torch.linalg.norm(x).item() - 1.0) < 1e-5
            assert abs(torch.dot(x, y).item()) < 1e-5

    def test_no_grad_forward(self, small_tokenizer, sample_protein):
        """Test that forward pass does not track gradients."""
        coords, labels, atom_mask = sample_protein
        coords.requires_grad = True

        C_backbone, divergence, local_frames = small_tokenizer(coords, labels, atom_mask)

        # Outputs should not require gradients
        assert not C_backbone.requires_grad
        assert not divergence.requires_grad
        assert not local_frames.requires_grad

    def test_invalid_labels(self, small_tokenizer, sample_protein):
        """Test handling of invalid amino acid labels."""
        coords, labels, atom_mask = sample_protein

        # Set one label to invalid (-1)
        labels[1] = -1

        # Should still run without error (invalid labels replaced with X)
        C_backbone, divergence, local_frames = small_tokenizer(coords, labels, atom_mask)

        assert C_backbone.shape == (3, 4, 3)
        assert divergence.shape == (3, 1, 8, 8, 8)
        assert torch.all(torch.isfinite(divergence))

    def test_atom_masking(self, small_tokenizer, sample_protein):
        """Test that atom masking properly zeros out contributions."""
        coords, labels, atom_mask = sample_protein

        # Run with full mask
        _, div_full, _ = small_tokenizer(coords, labels, atom_mask)

        # Run with partial mask (mask out some backbone atoms)
        atom_mask_partial = atom_mask.clone()
        atom_mask_partial[1, 2:] = 0.0  # Mask out C and O of residue 1
        _, div_partial, _ = small_tokenizer(coords, labels, atom_mask_partial)

        # Divergence should be different
        assert not torch.allclose(div_full, div_partial)

    def test_batch_processing(self, small_tokenizer, device):
        """Test processing multiple residues in batch."""
        torch.manual_seed(123)

        # Create a larger batch
        ZN = 10
        A = 14

        # Create simple extended chain with random perturbations
        coords = torch.randn(ZN, A, 3, device=device) * 5.0
        labels = torch.randint(0, 20, (ZN,), device=device)
        atom_mask = torch.zeros(ZN, A, device=device)
        atom_mask[:, :4] = 1.0

        # Should process without error
        C_backbone, divergence, local_frames = small_tokenizer(coords, labels, atom_mask)

        assert C_backbone.shape == (ZN, 4, 3)
        assert divergence.shape == (ZN, 1, 8, 8, 8)
        assert local_frames.shape == (ZN, 3, 3)
        

    def test_deterministic_output(self, small_tokenizer, device):
        """Test that tokenizer output is deterministic (no randomness)."""
        torch.manual_seed(999)
        np.random.seed(999)

        # Create sample protein
        ZN = 3
        A = 14
        coords = torch.randn(ZN, A, 3, device=device) * 5.0
        labels = torch.randint(0, 20, (ZN,), device=device)
        atom_mask = torch.zeros(ZN, A, device=device)
        atom_mask[:, :4] = 1.0

        # Run twice with same input
        out1 = small_tokenizer(coords.clone(), labels.clone(), atom_mask.clone())
        out2 = small_tokenizer(coords.clone(), labels.clone(), atom_mask.clone())

        # Should be identical
        torch.testing.assert_close(out1[0], out2[0])  # C_backbone
        torch.testing.assert_close(out1[1], out2[1])  # divergence
        torch.testing.assert_close(out1[2], out2[2])  # local_frames
