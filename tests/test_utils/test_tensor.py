"""Tests for tensor utilities (unpad/repad functions)."""

import pytest
import torch

from proteus.utils.tensor import unpad, repad


class TestUnpadRepad:
    """Test suite for unpad and repad tensor utilities."""

    def test_round_trip_single_tensor(self):
        """Test that repad(unpad(x)) == x for a single tensor."""
        Z, N = 3, 5
        d_model = 128

        # Create padded tensor
        x = torch.randn(Z, N, d_model)

        # Create pad_mask with varying sequence lengths
        pad_mask = torch.tensor([
            [True, True, True, False, False],  # length 3
            [True, True, True, True, True],    # length 5
            [True, True, False, False, False],  # length 2
        ])

        # Round trip
        [x_unpacked], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)
        [x_repacked] = repad(x_unpacked, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Check shapes
        assert x_repacked.shape == x.shape
        assert max_seqlen == N

        # Check values match in valid positions
        assert torch.allclose(x_repacked[pad_mask], x[pad_mask])

        # Check padding is zeros in invalid positions
        assert torch.allclose(x_repacked[~pad_mask], torch.zeros_like(x_repacked[~pad_mask]))

    def test_round_trip_multiple_tensors(self):
        """Test round trip with multiple tensors of different shapes."""
        Z, N = 2, 4
        d1, d2 = 64, 128

        x1 = torch.randn(Z, N, d1)
        x2 = torch.randn(Z, N, d2, 3)  # Extra trailing dimension

        pad_mask = torch.tensor([
            [True, True, True, False],  # length 3
            [True, True, False, False],  # length 2
        ])

        # Unpad multiple tensors
        [x1_u, x2_u], cu_seqlens, max_seqlen = unpad(x1, x2, pad_mask=pad_mask)

        # Check unpacked shapes
        assert x1_u.shape == (5, d1)  # 3 + 2 = 5
        assert x2_u.shape == (5, d2, 3)

        # Repad
        [x1_r, x2_r] = repad(x1_u, x2_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Check repadded shapes (max_seqlen = 3, not original N = 4)
        assert x1_r.shape == (Z, max_seqlen, d1)
        assert x2_r.shape == (Z, max_seqlen, d2, 3)

        # Create truncated mask for comparison
        truncated_mask = pad_mask[:, :max_seqlen]
        assert torch.allclose(x1_r[truncated_mask], x1[pad_mask])
        assert torch.allclose(x2_r[truncated_mask], x2[pad_mask])

    def test_single_sequence(self):
        """Test with Z=1 (single sequence)."""
        Z, N = 1, 3
        d_model = 64

        x = torch.randn(Z, N, d_model)
        pad_mask = torch.tensor([[True, True, False]])  # length 2

        [x_u], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)

        assert x_u.shape == (2, d_model)
        assert cu_seqlens.tolist() == [0, 2]
        assert max_seqlen == 2

        [x_r] = repad(x_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        assert torch.allclose(x_r[:, :2], x[:, :2])

    def test_all_valid(self):
        """Test when all positions are valid (no padding)."""
        Z, N = 2, 3
        d_model = 32

        x = torch.randn(Z, N, d_model)
        pad_mask = torch.ones(Z, N, dtype=torch.bool)

        [x_u], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)

        assert x_u.shape == (6, d_model)  # 2 * 3
        assert cu_seqlens.tolist() == [0, 3, 6]
        assert max_seqlen == 3

        [x_r] = repad(x_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        assert torch.allclose(x_r, x)

    def test_varying_lengths(self):
        """Test with varied sequence lengths."""
        Z, N = 4, 10
        d_model = 256

        x = torch.randn(Z, N, d_model)

        # Different sequence lengths: 2, 5, 10, 3
        pad_mask = torch.zeros(Z, N, dtype=torch.bool)
        pad_mask[0, :2] = True
        pad_mask[1, :5] = True
        pad_mask[2, :] = True
        pad_mask[3, :3] = True

        [x_u], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)

        assert x_u.shape == (20, d_model)  # 2 + 5 + 10 + 3
        assert cu_seqlens.tolist() == [0, 2, 7, 17, 20]
        assert max_seqlen == 10

        [x_r] = repad(x_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        assert torch.allclose(x_r[pad_mask], x[pad_mask])

    def test_multi_dimensional_trailing(self):
        """Test with multi-dimensional trailing shapes."""
        Z, N = 2, 3
        Vx, Vy, Vz = 8, 8, 8

        # Divergence-like tensor: (Z, N, 1, Vx, Vy, Vz)
        divergence = torch.randn(Z, N, 1, Vx, Vy, Vz)
        pad_mask = torch.tensor([
            [True, True, False],
            [True, False, False],
        ])

        [div_u], cu_seqlens, max_seqlen = unpad(divergence, pad_mask=pad_mask)

        assert div_u.shape == (3, 1, Vx, Vy, Vz)  # 2 + 1 = 3
        assert max_seqlen == 2  # Max actual length

        [div_r] = repad(div_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Repadded shape uses max_seqlen, not original N
        assert div_r.shape == (Z, max_seqlen, 1, Vx, Vy, Vz)

        # Create truncated mask for comparison
        truncated_mask = pad_mask[:, :max_seqlen]
        assert torch.allclose(div_r[truncated_mask], divergence[pad_mask])

    def test_gradient_flow(self):
        """Test that gradients flow correctly through unpad/repad."""
        Z, N = 2, 4
        d_model = 64

        x = torch.randn(Z, N, d_model, requires_grad=True)
        pad_mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
        ])

        # Forward pass
        [x_u], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)
        [x_r] = repad(x_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Create truncated mask for repadded tensor
        truncated_mask = pad_mask[:, :max_seqlen]

        # Compute loss (sum of valid positions)
        loss = x_r[truncated_mask].sum()

        # Backward pass
        loss.backward()

        # Check gradients exist
        assert x.grad is not None

        # Check gradients are non-zero only in valid positions
        assert (x.grad[pad_mask] != 0).any()
        # Padding positions should have zero gradient
        assert torch.allclose(x.grad[~pad_mask], torch.zeros_like(x.grad[~pad_mask]))

    def test_dtype_preservation(self):
        """Test that dtypes are preserved."""
        Z, N = 2, 3

        x_float = torch.randn(Z, N, 64, dtype=torch.float32)
        x_int = torch.randint(0, 20, (Z, N,), dtype=torch.long)
        pad_mask = torch.ones(Z, N, dtype=torch.bool)

        [x_f_u, x_i_u], cu_seqlens, max_seqlen = unpad(x_float, x_int, pad_mask=pad_mask)

        assert x_f_u.dtype == torch.float32
        assert x_i_u.dtype == torch.long

        [x_f_r, x_i_r] = repad(x_f_u, x_i_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        assert x_f_r.dtype == torch.float32
        assert x_i_r.dtype == torch.long

    def test_device_preservation(self):
        """Test that device is preserved (CPU only in this test)."""
        Z, N = 2, 3
        d_model = 32

        x = torch.randn(Z, N, d_model)
        pad_mask = torch.ones(Z, N, dtype=torch.bool)

        [x_u], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)
        [x_r] = repad(x_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        assert x_u.device == x.device
        assert x_r.device == x.device
        assert cu_seqlens.device == x.device

    def test_empty_sequence_filtering(self):
        """Test behavior when some sequences have zero length."""
        Z, N = 3, 4
        d_model = 64

        x = torch.randn(Z, N, d_model)
        pad_mask = torch.tensor([
            [True, True, False, False],   # length 2
            [False, False, False, False],  # length 0 (should be filtered upstream)
            [True, True, True, False],     # length 3
        ])

        # When a sequence has zero length, cu_seqlens will have repeated values
        [x_u], cu_seqlens, max_seqlen = unpad(x, pad_mask=pad_mask)

        assert x_u.shape == (5, d_model)  # 2 + 0 + 3 = 5
        assert cu_seqlens.tolist() == [0, 2, 2, 5]  # Note the repeat at position 2
        assert max_seqlen == 3  # Max actual length

        [x_r] = repad(x_u, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Create truncated mask for comparison
        truncated_mask = pad_mask[:, :max_seqlen]
        assert torch.allclose(x_r[truncated_mask], x[pad_mask])

    def test_error_on_shape_mismatch(self):
        """Test that errors are raised on shape mismatches."""
        Z, N = 2, 3

        x = torch.randn(Z, N, 64)
        wrong_mask = torch.ones(Z, N + 1, dtype=torch.bool)  # Wrong N

        with pytest.raises(ValueError, match="doesn't match pad_mask shape"):
            unpad(x, pad_mask=wrong_mask)

    def test_error_on_length_mismatch(self):
        """Test that errors are raised when ZN length doesn't match cu_seqlens."""
        x_unpacked = torch.randn(10, 64)
        cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)  # Total should be 7, not 10
        max_seqlen = 4

        with pytest.raises(ValueError, match="doesn't match cu_seqlens total"):
            repad(x_unpacked, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

    def test_empty_input(self):
        """Test that error is raised when no tensors provided."""
        pad_mask = torch.ones(2, 3, dtype=torch.bool)

        with pytest.raises(ValueError, match="At least one tensor must be provided"):
            unpad(pad_mask=pad_mask)

        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
        with pytest.raises(ValueError, match="At least one tensor must be provided"):
            repad(cu_seqlens=cu_seqlens, max_seqlen=3)


class TestUnpadDetails:
    """Detailed tests for unpad function."""

    def test_cu_seqlens_computation(self):
        """Test that cumulative sequence lengths are computed correctly."""
        Z, N = 3, 5
        pad_mask = torch.tensor([
            [True, True, True, False, False],  # length 3
            [True, True, False, False, False],  # length 2
            [True, True, True, True, True],    # length 5
        ])

        x = torch.randn(Z, N, 32)
        [_], cu_seqlens, _ = unpad(x, pad_mask=pad_mask)

        # Check cumulative sums
        assert cu_seqlens.dtype == torch.int32
        assert cu_seqlens.tolist() == [0, 3, 5, 10]

    def test_max_seqlen_computation(self):
        """Test that max_seqlen is computed correctly."""
        Z, N = 4, 8
        pad_mask = torch.zeros(Z, N, dtype=torch.bool)
        pad_mask[0, :2] = True   # length 2
        pad_mask[1, :6] = True   # length 6 (max)
        pad_mask[2, :3] = True   # length 3
        pad_mask[3, :5] = True   # length 5

        x = torch.randn(Z, N, 32)
        [_], _, max_seqlen = unpad(x, pad_mask=pad_mask)

        assert max_seqlen == 6


class TestRepadDetails:
    """Detailed tests for repad function."""

    def test_padding_values_are_zero(self):
        """Test that padding positions are filled with zeros."""
        Z = 2
        cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
        max_seqlen = 6

        x_unpacked = torch.randn(5, 64)
        [x_padded] = repad(x_unpacked, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # Check that padding positions are zero
        # First sequence: positions 3,4,5 should be zero
        assert torch.allclose(x_padded[0, 3:], torch.zeros(3, 64))
        # Second sequence: positions 2,3,4,5 should be zero
        assert torch.allclose(x_padded[1, 2:], torch.zeros(4, 64))

    def test_values_placed_correctly(self):
        """Test that values are placed in correct positions."""
        Z = 2
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        max_seqlen = 3

        # Create distinct values to track placement
        x_unpacked = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
        [x_padded] = repad(x_unpacked, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # First sequence should have rows 0, 1
        assert torch.allclose(x_padded[0, :2], x_unpacked[:2])
        # Second sequence should have rows 2, 3, 4
        assert torch.allclose(x_padded[1, :3], x_unpacked[2:5])
