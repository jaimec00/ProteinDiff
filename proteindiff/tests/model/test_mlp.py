"""
Sanity tests for MLP utility module.
"""
import pytest
import torch
import torch.nn as nn

from proteindiff.model.utils.mlp import MLP, MLPCfg, ActivationFn
from proteindiff.model.utils.mlp import init_orthogonal, init_kaiming, init_xavier, init_zeros


class TestMLP:
    """Test suite for the MLP module."""

    @pytest.fixture
    def default_mlp(self):
        """Create an MLP with default configuration."""
        cfg = MLPCfg()
        return MLP(cfg)

    @pytest.fixture
    def small_mlp(self):
        """Create a smaller MLP for faster tests."""
        cfg = MLPCfg(d_in=64, d_out=32, d_hidden=128, hidden_layers=0)
        return MLP(cfg)

    @pytest.fixture
    def deep_mlp(self):
        """Create a deeper MLP with multiple hidden layers."""
        cfg = MLPCfg(d_in=64, d_out=32, d_hidden=128, hidden_layers=3)
        return MLP(cfg)

    def test_mlp_initialization(self, default_mlp):
        """Test that MLP initializes with correct modules."""
        assert hasattr(default_mlp, 'in_proj')
        assert hasattr(default_mlp, 'hidden_proj')
        assert hasattr(default_mlp, 'out_proj')
        assert hasattr(default_mlp, 'in_dropout')
        assert hasattr(default_mlp, 'hidden_dropout')
        assert hasattr(default_mlp, 'act')

        # Check layer dimensions
        assert default_mlp.in_proj.in_features == 512
        assert default_mlp.in_proj.out_features == 1024
        assert default_mlp.out_proj.in_features == 1024
        assert default_mlp.out_proj.out_features == 512

    def test_mlp_hidden_layers(self, deep_mlp):
        """Test that MLP creates correct number of hidden layers."""
        assert len(deep_mlp.hidden_proj) == 3
        assert len(deep_mlp.hidden_dropout) == 3

        # Check hidden layer dimensions
        for layer in deep_mlp.hidden_proj:
            assert isinstance(layer, nn.Linear)
            assert layer.in_features == 128
            assert layer.out_features == 128

    def test_forward_pass_basic(self, small_mlp):
        """Test basic forward pass through MLP."""
        batch_size = 4
        x = torch.randn(batch_size, 64)

        output = small_mlp(x)

        # Check output shape
        assert output.shape == (batch_size, 32)

        # Check output is finite
        assert torch.all(torch.isfinite(output))

    def test_forward_pass_deep(self, deep_mlp):
        """Test forward pass through deep MLP with multiple hidden layers."""
        batch_size = 8
        x = torch.randn(batch_size, 64)

        output = deep_mlp(x)

        # Check output shape
        assert output.shape == (batch_size, 32)

        # Check output is finite
        assert torch.all(torch.isfinite(output))

    def test_forward_pass_batched(self, small_mlp):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 64)
            output = small_mlp(x)
            assert output.shape == (batch_size, 32)

    def test_forward_pass_3d_input(self, small_mlp):
        """Test forward pass with 3D input (e.g., sequence data)."""
        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, 64)

        output = small_mlp(x)

        # Check output shape preserves first dimensions
        assert output.shape == (batch_size, seq_len, 32)
        assert torch.all(torch.isfinite(output))

    def test_activation_gelu(self):
        """Test MLP with GELU activation."""
        cfg = MLPCfg(d_in=32, d_out=32, d_hidden=64, act=ActivationFn.GELU)
        mlp = MLP(cfg)

        x = torch.randn(4, 32)
        output = mlp(x)

        assert output.shape == (4, 32)
        assert torch.all(torch.isfinite(output))

    def test_activation_relu(self):
        """Test MLP with ReLU activation."""
        cfg = MLPCfg(d_in=32, d_out=32, d_hidden=64, act=ActivationFn.RELU)
        mlp = MLP(cfg)

        x = torch.randn(4, 32)
        output = mlp(x)

        assert output.shape == (4, 32)
        assert torch.all(torch.isfinite(output))

    def test_activation_silu(self):
        """Test MLP with SiLU activation."""
        cfg = MLPCfg(d_in=32, d_out=32, d_hidden=64, act=ActivationFn.SILU)
        mlp = MLP(cfg)

        x = torch.randn(4, 32)
        output = mlp(x)

        assert output.shape == (4, 32)
        assert torch.all(torch.isfinite(output))

    def test_activation_sigmoid(self):
        """Test MLP with Sigmoid activation."""
        cfg = MLPCfg(d_in=32, d_out=32, d_hidden=64, act=ActivationFn.SIGMOID)
        mlp = MLP(cfg)

        x = torch.randn(4, 32)
        output = mlp(x)

        assert output.shape == (4, 32)
        assert torch.all(torch.isfinite(output))

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        cfg = MLPCfg(d_in=32, d_out=32, d_hidden=64, act="BAD")

        with pytest.raises(ValueError, match="Invalid Activation"):
            MLP(cfg)

    def test_dropout_training_mode(self):
        """Test that dropout is applied in training mode."""
        cfg = MLPCfg(d_in=64, d_out=32, d_hidden=128, dropout=0.5)
        mlp = MLP(cfg)
        mlp.train()

        torch.manual_seed(42)
        x = torch.randn(100, 64)

        # Run forward pass twice with same input
        output1 = mlp(x)
        output2 = mlp(x)

        # With dropout, outputs should be different
        assert not torch.allclose(output1, output2)

    def test_dropout_eval_mode(self):
        """Test that dropout is not applied in eval mode."""
        cfg = MLPCfg(d_in=64, d_out=32, d_hidden=128, dropout=0.5)
        mlp = MLP(cfg)
        mlp.eval()

        torch.manual_seed(42)
        x = torch.randn(100, 64)

        # Run forward pass twice with same input
        output1 = mlp(x)
        output2 = mlp(x)

        # Without dropout, outputs should be identical
        torch.testing.assert_close(output1, output2)

    def test_no_dropout(self):
        """Test MLP with no dropout."""
        cfg = MLPCfg(d_in=64, d_out=32, d_hidden=128, dropout=0.0)
        mlp = MLP(cfg)
        mlp.train()

        x = torch.randn(4, 64)
        output1 = mlp(x)
        output2 = mlp(x)

        # With no dropout, outputs should be identical even in training mode
        torch.testing.assert_close(output1, output2)

    def test_zeros_initialization(self):
        """Test that zeros=True initializes output layer to zeros."""
        cfg = MLPCfg(d_in=32, d_out=16, d_hidden=64, zeros=True)
        mlp = MLP(cfg)

        # Check output layer is initialized to zeros
        assert torch.allclose(mlp.out_proj.weight, torch.zeros_like(mlp.out_proj.weight))
        assert torch.allclose(mlp.out_proj.bias, torch.zeros_like(mlp.out_proj.bias))

        # Check other layers are not zeros
        assert not torch.allclose(mlp.in_proj.weight, torch.zeros_like(mlp.in_proj.weight))

    def test_non_zeros_initialization(self):
        """Test that zeros=False doesn't initialize output layer to zeros."""
        cfg = MLPCfg(d_in=32, d_out=16, d_hidden=64, zeros=False)
        mlp = MLP(cfg)

        # Check output layer is not all zeros
        assert not torch.allclose(mlp.out_proj.weight, torch.zeros_like(mlp.out_proj.weight))

    def test_gradient_flow(self):
        """Test that gradients flow through MLP."""
        cfg = MLPCfg(d_in=32, d_out=16, d_hidden=64, hidden_layers=2)
        mlp = MLP(cfg)

        x = torch.randn(4, 32, requires_grad=True)
        output = mlp(x)
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for all parameters
        assert mlp.in_proj.weight.grad is not None
        assert mlp.out_proj.weight.grad is not None
        for layer in mlp.hidden_proj:
            assert layer.weight.grad is not None

        # Check that input has gradients
        assert x.grad is not None

    def test_deterministic_output(self):
        """Test that MLP output is deterministic in eval mode."""
        torch.manual_seed(123)
        cfg = MLPCfg(d_in=64, d_out=32, d_hidden=128, hidden_layers=2, dropout=0.1)
        mlp = MLP(cfg)
        mlp.eval()

        x = torch.randn(8, 64)

        # Run twice with same input
        output1 = mlp(x)
        output2 = mlp(x)

        # Should be identical in eval mode
        torch.testing.assert_close(output1, output2)

    def test_parameter_count(self):
        """Test that parameter count matches expected."""
        cfg = MLPCfg(d_in=64, d_out=32, d_hidden=128, hidden_layers=2)
        mlp = MLP(cfg)

        # Calculate expected parameters
        # in_proj: 64 * 128 + 128
        # hidden_proj[0]: 128 * 128 + 128
        # hidden_proj[1]: 128 * 128 + 128
        # out_proj: 128 * 32 + 32
        expected = (64 * 128 + 128) + 2 * (128 * 128 + 128) + (128 * 32 + 32)

        actual = sum(p.numel() for p in mlp.parameters())
        assert actual == expected


class TestInitializationFunctions:
    """Test suite for weight initialization functions."""

    def test_init_orthogonal(self):
        """Test orthogonal initialization."""
        layer = nn.Linear(64, 64)
        init_orthogonal(layer)

        # Check weight matrix is orthogonal (W @ W^T ≈ I)
        weight = layer.weight.data
        product = weight @ weight.T
        identity = torch.eye(64)
        assert torch.allclose(product, identity, atol=1e-5)

        # Check bias is zeros
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_init_kaiming(self):
        """Test Kaiming initialization."""
        layer = nn.Linear(64, 32)
        init_kaiming(layer)

        # Check weights are not zeros
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

        # Check bias is zeros
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

        # Check variance is roughly correct for Kaiming
        weight_std = layer.weight.std()
        assert weight_std > 0.01

    def test_init_xavier(self):
        """Test Xavier initialization."""
        layer = nn.Linear(64, 32)
        init_xavier(layer)

        # Check weights are not zeros
        assert not torch.allclose(layer.weight, torch.zeros_like(layer.weight))

        # Check bias is zeros
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

        # Check variance is roughly correct for Xavier
        weight_std = layer.weight.std()
        assert weight_std > 0.01

    def test_init_zeros(self):
        """Test zeros initialization."""
        layer = nn.Linear(64, 32)
        # Initialize to non-zero first
        nn.init.normal_(layer.weight)
        nn.init.normal_(layer.bias)

        # Apply zeros initialization
        init_zeros(layer)

        # Check both weight and bias are zeros
        assert torch.allclose(layer.weight, torch.zeros_like(layer.weight))
        assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))

    def test_init_with_non_linear_layer(self):
        """Test that init functions handle non-Linear layers gracefully."""
        conv = nn.Conv2d(3, 16, 3)

        # These should not raise errors, just do nothing
        init_orthogonal(conv)
        init_kaiming(conv)
        init_xavier(conv)
        init_zeros(conv)

        # Conv layer should be unchanged (not zeros)
        assert not torch.allclose(conv.weight, torch.zeros_like(conv.weight))

    def test_init_without_bias(self):
        """Test initialization functions with layers that have no bias."""
        layer = nn.Linear(64, 32, bias=False)

        # These should not raise errors
        init_orthogonal(layer)
        init_kaiming(layer)
        init_xavier(layer)
        init_zeros(layer)

        # Weight should be initialized
        assert layer.bias is None


class TestMLPEdgeCases:
    """Test suite for MLP edge cases."""

    def test_identity_dimensions(self):
        """Test MLP with same input, hidden, and output dimensions."""
        cfg = MLPCfg(d_in=128, d_out=128, d_hidden=128, hidden_layers=2)
        mlp = MLP(cfg)

        x = torch.randn(4, 128)
        output = mlp(x)

        assert output.shape == (4, 128)

    def test_very_small_dimensions(self):
        """Test MLP with very small dimensions."""
        cfg = MLPCfg(d_in=2, d_out=1, d_hidden=4, hidden_layers=0)
        mlp = MLP(cfg)

        x = torch.randn(8, 2)
        output = mlp(x)

        assert output.shape == (8, 1)

    def test_very_large_batch(self):
        """Test MLP with very large batch size."""
        cfg = MLPCfg(d_in=32, d_out=16, d_hidden=64)
        mlp = MLP(cfg)

        x = torch.randn(1000, 32)
        output = mlp(x)

        assert output.shape == (1000, 16)
        assert torch.all(torch.isfinite(output))

    def test_zero_input(self):
        """Test MLP with zero input."""
        cfg = MLPCfg(d_in=32, d_out=16, d_hidden=64)
        mlp = MLP(cfg)

        x = torch.zeros(4, 32)
        output = mlp(x)

        assert output.shape == (4, 16)
        assert torch.all(torch.isfinite(output))

    def test_single_sample(self):
        """Test MLP with single sample (no batch dimension)."""
        cfg = MLPCfg(d_in=32, d_out=16, d_hidden=64)
        mlp = MLP(cfg)

        x = torch.randn(32)
        output = mlp(x)

        assert output.shape == (16,)

    def test_many_hidden_layers(self):
        """Test MLP with many hidden layers."""
        cfg = MLPCfg(d_in=32, d_out=16, d_hidden=64, hidden_layers=10)
        mlp = MLP(cfg)

        assert len(mlp.hidden_proj) == 10

        x = torch.randn(4, 32)
        output = mlp(x)

        assert output.shape == (4, 16)
        assert torch.all(torch.isfinite(output))

    def test_no_hidden_layers(self):
        """Test MLP with no hidden layers (direct in -> out)."""
        cfg = MLPCfg(d_in=64, d_out=32, d_hidden=128, hidden_layers=0)
        mlp = MLP(cfg)

        assert len(mlp.hidden_proj) == 0

        x = torch.randn(4, 64)
        output = mlp(x)

        assert output.shape == (4, 32)
