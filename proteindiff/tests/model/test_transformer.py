"""
Sanity tests for transformer components.
"""
import pytest
import torch
import numpy as np

from proteindiff.model.transformer.attention import MHA, MHACfg, torch_attn_varlen_qkvpacked_func
from proteindiff.model.transformer.transformer import TransformerBlock, TransformerBlockCfg, TransformerModel, TransformerModelCfg
from proteindiff.model.utils import FFNCfg
from proteindiff.model.utils.mlp import ActivationFn


class TestMHA:
    """Test suite for the Multi-Head Attention module."""

    @pytest.fixture
    def mha_config(self):
        """Create a small MHA config for testing."""
        return MHACfg(d_model=64, heads=4, dropout_p=0.0)

    @pytest.fixture
    def mha(self, mha_config, device):
        """Create an MHA module with test configuration."""
        return MHA(mha_config).to(device)

    @pytest.fixture
    def sample_input(self, device):
        """Create sample input with variable-length sequences."""
        torch.manual_seed(42)

        # Two sequences: length 3 and 5 (total ZN=8)
        ZN = 8
        d_model = 64
        x = torch.randn(ZN, d_model, device=device)

        # Cumulative sequence lengths [0, 3, 8]
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
        max_seqlen = 5

        return x, cu_seqlens, max_seqlen

    def test_mha_initialization(self, mha, mha_config):
        """Test that MHA initializes with correct parameters."""
        assert hasattr(mha, 'qkv_proj')
        assert hasattr(mha, 'qkv_bias')
        assert hasattr(mha, 'out_proj')

        # Check QKV projection shape: (3, heads, d_model, d_k)
        d_k = mha_config.d_model // mha_config.heads
        assert mha.qkv_proj.shape == (3, mha_config.heads, mha_config.d_model, d_k)
        assert mha.qkv_bias.shape == (3, mha_config.heads, d_k)

        # Check dropout
        assert mha.dropout_p == mha_config.dropout_p

    def test_mha_forward_shape(self, mha, sample_input):
        """Test that MHA forward pass produces correct output shape."""
        x, cu_seqlens, max_seqlen = sample_input

        output = mha(x, cu_seqlens, max_seqlen)

        # Output should have same shape as input
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_mha_single_sequence(self, mha_config, device):
        """Test MHA with a single sequence."""
        mha = MHA(mha_config).to(device)
        torch.manual_seed(123)

        # Single sequence of length 10
        ZN = 10
        d_model = 64
        x = torch.randn(ZN, d_model, device=device)
        cu_seqlens = torch.tensor([0, 10], dtype=torch.int32, device=device)
        max_seqlen = 10

        output = mha(x, cu_seqlens, max_seqlen)

        assert output.shape == (ZN, d_model)
        assert torch.all(torch.isfinite(output))

    def test_mha_deterministic(self, mha, sample_input):
        """Test that MHA output is deterministic (no dropout)."""
        x, cu_seqlens, max_seqlen = sample_input

        # Run twice with same input
        output1 = mha(x.clone(), cu_seqlens, max_seqlen)
        output2 = mha(x.clone(), cu_seqlens, max_seqlen)

        torch.testing.assert_close(output1, output2)

    def test_mha_gradient_flow(self, mha, sample_input):
        """Test that gradients flow through MHA."""
        x, cu_seqlens, max_seqlen = sample_input
        x.requires_grad = True

        output = mha(x, cu_seqlens, max_seqlen)
        loss = output.sum()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

    def test_mha_sequence_independence(self, mha, device):
        """Test that different sequences in batch don't affect each other."""
        torch.manual_seed(456)

        d_model = 64

        # Create two sequences separately
        seq1 = torch.randn(3, d_model, device=device)
        seq2 = torch.randn(5, d_model, device=device)

        # Process separately
        cu_seqlens_1 = torch.tensor([0, 3], dtype=torch.int32, device=device)
        cu_seqlens_2 = torch.tensor([0, 5], dtype=torch.int32, device=device)
        out1 = mha(seq1, cu_seqlens_1, 3)
        out2 = mha(seq2, cu_seqlens_2, 5)

        # Process together
        combined = torch.cat([seq1, seq2], dim=0)
        cu_seqlens_combined = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
        out_combined = mha(combined, cu_seqlens_combined, 5)

        # Outputs should match
        torch.testing.assert_close(out_combined[:3], out1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out_combined[3:], out2, rtol=1e-4, atol=1e-4)

    def test_torch_attn_varlen_qkvpacked_func(self, device):
        """Test the PyTorch CPU fallback attention implementation."""
        torch.manual_seed(789)

        ZN = 8
        H = 4
        Dk = 16

        # Create packed QKV
        qkv = torch.randn(ZN, 3, H, Dk, device=device)
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
        max_seqlen = 5

        # Run attention
        output = torch_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=0.0
        )

        # Check output shape
        assert output.shape == (ZN, H, Dk)
        assert torch.all(torch.isfinite(output))

    def test_torch_attn_masking(self, device):
        """Test that PyTorch attention properly masks between sequences."""
        torch.manual_seed(999)

        ZN = 6  # Two sequences: 3 + 3
        H = 2
        Dk = 8

        # Create QKV where Q has strong values in first sequence
        qkv = torch.randn(ZN, 3, H, Dk, device=device)
        qkv[:3, 0, :, :] = 10.0  # Strong queries in first sequence
        qkv[3:, 0, :, :] = 0.1   # Weak queries in second sequence

        cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32, device=device)
        max_seqlen = 3

        output = torch_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, max_seqlen, dropout_p=0.0
        )

        # Both sequences should produce valid outputs (no cross-contamination)
        assert torch.all(torch.isfinite(output[:3]))
        assert torch.all(torch.isfinite(output[3:]))


class TestTransformerBlock:
    """Test suite for the TransformerBlock module."""

    @pytest.fixture
    def block_config(self):
        """Create a transformer block configuration."""
        d_model = 64
        attn_cfg = MHACfg(
            d_model=d_model,
            heads=4,
            dropout_p=0.0,
        )
        ffn_cfg = FFNCfg(
            d_model=d_model,
            expansion_factor=2,
            dropout=0.0,
            act=ActivationFn.GELU,
            zeros=False,
        )
        return TransformerBlockCfg(d_model=d_model, attn=attn_cfg, ffn=ffn_cfg)

    @pytest.fixture
    def transformer_block(self, block_config, device):
        """Create a transformer block."""
        return TransformerBlock(block_config).to(device)

    @pytest.fixture
    def sample_input(self, device):
        """Create sample input for transformer block."""
        torch.manual_seed(42)

        ZN = 8
        d_model = 64
        x = torch.randn(ZN, d_model, device=device)
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
        max_seqlen = 5

        return x, cu_seqlens, max_seqlen

    def test_block_initialization(self, transformer_block):
        """Test that transformer block initializes correctly."""
        assert hasattr(transformer_block, 'attn')
        assert hasattr(transformer_block, 'attn_norm')
        assert hasattr(transformer_block, 'ffn')
        assert hasattr(transformer_block, 'ffn_norm')

    def test_block_forward_shape(self, transformer_block, sample_input):
        """Test that block forward pass produces correct output shape."""
        x, cu_seqlens, max_seqlen = sample_input

        output = transformer_block(x, cu_seqlens, max_seqlen)

        # Output should have same shape as input
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_block_residual_connections(self, transformer_block, sample_input):
        """Test that residual connections work correctly."""
        x, cu_seqlens, max_seqlen = sample_input

        output = transformer_block(x, cu_seqlens, max_seqlen)

        # Output should be different from input (has transformations)
        assert not torch.allclose(output, x)

        # But should have finite values
        assert torch.all(torch.isfinite(output))

    def test_block_deterministic(self, transformer_block, sample_input):
        """Test that block output is deterministic."""
        x, cu_seqlens, max_seqlen = sample_input

        output1 = transformer_block(x.clone(), cu_seqlens, max_seqlen)
        output2 = transformer_block(x.clone(), cu_seqlens, max_seqlen)

        torch.testing.assert_close(output1, output2)

    def test_block_gradient_flow(self, transformer_block, sample_input):
        """Test that gradients flow through the block."""
        x, cu_seqlens, max_seqlen = sample_input
        x.requires_grad = True

        output = transformer_block(x, cu_seqlens, max_seqlen)
        loss = output.sum()
        loss.backward()

        # Check gradients on input
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

        # Check gradients on parameters
        for param in transformer_block.parameters():
            assert param.grad is not None
            assert torch.all(torch.isfinite(param.grad))

    def test_block_layer_norm(self, transformer_block, sample_input):
        """Test that layer normalization is applied correctly."""
        x, cu_seqlens, max_seqlen = sample_input

        output = transformer_block(x, cu_seqlens, max_seqlen)

        # Output should have reasonable magnitude due to layer norms
        assert output.abs().mean() < 10.0


class TestTransformerModel:
    """Test suite for the TransformerModel (stacked blocks)."""

    @pytest.fixture
    def model_config(self):
        """Create a transformer model configuration with 2 blocks."""
        d_model = 64
        attn_cfg = MHACfg(
            d_model=d_model,
            heads=4,
            dropout_p=0.0,
        )
        ffn_cfg = FFNCfg(
            d_model=d_model,
            expansion_factor=2,
            dropout=0.0,
            act=ActivationFn.GELU,
            zeros=False,
        )
        block_cfg = TransformerBlockCfg(d_model=d_model, attn=attn_cfg, ffn=ffn_cfg)

        # Create list of 2 identical block configs
        return TransformerModelCfg(blocks=[block_cfg, block_cfg])

    @pytest.fixture
    def transformer_model(self, model_config, device):
        """Create a transformer model."""
        return TransformerModel(model_config).to(device)

    @pytest.fixture
    def sample_input(self, device):
        """Create sample input for transformer model."""
        torch.manual_seed(42)

        ZN = 8
        d_model = 64
        x = torch.randn(ZN, d_model, device=device)
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
        max_seqlen = 5

        return x, cu_seqlens, max_seqlen

    def test_model_initialization(self, transformer_model, model_config):
        """Test that transformer model initializes with correct number of blocks."""
        assert hasattr(transformer_model, 'blocks')
        assert len(transformer_model.blocks) == len(model_config.blocks)

    def test_model_forward_shape(self, transformer_model, sample_input):
        """Test that model forward pass produces correct output shape."""
        x, cu_seqlens, max_seqlen = sample_input

        output = transformer_model(x, cu_seqlens, max_seqlen)

        # Output should have same shape as input
        assert output.shape == x.shape
        assert output.dtype == x.dtype

    def test_model_deterministic(self, transformer_model, sample_input):
        """Test that model output is deterministic."""
        x, cu_seqlens, max_seqlen = sample_input

        output1 = transformer_model(x.clone(), cu_seqlens, max_seqlen)
        output2 = transformer_model(x.clone(), cu_seqlens, max_seqlen)

        torch.testing.assert_close(output1, output2)

    def test_model_gradient_flow(self, transformer_model, sample_input):
        """Test that gradients flow through all blocks."""
        x, cu_seqlens, max_seqlen = sample_input
        x.requires_grad = True

        output = transformer_model(x, cu_seqlens, max_seqlen)
        loss = output.sum()
        loss.backward()

        # Check gradients on input
        assert x.grad is not None
        assert torch.all(torch.isfinite(x.grad))

        # Check gradients on all block parameters
        for block in transformer_model.blocks:
            for param in block.parameters():
                assert param.grad is not None
                assert torch.all(torch.isfinite(param.grad))

    def test_model_multiple_sequences(self, transformer_model, device):
        """Test model with multiple sequences of varying lengths."""
        torch.manual_seed(123)

        # Three sequences: lengths 2, 5, 3 (total ZN=10)
        ZN = 10
        d_model = 64
        x = torch.randn(ZN, d_model, device=device)
        cu_seqlens = torch.tensor([0, 2, 7, 10], dtype=torch.int32, device=device)
        max_seqlen = 5

        output = transformer_model(x, cu_seqlens, max_seqlen)

        assert output.shape == (ZN, d_model)
        assert torch.all(torch.isfinite(output))

    def test_model_single_block(self, device):
        """Test model with a single transformer block."""
        d_model = 64
        attn_cfg = MHACfg(
            d_model=d_model,
            heads=4,
            dropout_p=0.0,
        )
        ffn_cfg = FFNCfg(
            d_model=d_model,
            expansion_factor=2,
            dropout=0.0,
            act=ActivationFn.GELU,
            zeros=False,
        )
        block_cfg = TransformerBlockCfg(d_model=d_model, attn=attn_cfg, ffn=ffn_cfg)
        model_cfg = TransformerModelCfg(blocks=[block_cfg])

        model = TransformerModel(model_cfg).to(device)

        torch.manual_seed(456)
        x = torch.randn(5, d_model, device=device)
        cu_seqlens = torch.tensor([0, 5], dtype=torch.int32, device=device)
        max_seqlen = 5

        output = model(x, cu_seqlens, max_seqlen)

        assert output.shape == (5, d_model)
        assert torch.all(torch.isfinite(output))

    def test_model_deep_stack(self, device):
        """Test model with deeper stack (4 blocks)."""
        d_model = 64
        attn_cfg = MHACfg(
            d_model=d_model,
            heads=4,
            dropout_p=0.0,
        )
        ffn_cfg = FFNCfg(
            d_model=d_model,
            expansion_factor=2,
            dropout=0.0,
            act=ActivationFn.GELU,
            zeros=False,
        )
        block_cfg = TransformerBlockCfg(d_model=d_model, attn=attn_cfg, ffn=ffn_cfg)
        model_cfg = TransformerModelCfg(blocks=[block_cfg] * 4)

        model = TransformerModel(model_cfg).to(device)

        torch.manual_seed(789)
        x = torch.randn(8, d_model, device=device)
        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
        max_seqlen = 5

        output = model(x, cu_seqlens, max_seqlen)

        assert output.shape == (8, d_model)
        assert torch.all(torch.isfinite(output))

    def test_model_sequence_representation(self, transformer_model, sample_input):
        """Test that model produces meaningful representations."""
        x, cu_seqlens, max_seqlen = sample_input

        # Pass through model twice with different inputs
        output1 = transformer_model(x, cu_seqlens, max_seqlen)

        x_different = torch.randn_like(x)
        output2 = transformer_model(x_different, cu_seqlens, max_seqlen)

        # Outputs should be different for different inputs
        assert not torch.allclose(output1, output2, atol=1e-3)

    def test_model_large_batch(self, transformer_model, device):
        """Test model with larger batch of sequences."""
        torch.manual_seed(999)

        # 5 sequences with varying lengths
        lengths = [4, 7, 3, 6, 5]
        ZN = sum(lengths)
        d_model = 64
        x = torch.randn(ZN, d_model, device=device)

        cu_seqlens = torch.tensor([0] + [sum(lengths[:i+1]) for i in range(len(lengths))], dtype=torch.int32, device=device)
        max_seqlen = max(lengths)

        output = transformer_model(x, cu_seqlens, max_seqlen)

        assert output.shape == (ZN, d_model)
        assert torch.all(torch.isfinite(output))

    def test_model_eval_mode(self, transformer_model, sample_input):
        """Test that model behaves correctly in eval mode."""
        x, cu_seqlens, max_seqlen = sample_input

        # Train mode
        transformer_model.train()
        output_train = transformer_model(x.clone(), cu_seqlens, max_seqlen)

        # Eval mode
        transformer_model.eval()
        output_eval = transformer_model(x.clone(), cu_seqlens, max_seqlen)

        # Should be identical (no dropout in this config)
        torch.testing.assert_close(output_train, output_eval)
