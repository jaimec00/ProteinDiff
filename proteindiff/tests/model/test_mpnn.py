"""
Sanity tests for MPNN components.
"""
import pytest
import torch
import numpy as np

from proteindiff.model.mpnn.mpnn import (
    MPNNBlock, MPNNBlockCfg,
    EdgeEncoder, EdgeEncoderCfg,
    MPNNModel, MPNNModelCfg
)
from proteindiff.model.utils.mlp import MPNNMLPCfg, FFNCfg
from proteindiff.static.constants import aa_2_lbl


class TestEdgeEncoder:
    """Test suite for the EdgeEncoder module."""

    @pytest.fixture
    def edge_encoder(self):
        """Create edge encoder with default configuration."""
        cfg = EdgeEncoderCfg(
            d_model=128,
            top_k=8,
            num_rbf=16,
            edge_mlp=MPNNMLPCfg(d_model=128)
        )
        return EdgeEncoder(cfg)

    @pytest.fixture
    def small_edge_encoder(self):
        """Create smaller edge encoder for faster tests."""
        cfg = EdgeEncoderCfg(
            d_model=64,
            top_k=4,
            num_rbf=8,
            edge_mlp=MPNNMLPCfg(d_model=64)
        )
        return EdgeEncoder(cfg)

    @pytest.fixture
    def sample_protein(self):
        """Create a simple sample protein with backbone coordinates and frames."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Create a small protein with 5 residues
        ZN = 5

        # Create realistic backbone coordinates (N, CA, C, O)
        coords_bb = torch.zeros(ZN, 4, 3)
        for i in range(ZN):
            base_x = i * 3.8
            coords_bb[i, 0] = torch.tensor([base_x, 0.0, 0.0]) + torch.randn(3) * 0.1  # N
            coords_bb[i, 1] = torch.tensor([base_x + 1.46, 0.0, 0.0]) + torch.randn(3) * 0.1  # CA
            coords_bb[i, 2] = torch.tensor([base_x + 2.98, 0.0, 0.0]) + torch.randn(3) * 0.1  # C
            coords_bb[i, 3] = torch.tensor([base_x + 3.21, 1.23, 0.0]) + torch.randn(3) * 0.1  # O

        # Create orthonormal frames for each residue
        frames = torch.zeros(ZN, 3, 3)
        for i in range(ZN):
            # Simple rotation around z-axis by small random angle
            angle = torch.randn(1) * 0.1
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            frames[i, 0, :] = torch.tensor([cos_a, sin_a, 0.0])  # x-axis
            frames[i, 1, :] = torch.tensor([-sin_a, cos_a, 0.0])  # y-axis
            frames[i, 2, :] = torch.tensor([0.0, 0.0, 1.0])  # z-axis

        # Create sequence and chain positions (single chain)
        seq_pos = torch.arange(ZN)
        chain_pos = torch.zeros(ZN, dtype=torch.long)
        sample_idx = torch.zeros(ZN, dtype=torch.long)

        return coords_bb, frames, seq_pos, chain_pos, sample_idx

    def test_edge_encoder_initialization(self, edge_encoder):
        """Test that edge encoder initializes correctly."""
        assert hasattr(edge_encoder, 'rbf_centers')
        assert hasattr(edge_encoder, 'top_k')
        assert edge_encoder.rbf_centers.shape[0] == 16
        assert edge_encoder.top_k == 8

    def test_get_neighbors_single_sample(self, small_edge_encoder, sample_protein):
        """Test neighbor finding for a single sample."""
        coords_bb, _, _, _, sample_idx = sample_protein

        nbrs, nbr_mask = small_edge_encoder._get_neighbors(coords_bb, sample_idx)

        # Check output shapes
        ZN = coords_bb.shape[0]
        K = small_edge_encoder.top_k
        assert nbrs.shape == (ZN, K)
        assert nbr_mask.shape == (ZN, K)

        # Check that all neighbors are valid indices
        assert torch.all(nbrs >= 0)
        assert torch.all(nbrs < ZN)

        # Check that masked neighbors point to self
        for i in range(ZN):
            masked = ~nbr_mask[i]
            if masked.any():
                assert torch.all(nbrs[i, masked] == i)

    def test_get_neighbors_batched(self, small_edge_encoder):
        """Test neighbor finding with multiple samples in batch."""
        torch.manual_seed(123)

        # Create two separate chains with 3 residues each
        ZN = 6
        coords_bb = torch.randn(ZN, 4, 3) * 10.0

        # First 3 residues are sample 0, last 3 are sample 1
        sample_idx = torch.tensor([0, 0, 0, 1, 1, 1])

        nbrs, nbr_mask = small_edge_encoder._get_neighbors(coords_bb, sample_idx)

        # Check that neighbors respect sample boundaries
        for i in range(ZN):
            valid_nbrs = nbrs[i, nbr_mask[i]]
            sample_i = sample_idx[i]
            # All valid neighbors should be from the same sample
            assert torch.all(sample_idx[valid_nbrs] == sample_i)

    def test_get_rbfs(self, small_edge_encoder, sample_protein):
        """Test RBF distance encoding."""
        coords_bb, _, _, _, sample_idx = sample_protein
        nbrs, _ = small_edge_encoder._get_neighbors(coords_bb, sample_idx)

        rbfs = small_edge_encoder._get_rbfs(coords_bb, nbrs)

        # Check output shape
        ZN, K = nbrs.shape
        num_rbf = small_edge_encoder.rbf_centers.shape[0]
        expected_shape = (ZN, K, num_rbf * 4 * 4)  # 4 backbone atoms x 4 backbone atoms
        assert rbfs.shape == expected_shape

        # Check that RBFs are positive (Gaussian functions)
        assert torch.all(rbfs >= 0)
        assert torch.all(rbfs <= 1)  # RBF values should be <= 1

    def test_get_frames(self, small_edge_encoder, sample_protein):
        """Test relative frame computation."""
        coords_bb, frames, _, _, sample_idx = sample_protein
        nbrs, _ = small_edge_encoder._get_neighbors(coords_bb, sample_idx)

        rel_frames = small_edge_encoder._get_frames(frames, nbrs)

        # Check output shape
        ZN, K = nbrs.shape
        assert rel_frames.shape == (ZN, K, 9)  # 3x3 flattened

        # Check that relative frames are valid rotation matrices (when reshaped)
        for i in range(min(3, ZN)):  # Check first few
            for j in range(min(2, K)):
                R = rel_frames[i, j].reshape(3, 3)
                # Check that it's approximately orthogonal: R^T R ≈ I
                RTR = torch.matmul(R.T, R)
                assert torch.allclose(RTR, torch.eye(3), atol=1e-4)

    def test_get_seq_pos(self, small_edge_encoder, sample_protein):
        """Test relative sequence position encoding."""
        coords_bb, _, seq_pos, chain_pos, sample_idx = sample_protein
        nbrs, _ = small_edge_encoder._get_neighbors(coords_bb, sample_idx)

        rel_idx = small_edge_encoder._get_seq_pos(seq_pos, chain_pos, nbrs)

        # Check output shape
        ZN, K = nbrs.shape
        assert rel_idx.shape == (ZN, K)

        # Check that indices are in valid range [0, 65] (after offset of +32)
        assert torch.all(rel_idx >= 0)
        assert torch.all(rel_idx <= 65)

        # For same chain neighbors, check relative position calculation
        for i in range(ZN):
            for j in range(K):
                nbr_idx = nbrs[i, j]
                if chain_pos[i] == chain_pos[nbr_idx]:
                    expected_diff = torch.clamp(seq_pos[nbr_idx] - seq_pos[i], min=-32, max=32)
                    assert rel_idx[i, j] == expected_diff + 32

    def test_get_seq_pos_different_chains(self, small_edge_encoder):
        """Test that different chains get special token (33 + 32 = 65)."""
        torch.manual_seed(456)

        ZN = 6
        coords_bb = torch.randn(ZN, 4, 3) * 10.0
        seq_pos = torch.arange(ZN)
        chain_pos = torch.tensor([0, 0, 0, 1, 1, 1])  # Two chains
        sample_idx = torch.zeros(ZN, dtype=torch.long)

        nbrs, _ = small_edge_encoder._get_neighbors(coords_bb, sample_idx)
        rel_idx = small_edge_encoder._get_seq_pos(seq_pos, chain_pos, nbrs)

        # Check that cross-chain neighbors get special token (65)
        for i in range(ZN):
            for j in range(nbrs.shape[1]):
                nbr_idx = nbrs[i, j]
                if chain_pos[i] != chain_pos[nbr_idx]:
                    assert rel_idx[i, j] == 65

    def test_forward_pass(self, small_edge_encoder, sample_protein):
        """Test complete forward pass through edge encoder."""
        coords_bb, frames, seq_pos, chain_pos, sample_idx = sample_protein

        edges, nbrs, nbr_mask = small_edge_encoder(coords_bb, frames, seq_pos, chain_pos, sample_idx)

        # Check output shapes
        ZN = coords_bb.shape[0]
        K = small_edge_encoder.top_k
        d_model = 64

        assert edges.shape == (ZN, K, d_model)
        assert nbrs.shape == (ZN, K)
        assert nbr_mask.shape == (ZN, K)

        # Check that edges are finite
        assert torch.all(torch.isfinite(edges))

    def test_deterministic_output(self, small_edge_encoder):
        """Test that edge encoder output is deterministic."""
        torch.manual_seed(789)
        np.random.seed(789)

        # Create sample data
        ZN = 5
        coords_bb = torch.randn(ZN, 4, 3) * 10.0
        frames = torch.eye(3).unsqueeze(0).expand(ZN, 3, 3).clone()
        seq_pos = torch.arange(ZN)
        chain_pos = torch.zeros(ZN, dtype=torch.long)
        sample_idx = torch.zeros(ZN, dtype=torch.long)

        # Run twice with same input
        edges1, nbrs1, mask1 = small_edge_encoder(coords_bb, frames, seq_pos, chain_pos, sample_idx)
        edges2, nbrs2, mask2 = small_edge_encoder(coords_bb, frames, seq_pos, chain_pos, sample_idx)

        # Should be identical
        torch.testing.assert_close(edges1, edges2)
        torch.testing.assert_close(nbrs1, nbrs2)
        torch.testing.assert_close(mask1, mask2)


class TestMPNNBlock:
    """Test suite for the MPNNBlock module."""

    @pytest.fixture
    def mpnn_block(self):
        """Create MPNN block with default configuration."""
        cfg = MPNNBlockCfg(
            d_model=128,
            node_mlp=MPNNMLPCfg(d_model=128, hidden_layers=1),
            ffn_mlp=FFNCfg(d_model=128, expansion_factor=2),
            edge_mlp=MPNNMLPCfg(d_model=128, hidden_layers=1)
        )
        return MPNNBlock(cfg)

    @pytest.fixture
    def small_mpnn_block(self):
        """Create smaller MPNN block for faster tests."""
        cfg = MPNNBlockCfg(
            d_model=64,
            node_mlp=MPNNMLPCfg(d_model=64, hidden_layers=0),
            ffn_mlp=FFNCfg(d_model=64, expansion_factor=2),
            edge_mlp=MPNNMLPCfg(d_model=64, hidden_layers=0)
        )
        return MPNNBlock(cfg)

    @pytest.fixture
    def sample_graph(self):
        """Create sample graph data for testing."""
        torch.manual_seed(42)

        ZN = 5
        K = 4
        d_model = 64

        nodes = torch.randn(ZN, d_model)
        edges = torch.randn(ZN, K, d_model)
        nbrs = torch.randint(0, ZN, (ZN, K))
        nbr_mask = torch.ones(ZN, K, dtype=torch.bool)

        # Mask out some neighbors
        nbr_mask[0, -1] = False
        nbr_mask[1, -2:] = False

        return nodes, edges, nbrs, nbr_mask

    def test_mpnn_block_initialization(self, mpnn_block):
        """Test that MPNN block initializes correctly."""
        assert hasattr(mpnn_block, 'node_mlp')
        assert hasattr(mpnn_block, 'ffn')
        assert hasattr(mpnn_block, 'ln1')
        assert hasattr(mpnn_block, 'ln2')
        assert mpnn_block._update_edges

    def test_mpnn_block_no_edge_update(self):
        """Test MPNN block without edge updates."""
        cfg = MPNNBlockCfg(
            d_model=64,
            node_mlp=MPNNMLPCfg(d_model=64, hidden_layers=0),
            ffn_mlp=FFNCfg(d_model=64, expansion_factor=2),
            edge_mlp=None
        )
        block = MPNNBlock(cfg)

        assert not block._update_edges
        # Note: edge_mlp attribute won't exist when _update_edges is False

    def test_create_message(self, small_mpnn_block, sample_graph):
        """Test message creation from nodes and edges."""
        nodes, edges, nbrs, nbr_mask = sample_graph

        message = small_mpnn_block._create_msg(nodes, edges, nbrs)

        # Check output shape
        ZN, K, d_model = edges.shape
        assert message.shape == (ZN, K, 3 * d_model)

        # Check that message contains node_i, node_j, and edge features
        # This is implicit in the concatenation, but we can verify the shape

    def test_node_message_passing(self, small_mpnn_block, sample_graph):
        """Test node message passing."""
        nodes, edges, nbrs, nbr_mask = sample_graph

        nodes_out = small_mpnn_block._node_msg(nodes, edges, nbrs, nbr_mask)

        # Check output shape
        assert nodes_out.shape == nodes.shape

        # Check that output is different from input (message passing occurred)
        assert not torch.allclose(nodes_out, nodes)

        # Check that output is finite
        assert torch.all(torch.isfinite(nodes_out))

    def test_edge_message_passing(self, small_mpnn_block, sample_graph):
        """Test edge message passing."""
        nodes, edges, nbrs, nbr_mask = sample_graph

        edges_out = small_mpnn_block._edge_msg(nodes, edges, nbrs)

        # Check output shape
        assert edges_out.shape == edges.shape

        # Check that output is different from input (message passing occurred)
        assert not torch.allclose(edges_out, edges)

        # Check that output is finite
        assert torch.all(torch.isfinite(edges_out))

    def test_masking_effect(self, small_mpnn_block):
        """Test that masking properly zeros out neighbor contributions."""
        torch.manual_seed(999)

        ZN, K, d_model = 5, 4, 64
        nodes = torch.randn(ZN, d_model)
        edges = torch.randn(ZN, K, d_model)
        nbrs = torch.randint(0, ZN, (ZN, K))

        # Run with full mask
        nbr_mask_full = torch.ones(ZN, K, dtype=torch.bool)
        nodes_full, _ = small_mpnn_block(nodes.clone(), edges.clone(), nbrs, nbr_mask_full)

        # Run with partial mask
        nbr_mask_partial = nbr_mask_full.clone()
        nbr_mask_partial[0, :] = False  # Mask all neighbors of node 0
        nodes_partial, _ = small_mpnn_block(nodes.clone(), edges.clone(), nbrs, nbr_mask_partial)

        # Node 0 should receive different updates
        assert not torch.allclose(nodes_full[0], nodes_partial[0])

    def test_forward_pass(self, small_mpnn_block, sample_graph):
        """Test complete forward pass through MPNN block."""
        nodes, edges, nbrs, nbr_mask = sample_graph

        nodes_out, edges_out = small_mpnn_block(nodes, edges, nbrs, nbr_mask)

        # Check output shapes
        assert nodes_out.shape == nodes.shape
        assert edges_out.shape == edges.shape

        # Check that outputs are finite
        assert torch.all(torch.isfinite(nodes_out))
        assert torch.all(torch.isfinite(edges_out))

        # Check that outputs changed from inputs
        assert not torch.allclose(nodes_out, nodes)
        assert not torch.allclose(edges_out, edges)

    def test_residual_connections(self, small_mpnn_block, sample_graph):
        """Test that residual connections preserve information."""
        nodes, edges, nbrs, nbr_mask = sample_graph

        nodes_out, edges_out = small_mpnn_block(nodes, edges, nbrs, nbr_mask)

        # With residual connections, the change should be bounded
        node_change = torch.norm(nodes_out - nodes, dim=-1)
        edge_change = torch.norm(edges_out - edges, dim=-1)

        # Changes should be reasonable (not exploding)
        assert torch.all(node_change < 100.0)
        assert torch.all(edge_change < 100.0)


class TestMPNNModel:
    """Test suite for the complete MPNN model."""

    @pytest.fixture
    def mpnn_model(self):
        """Create MPNN model with default configuration."""
        edge_cfg = EdgeEncoderCfg(
            d_model=128,
            top_k=8,
            num_rbf=16,
            edge_mlp=MPNNMLPCfg(d_model=128)
        )
        block_cfg = MPNNBlockCfg(
            d_model=128,
            node_mlp=MPNNMLPCfg(d_model=128, hidden_layers=1),
            ffn_mlp=FFNCfg(d_model=128, expansion_factor=2),
            edge_mlp=MPNNMLPCfg(d_model=128, hidden_layers=1)
        )
        cfg = MPNNModelCfg(
            edge_encoder_cfg=edge_cfg,
            blocks=[block_cfg, block_cfg]
        )
        return MPNNModel(cfg)

    @pytest.fixture
    def small_mpnn_model(self):
        """Create smaller MPNN model for faster tests."""
        edge_cfg = EdgeEncoderCfg(
            d_model=64,
            top_k=4,
            num_rbf=8,
            edge_mlp=MPNNMLPCfg(d_model=64)
        )
        block_cfg = MPNNBlockCfg(
            d_model=64,
            node_mlp=MPNNMLPCfg(d_model=64, hidden_layers=0),
            ffn_mlp=FFNCfg(d_model=64, expansion_factor=2),
            edge_mlp=MPNNMLPCfg(d_model=64, hidden_layers=0)
        )
        cfg = MPNNModelCfg(
            edge_encoder_cfg=edge_cfg,
            blocks=[block_cfg]
        )
        return MPNNModel(cfg)

    @pytest.fixture
    def sample_protein(self):
        """Create a simple sample protein with all required inputs."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Create a small protein with 6 residues
        ZN = 6
        d_model = 64

        # Create realistic backbone coordinates (N, CA, C, O)
        coords_bb = torch.zeros(ZN, 4, 3)
        for i in range(ZN):
            base_x = i * 3.8
            coords_bb[i, 0] = torch.tensor([base_x, 0.0, 0.0]) + torch.randn(3) * 0.1  # N
            coords_bb[i, 1] = torch.tensor([base_x + 1.46, 0.0, 0.0]) + torch.randn(3) * 0.1  # CA
            coords_bb[i, 2] = torch.tensor([base_x + 2.98, 0.0, 0.0]) + torch.randn(3) * 0.1  # C
            coords_bb[i, 3] = torch.tensor([base_x + 3.21, 1.23, 0.0]) + torch.randn(3) * 0.1  # O

        # Create orthonormal frames
        frames = torch.zeros(ZN, 3, 3)
        for i in range(ZN):
            angle = torch.randn(1) * 0.1
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            frames[i, 0, :] = torch.tensor([cos_a, sin_a, 0.0])
            frames[i, 1, :] = torch.tensor([-sin_a, cos_a, 0.0])
            frames[i, 2, :] = torch.tensor([0.0, 0.0, 1.0])

        # Create sequence and chain positions
        seq_pos = torch.arange(ZN)
        chain_pos = torch.zeros(ZN, dtype=torch.long)
        sample_idx = torch.zeros(ZN, dtype=torch.long)

        # Create initial node features
        nodes = torch.randn(ZN, d_model)

        return coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes

    def test_mpnn_model_initialization(self, mpnn_model):
        """Test that MPNN model initializes correctly."""
        assert hasattr(mpnn_model, 'edge_encoder')
        assert hasattr(mpnn_model, 'mpnn_blocks')
        assert len(mpnn_model.mpnn_blocks) == 2

    def test_forward_pass(self, small_mpnn_model, sample_protein):
        """Test complete forward pass through MPNN model."""
        coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes = sample_protein

        nodes_out = small_mpnn_model(coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes)

        # Check output shape
        assert nodes_out.shape == nodes.shape

        # Check that output is finite
        assert torch.all(torch.isfinite(nodes_out))

        # Check that output changed from input
        assert not torch.allclose(nodes_out, nodes)

    def test_batched_processing(self, small_mpnn_model):
        """Test processing multiple proteins in batch."""
        torch.manual_seed(123)

        # Create two separate chains with 3 residues each
        ZN = 6
        d_model = 64

        coords_bb = torch.randn(ZN, 4, 3) * 10.0
        frames = torch.eye(3).unsqueeze(0).expand(ZN, 3, 3).clone()
        seq_pos = torch.tensor([0, 1, 2, 0, 1, 2])
        chain_pos = torch.tensor([0, 0, 0, 1, 1, 1])
        sample_idx = torch.tensor([0, 0, 0, 1, 1, 1])
        nodes = torch.randn(ZN, d_model)

        # Should process without error
        nodes_out = small_mpnn_model(coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes)

        assert nodes_out.shape == (ZN, d_model)
        assert torch.all(torch.isfinite(nodes_out))

    def test_sample_isolation(self, small_mpnn_model):
        """Test that different samples don't interact during message passing."""
        torch.manual_seed(456)

        # Create two well-separated chains
        ZN = 6
        d_model = 64

        coords_bb = torch.zeros(ZN, 4, 3)
        # First chain at x=0, second chain at x=1000 (very far)
        for i in range(3):
            coords_bb[i] = torch.randn(4, 3) + torch.tensor([0.0, 0.0, 0.0])
        for i in range(3, 6):
            coords_bb[i] = torch.randn(4, 3) + torch.tensor([1000.0, 0.0, 0.0])

        frames = torch.eye(3).unsqueeze(0).expand(ZN, 3, 3).clone()
        seq_pos = torch.tensor([0, 1, 2, 0, 1, 2])
        chain_pos = torch.tensor([0, 0, 0, 1, 1, 1])
        sample_idx = torch.tensor([0, 0, 0, 1, 1, 1])

        # Different initial features for each sample
        nodes = torch.randn(ZN, d_model)
        nodes[0:3] *= 10.0  # Sample 0 has large values
        nodes[3:6] *= 0.1   # Sample 1 has small values

        nodes_out = small_mpnn_model(coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes)

        # Check that sample 1 output doesn't have large values from sample 0
        # (i.e., no information leaked across samples)
        sample0_scale = torch.norm(nodes_out[0:3])
        sample1_scale = torch.norm(nodes_out[3:6])

        # Sample 1 should maintain relatively small scale
        assert sample1_scale < sample0_scale

    def test_multiple_blocks(self, mpnn_model, sample_protein):
        """Test that multiple MPNN blocks process sequentially."""
        coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes = sample_protein
        # Need to adjust node dimension to match mpnn_model's d_model=128
        nodes = torch.randn(coords_bb.shape[0], 128)

        nodes_out = mpnn_model(coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes)

        # Check that multi-block processing works
        assert nodes_out.shape == nodes.shape
        assert torch.all(torch.isfinite(nodes_out))

    def test_deterministic_output(self, small_mpnn_model):
        """Test that MPNN model output is deterministic."""
        torch.manual_seed(789)
        np.random.seed(789)

        # Create sample data
        ZN = 5
        d_model = 64

        coords_bb = torch.randn(ZN, 4, 3) * 10.0
        frames = torch.eye(3).unsqueeze(0).expand(ZN, 3, 3).clone()
        seq_pos = torch.arange(ZN)
        chain_pos = torch.zeros(ZN, dtype=torch.long)
        sample_idx = torch.zeros(ZN, dtype=torch.long)
        nodes = torch.randn(ZN, d_model)

        # Run twice with same input
        nodes_out1 = small_mpnn_model(
            coords_bb.clone(), frames.clone(), seq_pos.clone(),
            chain_pos.clone(), sample_idx.clone(), nodes.clone()
        )
        nodes_out2 = small_mpnn_model(
            coords_bb.clone(), frames.clone(), seq_pos.clone(),
            chain_pos.clone(), sample_idx.clone(), nodes.clone()
        )

        # Should be identical
        torch.testing.assert_close(nodes_out1, nodes_out2)

    def test_gradient_flow(self, small_mpnn_model, sample_protein):
        """Test that gradients flow through the model."""
        coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes = sample_protein

        # Enable gradients
        nodes.requires_grad = True

        nodes_out = small_mpnn_model(coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes)

        # Compute a simple loss
        loss = nodes_out.sum()
        loss.backward()

        # Check that gradients exist and are non-zero
        assert nodes.grad is not None
        assert torch.any(nodes.grad != 0)

    def test_varying_sequence_lengths(self, small_mpnn_model):
        """Test processing proteins of different lengths."""
        torch.manual_seed(999)

        for ZN in [3, 5, 10, 20]:
            d_model = 64

            coords_bb = torch.randn(ZN, 4, 3) * 10.0
            frames = torch.eye(3).unsqueeze(0).expand(ZN, 3, 3).clone()
            seq_pos = torch.arange(ZN)
            chain_pos = torch.zeros(ZN, dtype=torch.long)
            sample_idx = torch.zeros(ZN, dtype=torch.long)
            nodes = torch.randn(ZN, d_model)

            nodes_out = small_mpnn_model(coords_bb, frames, seq_pos, chain_pos, sample_idx, nodes)

            assert nodes_out.shape == (ZN, d_model)
            assert torch.all(torch.isfinite(nodes_out))
