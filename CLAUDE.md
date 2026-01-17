# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProteinDiff is a multimodal protein generative model that generates proteins conditioned on partial sequence and/or structure. Uses a VAE with per-residue latents followed by latent diffusion.

**Status**: Architecture design phase (early development)

## Build System & Environment

Uses **pixi** (modern conda alternative) for environment management:

```bash
pixi shell                    # Drop into environment (auto-activates)
pixi shell -e gpu             # GPU environment with CUDA 12 + flash-attn
pixi run local_mlflow         # Start MLflow tracking server on localhost:5000
```

Environment variables set automatically via pixi:
- `EXP_DIR`: Experiment directory
- `MLFLOW_TRACKING_URI`: SQLite DB for MLflow tracking

## Running Training

```bash
python -m proteindiff.training.training_run
```

Uses Hydra config system with `configs/debug.yaml` as default entry point. Key config structure:
- `configs/model/simple.yaml` - Model architecture (d_model, d_latent, voxel_dim, etc.)
- `configs/data/default.yaml` - Data paths, batch_tokens, max_seq_size
- `configs/losses/vae.yaml` - Loss weights and parameters
- `configs/optim/adam.yaml`, `configs/scheduler/static.yaml` - Optimizer settings

Override configs via Hydra CLI: `python -m proteindiff.training.training_run model.d_model=512`

## Running Tests

```bash
pixi run pytest tests/                           # All tests
pixi run pytest tests/test_losses/ -v            # Loss tests with verbose output
pytest tests/test_losses/test_struct_losses/test_anglogram_loss.py  # Single test file
```

Note: Some tests (anglogram, distogram) require CUDA.

## Architecture

### Two-Stage Training
1. **VAE Stage**: Train VAE end-to-end (reconstruction + KL divergence)
2. **Diffusion Stage**: Freeze VAE, train conditioning network + diffusion model (not yet implemented)

### Model Pipeline (`proteindiff/model/`)

**Tokenizer** (`tokenizer/tokenizer.py`): Converts atomic coords to voxelized electric field divergence representation. Outputs backbone coords, divergence field, and local frames.

**VAE Encoder** (`vae/encoder.py`):
```
Divergence → CNN Downsample → MPNN (on residue graph) → Transformer → Latent Projection (mu, logvar)
```
Produces per-residue latent vectors.

**VAE Decoder** (`vae/decoder.py`):
```
Latent → Up Projection → Transformer → Multi-task Heads
```

**Output Heads**:
- Divergence reconstruction (MSE loss)
- Sequence prediction (cross-entropy)
- Distogram (binned pairwise distances)
- Anglogram (6 non-redundant dot products, uses fused Triton kernels)
- PAE/pLDDT (partially implemented)

**Diffusion** (`diffusion/diffusion.py`): DiT-style architecture (planned)

### Training Pipeline (`proteindiff/training/`)
- **Data loading** (`data/data_loader.py`): Token-based dynamic batching from ProteinMPNN dataset (pdb_2021aug02)
- **Losses** (`losses/training_loss.py`): Weighted combination of KL, reconstruction, sequence, and structure losses
- **Logging** (`logger.py`): MLflow integration for experiment tracking

### Key Types
`proteindiff/types/__init__.py` defines jaxtyping annotations used throughout:
- `Float`, `Int`, `Bool` tensor types with shape annotations
- Tensor shapes use conventions: `ZN` = total residues in batch, `Z` = number of samples

## Data

Training data: ProteinMPNN dataset. Download:
```bash
DATA_PATH=/path/to/data && \
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz -P $DATA_PATH && \
tar -xzf $DATA_PATH/pdb_2021aug02.tar.gz -C $DATA_PATH && \
rm $DATA_PATH/pdb_2021aug02.tar.gz
```

Configure path in `configs/data/default.yaml`.

## GPU/CUDA Notes

- CUDA 12.0+ required for GPU training
- Flash attention 2.8+ for transformer
- Custom Triton kernels for anglogram loss (fused forward/backward)
- TF32 matmuls enabled by default
- CPU fallback available for attention (`model/transformer/attention_cpu.py`)

## Current Development State

Implemented:
- VAE training pipeline
- Tokenizer, encoder, decoder
- Structure losses (distogram, anglogram with Triton)
- MLflow logging

Not yet implemented:
- Diffusion training stage
- Model inference/generation
- Multi-GPU support

## Other

make sure to use the skills and subagents you have available to delegate tasks