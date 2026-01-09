# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ProteinDiff is a **multimodal protein generative model** that understands and generates proteins conditioned on partial sequence and/or partial structure. The model uses **Mixture of Queries (MoQ)** to compress variable-length protein representations into fixed-size latent vectors, enabling flexible conditional generation and latent space manipulation.

**Key Innovations**:
1. **Mixture of Queries (MoQ)**: Soft-weighted routing selects learnable query vectors that compress variable-length proteins into fixed-size latent representations via cross-attention
2. **Voxelized Side Chain Environments**: Transforms variable-atom side chains into fixed 3D electric field voxels, computes divergence to create a Euclidean-manifold representation
3. **Flexible Conditioning**: Hybrid MPNN/1D-CNN processes partial sequence and structure, enabling generation from any combination of known information
4. **Fixed-Size Latent Diffusion**: Operates on fixed number of latent vectors regardless of protein length, enabling faster training, latent interpolation, and multi-length generation

**Current Status**: Architecture redesign in progress. New model will replace previous architecture while reusing preprocessing pipeline (voxelization/divergence). See [idea.md](idea.md) for complete architecture specification.

## Development Setup

### Environment Setup

The project uses Docker for consistent development environments. Before starting:

1. **Set up data paths** in `config/setup/.env`:
```bash
DATA_PATH=/path/to/data
EXP_PATH=/path/where/experiments/are/written/to
```

2. **Download training data** (ProteinMPNN dataset):
```bash
DATA_PATH=/PATH/TO/YOUR/DATA && \
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz -P $DATA_PATH && \
tar -xzf $DATA_PATH/pdb_2021aug02.tar.gz -C $DATA_PATH && \
rm $DATA_PATH/pdb_2021aug02.tar.gz
```

3. **Pull/Build Docker image**:
```bash
# For sm80 GPUs (Ampere):
sudo docker pull jaimec00/proteindiff:cu12.8-sm80
sudo docker tag jaimec00/proteindiff:cu12.8-sm80 proteindiff:dev

# For sm90 GPUs (Hopper) - build locally:
sudo docker build -t proteindiff:dev -f config/setup/Dockerfile .
```

4. **Run development environment**:
```bash
# Interactive shell for debugging
./config/setup/start.sh debug

# Run training directly
./config/setup/start.sh train
```

### Running Tests

Tests are located in `src/tests/`. Run them using pytest inside the Docker container:

```bash
# Run all tests
pytest src/tests/

# Run specific test file
pytest src/tests/test_train_loop.py

# Run specific test
pytest src/tests/test_model.py::Test::test_sc_vae -v
```

Tests require the `DATA_PATH` environment variable to be set (handled by docker-compose).

### Training Commands

Training is configured via `config/train.yml`. The main entry point is `src/training/train/train.py`.

```bash
# Train with default config
python src/training/train/train.py --config config/train.yml

# Override config parameters (example)
python src/training/train/train.py --config config/train.yml \
    --training_parameters.train_type sc_vae \
    --training_parameters.epochs 100
```

**Two-Stage Training Process** (New Architecture):
1. **Stage 1 - VAE**: Train encoder + decoder end-to-end with multi-task losses
2. **Stage 2 - Diffusion**: Train conditioning network + diffusion model with frozen VAE

**Note**: Previous three-stage training (sc_vae → bb_vae → diffusion) is being replaced with new architecture. See [README.md](README.md) for complete details.

## Architecture Overview

**See [README.md](README.md) for complete architecture specification.**

### Architecture Change Summary

**Previous Architecture** (being replaced):
- Three-stage training: Side Chain VAE → Backbone VAE → Latent Diffusion
- Separate voxel and backbone encoders producing variable-length latents (ZN, d_latent)
- Diffusion operated on variable-length concatenated latents

**New Architecture** (current):
- Two-stage training: VAE → Latent Diffusion
- Unified encoder with **Mixture of Queries (MoQ)** producing fixed-size latents (N_latent, d_latent)
- Flexible conditioning network for partial sequence/structure
- Fixed-size latent diffusion enabling latent interpolation and multi-length generation

### High-Level Pipeline

```
ProteinDiff (New Architecture)
├── PreProcesser (reused from previous)
│   └── Voxelizes side chains → electric field divergence
├── VAE Encoder
│   ├── CNN: Flatten voxels → vectors (ZN, d_voxel)
│   ├── GNN: MPNN-style message passing with backbone edges
│   ├── Transformer: Self-attention on (ZN, d_model) tokens
│   ├── MoQ: Mixture of Queries → fixed (N_latent, d_model)
│   └── Latent Sampling: Linear → (N_latent, d_latent)
├── VAE Decoder
│   ├── Latent Self-Attention
│   ├── Cross-Attention: Empty positional tokens attend to latents
│   ├── Token Self-Attention
│   └── Multi-Task Heads: Voxels, AA labels, distogram, anglogram, coords, torsions
├── Conditioning Network (for diffusion)
│   ├── Hybrid MPNN/1D-CNN: Process partial seq/struct with masking
│   ├── Self-Attention: Integrate partial information
│   └── MoQ: Compress to (N_latent, d_model) conditioning vectors
└── Latent Diffusion
    ├── AdaLN Conditioning: Timestep + partial seq/struct
    └── DiT Blocks: Denoise fixed-size latents
```

### Key Architecture Components

**Preprocessing** (Reused from previous architecture):
- Compute virtual Cβ atoms from empirical constants
- Define local coordinate frames (y: Cα→Cβ, x: N→C projected, z: cross product)
- Voxelize side chains (default: 8×8×8 grid, cell_dim=1.0Å)
- Compute electric fields using AMBER ff19SB partial charges
- Normalize to unit vectors → compute divergence (scalar field)
- **Deterministic**: No randomness in preprocessing ensures reproducibility

**Mixture of Queries (MoQ)** (New component):
- Learnable query bank with N_query_bank vectors (16-32x larger than N_latent)
- **Soft-weighted router** selects N_latent queries based on input tokens
- Selected queries attend to tokens via cross-attention
- Compresses variable-length (ZN) → fixed-size (N_latent) representations
- **Two independent MoQ modules**: one in encoder, one in conditioning network
- Enables latent interpolation and multi-length generation

**VAE Encoder** (New design):
- Input: Full atomic coordinates `(ZN, 14, 3)` + labels
- Pipeline: Voxels → CNN flatten → GNN (backbone edges) → Transformer → MoQ → latents
- Output: Fixed-size latents `(N_latent, d_latent)` regardless of protein length
- Training: End-to-end with multi-task decoder losses

**VAE Decoder** (New design):
- Input: Fixed-size latents `(N_latent, d_latent)`
- Creates empty positional tokens `(ZN, d_model)` with sinusoidal PE
- Cross-attention: Empty tokens (queries) attend to latents (keys/values)
- Multi-task heads output: voxels, AA labels, distogram, anglogram, coordinates, torsions
- Reconstructs same number of residues as input

**Conditioning Network** (New component):
- Input: Partial sequence (masked AA labels) + partial structure (masked backbone coords)
- **Hybrid MPNN/1D-CNN**:
  - MPNN: Only nodes with coordinates participate in message passing (top-k neighbors)
  - 1D CNN: After each message passing, small conv (kernel 3-6) propagates info to masked neighbors
  - Careful masking ensures samples don't interact (no padding, use gather/scatter)
- Self-attention integrates partial information
- MoQ compresses to `(N_latent, d_model)` conditioning vectors matching latent dimensionality

**Latent Diffusion** (New design):
- Operates on fixed-size latents `(N_latent, d_latent)`
- AdaLN conditioning: modulates based on timestep + partial seq/struct from conditioning network
- DiT-style transformer blocks denoise latents
- Loss: MSE noise prediction

### Key Technical Details

**Variable-Length Batching**: Uses token-based batching (not fixed batch size). Batches accumulate sequences until `batch_tokens` residues are reached. FlashAttention handles variable lengths via `cu_seqlens` (cumulative sequence lengths) and `max_seqlen`.

**MPNN Edge Features**:
- 16 RBF basis functions for inter-residue distances (all backbone atom pairs: N, CA, C, O)
- 9 values from flattened rotation matrix (source→neighbor frame transformation)
- 1 relative sequence position embedding
- Used in both encoder GNN and conditioning network MPNN

**MoQ Routing Details**:
- Soft-weighted top-k selection (not hard selection)
- Router sees all tokens in sample and selects queries for entire sample (all tokens see same queries)
- Query diversity loss encourages orthogonal query representations: `||Q^T Q - I||^2`
- MoE-style load balancing and entropy regularization losses

**Masking Strategy** (Conditioning Network):
- Random masking percentage sampled from uniform distribution (possibly ramped)
- Sequence and structure can be independently masked
- GNN topology computed only from known coordinates
- 1D conv propagates structural info to masked neighbors after each message passing step
- Implementation: Masking + gather/scatter (no padding) integrates with FlashAttention

**Fixed-Size Latents Enable**:
- Latent interpolation between known proteins
- Multi-length generation (vary number of decoder empty tokens)
- Faster, more stable diffusion training
- Conditional/unconditional generation from same latent space

## Training Configuration

Training is configured via `config/train.yml`. Key parameters include:

**Model Architecture**:
- `d_model`: Base feature dimension
- `d_latent`: Latent vector dimension
- `N_latent`: Number of fixed latent vectors (hyperparameter)
- `N_query_bank`: Size of MoQ query bank
- `top_k`: k-NN neighbors for GNN/MPNN
- `voxel_dims`: Voxel grid size (e.g., 8×8×8)
- `cell_dim`: Voxel cell size in Angstroms

**Training Parameters**:
- `train_type`: `"vae"` or `"diffusion"` (two-stage training)
- `batch_tokens`: Tokens per batch (variable-length sequences)
- `checkpoint`: Load pretrained VAE for diffusion stage
- Loss weights: KL divergence (`beta`), query diversity, routing losses

**Data Parameters**:
- `min_seq_size`, `max_seq_size`: Sequence length filters
- `max_resolution`: PDB resolution filter (Angstroms)
- `homo_thresh`: Homology clustering threshold
- Masking parameters for conditioning network training

## Code Structure

**Model Architecture** (to be implemented):
- `ProteinDiff`: Main model class routing to VAE or diffusion
- `VAEEncoder`: CNN → GNN → Transformer → MoQ → latent sampling
- `VAEDecoder`: Latent self-attn → cross-attn with empty tokens → multi-task heads
- `ConditioningNetwork`: Hybrid MPNN/1D-CNN → self-attn → MoQ
- `LatentDiffusion`: DiT blocks with AdaLN conditioning
- `MixtureOfQueries`: Router + query bank + cross-attention module

**Data Loading** (`src/training/data/`) - keeping existing:
- `DataHolder`: Manages train/val/test splits by cluster (prevents homology leakage)
- `Data`: IterableDataset that samples chains deterministically with buffering
- `BatchBuilder`: Accumulates samples until batch_tokens threshold, returns DataBatch
- `DataBatch`: Contains coords, labels, masks, positional info, and FlashAttention metadata

**Loss Functions** (to be updated):
- `VAELoss`: Multi-task losses (KL, voxel recon, AA classification, distogram, anglogram, coords, torsions, mutual info, query diversity, routing)
- `DiffusionLoss`: MSE noise prediction with conditioning
- Losses are sums over residues/latents (not means), requiring gradient clipping

**Training Loop** (to be updated):
- Two-stage training: VAE → Diffusion (with frozen VAE)
- Checkpoint loading/saving
- Module freezing based on train_type
- Early stopping on validation metrics

**Preprocessing** (`src/model/utils/preprocesser.py`) - keeping existing:
- Voxelization and electric field divergence computation
- Deterministic preprocessing pipeline

**Static Data** (`src/static/`) - keeping existing:
- `constants.py`: Amino acid alphabets, atom mappings, empirical constants
- `amber/amino19.lib`: AMBER ff19SB partial charges for all amino acids

## Important Conventions

**Coordinate Systems**:
- All atomic coordinates are in Angstroms
- Local frames are right-handed (x: N→C projected, y: Cα→Cβ, z: x×y)
- Voxel origin is at Cβ position

**Tensor Shapes**:
- `ZN`: Total residues in batch (variable)
- `ZB`: Batch size (number of sequences)
- `ZA`: Number of atoms per residue (14 for all-atom, 4 for backbone)
- Batches use `sample_idx` to map residues to sequences

**Gradient Flow**:
- VAE is frozen during diffusion training stage
- Use gradient clipping (norm=1.0) due to sum-based losses
- Query diversity and routing losses affect MoQ module training

**Preprocessing is Deterministic**: Voxelization and divergence computation are pure functions of coordinates. No randomness in preprocessing ensures reproducibility.

## Generation Capabilities (New Architecture)

**Unconditional Generation**:
- Sample `z ~ N(0, I)` with fixed `N_latent` vectors
- Run diffusion reverse process with fully masked conditioning
- Decode latents → full protein structure

**Conditional Generation**:
- Provide partial sequence (some AA labels, mask others)
- Provide partial structure (some backbone coords, mask others)
- Any combination of partial seq/struct
- Conditioning network processes known information
- Diffusion generates latents conditioned on partial input

**Latent Interpolation**:
- Encode two known proteins: `z_A`, `z_B` using VAE encoder
- Interpolate: `z_interp = α * z_A + (1-α) * z_B`
- Option 1: Decode directly with VAE decoder
- Option 2: Use interpolated latents as initialization for conditional/unconditional diffusion
- Enables exploration of latent space between known proteins

**Multi-Length Generation**:
- Fixed latent size `(N_latent, d_latent)` decouples from protein length
- Vary target length by adjusting number of empty decoder tokens
- Enables generation of proteins with different lengths from same latent distribution

## Docker Volume Mounts

When running via `docker-compose`:
- Source code: `/ProteinDiff/src` (live mount for development)
- Data: `/mnt/data` (read-only, set via DATA_PATH)
- Experiments: `/mnt/experiments` (write, set via EXP_PATH)

Checkpoints and logs are written to `/mnt/experiments` and persist outside the container.
