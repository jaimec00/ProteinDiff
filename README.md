
# Multimodal Protein Model with Mixture of Queries (MoQ)

## Project Overview

This is a **multimodal protein generative model** that understands and generates proteins conditioned on partial sequence and/or partial structure. Unlike the current ProteinDiff architecture, this model uses **Mixture of Queries (MoQ)** to compress variable-length protein representations into fixed-size latent vectors, enabling:

- **Stable latent diffusion** on fixed-size representations
- **Faster training** due to fixed latent dimensionality
- **Latent interpolation** between known proteins for generation
- **Flexible conditioning** on any combination of partial sequence/structure

**Key Innovation**: MoQ routing selects a fixed number of learnable query vectors that compress variable-length protein embeddings via cross-attention, creating a bottleneck that forces the model to learn compact, information-rich representations.

**Current Status**: Architecture design phase. Will reuse preprocessing (voxelization/divergence) from ProteinDiff but implement new encoder/decoder/conditioning networks.

---

## Setup

Thus far, I am still working on setting up the model architecture and training logic. Once this is done, I will move on to making the model scalable by adding hybrid parallelism (FSDP, PP, TP). For now, here is how to setup the environment to develop.


First, you need to pull the image. I pushed a docker image that works on sm80, you can pull it like this

```shell
# pull
sudo docker pull jaimec00/proteindiff:cu12.8-sm80

# change the tag so it matches what docker compose expects
sudo docker tag jaimec00/proteindiff:cu12.8-sm80 proteindiff:dev
```

I have not tried sm90, but you can try building it on a node with hopper gpus:

```shell
sudo docker build -t proteindiff:dev -f config/setup/Dockerfile .
```

Once the image is pulled/built, you can run this helper script to run a shell in the environment (debug service) or to directly train the model (train service). the helper script is running docker compose under the hood.

```shell
./config/setup/start.sh <train/debug>
```

You may also want to download the training data. as of right now, I am using the dataset curated for ProteinMPNN. In the future, I will make my own dataset which will include sample from PDB, AFDB, and ESMAtlas. here is how to download the PMPNN dataset (do this BEFORE running start.sh):

```shell
DATA_PATH=/PATH/TO/YOUR/DATA && \
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz -P $DATA_PATH && \
tar -xzf $DATA_PATH/pdb_2021aug02.tar.gz -C $DATA_PATH && \
rm $DATA_PATH/pdb_2021aug02.tar.gz
```

make sure to edit the ```config/setup/.env``` file so that it matches the path to the data, as docker compose reads this to mount the volumes.

```
# .env
DATA_PATH=/path/to/data
EXP_PATH=/path/where/experiments/are/written/to
```

## Architecture Components

### 1. VAE Encoder

**Input**: Full atomic coordinates `(ZN, 14, 3)` + amino acid labels

**Pipeline**:
1. **Preprocessing** (reused from ProteinDiff):
   - Compute local coordinate frames per residue
   - Voxelize side chain space around each residue
   - Compute electric field using AMBER ff19SB partial charges
   - Compute divergence → scalar field `(ZN, 1, V, V, V)` capturing local environment

2. **CNN Flattening**:
   - 3D CNN to flatten voxel `(ZN, 1, V, V, V)` → vector `(ZN, d_voxel)`
   - Embeds local residue environment

3. **GNN (ProteinMPNN-style)**:
   - **Nodes**: Flattened divergence vectors `(ZN, d_voxel)`
   - **Edges**: Derived from backbone geometry (RBF distances, rotation matrices, relative positions)
   - Message passing aggregates structural neighborhood information
   - Output: `(ZN, d_model)` with mixed local + structural features

4. **Transformer Self-Attention**:
   - Multiple layers of self-attention on `(ZN, d_model)` token representations
   - This is the **largest component** of the encoder
   - Captures long-range dependencies across the full protein

5. **Mixture of Queries (MoQ)**:
   - **Learnable query bank**: `(N_query_bank, d_model)` where `N_query_bank = 16-32 × N_latent`
   - **Router**: Soft-weighted (top-k) selection of `N_latent` query vectors based on full input
   - Router sees all tokens and selects queries for the entire sample (all tokens attend to same queries)
   - **Cross-attention**: Selected queries `(N_latent, d_model)` attend to encoder tokens `(ZN, d_model)`
   - Output: Fixed-size representation `(N_latent, d_model)`

6. **Query Self-Attention**:
   - Few layers of self-attention on `(N_latent, d_model)` query representations
   - Refines information compressed into fixed number of vectors

7. **Latent Sampling**:
   - Linear projection → `mean` and `logvar` `(N_latent, d_latent)`
   - Sample latent vectors `z ~ N(mean, exp(0.5 * logvar))`
   - Output: `(N_latent, d_latent)` fixed-size latent representation

**Key Parameters**:
- `N_latent`: Fixed number of latent vectors (hyperparameter, e.g., 64)
- `N_query_bank`: Size of learnable query bank (e.g., 1024-2048)
- `d_model`: Feature dimension throughout encoder
- `d_latent`: Latent vector dimension

---

### 2. VAE Decoder

**Input**: Latent vectors `(N_latent, d_latent)`

**Pipeline**:
1. **Latent Self-Attention**:
   - Few layers of self-attention on `(N_latent, d_latent)` latent vectors
   - Processes compressed information

2. **Empty Token Initialization**:
   - Create `(ZN, d_model)` "empty" placeholder tokens
   - **Only positional information**: Sinusoidal positional encodings based on sequence index
   - No amino acid or structural information

3. **Cross-Attention (Latent → Tokens)**:
   - Empty tokens `(ZN, d_model)` are **queries**
   - Latent vectors `(N_latent, d_model)` are **keys/values**
   - Decoder attends to latent representation to fill in token information

4. **Token Self-Attention**:
   - Self-attention on `(ZN, d_model)` decoded tokens
   - Refines reconstructed representations

5. **Multi-Task Output Heads**:
   - **Voxel Reconstruction**: `(ZN, 1, V, V, V)` reconstructed divergence voxels
   - **AA Classification**: `(ZN, 20)` amino acid logits
   - **Distogram**: `(ZN, ZN, n_bins)` pairwise distance distributions
   - **Anglogram**: `(ZN, ZN, n_angle_bins)` pairwise angle distributions
   - **Distance Map**: `(ZN, ZN)` pairwise distances
   - **Torsion Angles**: `(ZN, n_torsions)` backbone/side chain torsions
   - **Atomic Coordinates**: `(ZN, 14, 3)` reconstructed all-atom coordinates

**Output**: Reconstructs original protein with same number of residues `(ZN)` as input

---

### 3. Conditioning Network (for Diffusion)

**Purpose**: Encode partial sequence and/or partial structure into fixed-size conditioning vectors

**Input**:
- Amino acid labels `(ZN,)` with MASK tokens for unknown positions
- Backbone coordinates `(ZN, 4, 3)` [N, CA, C, O] with missing coordinates for masked positions

**Pipeline**:
1. **Node Initialization**:
   - Embed amino acid labels (including MASK tokens) → `(ZN, d_model)`

2. **Hybrid MPNN/1D-CNN**:
   - **GNN Component**:
     - **Nodes**: Only residues with known coordinates participate in message passing
     - **Edges**: Computed from backbone geometry (top-k neighbors, RBF distances, rotations)
     - Masked nodes (no coordinates) do **not** participate in GNN

   - **1D CNN Component**:
     - After each GNN message passing operation, apply small 1D convolution
     - Kernel size: 3-6 residues
     - Purpose: Propagate information from nodes with coordinates to nearby masked neighbors
     - Implementation: Use masking + gather/scatter (no padding) to integrate with FlashAttention variable-length sequences
     - Ensures samples don't interact via careful masking

   - **Result**: Nodes with coordinates get strong structural signal from GNN; masked nodes get weaker signal from 1D conv propagation

3. **Self-Attention**:
   - Multiple layers of self-attention on `(ZN, d_model)` conditioned tokens
   - Further integrates partial sequence and structure information

4. **Mixture of Queries (MoQ)**:
   - **Separate learnable query bank** from encoder (independent learning)
   - Same number of selected queries as encoder (`N_latent`)
   - Soft-weighted top-k routing based on conditioned tokens
   - Cross-attention: Selected queries attend to conditioned tokens
   - Output: `(N_latent, d_model)` conditioning vectors in same space as latent vectors

5. **Conditioning Self-Attention**:
   - Few layers of self-attention on `(N_latent, d_model)` conditioning vectors
   - Refines conditioning representation

**Output**: `(N_latent, d_model)` conditioning vectors matching latent dimensionality

---

### 4. Latent Diffusion Model

**Input**:
- Noisy latent vectors `z_t` `(N_latent, d_latent)`
- Conditioning vectors `c` `(N_latent, d_model)` from conditioning network
- Timestep `t`

**Pipeline**:
1. **Conditioning Integration**:
   - Project conditioning vectors to latent space if needed
   - Combine with timestep embeddings

2. **Denoising Network**:
   - Multiple layers of self-attention on `(N_latent, d_latent)` latent vectors
   - **AdaLN (Adaptive Layer Normalization)** conditioning:
     - Modulate features based on timestep `t`
     - Modulate based on partial structure (from conditioning network)
     - Modulate based on partial sequence (from conditioning network)
   - Architecture: DiT-style (Diffusion Transformer) blocks

3. **Noise Prediction**:
   - Predict noise `ε_θ(z_t, t, c)`
   - Loss: MSE between predicted and actual noise

**Diffusion Process**:
- Forward: Add Gaussian noise to clean latents from encoder
- Reverse: Iteratively denoise starting from random Gaussian noise
- Conditioning: Partial sequence/structure via conditioning network

---

## Loss Functions

### VAE Losses (Multi-Task)

1. **KL Divergence**: `D_KL(q(z|x) || p(z))` where `p(z) = N(0, I)`
2. **Voxel Reconstruction**: MSE between predicted and ground truth divergence voxels
3. **AA Classification**: Cross-entropy loss for amino acid prediction
4. **Distogram**: Cross-entropy over distance bins
5. **Anglogram**: Cross-entropy over angle bins
6. **Distance Map**: MSE for pairwise distances
7. **Torsion Loss**: MSE or circular loss for torsion angles
8. **Mutual Information Loss**: Maximize MI between latent vectors and reconstructions
9. **MoQ Routing Losses**:
   - Load balancing loss (similar to MoE)
   - Routing entropy regularization
10. **Query Diversity Loss**: Encourage different query vectors via dot product loss
    - Penalize high similarity between selected (or all) query vectors
    - `L_diversity = ||Q^T Q - I||^2` where Q is query matrix

### Diffusion Loss

- **Noise Prediction**: MSE between predicted noise and ground truth Gaussian noise
- `L_diffusion = ||ε - ε_θ(z_t, t, c)||^2`

---

## Training Strategy

### Stage 1: VAE Training
- Train encoder + decoder end-to-end with multi-task losses
- Input: Full atomic coordinates + labels
- Output: Reconstructions for all modalities
- Goal: Learn compressed fixed-size latent representation

### Stage 2: Diffusion Training
- **Freeze VAE** (encoder + decoder)
- Train conditioning network + diffusion model
- Input: Partial sequence/structure → conditioning network
- Latents: Sampled from frozen encoder on full protein
- Goal: Learn to denoise latents conditioned on partial information

---

## Generation Capabilities

### Unconditional Generation
- Sample `z ~ N(0, I)` with fixed `N_latent` vectors
- Run diffusion reverse process with fully masked conditioning (or no conditioning)
- Decode latents → full protein structure

### Conditional Generation
- **Partial Sequence**: Provide some AA labels, mask others
- **Partial Structure**: Provide some backbone coordinates, mask others
- **Hybrid**: Any combination of partial seq/struct
- Conditioning network processes partial information
- Diffusion generates latents conditioned on known information

### Latent Interpolation
- Encode two known proteins: `z_A`, `z_B`
- Interpolate: `z_interp = α * z_A + (1-α) * z_B`
- Option 1: Decode directly with VAE decoder
- Option 2: Use as initialization for conditional/unconditional diffusion
- Can vary target sequence length by adjusting number of empty tokens in decoder

### Multi-Length Generation
- Fixed latent size enables generation of proteins with different lengths
- Decoder cross-attention can attend to latents for any number of output tokens
- Specify desired length via number of empty tokens

---

## Key Advantages

1. **Fixed-Size Latents**:
   - Faster training (no variable-length attention in diffusion)
   - More stable diffusion (consistent latent dimensionality)
   - Enables latent arithmetic (interpolation, manipulation)

2. **Mixture of Queries**:
   - Learned compression via soft query selection
   - Forces model to capture essential information in fixed vectors
   - Routing can specialize queries for different protein types

3. **Flexible Conditioning**:
   - Generate proteins from any combo of partial seq/struct
   - Scientists can prompt model like: "fold this sequence with this motif structure"
   - Future: Add functional annotations as conditioning

4. **Multimodal Understanding**:
   - Single model understands sequence, structure, and geometry
   - Multi-task learning improves representation quality
   - Can be extended to include function, binding sites, etc.

5. **Latent Space Manipulation**:
   - Interpolate between known proteins
   - Explore latent space for novel designs
   - Conditional/unconditional generation from same latent space

---

## Technical Details

### Masking Strategy (Training)
- **Random masking percentage**: Sample from uniform distribution (possibly with curriculum ramping)
- **Independent masking**: Sequence and structure can be independently masked
  - Example: Know residues 1-50 sequence, residues 30-80 structure
- **GNN handling**: Masked coordinate nodes excluded from graph construction
  - Topology computed only from known coordinates (top-k neighbors)
  - 1D conv propagates info to masked nodes after each message passing step

### Variable-Length Sequence Handling
- **No padding**: Use FlashAttention with `cu_seqlens` and `max_seqlen`
- **1D CNN masking**: Gather/scatter operations ensure samples don't interact
- **Careful masking**: Prevent cross-contamination between proteins in batch

### MoQ Implementation Details
- **Two independent query banks**: Encoder and conditioning network learn separate queries
- **Soft-weighted routing**: Top-k selection with routing probabilities
- **Same N_latent**: Conditioning must produce same number of vectors as encoder for compatibility
- **Query diversity loss**: Encourage orthogonal/diverse query representations

### Preprocessing (Reused from ProteinDiff)
- Local coordinate frame computation
- Voxelization of side chain environment
- Electric field calculation (AMBER ff19SB charges)
- Divergence computation → scalar field representation
- **Deterministic**: No randomness in preprocessing ensures reproducibility

---

## Implementation Roadmap

### Phase 1: VAE Development
1. Implement MoQ routing module
2. Build encoder: CNN → GNN → Transformer → MoQ → Latent sampling
3. Build decoder: Latent self-attn → Cross-attn with empty tokens → Multi-task heads
4. Implement all VAE losses
5. Training script for Stage 1

### Phase 2: Conditioning Network
1. Implement hybrid MPNN/1D-CNN with masking
2. Build conditioning MoQ (separate from encoder)
3. Test with various masking strategies
4. Integrate with existing preprocessing

### Phase 3: Diffusion Model
1. Implement AdaLN-conditioned diffusion transformer
2. Integrate conditioning network
3. Training script for Stage 2 (frozen VAE)
4. Implement sampling/generation scripts

### Phase 4: Evaluation & Extensions
1. Benchmark generation quality (TM-score, sequence recovery, etc.)
2. Test latent interpolation
3. Ablation studies on MoQ, conditioning, losses
4. Future: Add functional annotations, binding site conditioning, etc.

---

## Relationship to Current ProteinDiff

### Components to Keep
- Preprocessing pipeline (voxelization, divergence computation)
- Data loading infrastructure (DataHolder, DataBatch, variable-length batching)
- Static data (AMBER charges, constants)
- Docker setup, config system

### Components to Replace
- Encoder architecture (new: CNN → GNN → Transformer → MoQ)
- Decoder architecture (new: cross-attn with empty tokens, multi-task heads)
- Add conditioning network (new: hybrid MPNN/1D-CNN)
- Diffusion model (new: fixed-size latents, AdaLN conditioning)
- Loss functions (new: multi-task VAE losses, query diversity)

### Development Approach
- **Mostly a rewrite** of model architecture
- Keep proven components (preprocessing, data pipeline)
- Implement alongside current ProteinDiff for comparison
- Proof-of-concept before adding function/other modalities

---

## Open Questions / Future Work

1. **Optimal N_latent**: How many latent vectors for different protein sizes?
2. **Query bank size**: Ratio of `N_query_bank / N_latent` (currently 16-32x)
3. **1D CNN kernel size**: What neighborhood size for masked info propagation?
4. **Masking curriculum**: Fixed vs. ramped masking during training
5. **Query diversity loss weight**: Balance between diversity and task performance
6. **Extension to function**: How to incorporate GO terms, binding sites, etc.
7. **Multi-domain proteins**: Does fixed latent size handle large, multi-domain proteins?
8. **Latent space structure**: Does MoQ create interpretable latent dimensions?
