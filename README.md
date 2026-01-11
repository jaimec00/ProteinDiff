# Multimodal Protein Model with Mixture of Queries (MoQ)

A multimodal protein generative model that understands and generates proteins conditioned on partial sequence and/or structure. Uses **Mixture of Queries (MoQ)** to compress variable-length proteins into fixed-size latent vectors.

**Status**: Architecture design phase

## Key Features

- **Fixed-size latents** - stable diffusion training, latent interpolation
- **Flexible conditioning** - generate from any combo of partial sequence/structure
- **Multi-task learning** - reconstructs sequence, structure, geometry simultaneously

## Architecture

1. **VAE Encoder**: CNN → GNN → Transformer → MoQ compression → latent sampling
2. **VAE Decoder**: cross-attention from positional tokens to latents → multi-task heads
3. **Conditioning Network**: encodes partial seq/struct via hybrid GNN + 1D-CNN
4. **Latent Diffusion**: DiT-style denoising with AdaLN conditioning

## Training

- **Stage 1**: Train VAE end-to-end (reconstruction + KL)
- **Stage 2**: Freeze VAE, train conditioning network + diffusion model

## Setup

We use pixi for this repo, so just run

`pixi shell`

to drop into an interactive environment

Download training data (ProteinMPNN dataset):

```shell
DATA_PATH=/PATH/TO/YOUR/DATA && \
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz -P $DATA_PATH && \
tar -xzf $DATA_PATH/pdb_2021aug02.tar.gz -C $DATA_PATH && \
rm $DATA_PATH/pdb_2021aug02.tar.gz
```
