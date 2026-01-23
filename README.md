# Proteus

A multimodal protein generative model that generates proteins conditioned on partial sequence and/or structure. Uses a VAE with per-residue latents followed by latent diffusion.

**Status**: Architecture design phase

## Key Features

- **Per-residue latents** - each residue maps to a latent vector for fine-grained representation
- **Flexible conditioning** - generate from any combo of partial sequence/structure
- **Multi-task learning** - reconstructs sequence, structure, geometry simultaneously

## Architecture

1. **Tokenizer**: Converts atomic coordinates to voxelized electric field divergence representation
2. **VAE Encoder**: CNN Downsample → MPNN → Transformer → Latent Projection (mu, logvar)
3. **VAE Decoder**: Up Projection → Transformer → Multi-task heads (divergence, sequence, structure)
4. **Latent Diffusion**: DiT-style denoising (planned)

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
