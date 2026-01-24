# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System & Environment

Uses **pixi** (modern conda alternative) for environment management:

```bash
pixi shell                    # Drop into environment (auto-activates)
pixi shell -e gpu             # GPU environment with CUDA 12 + flash-attn
pixi shell -e cpu             # CPU env for local dev
pixi run local_mlflow         # Start MLflow tracking server on localhost:5000
```

Environment variables set automatically via pixi:
- `EXP_DIR`: Experiment directory
- `MLFLOW_TRACKING_URI`: SQLite DB for MLflow tracking

## Running Tests

```bash
pixi run pytest tests/                           # All tests
pixi run pytest tests/test_losses/ -v            # Loss tests with verbose output
pytest tests/test_losses/test_struct_losses/test_anglogram_loss.py  # Single test file
```

Note: Some tests (anglogram, distogram) require CUDA.

### Key Types
`proteus/types/__init__.py` defines jaxtyping annotations used throughout:
- `Float`, `Int`, `Bool` tensor types with shape annotations
- Tensor shapes use conventions: `BL` = total residues in batch, `B` = number of samples, `L` = sequence length

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

## Important Notes

make sure to use the skills and subagents you have available to delegate tasks. 
make the code clean and readable, prefer a single descriptive comment above a code block over comments on every line
use the conventions already in place (types, shapes, lowercase comments)
