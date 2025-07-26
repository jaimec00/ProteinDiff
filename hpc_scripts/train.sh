#!/bin/bash
#SBATCH --job-name=train_diff
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=runs/train_diff.out
#SBATCH --error=runs/train_diff.err

source ~/.bash_custom
conda activate ProteinDiff_env
python -u learn_seqs.py --config config/train.yml
