#!/bin/bash
#SBATCH --job-name=train_diff_eps
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=runs/train_diff_eps.out
#SBATCH --error=runs/train_diff_eps.err

source ~/.bash_custom
conda activate ProteinDiff_env
python -u learn_seqs.py --config config/train.yml
