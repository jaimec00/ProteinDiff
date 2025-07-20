#!/bin/bash
#SBATCH --job-name=overfit2_train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=train_overfit2.out
#SBATCH --error=train_overfit2.err

source ~/.bash_custom
conda activate ProteinDiff_env
python -u learn_seqs.py --config config/train.yml
