#!/bin/bash
#SBATCH --job-name=ProtDiff_train
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=train.out
#SBATCH --error=train.err

source ~/.bash_custom
conda activate ProteinDiff_env
python -u learn_seqs.py --config config/train.yml