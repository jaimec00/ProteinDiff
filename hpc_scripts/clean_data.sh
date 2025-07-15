#!/bin/bash

#SBATCH --job-name=ProtDiff_clean
#SBATCH --partition=common
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=clean.out
#SBATCH --error=clean.err

source ~/.bash_custom
conda activate ProteinDiff_env
python -u utils/train_utils/data_utils.py --data_path /work/hcardenas1/projects/ProteinDiff/data/raw/pdb_2021aug02 --new_data_path /work/hcardenas1/projects/ProteinDiff/data/processed