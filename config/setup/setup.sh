#!/bin/bash

# get the config dir
CONFIG_DIR=$(cd $(dirname ${BASH_SOURCE}) && pwd)
PROTEINDIFF_DIR=$(dirname $CONFIG_DIR)

# check if conda is available, if not then setup conda
source "$CONFIG_DIR/setup_conda.sh"

# if it does, create the environment
PROTEINDIFF_ENV="$CONFIG_DIR/ProteinDiff.yml"
conda env create -f $PROTEINDIFF_ENV

# activate the environment
conda activate ProteinDiff_env

# check where the env is located
ACT_SCRIPT="$CONDA_PREFIX/etc/conda/activate.d/conda_activation_script.sh"
cat << EOF > $ACT_SCRIPT

# setup env for proteus ai
export PYTHONPATH="\$PYTHONPATH;$PROTEINDIFF_DIR"

# cuda config
export CUDA_HOME=\$CONDA_PREFIX
export TORCH_NVCC_EXECUTABLE=\$CUDA_HOME/bin/nvcc
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

EOF 

# deactivate and activate so the activation script works
conda deactivate
conda activate ProteinDiff_env