#!/bin/bash

# Initialize conda for the current shell
eval "$(conda shell.bash hook)"

# Create a new conda environment named "bitnet_env" with Python 3.10
conda create -n bitnet_env python=3.10 -y

# Activate the environment
source activate bitnet_env

# Install required Python packages
conda install numpy -y
pip install sentencepiece transformers torch

# Environment setup complete
echo "Conda environment 'bitnet_env' created and all dependencies installed."