#!/bin/bash

# Load Conda if not already loaded
if ! command -v conda &> /dev/null
then
    echo "Conda is not available. Please load Conda and try again."
    exit 1
fi

# Create the new Conda environment
conda create -y -n env_deconv python=3.9

# Activate the new environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate env_deconv

# Install required packages
conda install -y \
    numpy \
    pandas \
    scanpy \
    matplotlib \
    seaborn \
    scikit-learn \
    anndata \
    h5py

# Install additional packages with pip
pip install --no-cache-dir \
    tqdm \
    umap-learn

# Verify the installation
echo "****** Verifying environment setup... ******"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Installed packages:"
pip list

# Deactivate the environment
conda deactivate

echo "Conda environments 'env_deconv' has been created and is ready to use."