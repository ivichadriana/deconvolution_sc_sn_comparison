#!/bin/bash
#SBATCH --job-name=Bayesprism_sim
#SBATCH --output=output_Bayesprism_sim_%A_%a.log
#SBATCH --error=error_Bayesprism__sim_%A_%a.err
#SBATCH --time=15:00:00  
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --array=0-3 

# Define the datasets
DATASETS=("ADP" "MSB" "MBC" "PBMC")  # List your datasets here

# Activate Conda environment
source ~/.bashrc
conda activate env_deconv_R

# Get the dataset for this array job
DATA_TYPE=${DATASETS[$SLURM_ARRAY_TASK_ID]}

# Run the deconvolution script
Rscript BayesPrism_sim.R $DATA_TYPE
