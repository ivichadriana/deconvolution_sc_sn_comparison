#!/bin/bash
#SBATCH --job-name=Process_results_sim
#SBATCH --output=log_Process_results_sim.log 
#SBATCH --error=error_Process_results_sim.err 
#SBATCH --mail-type=ALL
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=01:00:00                              
#SBATCH --ntasks-per-node=30
#SBATCH --nodes=1  # Use a single node

# Define the datasets
DATASETS=("ADP" "MSB" "MBC" "PBMC")  # List your datasets here

# Activate Conda environment with R
source ~/.bashrc
conda activate env_deconv

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Run the Python script with the dataset as an argument
    python ${BASE_DIR}/scripts/process_results.py \
        --dataset="$dataset" \
        --simulation
done

echo "All datasets processed."

conda deactivate
