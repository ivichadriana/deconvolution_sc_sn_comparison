#!/bin/bash
#SBATCH --job-name=Process_results_sim
#SBATCH --output=log_Process_results_sim.log 
#SBATCH --error=error_Process_results_sim.err 
#SBATCH --time=00:40:00                              
#SBATCH --nodes=1  
#SBATCH --mem=5G

# Define the datasets and method
datasets=("ADP" "MSB" "MBC" "PBMC")  # List your datasets here
methods=("BayesPrism" "SCADEN" "SCDC")  

# Activate Conda environment with R
source ~/.bashrc
conda activate env_deconv

# Get the directory of this script and define base paths relative to it
BASE_DIR=$(realpath "$(pwd)")
SCRIPT_DIR="${BASE_DIR}/scripts"

for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    
    for method in "${methods[@]}"; do
        echo "Processing method: $method, $dataset"
        # Run the Python script with the dataset as an argument
        python ${BASE_DIR}/scripts/process_results.py \
            --dataset="$dataset" \
            --method="$method" \
            --simulation
    done
done

echo "All datasets processed."

conda deactivate
