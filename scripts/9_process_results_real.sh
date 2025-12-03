#!/bin/bash
#SBATCH --job-name=Process_results_real
#SBATCH --output=log_Process_results_real.log 
#SBATCH --error=error_Process_results_real.err 
#SBATCH --qos=normal
#SBATCH --time=00:25:00                              
#SBATCH --mem=5G
#SBATCH --nodes=1  

# Activate Conda environment with R
source ~/.bashrc
conda activate env_deconv

# Get the directory of this script and define base paths relative to it
BASE_DIR=$(realpath "$(pwd)")
SCRIPT_DIR="${BASE_DIR}/scripts"

# List of dataset names to process
datasets="Real_ADP"
methods=("BayesPrism" "SCADEN" "SCDC")  

for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        echo "Processing real, method: $method, $dataset"
        # Run the Python script with the dataset as an argument
        python ${BASE_DIR}/scripts/process_results.py \
            --dataset="$dataset" \
            --method="$method" \
            --perdonor
    done
done

# and per donor
for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        echo "Processing per donor, method: $method, $dataset"
        # Run the Python script with the dataset as an argument
        python ${BASE_DIR}/scripts/process_results.py \
            --dataset="$dataset" \
            --method="$method" 
    done
done

echo "All datasets and results processed."

conda deactivate