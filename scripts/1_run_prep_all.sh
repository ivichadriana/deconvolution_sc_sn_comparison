#!/bin/bash
#SBATCH --job-name=Prep_deconvolution_files
#SBATCH --account=amc-general
#SBATCH --output=output_Prep_deconvolution_files.log
#SBATCH --error=error_Prep_deconvolution_files.log 
#SBATCH --mail-type=ALL
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=24:00:00                              
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16

# Exit if any command fails
set -e

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Define relative data and output paths
data_path="${BASE_DIR}/data"

# In our case, the output directory for deconvolution files is under BASE_DIR/deconvolution
output_root="${BASE_DIR}/data/deconvolution"

# Activate the conda environment
echo "****** Activating environment... ******"
source ~/.bashrc
conda deactivate
conda activate env_deconv

sleep 5 

# Define common parameters
pseudobulks_props="{\"realistic\": 500, \"random\": 500}"
num_cells=1000
noise=True
deconvolution_method="bayesprism"
deseq_alpha=0.01

# List of dataset names to process
datasets=("ADP" "PBMC" "MBC" "MSB")

for res_name in "${datasets[@]}"; do
    echo "****** Running dataset: ${res_name} ******"
    output_path="${output_root}/${res_name}"
    mkdir -p "$output_path"

    echo "****** Running prepare_deconvolution.py for ${res_name} ******"
    python "${BASE_DIR}/scripts/prepare_deconvolution.py" \
        --res_name="$res_name" \
        --data_path="$data_path" \
        --output_path="$output_path" \
        --pseudobulks_props="$pseudobulks_props" \
        --num_cells="$num_cells" \
        --noise="$noise" \
        --deconvolution_method="$deconvolution_method" \
        --deseq_alpha="$deseq_alpha"
done

conda deactivate
