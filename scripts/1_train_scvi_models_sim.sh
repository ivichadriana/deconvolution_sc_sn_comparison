#!/bin/bash
#SBATCH --job-name=Train_nodeg
#SBATCH --account=amc-general
#SBATCH --output=output_Train_nodeg_%A_%a.log
#SBATCH --error=error_Train_nodeg_%A_%a.log 
#SBATCH --time=02:00:00                              
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3

# Exit if any command fails
set -e

# Get the directory of this script and define base paths relative to it
BASE_DIR="${SLURM_SUBMIT_DIR}"
scripts_path="${BASE_DIR}/scripts"
# Define relative data and output paths
data_path="${BASE_DIR}/data"
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
min_cells_per_type=50

# List of dataset names
datasets=("ADP" "PBMC" "MBC" "MSB")

# Select the dataset for this array task
res_name=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "****** Running dataset: ${res_name} ******"
output_path="${output_root}/${res_name}"
mkdir -p "$output_path"

echo "****** Running differential_gene_expression.py for ${res_name} ******"

python "${scripts_path}/differential_gene_expression.py" \
    --res_name="$res_name" \
    --data_path="$data_path" \
    --output_path="$output_path" \
    --deseq_alpha="$deseq_alpha" \
    --min_cells_per_type="$min_cells_per_type"

echo "****** Running train_scvi_models_allgenes.py for ${res_name} ******"
python "${scripts_path}/train_scvi_models_allgenes.py" \
    --res_name="$res_name" \
    --data_path="$data_path" \
    --output_path="$output_path" \
    --deseq_alpha="$deseq_alpha"

echo "****** Running train_scvi_models_nodeg.py for ${res_name} ******"
python "${scripts_path}/train_scvi_models_nodeg.py" \
    --res_name="$res_name" \
    --data_path="$data_path" \
    --output_path="$output_path" \
    --deseq_alpha="$deseq_alpha"

conda deactivate
