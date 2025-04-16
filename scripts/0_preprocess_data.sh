#!/bin/sh

#SBATCH --job-name=Data_preprocessing
#SBATCH --account=amc-general
#SBATCH --output=output_Data_preprocessing.log
#SBATCH --error=error_Data_preprocessing.log
#SBATCH --time=05:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=64
#SBATCH --nodes=1 

# Exit if any command fails
set -e

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Define relative data and output paths
notebooks_path="${BASE_DIR}/notebooks"
data_path="${BASE_DIR}/data"

# Conda Environment:
source ~/.bashrc
conda deactivate
conda activate env_deconv\

##########################################################################################################
##########################################################################################################

echo "****** Running preprocessing for Adipose (ADP) data...******"

res_name="ADP"

input_notebook=${notebooks_path}/${res_name}_preprocessing.ipynb
output_notebook=${notebooks_path}/${res_name}_preprocessing.ipynb

papermill "$input_notebook" "$output_notebook" \
          -p res_name "$res_name"   

#########################################################################################################

echo "****** Running preprocessing for 10x Genomics datasets (PBMC and Mouse Brain)...******"

datasets=("PBMC" "MSB")

for res_name in "${datasets[@]}"; do
    echo "****** dataset: ${res_name} ******"

    input_notebook=${notebooks_path}/10xGen_preprocessing.ipynb
    output_notebook=${notebooks_path}/${res_name}_preprocessing.ipynb

    papermill "$input_notebook" "$output_notebook" \
            -p res_name "$res_name"   

##########################################################################################################

echo "****** Running preprocessing for Metastatic Breast Cancer (MBC) data...******"

res_name="MBC"

input_notebook=${notebooks_path}/TumorToolbox_preprocessing.ipynb
output_notebook=${notebooks_path}/${res_name}_preprocessing.ipynb

papermill "$input_notebook" "$output_notebook" \
          -p res_name "$res_name"   

##########################################################################################################
##########################################################################################################

# Deactivate the virtual environment w/ conda
conda deactivate
