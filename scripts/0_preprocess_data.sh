#!/bin/bash

#SBATCH --job-name=Data_preprocessing
#SBATCH --output=output_Data_preprocessing.log
#SBATCH --error=error_Data_preprocessing.log
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   

set -e

# Get the directory of this script and define base paths relative to it
BASE_DIR="${SLURM_SUBMIT_DIR}"
notebooks_path="${BASE_DIR}/notebooks"
cd "$notebooks_path"

# Conda Environment:
source ~/.bashrc
conda deactivate
conda activate env_deconv

#########################################################################################################
#########################################################################################################

echo "****** Running preprocessing for Adipose (ADP) data...******"

res_name="ADP"

input_notebook=${notebooks_path}/ADP_preprocessing.ipynb
output_notebook=${notebooks_path}/ADP_preprocessing_DONE.ipynb

papermill "$input_notebook" "$output_notebook" \
          -p res_name "$res_name"   

#########################################################################################################

echo "****** Running preprocessing for 10x Genomics datasets (PBMC and Mouse Brain)...******"

datasets=("PBMC" "MSB")

for res_name in "${datasets[@]}"; do
    echo "****** dataset: ${res_name} ******"

    input_notebook=${notebooks_path}/10xGen_preprocessing.ipynb
    output_notebook=${notebooks_path}/${res_name}_10xGen_preprocessing_DONE.ipynb

    papermill "$input_notebook" "$output_notebook" \
            -p res_name "$res_name"   
done

#########################################################################################################

echo "****** Running preprocessing for Metastatic Breast Cancer (MBC) data...******"

res_name="MBC"

input_notebook=${notebooks_path}/TumorToolbox_preprocessing.ipynb
output_notebook=${notebooks_path}/${res_name}_TumorToolbox_preprocessing_DONE.ipynb

papermill "$input_notebook" "$output_notebook" \
          -p res_name "$res_name"   

#########################################################################################################

# Deactivate the virtual environment w/ conda
conda deactivate
