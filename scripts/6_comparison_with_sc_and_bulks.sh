#!/bin/bash
#SBATCH --job-name=RealBulk_andSC_Comparison
#SBATCH --account=amc-general
#SBATCH --output=output_RealBulk_andSC_Comparison_%A_%a.log
#SBATCH --error=error_RealBulk_andSC_Comparison_%A_%a.log 
#SBATCH --time=24:00:00                              
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# Activate the conda environment
echo "****** Activating environment... ******"
source ~/.bashrc
conda deactivate
conda activate env_deconv

sleep 5

echo "****** Running dataset: ${res_name} ******"
output_path="${output_root}/${res_name}"
mkdir -p "$output_path"

##########################################################################################################
##########################                Two SCRIPTS             #########################################

echo "****** Running process_bulks_train_models.py ******"
python "${BASE_DIR}/scripts/process_bulks_train_models.py" 

echo "****** Running prepare_real_bulk_clustering.py ******"
python "${BASE_DIR}/scripts/prepare_real_bulk_clustering.py"

# Define relative data and output paths
nb_path="${BASE_DIR}/notebooks"

##########################################################################################################
##########################                NOTEBOOK              #########################################

input_notebook=${nb_path}/results_bulks_comparison.ipynb
output_notebook=${nb_path}/results_bulks_comparison.ipynb

##########################               EXECUTION             #########################################

echo "****** Running notebook of bulk similarity: ******"

papermill "$input_notebook" "$output_notebook"
##########################################################################################################

##########################################################################################################
##########################                NOTEBOOK              #########################################

input_notebook=${nb_path}/results_sc_comparison.ipynb
output_notebook=${nb_path}/results_sc_comparison.ipynb

##########################               EXECUTION             #########################################

echo "****** Running notebook of sc similarity: ******"

papermill "$input_notebook" "$output_notebook"
##########################################################################################################

# Deactivate the environment
conda deactivate
