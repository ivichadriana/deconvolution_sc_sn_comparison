#!/bin/sh

#SBATCH --job-name=Results_notebook_composite
#SBATCH --output=output_Results_notebook_composite.log
#SBATCH --error=error_Results_notebook_composite.log
#SBATCH --time=00:40:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   

# For Conda Environment:
source ~/.bashrc
conda deactivate
conda activate env_deconv

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")

# Define relative data and output paths
nb_path="${BASE_DIR}/notebooks"

##########################################################################################################

##########################                NOTEBOOKS              #########################################

input_notebook=${nb_path}/results_composite.ipynb
output_notebook=${nb_path}/results_composite_DONE.ipynb

input_notebook_methods=${nb_path}/results_composite_methods.ipynb
output_notebook_methods=${nb_path}/results_composite_methods_DONE.ipynb

##########################               EXECUTION             #########################################

echo "****** Running notebooks: ******"

papermill "$input_notebook" "$output_notebook"     
papermill "$input_notebook_methods" "$output_notebook_methods"     

##########################################################################################################

# Deactivate the environment
conda deactivate
