#!/bin/sh

#SBATCH --job-name=Results_notebook_sim
#SBATCH --account=amc-general
#SBATCH --output=output_Results_notebook_sim.log
#SBATCH --error=error_Results_notebook_sim.log
#SBATCH --time=03:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=24
#SBATCH --nodes=1  # Use a single node
#SBATCH --mail-type=ALL

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

##########################                NOTEBOOK              #########################################

input_notebook=${nb_path}/results_sim.ipynb
output_notebook=${nb_path}/results_sim.ipynb

##########################               EXECUTION             #########################################

echo "****** Running dataset: ${res_name} ******"

papermill "$input_notebook" "$output_notebook"
            
##########################################################################################################

# Deactivate the environment
conda deactivate
