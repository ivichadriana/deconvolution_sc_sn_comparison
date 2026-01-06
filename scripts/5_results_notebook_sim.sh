#!/bin/sh

#SBATCH --job-name=Results_notebook_sim
#SBATCH --output=output_Results_notebook_sim.log
#SBATCH --error=error_Results_notebook_sim.log
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8 

# For Conda Environment:
source ~/.bashrc
conda deactivate
conda activate env_deconv

# Get the directory of this script and define base paths relative to it
BASE_DIR=$(realpath "$(pwd)")
SCRIPT_DIR="${BASE_DIR}/scripts"

# Define relative data and output paths
nb_path="${BASE_DIR}/notebooks"
methods=("BayesPrism" "DWLS" "SCDC")  

###################################################################################

for method in "${methods[@]}"; do:

    input_notebook=${nb_path}/results_sim.ipynb
    output_notebook=${nb_path}/results_sim_${method}_DONE.ipynb

    papermill "$input_notebook" "$output_notebook"

done 
            
###################################################################################

# Deactivate the environment
conda deactivate
