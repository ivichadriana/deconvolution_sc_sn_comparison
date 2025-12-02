#!/bin/bash

#SBATCH --job-name=Results_notebook_real
#SBATCH --output=output_Results_notebook_real.log
#SBATCH --error=error_Results_notebook_real.log
#SBATCH --time=03:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8   

# For Conda Environment:
source ~/.bashrc
set -euo pipefail
conda deactivate
conda activate env_deconv

# Define relative data and output paths
nb_path="${SLURM_SUBMIT_DIR}/notebooks"
methods=("BayesPrism" "DWLS" "SCDC")  

##########################################################################################################

##########################                NOTEBOOKS              #########################################

input_notebook_real=${nb_path}/results_real.ipynb
input_notebook_perdonor=${nb_path}/results_perdonor.ipynb

##########################               EXECUTION             #########################################

echo "****** Running notebooks: ******"

for method in "${methods[@]}"; do
    echo "Runnning real results notebook, method: $method"
    output_notebook_real=${nb_path}/results_real_${method}_DONE.ipynb
    papermill "$input_notebook_real" "$output_notebook_real" -p method "$method"   
done

for method in "${methods[@]}"; do
    echo "Runnning real results notebook, method: $method"
    output_notebook_perdonor=${nb_path}/results_perdonor_${method}_DONE.ipynb
    papermill "$input_notebook_perdonor" "$output_notebook_perdonor" -p method "$method"   
done
            
##########################################################################################################

# Deactivate the environment
conda deactivate
