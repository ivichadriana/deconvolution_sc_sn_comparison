#!/bin/bash
#SBATCH --job-name=RealBulks_DeconvPrep
#SBATCH --output=output_RealBulks_DeconvPrep.log
#SBATCH --error=error_RealBulks_DeconvPrep.log
#SBATCH --time=10:00:00                              
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

# Get the directory of this script and define base paths relative to it
SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR/..")
output_root="${BASE_DIR}/data/deconvolution"

# Activate the conda environment
echo "****** Activating environment... ******"
source ~/.bashrc
conda deactivate
conda activate env_deconv

sleep 5

deconvolution_method="bayesprism"
deseq_alpha=0.01

res_name="ADP"
output_path="${output_root}/Real_ADP"

echo "****** Running prepare_deconvolution_real.py ******"
python "${BASE_DIR}/scripts/prepare_deconvolution_real.py"  \
    --res_name="$res_name" \
    --output_path="$output_path" \
    --degs_path="$output_root" \
    --deconvolution_method="$deconvolution_method" \
    --deseq_alpha="$deseq_alpha"

conda deactivate
