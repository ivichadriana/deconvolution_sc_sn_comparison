#!/bin/bash
#SBATCH --job-name=SCADEN_Deconvolution_perdonreal
#SBATCH --output=output_SCADEN_Deconvolution_perdonreal_%A_%a.log
#SBATCH --error=error_SCADEN_Deconvolution_realperdon_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --qos=normal
#SBATCH --time=11:00:00  
#SBATCH --mem=70G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --array=0-5

# Activate Conda environment
source ~/.bashrc
conda activate env_deconv_R
# Clear any R profile settings that might interfere
export R_PROFILE_USER=NONE
export R_ENVIRON_USER=NONE

# Define the dataset
DATA_TYPE="Real_ADP"  

echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    METHOD="SCADEN"
    echo "Running ${METHOD} for ${DATA_TYPE} (real)"
    Rscript scripts/deconvolution_real.R "${DATA_TYPE}" "${METHOD}"
    ;;
  1)
    METHOD="SCDC"
    echo "Running ${METHOD} for ${DATA_TYPE} (real)"
    Rscript scripts/deconvolution_real.R "${DATA_TYPE}" "${METHOD}"
    ;;
  1)
    METHOD="SCADEN"
    echo "Running ${METHOD} for ${DATA_TYPE} (per-donor)"
    Rscript scripts/deconvolution_perdonor.R "${DATA_TYPE}" "${METHOD}"
    ;;
  3)
    METHOD="SCDC"
    echo "Running ${METHOD} for ${DATA_TYPE} (per-donor)"
    Rscript scripts/deconvolution_perdonor.R "${DATA_TYPE}" "${METHOD}"
    ;;
  4)
    echo "Running BayesPrism per-donor for ${DATA_TYPE}"
    Rscript scripts/BayesPrism_perdonor.R "${DATA_TYPE}"
    ;;
  5)
    echo "Running BayesPrism real for ${DATA_TYPE}"
    Rscript scripts/BayesPrism_real.R "${DATA_TYPE}"
    ;;
  *)
    echo "Unknown SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
    ;;
esac

echo "Task ${SLURM_ARRAY_TASK_ID} completed for ${DATA_TYPE}."