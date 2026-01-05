#!/bin/bash
#SBATCH --job-name=Deconvolution_sim
#SBATCH --output=output_Deconvolution_sim_%A_%a.log
#SBATCH --error=error_Deconvlution_sim_%A_%a.err
#SBATCH --time=10:00:00  
#SBATCH --cpus-per-task=16
#SBATCH --mem=85G
#SBATCH --nodes=1
#SBATCH --array=0-11

# Define the datasets
DATASETS=("ADP" "PBMC" "MBC" "MSB")  # List your datasets here
METHODS=("Scaden" "SCDC" "BayesPrism")

# Activate Conda environment
source ~/.bashrc
conda activate env_deconv_R

export R_PROFILE_USER=NONE
export R_ENVIRON_USER=NONE

# ***  per-job tmp directory on scratch ***
JOB_TMP=/scratch/alpine/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$JOB_TMP"
export TMPDIR="$JOB_TMP"

echo "Using TMPDIR=$TMPDIR"
# Figure out which dataset/method this array index corresponds to
DATASET_INDEX=$(( SLURM_ARRAY_TASK_ID / ${#METHODS[@]} ))
METHOD_INDEX=$(( SLURM_ARRAY_TASK_ID % ${#METHODS[@]} ))

DATA_TYPE=${DATASETS[$DATASET_INDEX]}
METHOD=${METHODS[$METHOD_INDEX]}

echo "Task $SLURM_ARRAY_TASK_ID running dataset=$DATA_TYPE, method=$METHOD"

if [ "$METHOD" == "BayesPrism" ]; then
  echo "Running BayesPrism for $DATA_TYPE"
  Rscript scripts/BayesPrism_sim.R "$DATA_TYPE"
else
  echo "Running $METHOD for $DATA_TYPE via deconvolution_sim.R"
  Rscript scripts/deconvolution_sim.R "$DATA_TYPE" "$METHOD"
fi

echo "Completed $METHOD for $DATA_TYPE."