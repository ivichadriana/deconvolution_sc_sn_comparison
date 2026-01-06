#!/bin/bash
#SBATCH --job-name=Prep_deconvolution_sim
#SBATCH --account=amc-general
#SBATCH --output=output_Prep_deconvolution_sim_%a.log
#SBATCH --error=error_Prep_deconvolution_sim_%a.log 
#SBATCH --mail-type=ALL
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=20:00:00                              
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --array=0


# Define the datasets
# datasets=("ADP" "MSB" "MBC" "PBMC")  # List your datasets here
datasets=("PBMC")  # List your datasets here

# Get the directory of this script and define base paths relative to it
BASE_DIR=$(realpath "$SLURM_SUBMIT_DIR")
SCRIPT_DIR="${BASE_DIR}/scripts"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Define relative data and output paths
data_path="${BASE_DIR}/data"
output_root="${BASE_DIR}/data/deconvolution"

# Activate the conda environment
echo "****** Activating environment... ******"
source ~/.bashrc
conda deactivate
conda activate env_deconv

sleep 5

# Define common parameters
pseudobulks_props="{\"realistic\": 500, \"random\": 500}"
num_cells=1000
deconvolution_method="bayesprism"
deseq_alpha=0.01

# Select the dataset for this array task
res_name=${datasets[$SLURM_ARRAY_TASK_ID]}

echo "****** Running dataset: ${res_name} ******"
output_path="${output_root}/${res_name}"
degs_path="${BASE_DIR}/data/deconvolution"
mkdir -p "$output_path"
cd "$SCRIPT_DIR"

echo "****** Running prepare_deconvolution.py for ${res_name} ******"
python "${SCRIPT_DIR}/prepare_deconvolution_sim.py" \
    --res_name="$res_name" \
    --data_path="$data_path" \
    --output_path="$output_path" \
    --pseudobulks_props="$pseudobulks_props" \
    --num_cells="$num_cells" \
    --noise \
    --deconvolution_method="$deconvolution_method" \
    --deseq_alpha="$deseq_alpha" \
    --degs_path="$degs_path" \

python "${SCRIPT_DIR}/check_references.py" \
    --output_path="$output_path"

conda deactivate
