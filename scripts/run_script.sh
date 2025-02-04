#!/bin/sh

#SBATCH --job-name=MBC_Prep_Deconvolution
#SBATCH --account=amc-general
#SBATCH --output=output_Prep_deconvolution_MBC.log
#SBATCH --error=error_Prep_deconvolution_MBC.log 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=adriana.ivich@cuanschutz.edu
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=01:00:00                              
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16

# Activate the virtual environment
echo "****** Activating environment... ******"

# For Conda Environment:
source ~/.bashrc
conda deactivate
conda activate env_deconv

## Verify that the correct Python environment is loaded
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Define parameters
res_name="MBC"
data_path="/projects/aivich@xsede.org/deconvolution_differences/data/"
output_path="/projects/aivich@xsede.org/deconvolution_differences/data/deconvolution/$res_name/"
pseudobulks_props="{\"realistic\": 500, \"random\": 500}"
num_cells=1000
noise=True
deconvolution_method="bayesprism"

# Ensure output directory exists
mkdir -p $output_path

# Run the Python script
echo "****** Running prepare_deconvolution.py script... ******"
python /projects/aivich@xsede.org/deconvolution_differences/scripts/prepare_deconvolution.py \
    --res_name=$res_name \
    --data_path=$data_path \
    --pseudobulks_props="$pseudobulks_props" \
    --output_path=$output_path \
    --num_cells=$num_cells \
    --noise=$noise \
    --deconvolution_method=$deconvolution_method

# Deactivate the conda environment
conda deactivate