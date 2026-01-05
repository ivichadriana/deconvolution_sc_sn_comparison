#!/bin/bash

#SBATCH --job-name=create_envs
#SBATCH --account=amc-general
#SBATCH --output=output_create_envs.log
#SBATCH --error=error_create_envs.log
#SBATCH --time=02:00:00
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1 

# Exit if any command fails
set -e
#Clean ups
source ~/.bashrc 2>/dev/null || source /etc/profile.d/conda.sh
rm -rf ~/.conda/pkgs/r-base-4.3.3-h65010dc_18
conda clean -a -y
conda deactivate

conda env create -f environment/env_deconv.yml

sleep 10

#Create a env with all the R/Bioc 
conda create -n env_deconv_R -y -c conda-forge -c bioconda \
  r-base=4.3 \
  r-biocmanager r-devtools r-remotes r-rcpp r-data.table r-dplyr r-tidyr \
  r-ggplot2 r-lattice r-pheatmap r-abind r-pbapply r-gplots r-nmf r-caret \
  r-matrixstats \
  r-seurat r-seuratobject \
  bioconductor-singlecellexperiment \
  bioconductor-summarizedexperiment \
  bioconductor-scuttle \
  bioconductor-genomicranges \
  bioconductor-iranges \
  bioconductor-genomeinfodb \
  bioconductor-matrixgenerics \
  bioconductor-s4vectors \
  bioconductor-biocparallel \
  bioconductor-biocgenerics \
  git make pkg-config libcurl openssl libxml2 \
  gcc_linux-64 gxx_linux-64 \
  zlib xz

# Small pause so env fully registers on some HPCs
sleep 5

conda activate env_deconv_R
export R_PROFILE_USER=NONE
export R_ENVIRON_USER=NONE
export RETICULATE_MINICONDA_ENABLED=FALSE
export RETICULATE_PYTHON="$CONDA_PREFIX/bin/python"
export RETICULATE_CONDA="$(which conda)"

# Binary installs to avoid compiling nloptr/lme4 stack + provide anndata
conda install -n env_deconv_R -y -c conda-forge \
  r-nloptr r-lme4 r-car r-rstatix r-ggpubr r-igraph r-readr r-uuid r-anndata \
  r-reticulate r-leidenbase python anndata

# BisqueRNA and MAST (SCOPfunctions needs MAST)
conda install -n env_deconv_R -y -c bioconda \
  r-bisquerna bioconductor-mast

#and dependencies that crash
conda install -n env_deconv_R -y -c conda-forge r-snow r-snowfall
conda install -n env_deconv_R -c conda-forge -y r-openssl openssl pkg-config

# Install only the packages that are NOT on conda 
Rscript "$SLURM_SUBMIT_DIR/scripts/remaining_R_packages.R"

conda deactivate