''' The following file is used in create_env.sh in order to install the R packages that conda can not.'''

# We use the conda env's Python and the base conda binary; block Miniconda bootstrap
Sys.setenv(RETICULATE_MINICONDA_ENABLED = "FALSE")
Sys.setenv(RETICULATE_PYTHON = file.path(Sys.getenv("CONDA_PREFIX"), "bin", "python"))
Sys.setenv(RETICULATE_CONDA  = Sys.which("conda"))

# Avoid interactive prompts
options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

# If needed, ensure the required packages are installed
if (!requireNamespace("scran", quietly = TRUE)) {
  BiocManager::install("scran")
}
if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
  BiocManager::install("SingleCellExperiment")
}

if (!require("devtools", quietly = TRUE))
    install.packages("devtools")
    
library("devtools");
devtools::install_github(
  "Danko-Lab/BayesPrism",
  subdir = "BayesPrism",
  dependencies = FALSE,   # keep conda packages untouched
  upgrade = "never"
)

if (!require("Biobase", quietly = TRUE))
    BiocManager::install("Biobase")

devtools::install_github("humengying0907/InstaPrism")

if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
  BiocManager::install("SummarizedExperiment", ask = FALSE)
}
if (!requireNamespace("pak", quietly = TRUE)) {
  install.packages("pak", repos = "https://r-lib.github.io/p/pak/stable")
}

install.packages("anndata") 
devtools::install_github("bbc/bbplot", upgrade = "never", dependencies = FALSE)
devtools::install_github("CBMR-Single-Cell-Omics-Platform/SCOPfunctions",
                         upgrade = "never", dependencies = FALSE)
pak::pkg_install("omnideconv/omnideconv", dependencies = FALSE, upgrade = FALSE)

# deps for SCDC
install.packages(c("L1pack", "nnls"))          # CRAN
if (!requireNamespace("pak", quietly=TRUE))
  install.packages("pak", repos="https://r-lib.github.io/p/pak/stable")
pak::pkg_install("renozao/xbioc", upgrade = FALSE)

# install SCDC from the omnideconv fork explicitly
remotes::install_github("omnideconv/SCDC", dependencies = FALSE, upgrade = "never")
install.packages("scBio") 
# And checking installation
library(SingleCellExperiment)
library(omnideconv)
library(scran)
library(SingleCellExperiment)
library(BayesPrism)
library(InstaPrism)
