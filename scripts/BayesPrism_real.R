#!/usr/bin/env Rscript

# ============================================================
# R script to run BayesPrism on all references created
# by the "prepare_realbulks_references.py" script (or similar).
#
# Usage:
#   Rscript BayesPrism_realbulks.R <DATASET_NAME>
#
# We parallelize the InstaPrism function with n.cores in the cluster environment.
# ============================================================

# Avoid interactive prompts
options(repos = c(CRAN = "https://cloud.r-project.org"))

# If needed, ensure the required packages are installed
if (!requireNamespace("scran", quietly = TRUE)) {
  BiocManager::install("scran")
}
if (!requireNamespace("SingleCellExperiment", quietly = TRUE)) {
  BiocManager::install("SingleCellExperiment")
}
if (!requireNamespace("BayesPrism", quietly = TRUE)) {
  # install via CRAN or devtools as needed
  install.packages("BayesPrism")
}
if (!requireNamespace("InstaPrism", quietly = TRUE)) {
  # from dev or local install
  # devtools::install_github("Danko-Lab/InstaPrism")  # example
  stop("InstaPrism is not installed. Please install it before proceeding.")
}

# Load libraries
library(scran)
library(SingleCellExperiment)
library(BayesPrism)
library(InstaPrism)

# --- Parse arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Error: No dataset name provided. <DATASET_NAME>")
}
data_type <- args[1]

# --- Set up directories ---
script_dir <- getwd() # or another logic to detect the script location
bulks_path <- import_path <- file.path(script_dir, "..", "data", "deconvolution", data_type)
export_path <- file.path(script_dir, "..", "results", data_type)
cat("Import path:", import_path, "\n")
cat("Export path:", export_path, "\n")

if (!dir.exists(export_path)) dir.create(export_path, recursive = TRUE)

# --- 1) Load the real bulk mixture data (processed_bulks.csv) ---
realbulks_file <- file.path(bulks_path, "processed_bulks.csv")
if (!file.exists(realbulks_file)) {
  stop("Cannot find the realbulks.csv file at: ", realbulks_file)
}
realbulks_data <- read.csv(realbulks_file, stringsAsFactors = FALSE, row.names = 1)
cat("Loaded realbulks_data with dim:", dim(realbulks_data), "\n")

# We'll treat this as the mixture_data
mixture_data <- realbulks_data

# --- 2) Also handle the baseline references (sc_raw_real, sn_raw_real) if they exist ---
base_refs <- c("sc_raw_real", "sn_raw_real")

for (base_ref in base_refs) {
  signal_file <- file.path(import_path, paste0(base_ref, "_signal.csv"))
  cell_state_file <- file.path(import_path, paste0(base_ref, "_cell_state.csv"))

  if (!file.exists(signal_file) || !file.exists(cell_state_file)) {
    cat(
      "Skipping base reference:", base_ref,
      "since signal/cell_state file not found.\n"
    )
    next
  }

  # Read the reference
  cat("\nProcessing base reference:", base_ref, "\n")
  signal_data <- read.csv(signal_file, row.names = 1, stringsAsFactors = FALSE)
  cell_state <- read.csv(cell_state_file, stringsAsFactors = FALSE)

  # Subset mixture_data to match genes
  common_genes <- intersect(rownames(mixture_data), rownames(signal_data))
  if (length(common_genes) < 500) {
    cat(
      "Skipping base reference:", base_ref,
      " - fewer than 500 overlapping genes.\n"
    )
    next
  }

  mixture_sub <- mixture_data[common_genes, , drop = FALSE]
  signal_sub <- signal_data[common_genes, , drop = FALSE]

  # Build SingleCellExperiment
  scExpr <- SingleCellExperiment(assays = list(counts = as.matrix(signal_sub)))

  # Typically, the cell_state.csv has columns: cell_type, cell_subtype, ...
  # We'll assume row 1 is cell_type, row 2 is cell_subtype, etc.
  # Check how the scripts generate it
  cell_type_labels <- t(cell_state[1])
  cell_state_labels <- t(cell_state[2])

  # Prepare reference
  refPhi_obj <- refPrepare(
    sc_Expr = assay(scExpr, "counts"),
    cell.type.labels = cell_type_labels,
    cell.state.labels = cell_state_labels
  )

  # Run InstaPrism
  results <- InstaPrism(
    bulk_Expr = mixture_sub,
    refPhi_cs = refPhi_obj,
    n.core = 16,
    n.iter = 5000
  )

  # Save results
  cell_frac <- results@Post.ini.cs@theta
  out_prop <- file.path(export_path, paste0(base_ref, "_BayesPrism_proportions.csv"))
  write.table(cell_frac, out_prop, sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)

  cell_ref <- results@initial.reference@phi.cs
  out_ref <- file.path(export_path, paste0(base_ref, "_BayesPrism_usedref.csv"))
  write.table(cell_ref, out_ref, sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)

  cat("Done with base reference:", base_ref, "\n")
}

# --- 3) Identify references for other "real" transforms, e.g. ref_real_rawSN, ref_real_pcaSN, etc. ---
# We'll do a loop over a set of known transformations
transformations <- c(
  "rawSN",
  "pcaSN",
  "degSN",
  "degOtherSN",
  "degIntSN",
  "degRandSN",
  "degpcaSN",
  "scviSN",
  "scvi_LSshift_SN",
  "degScviSN",
  "degScviLSshift_SN"
)

# each transform might produce "ref_real_<transform>_signal.csv"
# We'll just do them all in a loop
for (trans in transformations) {
  # Build the file names
  prefix <- paste0("ref_real_", trans)
  signal_file <- file.path(import_path, paste0(prefix, "_signal.csv"))
  cell_state_file <- file.path(import_path, paste0(prefix, "_cell_state.csv"))

  if (!file.exists(signal_file) || !file.exists(cell_state_file)) {
    cat(
      "Skipping reference for transform:", trans,
      "because files not found.\n"
    )
    next
  }

  cat("\nProcessing transform:", trans, "\n")

  # Read them
  signal_data <- read.csv(signal_file, row.names = 1, stringsAsFactors = FALSE)
  cell_state <- read.csv(cell_state_file, stringsAsFactors = FALSE)

  # Subset mixture_data to match genes
  common_genes <- intersect(rownames(mixture_data), rownames(signal_data))
  if (length(common_genes) < 500) {
    cat("Skipping transform:", trans, " - fewer than 500 overlapping genes.\n")
    next
  }

  mixture_sub <- mixture_data[common_genes, , drop = FALSE]
  signal_sub <- signal_data[common_genes, , drop = FALSE]

  # Build SingleCellExperiment
  scExpr <- SingleCellExperiment(assays = list(counts = as.matrix(signal_sub)))

  # Format the cell_state
  cell_type_labels <- t(cell_state[1])
  cell_state_labels <- t(cell_state[2])

  refPhi_obj <- refPrepare(
    sc_Expr = assay(scExpr, "counts"),
    cell.type.labels = cell_type_labels,
    cell.state.labels = cell_state_labels
  )

  # Run InstaPrism
  results <- InstaPrism(
    bulk_Expr = mixture_sub,
    refPhi_cs = refPhi_obj,
    n.iter = 5000,
    n.core = 16
  )

  # Save output
  cell_frac <- results@Post.ini.cs@theta
  out_prop <- file.path(export_path, paste0(prefix, "_BayesPrism_proportions.csv"))
  write.table(cell_frac, out_prop, sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)

  # Also save the actual reference used after any possible filtering in InstaPrism
  cell_ref <- results@initial.reference@phi.cs
  out_ref <- file.path(export_path, paste0(prefix, "_BayesPrism_usedref.csv"))
  write.table(cell_ref, out_ref, sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)

  cat("Done with transform:", trans, "\n")
}

cat("\nAll references processed.\n")
