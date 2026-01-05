#!/usr/bin/env Rscript

# ============================================================
# R script to run BayesPrism on all references created
# by the "prepare_deconvolution.py" script.
#
# Usage:
#   Rscript BayesPrism_sim.R <DATA_TYPE>
# ============================================================

# --------------------------------------------------------------------
# Settings (add your transform name here!) (if human specific, add to human.)
# --------------------------------------------------------------------
transformations_not_human <- c(
  "rawSN",
  "pcaSN",
  "degSN",
  "degRandSN",
  "degPCA_SN",
  "scviSN",
  "scvi_LSshift_SN",
  "degScviSN",
  "degScviLSshift_SN"
)
transformations_human <- c(
  "rawSN",
  "pcaSN",
  "degSN",
  "degIntSN",
  "degRandSN",
  "degOtherSN",
  "degPCA_SN",
  "scviSN",
  "scvi_LSshift_SN",
  "degScviSN",
  "degScviLSshift_SN"
)

library(scran)
library(BayesPrism)
library(InstaPrism) 
library(SingleCellExperiment)

# --- Parse arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Error: No data type provided. Usage: Rscript BayesPrism.R <DATA_TYPE>")
}
data_type <- args[1]

# --- Set up directories ---
script_dir <- file.path(getwd(), "scripts")
import_path <- file.path(script_dir, "..", "data/deconvolution", data_type)
export_path <- file.path(script_dir, "..", "results", data_type)
cat("Import path:", import_path, "\n")
cat("Export path:", export_path, "\n")
if (!dir.exists(export_path)) dir.create(export_path, recursive = TRUE)

# --- 1) Load the pseudobulk mixture (pseudobulks.csv) ---
mixture_file <- file.path(import_path, "pseudobulks.csv")
if (!file.exists(mixture_file)) {
  stop("Cannot find the pseudobulks.csv file at: ", mixture_file)
}
mixture_data <- read.csv(mixture_file, stringsAsFactors = FALSE, row.names = 1)
cat("Loaded mixture_data with dim:", dim(mixture_data), "\n")

# --- 2) Also handle the standard references sc_raw and sn_raw ---
# We'll do these first, outside the holdout logic because it gets more complicated:
base_refs <- c("sc_raw", "sn_raw", "degIntAllSN")

for (base_ref in base_refs) {
  signal_file <- file.path(import_path, paste0(base_ref, "_signal.csv"))
  cell_state_file <- file.path(import_path, paste0(base_ref, "_cell_state.csv"))

  if (!file.exists(signal_file) || !file.exists(cell_state_file)) {
    cat("Skipping base reference:", base_ref, "since signal/cell state not found.\n")
    next
  }

  # Read the reference
  cat("\nProcessing base reference:", base_ref, "\n")
  signal_data <- read.csv(signal_file, row.names = 1, stringsAsFactors = FALSE)
  cell_state <- read.csv(cell_state_file, stringsAsFactors = FALSE)

  # Subset mixture_data to match genes
  common_genes <- intersect(rownames(mixture_data), rownames(signal_data))
  if (length(common_genes) < 500) {
    cat("Skipping base reference:", base_ref, " - fewer than 500 overlapping genes.\n")
    next
  }

  mixture_sub <- mixture_data[common_genes, , drop = FALSE]
  signal_sub <- signal_data[common_genes, , drop = FALSE]

  # Build SingleCellExperiment
  scExpr <- SingleCellExperiment(assays = list(counts = as.matrix(signal_sub)))

  # cell_type_labels is the first row
  # cell_state_labels is the second row
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
  write.table(cell_frac, out_prop,
    sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA
  )

  cell_ref <- results@initial.reference@phi.cs
  out_ref <- file.path(export_path, paste0(base_ref, "_BayesPrism_usedref.csv"))
  write.table(cell_ref, out_ref, sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)

  cat("Done with base reference:", base_ref, "\n")
}

# --- 3) Identify cell types from the 'sc_raw' cell_state file ---
# We'll read sc_raw_cell_state.csv to parse all cell types.
sc_raw_cell_state_file <- file.path(import_path, "sc_raw_cell_state.csv")
if (!file.exists(sc_raw_cell_state_file)) {
  stop("Cannot find sc_raw_cell_state.csv to retrieve cell types.")
}
sc_raw_cell_state <- read.csv(sc_raw_cell_state_file, stringsAsFactors = FALSE)
unique_cell_types <- unique(sc_raw_cell_state$cell_type)
cat("\nUnique cell types in sc_raw:", unique_cell_types, "\n")

# For each cell type, we look for references with patterns like:
#   ref_<CELLTYPE>_<transform>_signal.csv
#   ref_<CELLTYPE>_<transform>_cell_state.csv
# where <transform> is one of: rawSN, pcaSN, degSN, degPCA_SN or scviSN

if (data_type != "MSB") {
  transformations <- transformations_human
} else {
  transformations <- transformations_not_human
}
# --- 4) Loop over each cell type & each transformation ---
for (ct in unique_cell_types) {
  # sanitize
  ct_clean <- gsub(" ", "_", ct)

  for (trans in transformations) {
    # Build file names
    signal_file <- file.path(import_path, paste0("ref_", ct_clean, "_", trans, "_signal.csv"))
    cell_state_file <- file.path(import_path, paste0("ref_", ct_clean, "_", trans, "_cell_state.csv"))

    # Check if they exist
    if (!file.exists(signal_file) || !file.exists(cell_state_file)) {
      cat(
        "Skipping reference for cell type:", ct, "transform:", trans,
        "because files not found.\n"
      )
      next
    }

    cat("\nProcessing cell type:", ct, "transform:", trans, "\n")

    # Read them
    signal_data <- read.csv(signal_file, row.names = 1, stringsAsFactors = FALSE)
    cell_state <- read.csv(cell_state_file, stringsAsFactors = FALSE)

    # Subset mixture_data to match genes
    common_genes <- intersect(rownames(mixture_data), rownames(signal_data))
    if (length(common_genes) < 500) {
      cat("Skipping cell type:", ct, "transform:", trans, " - fewer than 500 overlapping genes.\n")
      next
    }

    mixture_sub <- mixture_data[common_genes, , drop = FALSE]
    signal_sub <- signal_data[common_genes, , drop = FALSE]

    # Build SingleCellExperiment
    scExpr <- SingleCellExperiment(assays = list(counts = as.matrix(signal_sub)))

    # Format the cell_state into what refPrepare expects
    cell_type_labels <- t(cell_state[2]) # second column is 'cell_type'
    cell_state_labels <- t(cell_state[3]) # third column is 'cell_subtype'

    refPhi_obj <- refPrepare(
      sc_Expr = assay(scExpr, "counts"),
      cell.type.labels = cell_type_labels,
      cell.state.labels = cell_state_labels
    )

    # Run InstaPrism
    results <- InstaPrism(bulk_Expr = mixture_sub, refPhi_cs = refPhi_obj, n.iter = 5000, n.core = 8)

    # Save output
    cell_frac <- results@Post.ini.cs@theta
    out_prop <- file.path(export_path, paste0("ref_", ct_clean, "_", trans, "_BayesPrism_proportions.csv"))
    write.table(cell_frac, out_prop, sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)

    cell_ref <- results@initial.reference@phi.cs
    out_ref <- file.path(export_path, paste0("ref_", ct_clean, "_", trans, "_BayesPrism_usedref.csv"))
    write.table(cell_ref, out_ref, sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)

    cat("Done with cell type:", ct, "transform:", trans, "\n")
  }
}

cat("\nAll references processed.\n")
