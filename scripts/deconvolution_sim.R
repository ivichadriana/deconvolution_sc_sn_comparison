#!/usr/bin/env Rscript

# ============================================================
# R script to run Scaden and SCDC on all references created
# by the "prepare_deconvolution_sim.py" script.

# Usage:
#   Rscript deconvolution_sim.R <DATASET_NAME> <METHOD>
#     METHOD ∈ {scaden, scdc} <<< as a user, you can add more through omnideconv.
## ============================================================

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
  "scviSN",
  "degScviLSshift_SN",
  "degScviSN",
  "scvi_LSshift_SN",
  "scviSN",
  "rawSN",
  "pcaSN",
  "degSN",
  "degIntSN",
  "degRandSN",
  "degOtherSN",
  "degPCA_SN"
)

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(omnideconv)
})

# --- Parse arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript deconvolution_simreal.R <DATASET_NAME> <METHOD>; METHOD ∈ {scaden, scdc}")
}
data_type <- args[1]

raw_method   <- args[2]
method_norm  <- tolower(raw_method)        # normalize for logic / omnideconv
if (!method_norm %in% c("scaden", "scdc", "cpm")) {
  stop("METHOD must be 'scaden' or 'scdc' or 'cpm'")
}
method_key   <- method_norm                # what omnideconv expects
method_tag   <- toupper(method_norm)       # for filenames (scaden/SCDC)

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
mixture_data <- as.matrix(
  read.csv(mixture_file, stringsAsFactors = FALSE, row.names = 1, check.names = FALSE)
)
cat("Loaded mixture_data with dim:", dim(mixture_data), "\n")

# --- 2) Also handle the standard references sc_raw and sn_raw ---
# We'll do these first, outside the holdout logic because it gets more complicated:
base_refs <- c("sc_raw", "sn_raw", "degIntAllSN")

for (base_ref in base_refs) {
  signal_file <- file.path(import_path, paste0(base_ref, "_signal.csv"))
  cell_state_file <- file.path(import_path, paste0(base_ref, "_cell_state.csv"))

  if (!file.exists(signal_file) || !file.exists(cell_state_file)) {
    cat("Skipping:", base_ref, "(missing files)\n")
    next
  }
  # ##REMOVE if you want to rerun
  out_file <- file.path(export_path, paste0(base_ref, "_", method_tag, "_proportions.csv"))
  if (file.exists(out_file)) {
    cat("Skipping:", base_ref, "(already done!)\n")
    next
  }
  ##

  cat("\nProcessing:", base_ref, "with", method_tag, "\n")
  signal_data <- read.csv(signal_file, row.names = 1, check.names = FALSE)
  cell_state  <- read.csv(cell_state_file, stringsAsFactors = FALSE, check.names = FALSE)

  # Intersect genes
  common_genes <- intersect(rownames(mixture_data), rownames(signal_data))
  if (length(common_genes) < 500) {
    cat("  Fewer than 500 overlapping genes; skipping.\n")
    next
  }
  mixture_sub <- mixture_data[common_genes, , drop = FALSE]
  signal_sub  <- signal_data[common_genes, , drop = FALSE]

  # --- Build single-cell matrix (genes × cells) ---
  sc_mat <- as.matrix(signal_sub)        # genes × cells
  storage.mode(sc_mat) <- "double"

  # --- cell_type comes from the named column 'cell_type' ---
  if (!"cell_type" %in% names(cell_state)) stop("cell_state.csv must have a 'cell_type' column.")
  labels <- as.character(cell_state[["cell_type"]])

  # --- Bulk matrix must be genes × samples ---
  bulk_gxs <- mixture_sub
  storage.mode(bulk_gxs) <- "double"
  batch_ids <- rep("batch1", ncol(sc_mat))

  # --- One-step omnideconv---
  if (method_key %in% c("scdc", "cpm")) {    batch_ids <- rep("batch1", ncol(sc_mat))
    res <- omnideconv::deconvolute(
      bulk_gene_expression  = bulk_gxs,   # genes x samples
      method                = method_key,
      single_cell_object    = sc_mat,     # genes x cells
      cell_type_annotations = labels,     # length = ncol(sc_mat)
      batch_ids             = batch_ids,   # for SCDC but they are constant
      verbose = TRUE
    )
  } else {
    # scaden path (no batch_ids)
    signature <- omnideconv::build_model(sc_mat, labels,
                                  method=method_key, bulk_gene_expression=bulk_gxs,
                                  verbose = TRUE)
    res <- omnideconv::deconvolute(bulk_gene_expression = bulk_gxs, model = signature, method=method_key, verbose = TRUE)
  }
  # res is samples × cell_types; write as-is
  out_file <- file.path(export_path, paste0(base_ref, "_", method_tag, "_proportions.csv"))
  write.table(res, out_file,sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)
  cat("  Saved:", out_file, "\n")
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

    ##REMOVE if you want to rerun
    out_file <- file.path(export_path, paste0("ref_",  ct_clean, "_", trans, "_", method_tag, "_proportions.csv"))
    if (file.exists(out_file)) {
      cat("Skipping:", trans, ct, "(already done!)\n")
      next
    }
    ##

    cat("\nProcessing:", ct_clean, trans, "with", method_tag, "\n")
    signal_data <- read.csv(signal_file, row.names = 1, check.names = FALSE)
    cell_state  <- read.csv(cell_state_file, stringsAsFactors = FALSE, check.names = FALSE)

    # Intersect genes
    common_genes <- intersect(rownames(mixture_data), rownames(signal_data))
    if (length(common_genes) < 500) {
      cat("  Fewer than 500 overlapping genes; skipping.\n")
      next
    }
    mixture_sub <- mixture_data[common_genes, , drop = FALSE]
    signal_sub  <- signal_data[common_genes, , drop = FALSE]

    # --- Build single-cell matrix (genes × cells) ---
    sc_mat <- as.matrix(signal_sub)        # genes × cells
    storage.mode(sc_mat) <- "double"

    # --- cell_type comes from the named column 'cell_type' ---
    if (!"cell_type" %in% names(cell_state)) stop("cell_state.csv must have a 'cell_type' column.")
    labels <- as.character(cell_state[["cell_type"]])

    # --- Bulk matrix must be genes × samples ---
    bulk_gxs <- mixture_sub
    storage.mode(bulk_gxs) <- "double"
    batch_ids <- rep("batch1", ncol(sc_mat))

    # --- One-step omnideconv---
    if (method_key %in% c("scdc", "cpm")) {
      batch_ids <- rep("batch1", ncol(sc_mat))
      res <- omnideconv::deconvolute(
        bulk_gene_expression  = bulk_gxs,   # genes x samples
        method                = method_key,
        single_cell_object    = sc_mat,     # genes x cells
        cell_type_annotations = labels,     # length = ncol(sc_mat)
        batch_ids             = batch_ids,   # for SCDC but they are constant
        verbose = TRUE
      )
    } else {
      # scaden path (no batch_ids)
      signature <- omnideconv::build_model(sc_mat, labels,
                                    method=method_key, bulk_gene_expression=bulk_gxs,
                                    verbose = TRUE)
      res <- omnideconv::deconvolute(bulk_gene_expression = bulk_gxs, model = signature, method=method_key, verbose = TRUE)
    }
    # res is samples × cell_types; write as-is
    write.table(res, out_file, sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)
    cat("  Saved:", out_file, "\n")
}}

cat("\nAll references processed.\n")
