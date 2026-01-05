
#!/usr/bin/env Rscript

# ============================================================
# Run omnideconv deconvolution (scaden/scdc) on "real" references
# Usage:
#   Rscript deconvolution_real.R <DATASET_NAME> <METHOD>
#     METHOD ∈ {scaden, scdc} <<< as a user, you can add more through omnideconv.
# ============================================================

suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(omnideconv)
})
# ------------------------- args -------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Usage: Rscript deconvolution_real.R <DATASET_NAME> <METHOD>; METHOD ∈ {scaden, scdc}")
}
data_type <- args[1]

raw_method   <- args[2]
method_norm  <- tolower(raw_method)        # normalize for logic / omnideconv
if (!method_norm %in% c("scaden", "scdc", "cpm")) {
  stop("METHOD must be 'scaden' or 'scdc' or 'cpm'")
}
method_key   <- method_norm                # what omnideconv expects
method_tag   <- toupper(method_norm)       # for filenames (scaden/SCDC)

# --- paths ---
script_dir <- file.path(getwd(), "scripts")
bulks_path  <- import_path <- file.path(script_dir, "..", "data", "deconvolution", data_type)
export_path <- file.path(script_dir, "..", "results", data_type)
cat("Import path:", import_path, "\n")
cat("Export path:", export_path, "\n")
if (!dir.exists(export_path)) dir.create(export_path, recursive = TRUE)

# --- bulk mixtures (same file your BP script uses) ---
bulk_file <- file.path(bulks_path, "processed_bulks.csv")
if (!file.exists(bulk_file)) stop("Cannot find processed_bulks.csv at: ", bulk_file)
mixture_data <- as.matrix(read.csv(bulk_file, row.names = 1, check.names = FALSE))
cat("Loaded bulks:", dim(mixture_data), "\n")

# --- references to process (same set as BP script) ---
base_refs <- c("sc_raw_real", "sn_raw_real", "ref_real_degIntAllSN")
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
all_refs <- c(base_refs, paste0("ref_real_", transformations))

run_one <- function(prefix, mixture_data, import_path, export_path, method_key, method_tag) {
  signal_file     <- file.path(import_path, paste0(prefix, "_signal.csv"))
  cell_state_file <- file.path(import_path, paste0(prefix, "_cell_state.csv"))

  if (!file.exists(signal_file) || !file.exists(cell_state_file)) {
    cat("Skipping:", prefix, "(missing files)\n")
    return(invisible(NULL))
  }
  ##REMOVE if you want to rerun
  out_file <- file.path(export_path, paste0(prefix, "_", method_tag, "_proportions.csv"))
  if (file.exists(out_file)) {
    cat("Skipping:", prefix, "(already done!)\n")
    return(invisible(NULL))
  }
  ##

  cat("\nProcessing:", prefix, "with", method_tag, "\n")
  signal_data <- read.csv(signal_file, row.names = 1, check.names = FALSE)
  cell_state  <- read.csv(cell_state_file, check.names = FALSE)

  # Intersect genes
  common_genes <- intersect(rownames(mixture_data), rownames(signal_data))
  if (length(common_genes) < 500) {
    cat("  Fewer than 500 overlapping genes; skipping.\n")
    return(invisible(NULL))
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
  out_file <- file.path(export_path, paste0(prefix, "_", method_tag, "_proportions.csv"))
  write.table(res, out_file, sep = "\t", quote = FALSE, row.names = TRUE, col.names = NA)
  cat("  Saved:", out_file, "\n")
}

for (ref in all_refs) {
  run_one(ref, mixture_data, import_path, export_path, method_key, method_tag)
}

cat("\nAll references processed with omnideconv (", method_tag, ").\n", sep = "")
