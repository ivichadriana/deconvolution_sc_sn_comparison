"""
Differential Expression: Single-Cell (sc) vs Single-Nucleus (sn)

This script identifies genes that are differentially expressed between
single-cell and single-nucleus RNA-seq profiles **within the same cell type**.

Workflow
--------
1. **Reproducibility** – Seeds NumPy, PyTorch, and Python’s RNG, and fixes
   `PYTHONHASHSEED`.
2. **Argument parsing** –  
   * `--res_name` : dataset code (e.g. “PBMC”).  
   * `--data_path` / `--output_path` – I/O locations.  
   * `--deseq_alpha` – FDR threshold for DEGs (default 0.01).  
   * `--min_cells_per_type` – minimum cells to keep a cell type (default 50).
3. **Load data** – `prepare_data` returns separate AnnData objects for sc and sn.
4. **Hold-out split (sc only)** –  
   * `split_single_cell_data` creates a **pseudobulk** set (for DE testing) and a
     **reference** set (for deconvolution).
5. **Cell-type filtering** – `pick_cells` keeps cell types present in both
   modalities with enough cells.
6. **Reference construction** – `make_references` downsamples each modality to
   ≤ 1 500 cells per type to balance comparisons.
7. **DEG computation** –  
   * `load_or_calc_degs` runs DESeq2 (via *pydeseq2*) for each cell type,
     comparing sc vs sn counts, or loads cached results if available.
   * Results (gene lists, log₂FC, adjusted p) are saved to `output_path`.

Use this script to benchmark modality-specific biases or to exclude sc/sn
marker/different genes before downstream deconvolution.
"""

import json
import scanpy as sc
import pandas as pd
import json
import scvi
import numpy as np
import os
import torch
import random
import sys
import gc
from scipy.sparse import issparse
from multiprocessing import Pool
from sklearn.decomposition import PCA
import scvi
from pydeseq2.default_inference import DefaultInference
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

import sys

sys.path.insert(1, "../../")
sys.path.insert(1, "../")
sys.path.insert(1, "../../../../")
from src.helpers import prepare_data, split_single_cell_data
from src.helpers import pick_cells, make_references
from src.helpers import (
    save_cibersort,
    save_bayesprism_references,
    save_bayesprism_pseudobulks,
)
from src.deg_funct import (
    create_fixed_pseudobulk,
    load_others_degs,
    run_deseq2_for_cell_type,
    load_or_calc_degs,
)
from src.deg_funct import (
    differential_expression_analysis,
    remove_diff_genes,
    differential_expression_analysis_parallel,
)

SEED = 42
# Set Python built-in random seed
random.seed(SEED)
# Set NumPy random seed
np.random.seed(SEED)
# Set PyTorch random seeds for CPU and GPU
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Ensure reproducibility in hash-based operations
os.environ["PYTHONHASHSEED"] = str(SEED)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Holdout cell-type deconvolution experiment"
    )
    parser.add_argument(
        "--res_name", type=str, required=True, help="Name of the dataset"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save outputs"
    )
    parser.add_argument(
        "--deseq_alpha",
        type=float,
        default=0.01,
        help="Alpha threshold for DEG analysis",
    )
    parser.add_argument(
        "--min_cells_per_type",
        type=float,
        default=50,
        help="Minimum number of cells for a cell type to be considered",
    )

    args = parser.parse_args()

    if args.res_name in ["MSB", "ADP", "PBMC", "MBC"]:
        check_min_cells = True
    else:
        check_min_cells = False

    # 1) Load data
    print("Preparing data...")
    adata_sc, adata_sn = prepare_data(args.res_name, args.data_path)

    # 2) Split for pseudobulks & reference
    print("Splitting data...")
    adata_sc_pseudo, adata_sc_ref = split_single_cell_data(
        adata_sc, data_type=args.res_name
    )

    # 3) Pick common cell types
    cell_types = pick_cells(
        adata_sc_pseudo,
        adata_sc_ref,
        adata_sn,
        min_cells_per_type=args.min_cells_per_type,
        check_min_cells=check_min_cells,
    )
    print("Cell types selected:", cell_types)

    # Create references for single-cell and single-nucleus
    max_cells_per_type = 1500  # Limit to 1500 cells per cell type
    adata_sc_ref, adata_sn_ref = make_references(
        adata_sc_ref,
        adata_sn,
        max_cells_per_type=max_cells_per_type,
        cell_types=cell_types,
    )

    _ = load_or_calc_degs(
        output_path=args.output_path,
        adata_sc_ref=adata_sc_ref,
        adata_sn_ref=adata_sn_ref,
        deseq_alpha=args.deseq_alpha,
    )
