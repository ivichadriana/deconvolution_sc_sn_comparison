import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import issparse
from pydeseq2.default_inference import DefaultInference
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import os
import sys
import gc
from multiprocessing import Pool

# Ensure PYTHONPATH includes the parent directory of 'src'
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
sys.path.insert(1, '../../../../../')
print("Updated PYTHONPATH:", sys.path)  # Debugging line

from src.helpers import prepare_data, split_single_cell_data 
from src.helpers import pick_cells, make_references
from src.helpers import make_pseudobulks
from src.helpers import run_deseq2_for_cell_type
from src.helpers import differential_expression_analysis, remove_diff_genes
from src.helpers import differential_expression_analysis_parallel
from src.helpers import save_cibersort, save_bayesprism_references, save_bayesprism_pseudobulks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Make pseudobulks")
    parser.add_argument("--res_name", type=str, required=True, help="Name of the data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data")
    parser.add_argument("--pseudobulks_props", type=str, required=True, help="Proportions of pseudobulks in JSON format")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output files")
    parser.add_argument("--num_cells", type=int, required=True, help="Number of cells per pseudobulk")
    parser.add_argument("--noise", type=bool, default=True, help="Add Gaussian noise to pseudobulks")
    parser.add_argument("--deconvolution_method", type=str, default="bayesprism", help="Type of deconvolution method to prepare files for.")

    args = parser.parse_args()

    # Prepare data
    print("Preparing data...")
    adata_sc, adata_sn = prepare_data(args.res_name, args.data_path)

    print("Splitting data...")
    # Split single-cell data into pseudobulks and reference
    adata_sc_pseudo, adata_sc_ref = split_single_cell_data(adata_sc, test_ratio=0.3, data_type=args.res_name)
    min_cells_per_type = 50
    cell_types = pick_cells(adata_sc_pseudo, adata_sc_ref, adata_sn, min_cells_per_type=50)
    print("We will keep these cell types:", cell_types)

    print("Create references without filtering...")

    # Create references for single-cell and single-nucleus
    max_cells_per_type = 1500  # Limit to 1000 cells per cell type
    adata_sc_ref, adata_sn_ref = make_references(adata_sc_ref, adata_sn, max_cells_per_type=max_cells_per_type, cell_types=cell_types)

    print("Differential gene expression...")
    # Calculate differentially expressed genes
    diff_genes = differential_expression_analysis_parallel(adata_sc_ref, adata_sn_ref)
    print("Found these many differentially expressed genes:")
    for key in diff_genes.keys():
        print(key)
        print(diff_genes[key].shape)

    print("Create references with filtering...")
    # Now remove diff genes and remake referece
    sc_adata_filtered, sn_adata_filtered = remove_diff_genes(sc_adata=adata_sc_ref, sn_adata=adata_sn_ref, diff_genes=diff_genes)
    adata_sc_ref_removed, adata_sn_ref_removed = make_references(sc_adata_filtered, sn_adata_filtered, max_cells_per_type=max_cells_per_type)

    print("Make pseudobulks...")
    # Generate pseudobulks
    pseudobulk_config = eval(args.pseudobulks_props)
    pseudobulks_df, proportions_df = make_pseudobulks(adata_sc_pseudo, pseudobulk_config, args.num_cells, args.noise, cell_types=cell_types)

    print("And save all...")

    # Save results
    os.makedirs(args.output_path, exist_ok=True)

    if args.deconvolution_method == "cibersortx":
        save_cibersort(pseudobulks_df=pseudobulks_df, 
                        proportions_df=proportions_df, 
                        adata_sc_ref=adata_sc_ref, 
                        sc_adata_filtered=sc_adata_filtered,
                        adata_sn_ref=adata_sn_ref,
                        sn_adata_filtered=sn_adata_filtered,)
    elif args.deconvolution_method == "bayesprism":
        # **Save BayesPrism Files**
        save_bayesprism_references(adata_sc_ref, args.output_path, "sc_notfiltered")
        save_bayesprism_references(adata_sn_ref, args.output_path, "sn_notfiltered")
        save_bayesprism_references(adata_sc_ref_removed, args.output_path, "sc_filtered")
        save_bayesprism_references(adata_sn_ref_removed, args.output_path, "sn_filtered")
        save_bayesprism_pseudobulks(pseudobulks_df, proportions_df, args.output_path)
    else:
        raise ValueError('deconvolution_methods must be either "bayesprism" or "cibersortx.')

