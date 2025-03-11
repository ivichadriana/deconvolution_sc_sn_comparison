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
from sklearn.decomposition import PCA

# Ensure PYTHONPATH includes the parent directory of 'src'
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
sys.path.insert(1, '../../../../../')
print("Updated PYTHONPATH:", sys.path)  # Debugging line

from src.helpers import prepare_data, split_single_cell_data 
from src.helpers import pick_cells, make_references
from src.helpers import make_pseudobulks, filter_out_cell_markers
from src.helpers import run_deseq2_for_cell_type, create_fixed_pseudobulk
from src.helpers import differential_expression_analysis, remove_diff_genes
from src.helpers import differential_expression_analysis_parallel, transform_heldout_sn_to_mean_sc
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
    parser.add_argument("--deseq_alpha", type=float, default=0.01, help="Alpha to use to categorize DEGs")
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
    max_cells_per_type = 1500  # Limit to 1500 cells per cell type
    adata_sc_ref, adata_sn_ref = make_references(adata_sc_ref, adata_sn, max_cells_per_type=max_cells_per_type, cell_types=cell_types)

    print("Creating pseudobulk of size 10 for SC and SN data...")

    pseudo_sc_adata = create_fixed_pseudobulk(adata_sc_ref, group_size=10)
    pseudo_sn_adata = create_fixed_pseudobulk(adata_sn_ref, group_size=10)

    print("Differential gene expression...")
    # Calculate differentially expressed genes
    diff_genes = differential_expression_analysis_parallel(sn_adata=pseudo_sn_adata,
                                                            sc_adata=pseudo_sc_adata, 
                                                            deseq_alpha=args.deseq_alpha,
                                                            num_threads=4,          # matches SLURM --ntasks
                                                            n_cpus_per_thread=16,   # matches SLURM --cpus-per-task
                                                            )
    print("Found these many differentially expressed genes:")
    for key in diff_genes.keys():
        print(key)
        print(diff_genes[key].shape)

    print("Saving diff expressed genes:")
    # Define the "universal" list of genes for final CSV
    all_genes = adata_sc_pseudo.var_names  

    # Create an empty DataFrame of shape [all_genes × cell_types], filled with NaN
    df_wide = pd.DataFrame(index=all_genes, 
                        columns=diff_genes.keys(), 
                        data=np.nan)

    # Fill each column with p-val
    for cell_type, de_df in diff_genes.items():
        # 'de_df' has shape [num_DE_genes, 6], with index = DE gene names
        df_wide.loc[de_df.index, cell_type] = de_df["padj"].values
        
    # Save to CSV
    df_wide.to_csv(f"{args.output_path}/differentially_expressed_genes.csv")
    
    print("Create references with filtering...")
    # Now remove diff genes and remake referece
    adata_sc_filtered, adata_sn_filtered = remove_diff_genes(sc_adata=adata_sc_ref, sn_adata=adata_sn_ref, diff_genes=diff_genes)
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
                        sc_adata_filtered=adata_sc_filtered,
                        adata_sn_ref=adata_sn_ref,
                        sn_adata_filtered=adata_sn_filtered,)
    elif args.deconvolution_method == "bayesprism":
        # **Save BayesPrism Files**
        save_bayesprism_references(adata_sc_ref, args.output_path, "sc_notfiltered")
        save_bayesprism_references(adata_sn_ref, args.output_path, "sn_notfiltered")
        save_bayesprism_references(adata_sc_filtered, args.output_path, "sc_filtered")
        save_bayesprism_references(adata_sn_filtered, args.output_path, "sn_filtered")
        save_bayesprism_pseudobulks(pseudobulks_df, proportions_df, args.output_path)
    else:
        raise ValueError('deconvolution_methods must be either "bayesprism" or "cibersortx')

    print("Now starting the held out references:")
    print("####################################")
    print("####################################")

    # 1) Load data
    print("Preparing data...")
    adata_sc, adata_sn = prepare_data(args.res_name, args.data_path)

    # 2) Split for pseudobulks & reference
    print("Splitting data...")
    adata_sc_pseudo, adata_sc_ref = split_single_cell_data(
        adata_sc, test_ratio=0.3, data_type=args.res_name
    )

    # 3) Pick common cell types
    cell_types = pick_cells(adata_sc_pseudo, adata_sc_ref, adata_sn, min_cells_per_type=50)
    print("Cell types selected:", cell_types)

    # 4) For each cell type => hold out in SC, replace with SN in 4 ways
    for cell_type in cell_types:
        print(f"\n### Holdout experiment for cell type: {cell_type} ###")

        # (A) SC minus the held-out cell type
        adata_sc_holdout = adata_sc_ref[adata_sc_ref.obs["cell_types"] != cell_type].copy()

        # (B) The SN subset for the held-out cell type
        adata_sn_ct = adata_sn[adata_sn.obs["cell_types"] == cell_type].copy()
        if adata_sn_ct.n_obs == 0:
            print(f"Skipping {cell_type}: Not found in SN dataset.")
            continue

        # (C) The SN reference except the held-out type
        adata_sn_except_ct = adata_sn[adata_sn.obs["cell_types"] != cell_type].copy()

        # Convert them to memory just in case
        adata_sc_holdout = adata_sc_holdout.copy()
        adata_sn_ct = adata_sn_ct.copy()
        adata_sn_except_ct = adata_sn_except_ct.copy()

        ### 1) Raw SN ####
        sc_plus_sn_raw = sc.concat([adata_sc_holdout, adata_sn_ct], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_raw, args.output_path, f"ref_{cell_type}_rawSN")

        ### 2) PCA SN ####
        adata_sn_transformed = transform_heldout_sn_to_mean_sc(
            adata_sc_ref=adata_sc_holdout,
            adata_sn_ref=adata_sn_except_ct,
            adata_sn_heldout=adata_sn_ct,
            variance_threshold=0.60
        )
        sc_plus_sn_pca = sc.concat([adata_sc_holdout, adata_sn_transformed], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_pca, args.output_path, f"ref_{cell_type}_pcaSN")

        ### 3) DEG-Filtered SN ####
        # Do DE analysis: SN (held-out ctype) vs. SC holdout (all ctypes).
        print("Running differential expression analysis (DE) ...")
        diff_genes = differential_expression_analysis_parallel(adata_sc_holdout, adata_sn_ct, args.deseq_alpha)

        sc_filtered, sn_filtered = remove_diff_genes(
            sc_adata=adata_sc_holdout, sn_adata=adata_sn_ct, diff_genes=diff_genes
        )
        # Combine
        sc_plus_sn_deg = sc.concat([sc_filtered, sn_filtered], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_deg, args.output_path, f"ref_{cell_type}_degSN")

        ### 4) DEG-Filtered + PCA ####
        # First, do the same filter on the SC holdout + SN-except-ct (for PCA)
        sc_filtered_for_pca, sn_filtered_for_pca = remove_diff_genes(
            sc_adata=adata_sc_holdout, sn_adata=adata_sn_except_ct, diff_genes=diff_genes
        )
        # Then transform the *held-out SN* after filtering the same genes
        # => We'll subset adata_sn_ct to that gene set as well for consistency
        common_genes_pca = sn_filtered_for_pca.var_names.intersection(sc_filtered_for_pca.var_names)
        adata_sn_ct_sub = adata_sn_ct[:, common_genes_pca].copy()

        adata_sn_transformed_deg = transform_heldout_sn_to_mean_sc(
            adata_sc_ref=sc_filtered_for_pca,
            adata_sn_ref=sn_filtered_for_pca,
            adata_sn_heldout=adata_sn_ct_sub,
            variance_threshold=0.60
        )

        # Merge the final references
        sc_plus_sn_deg_pca = sc.concat([sc_filtered_for_pca, adata_sn_transformed_deg], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_deg_pca, args.output_path, f"ref_{cell_type}_degPCA_SN")

    print("\n=== Holdout experiment completed. ===")

