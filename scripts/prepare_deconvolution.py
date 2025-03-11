"""
Holdout Cell-Type Deconvolution Experiment

This script performs a series of deconvolution experiments by systematically holding out 
one cell type from single-cell (SC) data and replacing it with transformed single-nucleus 
(SN) data. The goal is to assess different transformation techniques for integrating SN 
data into SC datasets.

### Overview of Steps:
1. **Load & Prepare Data**: 
   - Reads preprocessed single-cell and single-nucleus data.
   - Splits SC data into reference and pseudobulk sets.
   - Selects common cell types shared between SC and SN datasets.

2. **Generate Reference Datasets**: 
   - Creates multiple versions of SC + SN datasets, where the held-out SC cell type is replaced 
     with SN-derived equivalents using different transformation methods:
     - **Raw SN**: Directly adding SN data for the missing cell type.
     - **PCA Transformed SN**: Adjusts SN expression profiles using PCA shifts.
     - **DEG-Filtered SN**: Removes differentially expressed genes (DEGs) between SC and SN.
     - **DEG-Filtered + PCA SN**: Applies PCA after DEG removal.
     - **scVI-VAE Transformed SN**: Uses a deep generative model to harmonize SN data.

3. **Train & Apply scVI Model**:
   - Trains an scVI model on SC + SN (excluding the held-out cell type).
   - Applies the trained model to transform SN-heldout data into SC-like expression.
   - Saves the trained scVI model for reuse.

4. **Generate Pseudobulk Samples**:
   - Creates pseudobulk datasets using predefined proportions.
   - Saves references and pseudobulks in formats compatible with BayesPrism.

5. **Output Files**:
   - **Processed SC & SN References** (raw, transformed, and filtered versions).
   - **Trained scVI Model** for each held-out cell type.
   - **Generated Pseudobulks** for deconvolution experiments.

### Usage:
This script is designed to be run via a SLURM job submission (use run_prep_all.sh) or from the command line.
Example command:
python prepare_deconv_all.py --res_name "ADP" --data_path "data/" --output_path "results/"
--pseudobulks_props '{"realistic": 500, "random": 500}'
--num_cells 1000 --noise True --deconvolution_method "bayesprism" 

"""

import json
import scanpy as sc
import pandas as pd
import numpy as np
import os
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
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
sys.path.insert(1, '../../../../../')
from src.helpers import prepare_data, split_single_cell_data 
from src.helpers import pick_cells, make_references
from src.helpers import transform_heldout_sn_to_mean_sc
from src.helpers import make_pseudobulks, create_fixed_pseudobulk
from src.helpers import run_deseq2_for_cell_type
from src.helpers import differential_expression_analysis, remove_diff_genes
from src.helpers import differential_expression_analysis_parallel
from src.helpers import save_cibersort, save_bayesprism_references, save_bayesprism_pseudobulks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Holdout cell-type deconvolution experiment")
    parser.add_argument("--res_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--pseudobulks_props", type=str, required=True, help="Pseudobulk proportions in JSON format")
    parser.add_argument("--num_cells", type=int, required=True, help="Number of cells per pseudobulk")
    parser.add_argument("--noise", type=bool, default=True, help="Add Gaussian noise to pseudobulks")
    parser.add_argument("--deseq_alpha", type=float, default=0.01, help="Alpha threshold for DEG analysis")
    parser.add_argument("--deconvolution_method", type=str, default="bayesprism", help="Deconvolution method")

    args = parser.parse_args()

    def load_best_config(res_name, output_path):
        best_config_file = f"{output_path}/best_config_scvi_tune.json"
        with open(best_config_file, "r") as f:
            best_config = json.load(f)
        return best_config

    # 1) Load data
    print("Preparing data...")
    adata_sc, adata_sn = prepare_data(args.res_name, args.data_path)

    # 2) Split for pseudobulks & reference
    print("Splitting data...")
    adata_sc_pseudo, adata_sc_ref = split_single_cell_data(
        adata_sc, data_type=args.res_name
    )

    # 3) Pick common cell types
    cell_types = pick_cells(adata_sc_pseudo, adata_sc_ref, adata_sn, min_cells_per_type=50)
    print("Cell types selected:", cell_types)

    # Create references for single-cell and single-nucleus
    max_cells_per_type = 1500  # Limit to 1500 cells per cell type
    adata_sc_ref, adata_sn_ref = make_references(adata_sc_ref, adata_sn, max_cells_per_type=max_cells_per_type, cell_types=cell_types)

    print("Creating pseudobulk of size 10 for SC and SN data for DGE analysis...")
    pseudo_sc_adata = create_fixed_pseudobulk(adata_sc_ref, group_size=10)
    pseudo_sn_adata = create_fixed_pseudobulk(adata_sn_ref, group_size=10)

    print("Calculate differential gene expression...")
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
    
    print(f"\n### Holdout experiment files... ###")

    # 4) Now our experimental: For each cell type => hold out in SC, replace with SN in different ways ()
    for cell_type in cell_types:

        print(f"\nCreating refs for cell type: {cell_type}..")

        #### First our baseline datsets: ####

        # (A) SC minus the held-out cell type
        adata_sc_except_ct = adata_sc_ref[adata_sc_ref.obs["cell_types"] != cell_type].copy()

        # (B and C) The SN and SC subset for the held-out cell type
        adata_sn_ct = adata_sn[adata_sn.obs["cell_types"] == cell_type].copy()
        adata_sc_ct = adata_sc[adata_sc.obs["cell_types"] == cell_type].copy()

        if adata_sn_ct.n_obs == 0:
            print(f"Skipping {cell_type}: Not found in SN dataset.")
            continue

        # (D) The SN reference except the held-out type
        adata_sn_except_ct = adata_sn[adata_sn.obs["cell_types"] != cell_type].copy()

        # Convert them to memory  
        adata_sc_except_ct = adata_sc_except_ct.copy()
        adata_sn_ct = adata_sn_ct.copy()
        adata_sc_ct = adata_sc_ct.copy()
        adata_sn_except_ct = adata_sn_except_ct.copy()

        ### Now our references we'll be testing: ###

        ### 1) Raw SN added to SC ####
        ##############################
        sc_plus_sn_raw = sc.concat([adata_sc_except_ct, adata_sn_ct], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_raw, args.output_path, f"ref_{cell_type.replace(' ', '_')}_rawSN")

        ### 2) PCA-transformed SN ####
        ##############################
        df_transformed_sn_ct = transform_heldout_sn_to_mean_sc(
            sc_data=adata_sc_except_ct.to_df(),
            sn_data=adata_sn_except_ct.to_df(),
            sn_heldout_data=adata_sn_ct.to_df(),
            variance_threshold=0.75, 
            sc_celltype_labels=adata_sc_except_ct.obs["cell_types"].astype(str),
            sn_celltype_labels=adata_sn_except_ct.obs["cell_types"].astype(str),
            heldout_label = cell_type
        )
        # Convert the DataFrame to an AnnData object. This is the PCA SN transformed.
        adata_sn_transformed = sc.AnnData(X=df_transformed_sn_ct.values)
        adata_sn_transformed.var_names = df_transformed_sn_ct.columns 
        adata_sn_transformed.obs = adata_sn_ct.obs.copy()

        # Concatenate with SC (SC with held-out cell removed)
        sc_plus_sn_pca = sc.concat([adata_sc_except_ct, adata_sn_transformed], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_pca, args.output_path, f"ref_{cell_type.replace(' ', '_')}_pcaSN")

        ### 3) DEG-Filtered SN ####
        ###########################
        #We are mimicking what happens in real life; not using the DEG from that cell type in questions (missing from SC)
        diff_genes_ct = {k: v for k, v in diff_genes.items() if k != cell_type}

        sc_filtered, sn_filtered = remove_diff_genes(
            sc_adata=adata_sc_except_ct, sn_adata=adata_sn_ct, diff_genes=diff_genes_ct
        )
        # Combine
        sc_plus_sn_deg = sc.concat([sc_filtered, sn_filtered], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_deg, args.output_path, f"ref_{cell_type.replace(' ', '_')}_degSN")

        ### 4) DEG-Filtered + PCA ####
        # First, do the same filter on the SC-except-ct + SN-except-ct (for PCA)
        sc_filtered_for_pca, sn_filtered_for_pca = remove_diff_genes(
            sc_adata=adata_sc_except_ct, sn_adata=adata_sn_except_ct, diff_genes=diff_genes_ct
        )

        # Then transform the *held-out SN* after filtering the same genes
        # => We'll subset adata_sn_ct to that gene set as well
        common_genes_pca = sn_filtered_for_pca.var_names.intersection(sc_filtered_for_pca.var_names)
        adata_sn_ct_filtered = adata_sn_ct[:, common_genes_pca].copy()

        df_sn_transformed_deg = transform_heldout_sn_to_mean_sc(
            sc_data=sc_filtered_for_pca.to_df(),
            sn_data=sn_filtered_for_pca.to_df(),
            sn_heldout_data=adata_sn_ct_filtered.to_df(),
            variance_threshold=0.75,
            sc_celltype_labels=sc_filtered_for_pca.obs.cell_types.astype(str),
            sn_celltype_labels=sn_filtered_for_pca.obs.cell_types.astype(str),
            heldout_label = cell_type
        )

        # Convert the DataFrame to an AnnData object.
        adata_sn_transformed_deg = sc.AnnData(X=df_sn_transformed_deg.values)
        adata_sn_transformed_deg.var_names = df_sn_transformed_deg.columns
        adata_sn_transformed_deg.obs = adata_sn_ct_filtered.obs.copy()

        # Merge the final references
        sc_plus_sn_deg_pca = sc.concat([sc_filtered_for_pca, adata_sn_transformed_deg], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_deg_pca, args.output_path, f"ref_{cell_type.replace(' ', '_')}_degPCA_SN")

        ### 5) VAE Transform ####

        # Create "training" AnnData: SC minus this cell type + SN minus this cell type ("testing" data is our missing ct)
        adata_train = sc.concat([adata_sc_except_ct, adata_sn_except_ct], axis=0)
        adata_train.obs_names_make_unique()

        # scVI setup
        scvi.model.SCVI.setup_anndata(
            adata_train,
            batch_key="data_type"
        )

        # Define the model save path
        model_save_path = f"{args.output_path}/scvi_trained_model_NO{cell_type}"

        # Check if a trained model exists, training otherwise
        if os.path.exists(model_save_path):
            print(f"Loading pre-trained scVI model from {model_save_path}...")
            model = scvi.model.SCVI.load(model_save_path, adata=adata_train)
        else:
            # Training
            model = scvi.model.SCVI(adata_train, dispersion="gene-batch", use_observed_lib_size=False, encode_covariates=True, deeply_inject_covariates=True )
            model.train(max_epochs=300, train_size=0.8, early_stopping=True, early_stopping_patience=10)
            # raise FileNotFoundError(f"No pre-trained scVI model found at {model_save_path}. Please train and save the model first.")

        # Transform the SN cells of the held-out type with this target library size
        transformed_expr = model.get_normalized_expression(
            adata=adata_sn_ct,                
            transform_batch="single_cell",    
            # library_size parameter here will now use the latent estimates from scVI
            return_numpy=True
        )
        # (Build an AnnData of the transformed SN cells
        adata_sn_transformed_scvi = sc.AnnData(X=transformed_expr)
        adata_sn_transformed_scvi.var_names = adata_sn_ct.var_names
        adata_sn_transformed_scvi.obs = adata_sn_ct.obs.copy()

        # Merge with SC minus the held-out cell -> final reference
        sc_plus_sn_scvi = sc.concat([adata_sc_except_ct, adata_sn_transformed_scvi], axis=0, merge="same")

        # Save for deconvolution, reference and trained model.
        model.save(f"{args.output_path}/scvi_trained_model_NO{cell_type.replace(' ', '_')}", overwrite=True)
        out_prefix = f"ref_{cell_type.replace(' ', '_')}_scviSN"
        save_bayesprism_references(sc_plus_sn_scvi, args.output_path, out_prefix)

    print("Make pseudobulks...")
    # Generate pseudobulks
    pseudobulk_config = eval(args.pseudobulks_props)
    pseudobulks_df, proportions_df = make_pseudobulks(adata_sc_pseudo, pseudobulk_config, args.num_cells, args.noise, cell_types=cell_types)

    print("And save the controls SN and SC...")
    os.makedirs(args.output_path, exist_ok=True)
    if args.deconvolution_method == "bayesprism":
        # **Save BayesPrism Files**
        save_bayesprism_references(adata_sc_ref, args.output_path, "sc_raw")
        save_bayesprism_references(adata_sn_ref, args.output_path, "sn_raw")
        save_bayesprism_pseudobulks(pseudobulks_df, proportions_df, args.output_path)
    else:
        raise ValueError('deconvolution_methods must be "bayesprism" (others not supported yet.)')

    print("\n=== Holdout experiment files completed. ===")

