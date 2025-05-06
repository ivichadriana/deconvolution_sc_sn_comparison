"""
Holdout Cell-Type Deconvolution: scVI Training with & without DEG Filtering
---------------------------------------------------------------------------

This script performs a holdout-based deconvolution benchmark by training scVI 
variational autoencoders on paired single-cell (SC) and single-nucleus (SN) RNA-seq 
data, both with and without filtering out differentially expressed genes (DEGs).

Workflow:
1. **Data Loading & Preprocessing**  
   - Load SC and SN AnnData objects based on `--res_name` and `--data_path`.  
   - Split SC into “pseudobulk” and “reference” subsets.  
   - Select common cell types (≥50 cells in both SC & SN) and downsample to ≤1500 cells/type.

2. **Pseudobulk Generation & DEG Analysis**  
   - Create fixed-size pseudobulks (10 cells) for SC and SN.  
   - Compute or load cached DEGs (FDR ≤ `--deseq_alpha`) via parallelized DESeq2.  
   - Store DEG lists in `degs.json`.

3. **scVI Training – DEG-Filtered (Conditional) Models**  
   For each held-out cell type:  
     a. Remove its DEGs from SC and SN reference sets.  
     b. Concatenate remaining cells (SC + SN) and set `batch_key="data_type"`.  
     c. Train scVI with early stopping and save to  
        `scvi_NODEG_trained_model_NO<cell_type>`.

4. **scVI Training – Full Gene Set (Unconditional) Models**  
   For each held-out cell type:  
     a. Use all genes (no DEG removal).  
     b. Concatenate SC + SN reference and set `batch_key=None`.  
     c. Train scVI with early stopping and save to  
        `scvi_NODEG_notcond_trained_model_NO<cell_type>`.

Usage:
    python script.py \
        --res_name    <dataset_name> \
        --data_path   <path/to/input/data> \
        --output_path <path/to/save/results> \
        --deseq_alpha 0.01

Arguments:
    --res_name      Identifier for the dataset (e.g., “human_brain”)
    --data_path     Directory or file containing raw AnnData objects
    --output_path   Directory to write pseudobulks, DEGs, and trained models
    --deseq_alpha   FDR threshold for DEG calling (default: 0.01)
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
from src.helpers import run_deseq2_for_cell_type, transform_heldout_sn_to_mean_sc_local
from src.helpers import differential_expression_analysis, remove_diff_genes
from src.helpers import differential_expression_analysis_parallel
from src.helpers import save_cibersort, save_bayesprism_references, save_bayesprism_pseudobulks

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Holdout cell-type deconvolution experiment")
    parser.add_argument("--res_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--deseq_alpha", type=float, default=0.01, help="Alpha threshold for DEG analysis")

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
    
    print("Calculate differential gene expression if needed...")
    genes_save_path = f"{args.output_path}/degs.json"

    if os.path.exists(genes_save_path):
        try:
            with open(genes_save_path, 'r') as file:
                diff_genes_json = json.load(file)
            # Convert the JSON representation back to DataFrames
            diff_genes = {}
            for key, value in diff_genes_json.items():
                if isinstance(value, dict) and "index" in value and "columns" in value and "data" in value:
                    diff_genes[key] = pd.DataFrame(**value)
                else:
                    diff_genes[key] = value
        except json.JSONDecodeError:
            print("Warning: JSON file is empty or corrupted. Recalculating diff_genes...")
            diff_genes = differential_expression_analysis_parallel(
                sn_adata=pseudo_sn_adata,
                sc_adata=pseudo_sc_adata, 
                deseq_alpha=args.deseq_alpha,
                num_threads=4,          # matches SLURM --ntasks
                n_cpus_per_thread=16    # matches SLURM --cpus-per-task
            )
            # Write a JSON-serializable copy to file
            diff_genes_json = {
                key: value.to_dict(orient='split') if isinstance(value, pd.DataFrame) else value
                for key, value in diff_genes.items()
            }
            with open(genes_save_path, 'w') as file:
                json.dump(diff_genes_json, file)
    else:
        # Calculate differentially expressed genes
        diff_genes = differential_expression_analysis_parallel(
            sn_adata=pseudo_sn_adata,
            sc_adata=pseudo_sc_adata, 
            deseq_alpha=args.deseq_alpha,
            num_threads=4,          # matches SLURM --ntasks
            n_cpus_per_thread=16    # matches SLURM --cpus-per-task
        )
        print("Found these many differentially expressed genes:")
        for key in diff_genes.keys():
            print(key)
            print(diff_genes[key].shape)
        
        # Create a JSON-serializable copy for dumping while keeping diff_genes as DataFrames in memory
        diff_genes_json = {
            key: value.to_dict(orient='split') if isinstance(value, pd.DataFrame) else value
            for key, value in diff_genes.items()
        }
        with open(genes_save_path, 'w') as file:
            json.dump(diff_genes_json, file)

    for cell_type in cell_types:

        diff_genes_ct = {k: v for k, v in diff_genes.items() if k != cell_type}

        print(f"\nTraining (-) DEG genes: conditional models for cell type: {cell_type}..")
        # Define the model save path
        model_save_path = f"{args.output_path}/scvi_nodeg_trained_model_NO{cell_type.replace(' ', '_')}"

        # Check if a trained model exists, training otherwise
        if os.path.exists(model_save_path):
            print(f"Model exists at: {model_save_path}...")
            continue
        else:

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
            ### 5) VAE Transform ####

            # Create "training" AnnData: SC minus this cell type + SN minus this cell type ("testing" data is our missing ct)
            adata_sc_except_ct, adata_sn_except_ct = remove_diff_genes(sc_adata=adata_sc_except_ct, 
                                                                        sn_adata=adata_sn_except_ct, 
                                                                        diff_genes=diff_genes_ct)
            
            adata_train = sc.concat([adata_sc_except_ct, adata_sn_except_ct], axis=0)
            adata_train.obs_names_make_unique()

            # scVI setup
            scvi.model.SCVI.setup_anndata(
                adata_train,
                batch_key="data_type"
            )
            # Quick checks
            print(adata_train.obs["data_type"].unique())
            print(adata_train.obs["data_type"].dtype)

            # Training
            model = scvi.model.SCVI(adata_train, encode_covariates=True, 
                                                deeply_inject_covariates=True,
                                                n_layers=2, 
                                                n_latent=30, dispersion='gene-batch',
                                                gene_likelihood="nb")
            model.view_anndata_setup()
            #model.train(max_epochs=300, train_size=0.8, early_stopping=True, early_stopping_patience=10)
            model.train(early_stopping=True, early_stopping_patience=10)
            # Save for deconvolution, reference and trained model.
            model.save(f"{args.output_path}/scvi_NODEG_trained_model_NO{cell_type.replace(' ', '_')}", overwrite=True)

    print(f"\nTraining all EG genes: NOT conditional models models for cell type: {cell_type}..")

    for cell_type in cell_types:

        diff_genes_ct = {k: v for k, v in diff_genes.items() if k != cell_type}

        # Define the model save path
        model_save_path = f"{args.output_path}/scvi_nodeg_notcond_trained_model_NO{cell_type.replace(' ', '_')}"

        # Check if a trained model exists, training otherwise
        if os.path.exists(model_save_path):
            print(f"Model exists at: {model_save_path}...")
            continue
        else:

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
            ### 5) VAE Transform ####
            adata_sc_except_ct, adata_sn_except_ct = remove_diff_genes(sc_adata=adata_sc_except_ct, 
                                                                        sn_adata=adata_sn_except_ct, 
                                                                        diff_genes=diff_genes_ct)

            # Create "training" AnnData: SC minus this cell type + SN minus this cell type ("testing" data is our missing ct)
            adata_train = sc.concat([adata_sc_except_ct, adata_sn_except_ct], axis=0)
            adata_train.obs_names_make_unique()

            # scVI setup
            scvi.model.SCVI.setup_anndata(
                adata_train,
                batch_key=None
            )

            # Training
            model = scvi.model.SCVI(adata_train,n_layers=2, 
                                                n_latent=30, 
                                                gene_likelihood="nb")
            model.view_anndata_setup()
            #model.train(max_epochs=300, train_size=0.8, early_stopping=True, early_stopping_patience=10)
            model.train(early_stopping=True, early_stopping_patience=10)

            # Save for deconvolution, reference and trained model.
            model.save(f"{args.output_path}/scvi_NODEG_notcond_trained_model_NO{cell_type.replace(' ', '_')}", overwrite=True)

    print("\n=== All models trained. ===")