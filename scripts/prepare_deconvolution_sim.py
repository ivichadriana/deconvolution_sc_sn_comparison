"""
Holdout Cell-Type Deconvolution Experiment

This script generates multiple references for deconvolution by systematically holding out 
one cell type in single-cell (SC) data and replacing it with single-nucleus (SN) data 
using various transformation approaches. It also optionally trains/loads scVI models 
(conditional and non-conditional, with or without DEGs removed) to produce references that 
mimic the missing cell type as if it were single-cell.

Steps Overview:
1. **Load Input Data**: Reads single-cell (SC) and single-nucleus (SN) AnnData objects, 
   splitting the SC data into a reference set and a pseudobulk set.
2. **Determine Differentially Expressed Genes (DEGs)**: 
   - If a saved DEGs file (`degs.json`) is found, it is loaded; otherwise, DEGs are 
     computed by creating small pseudobulks of SC and SN, then running a 
     parallelized differential expression analysis.
3. **For Each Cell Type**:
   - Remove the selected cell type from SC (and from SN except for that held-out group).
   - Generate these references:
     a) **rawSN**: SC plus raw SN cells of the missing type  
     b) **pcaSN**: PCA-based transform to align missing SN to SC  
     c) **degSN**: DEG-filtered SN  
     d) **degPCA_SN**: DEG-filtered then PCA transform  
     e) **scviSN**: scVI-based transform (conditional, all genes)  
     f) **scvi_LSshift_SN**: scVI-based local latent shift (non-conditional, all genes)  
     g) **degScviSN**: DEG-filtered scVI-based transform (conditional)  
     h) **degScviLSshift_SN**: DEG-filtered scVI-based local latent shift (non-conditional)  
   - Each reference is saved in formats compatible with BayesPrism.
   - If scVI models for any transform do not exist, they are trained and saved; 
     otherwise they are loaded from disk.
4. **Pseudobulk Generation**: 
   - Creates pseudobulks from the SC pseudobulk subset for downstream deconvolution, 
     storing both the pseudobulk expression and the known proportions.
5. **Final Controls**: Saves the full SC reference (`sc_raw`) and full SN reference (`sn_raw`) 
   for baseline deconvolution comparisons.

Usage:
    python holdout_deconvolution.py \
        --res_name=DATASET_NAME \
        --data_path=/path/to/data \
        --output_path=/path/to/save/results \
        --pseudobulks_props='{"realistic": 500, "random": 500}' \
        --num_cells=1000 \
        --noise=True \
        --deseq_alpha=0.01 \
        --deconvolution_method=bayesprism

Outputs:
    - Multiple CSV files named `ref_<celltype>_<transform>_signal.csv/cell_state.csv` 
      representing each holdout reference.
    - Trained scVI models in the output_path (if they did not exist).
    - `degs.json` storing DEGs if calculated.
    - `pseudobulks.csv` with synthetic mixtures, plus `proportions.csv` with ground truth 
      for each mixture.
    - `sc_raw` and `sn_raw` references for baseline usage.

"""

import json
import scanpy as sc
import pandas as pd
import numpy as np
import os
import torch 
import random
from sklearn.neighbors import NearestNeighbors
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
sys.path.insert(1, '../../../../')

from src.helpers import prepare_data, split_single_cell_data, pick_cells
from src.helpers import downsample_cells_by_type,  make_references, save_bayesprism_references
from src.helpers import make_pseudobulks, save_bayesprism_pseudobulks, save_cibersort, scvi_train_model
from src.deg_funct import create_fixed_pseudobulk, load_others_degs, run_deseq2_for_cell_type, load_or_calc_degs, load_gene_list
from src.deg_funct import differential_expression_analysis, remove_diff_genes, differential_expression_analysis_parallel
from src.transforms import transform_heldout_sn_to_mean_sc_VAE, transform_heldout_sn_to_mean_sc_local
from src.transforms import transform_heldout_sn_to_mean_sc, calculate_median_library_size, get_normalized_expression_from_latent

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
os.environ['PYTHONHASHSEED'] = str(SEED)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Holdout cell-type deconvolution experiment")
    parser.add_argument("--res_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save outputs")
    parser.add_argument("--degs_path", type=str, required=True, help="Path to degs for all datasets (will add dataset name)")
    parser.add_argument("--pseudobulks_props", type=str, required=True, help="Pseudobulk proportions in JSON format")
    parser.add_argument("--num_cells", type=int, required=True, help="Number of cells per pseudobulk")
    parser.add_argument("--noise", action="store_true",
    parser.add_argument("--noise", nargs='?', const=True, default=False, type=str2bool,
                        help="Add Gaussian noise to pseudobulks (can be used as a flag or with True/False)")
    parser.add_argument("--deseq_alpha", type=float, default=0.01, help="Alpha threshold for DEG analysis")
    parser.add_argument("--deconvolution_method", type=str, default="bayesprism", help="Deconvolution method")

    args = parser.parse_args()

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

    # List of all human datasets used in the study.
    ALL_DATASETS = ["PBMC", "MBC", "ADP"]
    others_degs = load_others_degs(args.res_name, args.degs_path)

    diff_genes = load_or_calc_degs(output_path=args.output_path, 
                                    adata_sc_ref=adata_sc_ref, 
                                    adata_sn_ref=adata_sn_ref, 
                                    deseq_alpha=args.deseq_alpha)

    # ───────────────────────────────────────────────────────────
    # All DEG/intersection/union CSVs live here
    genes_dir = os.path.join(os.getcwd(), "../data")
    # ───────────────────────────────────────────────────────────

    if args.res_name == "MSB":
            pass
    else:
        # A)  full intersection (unchanged)
        all_intersect_genes = load_gene_list(os.path.join(genes_dir, "intersect_3ds.csv"))

        # B)  new long table:  gene | dataset | cell_type
        bydsct_df = pd.read_csv(
            os.path.join(genes_dir, "intersect_3ds_bydsct.csv"),
            dtype=str,
        )

        # C)  pre-compute  gene → set{(dataset, cell_type)}
        gene_to_pairs = (
            bydsct_df.groupby("gene")[["dataset", "cell_type"]]
                    .apply(lambda x: set(map(tuple, x.values)))   # {(ds,ct), … }
                    .to_dict()
        )

        for ct, df in diff_genes.items():
            diff_genes[ct].index = diff_genes[ct].index.astype(str)

    print(f"\n### Holdout experiment files... ###")

    # 4) Now our experimental: For each cell type => hold out in SC, replace with SN in different ways
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
        df_transformed_sn_ct = transform_heldout_sn_to_mean_sc_local(
            sc_data=adata_sc_except_ct.to_df(),
            sn_data=adata_sn_except_ct.to_df(),
            sn_heldout_data=adata_sn_ct.to_df(),
            variance_threshold=0.75, 
            sc_celltype_labels=adata_sc_except_ct.obs["cell_types"].astype(str),
            sn_celltype_labels=adata_sn_except_ct.obs["cell_types"].astype(str),
            heldout_label = cell_type,
        )

        # Convert the DataFrame to an AnnData object. This is the PCA SN transformed.
        adata_sn_transformed = sc.AnnData(X=df_transformed_sn_ct.values)
        adata_sn_transformed.var_names = df_transformed_sn_ct.columns 
        adata_sn_transformed.obs = adata_sn_ct.obs.copy()

        # Concatenate with SC (SC with held-out cell removed)
        sc_plus_sn_pca = sc.concat([adata_sc_except_ct, adata_sn_transformed], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_pca, args.output_path, f"ref_{cell_type.replace(' ', '_')}_pcaSN")

        ### 3) DEG-Filtered SN (this dataset's) ####
        ###########################
        #We are mimicking what happens in real life; not using the DEG from that cell type in questions (missing from SC)
            
        diff_genes_ct = {k: v for k, v in diff_genes.items() if k != cell_type}

        sc_filtered, sn_filtered = remove_diff_genes(
            sc_adata=adata_sc_except_ct, sn_adata=adata_sn_ct, diff_genes=diff_genes_ct
        )
        # Combine
        sc_plus_sn_deg = sc.concat([sc_filtered, sn_filtered], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_deg, args.output_path, f"ref_{cell_type.replace(' ', '_')}_degSN")

        ### 3) DEG-Filtered SN (other dataset's) ####
        ###########################
        if args.res_name == "MSB":
            pass
        else:
            sc_filtered_o, sn_filtered_o = remove_diff_genes(
                sc_adata=adata_sc_except_ct, sn_adata=adata_sn_ct, diff_genes=others_degs
            )
            # Combine
            sc_plus_sn_deg_o = sc.concat([sc_filtered_o, sn_filtered_o], axis=0, merge="same")
            save_bayesprism_references(sc_plus_sn_deg_o, args.output_path, f"ref_{cell_type.replace(' ', '_')}_degOtherSN")

        ### 5) DEG-Filtered SN (intersect of 3 datsets) ####
        ######################################################
        if args.res_name == "MSB":
            pass
        else:
            # genes whose ONLY appearance is exactly (dataset name, cell_type)
            skip_genes = {
                g for g, pairs in gene_to_pairs.items()
                if pairs == {(args.res_name, cell_type)}
            }

            # everything else stays on the “remove” list
            genes_to_remove = [g for g in all_intersect_genes if g not in skip_genes]

            # wrap exactly as remove_diff_genes() expects
            intersect_degs = {"all": pd.DataFrame(index=genes_to_remove)}

            sc_filtered_i, sn_filtered_i = remove_diff_genes(
                sc_adata=adata_sc_except_ct, sn_adata=adata_sn_ct, diff_genes=intersect_degs
            )
            # Combine
            sc_plus_sn_deg_i = sc.concat([sc_filtered_i, sn_filtered_i], axis=0, merge="same")
            save_bayesprism_references(sc_plus_sn_deg_i, args.output_path, f"ref_{cell_type.replace(' ', '_')}_degIntSN")

        ### And Random genes equal to this datasets DEGS ####
        ######################################################
        # Now let's combine all those gene names into a single set:
        all_diff_genes = set()
        for df in diff_genes_ct.values():
            all_diff_genes |= set(df.index)

        random_genes = np.random.choice(adata_sc_except_ct.var_names, size=len(all_diff_genes), replace=False)

        # Wrap in a DataFrame so remove_diff_genes(...) sees a dict with 'random' -> DataFrame
        random_df = pd.DataFrame(index=random_genes)       # an empty DataFrame with row index = gene names
        random_dict = {"random": random_df}

        sc_filtered_r, sn_filtered_r = remove_diff_genes(
            sc_adata=adata_sc_except_ct, 
            sn_adata=adata_sn_ct, 
            diff_genes=random_dict
        )       
        # Combine
        sc_plus_sn_deg_r = sc.concat([sc_filtered_r, sn_filtered_r], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_deg_r, args.output_path, f"ref_{cell_type.replace(' ', '_')}_degRandSN")
        
        ### 4) DEG-Filtered + PCA ####

        # First, do the same filter on the SC-except-ct + SN-except-ct (for PCA)
        sc_filtered_for_pca, sn_filtered_for_pca = remove_diff_genes(
            sc_adata=adata_sc_except_ct, sn_adata=adata_sn_except_ct, diff_genes=diff_genes_ct
        )

        # Then transform the *held-out SN* after filtering the same genes
        # => We'll subset adata_sn_ct to that gene set as well
        common_genes_pca = sn_filtered_for_pca.var_names.intersection(sc_filtered_for_pca.var_names)
        adata_sn_ct_filtered = adata_sn_ct[:, common_genes_pca].copy()

        df_sn_transformed_deg = transform_heldout_sn_to_mean_sc_local(
            sc_data=sc_filtered_for_pca.to_df(),
            sn_data=sn_filtered_for_pca.to_df(),
            sn_heldout_data=adata_sn_ct_filtered.to_df(),
            variance_threshold=0.75,
            sc_celltype_labels=sc_filtered_for_pca.obs.cell_types.astype(str),
            sn_celltype_labels=sn_filtered_for_pca.obs.cell_types.astype(str),
            heldout_label=cell_type
        )

        # Convert the DataFrame to an AnnData object.
        adata_sn_transformed_deg = sc.AnnData(X=df_sn_transformed_deg.values)
        adata_sn_transformed_deg.var_names = df_sn_transformed_deg.columns
        adata_sn_transformed_deg.obs = adata_sn_ct_filtered.obs.copy()

        # Merge the final references
        sc_plus_sn_deg_pca = sc.concat([sc_filtered_for_pca, adata_sn_transformed_deg], axis=0, merge="same")
        save_bayesprism_references(sc_plus_sn_deg_pca, args.output_path, f"ref_{cell_type.replace(' ', '_')}_degPCA_SN")

        ### 5) VAE Transform (conditional) #### 

        # Create "training" AnnData: SC minus this cell type + SN minus this cell type ("testing" data is our missing ct)
        adata_train = sc.concat([adata_sc_except_ct, adata_sn_except_ct], axis=0)
        adata_train.obs_names_make_unique()

        # scVI setup
        # For the conditional model
        adata_train_cond = sc.concat([adata_sc_except_ct, adata_sn_except_ct], axis=0)
        adata_train_cond.obs_names_make_unique()
        scvi.model.SCVI.setup_anndata(adata_train_cond, batch_key="data_type")
        # Quick checks
        print(adata_train_cond.obs["data_type"].unique())
        print(adata_train_cond.obs["data_type"].dtype)

        # Define the model save path
        model_save_path = f"{args.output_path}/scvi_trained_model_NO{cell_type.replace(' ', '_')}"

        # Check if a trained model exists, training otherwise
        if os.path.exists(model_save_path):
            print(f"Loading pre-trained scVI model from {model_save_path}...")
            model = scvi.model.SCVI.load(model_save_path, adata=adata_train_cond)
            model.view_anndata_setup()
        else:
            # Training
            model = scvi.model.SCVI(adata_train_cond, encode_covariates=True, 
                                                 deeply_inject_covariates=True,
                                                 n_layers=2, 
                                                 n_latent=30, dispersion='gene-batch',
                                                 gene_likelihood="nb")
            model.view_anndata_setup()
            model.train(early_stopping=True, early_stopping_patience=10)

        # Summation across genes => returns total counts per cell
        median_sc_lib = calculate_median_library_size(adata_sc_except_ct)

        # 3) scVI call with library_size
        transformed_expr = model.get_normalized_expression(
            adata=adata_sn_ct,
            transform_batch="single_cell",
            library_size=median_sc_lib,  # python float
            return_numpy=True
        )

        #  build an AnnData 
        adata_sn_transformed_scvi = sc.AnnData(X=transformed_expr)
        adata_sn_transformed_scvi.var_names = adata_sn_ct.var_names
        adata_sn_transformed_scvi.obs = adata_sn_ct.obs.copy()

        # Merge with SC minus the held-out cell -> final reference
        sc_plus_sn_scvi = sc.concat([adata_sc_except_ct, adata_sn_transformed_scvi], axis=0, merge="same")

        # Save for deconvolution, reference and trained model.
        model.save(f"{args.output_path}/scvi_trained_model_NO{cell_type.replace(' ', '_')}", overwrite=True)
        out_prefix = f"ref_{cell_type.replace(' ', '_')}_scviSN"
        save_bayesprism_references(sc_plus_sn_scvi, args.output_path, out_prefix)

        ### Latent space VAE Transform (not conditional) ####
        ## 6: VAE latent space arithmetic transform :
        del model

        # Create "training" AnnData: SC minus this cell type + SN minus this cell type ("testing" data is our missing ct)
        # For the non-conditional model
        adata_train_notcond = sc.concat([adata_sc_except_ct, adata_sn_except_ct], axis=0)

        for key in ["_scvi_batch", "_scvi_labels", "_scvi_size_factor", "_scvi_local_l_mean", "_scvi_local_l_var"]:
            if key in adata_train_notcond.obs.columns:
                adata_train_notcond.obs.pop(key)
            if key in adata_train_notcond.uns:
                adata_train_notcond.uns.pop(key)
        for key in list(adata_train_notcond.obsm.keys()):
            if "_scvi" in key:
                del adata_train_notcond.obsm[key]
        # Remove the batch column if it exists
        if "data_type" in adata_train_notcond.obs.columns:
            adata_train_notcond.obs.drop("data_type", axis=1, inplace=True)
            
        adata_train_notcond.obs_names_make_unique()
        scvi.model.SCVI.setup_anndata(adata_train_notcond, batch_key=None)

        # Define the model save path
        model_save_path = f"{args.output_path}/scvi_notcond_trained_model_NO{cell_type.replace(' ', '_')}"

        # Check if a trained model exists, training otherwise
        if os.path.exists(model_save_path):
            print(f"Loading pre-trained not conditional scVI model from {model_save_path}...")
            model = scvi.model.SCVI.load(model_save_path, adata=adata_train_notcond)
            model.view_anndata_setup()
        else:
            # Training
            model = scvi.model.SCVI(adata_train_notcond, 
                                    n_layers=2, 
                                    n_latent=30, 
                                    gene_likelihood="nb")
            model.view_anndata_setup()
            model.train(early_stopping=True, early_stopping_patience=10)

        transformed_expr =  transform_heldout_sn_to_mean_sc_VAE(model=model,
                                                            sc_adata = adata_sc_except_ct,
                                                            sn_adata = adata_sn_except_ct,
                                                            sn_heldout_adata = adata_sn_ct,
                                                            heldout_label=cell_type)

        # build an AnnData 
        adata_sn_transformed_scvi_notcond = sc.AnnData(X=transformed_expr)
        adata_sn_transformed_scvi_notcond.var_names = adata_sn_ct.var_names
        adata_sn_transformed_scvi_notcond.obs = adata_sn_ct.obs.copy()

        # Merge with SC minus the held-out cell -> final reference
        sc_plus_sn_scvi = sc.concat([adata_sc_except_ct, adata_sn_transformed_scvi_notcond], axis=0, merge="same")

        # Save for deconvolution, reference and trained model.
        model.save(f"{args.output_path}/scvi_notcond_trained_model_NO{cell_type.replace(' ', '_')}", overwrite=True)
        out_prefix = f"ref_{cell_type.replace(' ', '_')}_scvi_LSshift_SN"
        save_bayesprism_references(sc_plus_sn_scvi, args.output_path, out_prefix)
            
        ############################################################################
        ### (G) (-)DEG + SCVI transform (conditional) => ref_{ct}_degScviSN
        ############################################################################

        # 1) Filter SC minus hold-out + SN minus hold-out
        sc_filtered_scvi, sn_filtered_scvi = remove_diff_genes(
            sc_adata=adata_sc_except_ct,
            sn_adata=adata_sn_except_ct,
            diff_genes=diff_genes_ct
        )

        # 2) Subset the held-out SN cells to the same gene set
        common_genes_scvi = sc_filtered_scvi.var_names.intersection(sn_filtered_scvi.var_names)
        adata_sn_ct_filtered_scvi = adata_sn_ct[:, common_genes_scvi].copy()

        # 3) Create the AnnData that the model expects (SC + SN, DEGs removed)
        adata_train_deg_scvi = sc.concat([sc_filtered_scvi, sn_filtered_scvi], axis=0)
        adata_train_deg_scvi.obs_names_make_unique()

        scvi.model.SCVI.setup_anndata(
            adata_train_deg_scvi, batch_key="data_type"
        )

        # 4) Load the conditional scVI (DEG-filtered) model
        deg_cond_model_path = f"{args.output_path}/scvi_NODEG_trained_model_NO{cell_type.replace(' ', '_')}"
        if os.path.exists(deg_cond_model_path):
            print(f"Loading deg conditional scVI model from: {deg_cond_model_path}")
            model_deg_cond = scvi.model.SCVI.load(deg_cond_model_path, adata=adata_train_deg_scvi)
            
        else:
            # Training
            model_deg_cond = scvi.model.SCVI(adata_train_deg_scvi, encode_covariates=True, 
                                                 deeply_inject_covariates=True,
                                                 n_layers=2, 
                                                 n_latent=30, dispersion='gene-batch',
                                                 gene_likelihood="nb")
            model_deg_cond.view_anndata_setup()
            model_deg_cond.train(early_stopping=True, early_stopping_patience=10)

        # 5) Compute library size from SC minus holdout (DEG-filtered)
        median_sc_lib_deg = calculate_median_library_size(adata_sc_except_ct)

        # 6) Transform the held-out SN data
        transformed_expr_deg = model_deg_cond.get_normalized_expression(
            adata=adata_sn_ct_filtered_scvi,
            transform_batch="single_cell",
            library_size=median_sc_lib_deg,
            return_numpy=True
        )

        # 7) Build an AnnData for the transformed held-out SN
        adata_sn_deg_cond_transform = sc.AnnData(X=transformed_expr_deg)
        adata_sn_deg_cond_transform.var_names = adata_sn_ct_filtered_scvi.var_names
        adata_sn_deg_cond_transform.obs = adata_sn_ct_filtered_scvi.obs.copy()

        # 8) Concatenate with the SC (DEG-filtered)
        sc_plus_sn_deg_cond_scvi = sc.concat([sc_filtered_scvi, adata_sn_deg_cond_transform], axis=0, merge="same")

        # 9) Save the reference
        out_prefix_deg_cond = f"ref_{cell_type.replace(' ', '_')}_degScviSN"
        save_bayesprism_references(sc_plus_sn_deg_cond_scvi, args.output_path, out_prefix_deg_cond)
        model_deg_cond.save(deg_cond_model_path, overwrite=True)

        ############################################################################
        ### (H) (-)DEG + SCVI latent neighbor transform (non-conditional) => ref_{ct}_degScviLSshift_SN
        ############################################################################

        # Reuse sc_filtered_scvi, sn_filtered_scvi, and adata_sn_ct_filtered_scvi from above

        # 1) Create the AnnData used for the non-conditional scVI model (DEG-filtered)
        adata_train_deg_notcond = sc.concat([sc_filtered_scvi, sn_filtered_scvi], axis=0)
        adata_train_deg_notcond.obs_names_make_unique()

        scvi.model.SCVI.setup_anndata(adata_train_deg_notcond, batch_key=None)

        # 2) Load the non-conditional scVI (DEG-filtered) model
        deg_notcond_model_path = f"{args.output_path}/scvi_NODEG_notcond_trained_model_NO{cell_type.replace(' ', '_')}"
        # if os.path.exists(deg_notcond_model_path):
        #     print(f"Loading deg non-conditional scVI model from: {deg_notcond_model_path}")
        #     model_deg_notcond = scvi.model.SCVI.load(deg_notcond_model_path, adata=adata_train_deg_notcond)
        # else:
        # Training
        model_deg_notcond = scvi.model.SCVI(adata_train_deg_notcond, 
                                                n_layers=2, 
                                                n_latent=30, 
                                                gene_likelihood="nb")
        model_deg_notcond.view_anndata_setup()
        model_deg_notcond.train(early_stopping=True, early_stopping_patience=10)

        # 3) Apply the local latent shift function
        #    (We assume you have transform_heldout_sn_to_mean_sc_VAE(...) defined in your script)
        df_transformed_deg_notcond = transform_heldout_sn_to_mean_sc_VAE(
            model=model_deg_notcond,
            sc_adata=sc_filtered_scvi,
            sn_adata=sn_filtered_scvi,
            sn_heldout_adata=adata_sn_ct_filtered_scvi,
            heldout_label=cell_type,
            k_neighbors=10  # or whatever you set
        )

        # 4) Build the final AnnData
        adata_sn_deg_notcond_LS = sc.AnnData(X=df_transformed_deg_notcond.values)
        adata_sn_deg_notcond_LS.var_names = adata_sn_ct_filtered_scvi.var_names
        adata_sn_deg_notcond_LS.obs = adata_sn_ct_filtered_scvi.obs.copy()

        # 5) Concatenate with SC (DEG-filtered)
        sc_plus_sn_deg_notcond_LS = sc.concat([sc_filtered_scvi, adata_sn_deg_notcond_LS], axis=0, merge="same")

        # 6) Save the reference
        out_prefix_deg_notcond = f"ref_{cell_type.replace(' ', '_')}_degScviLSshift_SN"
        save_bayesprism_references(sc_plus_sn_deg_notcond_LS, args.output_path, out_prefix_deg_notcond)
        model_deg_notcond.save(deg_notcond_model_path, overwrite=True)

        '''
        
        Add your own transformation here!

        '''

    ###DEG-Filtered SN (all cells are SN!)####
    ###########################
    if args.res_name == "MSB":
        pass
    else:
        # everything else stays on the “remove” list
        genes_to_remove = [g for g in all_intersect_genes if g not in {}]

        # wrap exactly as remove_diff_genes() expects
        intersect_degs = {"all": pd.DataFrame(index=genes_to_remove)}

        _, sn_filtered_i = remove_diff_genes(
            sc_adata=adata_sn_ct, sn_adata=adata_sn_ct, diff_genes=intersect_degs
        )
    save_bayesprism_references(sn_filtered_i, args.output_path, "degIntAllSN")

    print("Making pseudobulks...")
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
