#!/usr/bin/env python
"""
Per-Donor Adipocyte Reference Generation

For each SN donor (batch), sample 95 adipocytes and pool neutrophils across all donors,
then generate multiple BayesPrism references:
  - rawSN
  - pcaSN
  - degSN
  - degOtherSN
  - degIntSN
  - degRandSN
  - degPCA_SN
  - scviSN (conditional)
  - scvi_LSshift_SN (non-conditional)
  - degScviSN
  - degScviLSshift_SN

DEGs and scVI models are recomputed per donor so that models are tailored to each donor’s
technical profile. Outputs are prefixed with the donor ID.

Usage:
    python per_donor_adipo_refs.py \
      --res_name Real_ADP \
      --degs_path /path/to/degs_root \
      --output_path /path/to/save/refs \
      --deseq_alpha 0.01
"""
import os, sys, json, csv
import random
import numpy as np
import scanpy as sc
import pandas as pd
import scvi
from sklearn.decomposition import PCA

sys.path.insert(1, "../../")
sys.path.insert(1, "../")
from src.helpers import (
    prepare_data,
    split_single_cell_data,
    pick_cells,
    open_adipose_datasets_all,
)
from src.helpers import (
    downsample_cells_by_type,
    make_references,
    save_bayesprism_references,
)
from src.helpers import (
    make_pseudobulks,
    save_bayesprism_pseudobulks,
    save_cibersort,
    scvi_train_model,
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
from src.transforms import (
    transform_heldout_sn_to_mean_sc_VAE,
    transform_heldout_sn_to_mean_sc_local,
)
from src.transforms import (
    transform_heldout_sn_to_mean_sc,
    calculate_median_library_size,
    get_normalized_expression_from_latent,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
st = 1
if torch := __import__("torch"):  # set torch seed if available
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--res_name", required=True)
    parser.add_argument(
        "--degs_path",
        required=True,
        help="Root folder containing per-dataset degs.json files",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--deseq_alpha", type=float, default=0.01)
    parser.add_argument(
        "--deconvolution_method",
        type=str,
        default="bayesprism",
        help="Deconvolution method",
    )

    args = parser.parse_args()

    # Prepare directories
    os.makedirs(args.output_path, exist_ok=True)

    # 1) Load adipose data
    bulk_df, adata_sc, adata_sn_all, sn_missing_list = open_adipose_datasets_all(
        res_name=args.res_name, base_dir=".."
    )
    # sn_missing is one AnnData with both fat cells and neutrophils.
    adata_adipo_all = sn_missing_list[0]
    adata_neut_all = sn_missing_list[1]
    adata_neut = adata_neut_all.copy()

    # Precompute median SC library size
    median_sc_lib = calculate_median_library_size(adata_sc)

    # Datasets are huge, we downsample.
    adata_sc = downsample_cells_by_type(adata_sc, max_cells=1500)

    # Pre-load DEG sets from other datasets
    ALL_DATASETS = ["PBMC", "MBC", "ADP"]
    others_degs = load_others_degs(args.res_name, args.degs_path)

    # Determine donors (batches) in adipocyte SN
    donors = adata_adipo_all.obs["batch"].unique().tolist()

    # Saving for later use
    donors_path = f"{args.output_path}/donors.csv"

    with open(donors_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(donors)

    for donor in donors:

        print(f"\n--- Processing donor: {donor} ---")

        # Subset donor adipocytes
        adata_adipo = adata_adipo_all[adata_adipo_all.obs["batch"] == donor].copy()
        n_avail = adata_adipo.n_obs
        if n_avail < 95:
            raise ValueError(f"Not enough adipocytes in donor {donor}: {n_avail}")
        # Randomly sample 95 adipocytes
        sc.pp.subsample(adata_adipo, n_obs=95, random_state=SEED)

        adata_sn = adata_sn_all[adata_sn_all.obs["batch"] == donor].copy()

        # 2) Compute donor-specific DEGs
        diff_genes = load_or_calc_degs(
            output_path=args.output_path,
            adata_sc_ref=adata_sc,
            adata_sn_ref=adata_sn,
            deseq_alpha=args.deseq_alpha,
            patient_id=donor,
        )

        # Build SN reference for this donor: pooled neutrophils + sampled adipocytes
        sn_missing = sc.concat([adata_neut, adata_adipo], axis=0, merge="same")

        # 3) Generate each transform
        # ---- rawSN ----
        ref_raw = sc.concat([adata_sc, sn_missing], axis=0, merge="same")
        save_bayesprism_references(ref_raw, args.output_path, f"ref_{donor}_rawSN")

        # ---- PCA SN ----
        df_transformed_sn_ct = transform_heldout_sn_to_mean_sc_local(
            sc_data=adata_sc.to_df(),
            sn_data=adata_sn.to_df(),
            sn_heldout_data=sn_missing.to_df(),
            variance_threshold=0.75,
            sc_celltype_labels=adata_sc.obs["cell_types"].astype(str),
            sn_celltype_labels=adata_sn.obs["cell_types"].astype(str),
        )

        # Convert the DataFrame to an AnnData object. This is the PCA SN transformed.
        adata_sn_transformed = sc.AnnData(X=df_transformed_sn_ct.values)
        adata_sn_transformed.var_names = df_transformed_sn_ct.columns
        adata_sn_transformed.obs = sn_missing.obs.copy()

        # Concatenate both with SC
        sc_plus_sn_pca = sc.concat(
            [adata_sc, adata_sn_transformed], axis=0, merge="same"
        )
        save_bayesprism_references(
            sc_plus_sn_pca, args.output_path, f"ref_{donor}_pcaSN"
        )

        # ---- degSN ----
        sc_filtered, sn_filtered = remove_diff_genes(
            sc_adata=adata_sc, sn_adata=sn_missing, diff_genes=diff_genes
        )
        # Combine
        sc_plus_sn_deg = sc.concat([sc_filtered, sn_filtered], axis=0, merge="same")
        save_bayesprism_references(
            sc_plus_sn_deg, args.output_path, f"ref_{donor}_degSN"
        )

        # ---- degOtherSN ----
        sc_filtered_o, sn_filtered_o = remove_diff_genes(
            sc_adata=adata_sc, sn_adata=sn_missing, diff_genes=others_degs
        )
        # Combine
        sc_plus_sn_deg_o = sc.concat(
            [sc_filtered_o, sn_filtered_o], axis=0, merge="same"
        )
        save_bayesprism_references(
            sc_plus_sn_deg_o, args.output_path, f"ref_{donor}_degOtherSN"
        )

        # ---- degIntSN ----
        # assume intersect_3ds.csv available
        if os.path.exists(os.path.join(os.getcwd(), "../data/intersect_3ds.csv")):
            intersect_degs = pd.read_csv(
                os.path.join(os.getcwd(), "../data/intersect_3ds.csv"), index_col=0
            )
            print("this are the DEGs intersection:", intersect_degs)
            intersect_degs = {"all": pd.DataFrame(index=intersect_degs.values)}
        else:
            raise FileExistsError(
                "intersect_3ds.csv doesn't exist. Run notebook: notebooks/differential_gene_expression.ipynb"
            )
        sc_filtered_i, sn_filtered_i = remove_diff_genes(
            sc_adata=adata_sc, sn_adata=sn_missing, diff_genes=intersect_degs
        )

        sc_plus_sn_deg_i = sc.concat(
            [sc_filtered_i, sn_filtered_i], axis=0, merge="same"
        )
        save_bayesprism_references(
            sc_plus_sn_deg_i, args.output_path, f"ref_{donor}_degIntSN"
        )

        # ---- degRandSN ----
        # gather the union of DEGs across all cell types
        all_diff_genes = set()
        for df in diff_genes.values():
            all_diff_genes |= set(df.index)

        random_genes = np.random.choice(
            adata_sc.var_names, size=len(all_diff_genes), replace=False
        )

        # Wrap in a DataFrame so remove_diff_genes(...) sees a dict with 'random' -> DataFrame
        random_df = pd.DataFrame(
            index=random_genes
        )  # an empty DataFrame with row index = gene names
        random_dict = {"random": random_df}

        sc_filtered_r, sn_filtered_r = remove_diff_genes(
            sc_adata=adata_sc, sn_adata=sn_missing, diff_genes=random_dict
        )
        sc_plus_sn_deg_r = sc.concat(
            [sc_filtered_r, sn_filtered_r], axis=0, merge="same"
        )
        print("shape of combo anndata AFTER filtering: ", sc_plus_sn_deg_r.shape)

        save_bayesprism_references(
            sc_plus_sn_deg_r, args.output_path, f"ref_{donor}_degRandSN"
        )

        # ---- degPCA_SN ----
        sc_filtered_for_pca, sn_missing_filtered_for_pca = remove_diff_genes(
            sc_adata=adata_sc, sn_adata=sn_missing, diff_genes=diff_genes
        )
        _, sn_filtered_for_pca = remove_diff_genes(
            sc_adata=adata_sc, sn_adata=adata_sn, diff_genes=diff_genes
        )
        df_deg_transformed_sn_ct = transform_heldout_sn_to_mean_sc_local(
            sc_data=sc_filtered_for_pca.to_df(),
            sn_data=sn_filtered_for_pca.to_df(),
            sn_heldout_data=sn_missing_filtered_for_pca.to_df(),
            variance_threshold=0.75,
            sc_celltype_labels=adata_sc.obs["cell_types"].astype(str),
            sn_celltype_labels=adata_sn.obs["cell_types"].astype(str),
        )

        # Convert the DataFrame to an AnnData object. This is the PCA SN transformed.
        adata_deg_sn_transformed = sc.AnnData(X=df_deg_transformed_sn_ct.values)
        adata_deg_sn_transformed.var_names = df_deg_transformed_sn_ct.columns
        adata_deg_sn_transformed.obs = sn_missing.obs.copy()

        # Concatenate both with SC
        sc_deg_plus_sn_pca = sc.concat(
            [sc_filtered_for_pca, adata_deg_sn_transformed], axis=0, merge="same"
        )
        save_bayesprism_references(
            sc_deg_plus_sn_pca, args.output_path, f"ref_{donor}_degPCA_SN"
        )

        # ---- scviSN (conditional) ----

        # scVI setup
        # For the conditional model
        adata_sc_fortrain = downsample_cells_by_type(adata_sc, max_cells=200)
        adata_train = sc.concat([adata_sc_fortrain, adata_sn], axis=0)
        adata_train_cond = adata_train.copy()
        adata_train_cond.obs_names_make_unique()
        scvi.model.SCVI.setup_anndata(adata_train_cond, batch_key="data_type")

        # Quick checks
        print(adata_train_cond.obs["data_type"].unique())
        print(adata_train_cond.obs["data_type"].dtype)

        # Define the model save path
        model_save_path = f"{args.output_path}/scvi_trained_model_{donor}"

        # Check if a trained model exists, training otherwise
        if os.path.exists(model_save_path):
            print(f"Loading pre-trained scVI model from {model_save_path}...")
            model = scvi.model.SCVI.load(model_save_path, adata=adata_train_cond)
            model.view_anndata_setup()
        else:
            # Training
            model = scvi.model.SCVI(
                adata_train_cond,
                encode_covariates=True,
                deeply_inject_covariates=True,
                n_layers=2,
                n_latent=30,
                dispersion="gene-batch",
                gene_likelihood="nb",
            )
            model.view_anndata_setup()
            model.train(early_stopping=True, early_stopping_patience=10)

        # 3) scVI call with library_size
        transformed_expr = model.get_normalized_expression(
            adata=sn_missing,
            transform_batch="single_cell",
            library_size=median_sc_lib,  # python float
            return_numpy=True,
        )

        #  build an AnnData
        adata_sn_transformed_scvi = sc.AnnData(X=transformed_expr)
        adata_sn_transformed_scvi.var_names = sn_missing.var_names
        adata_sn_transformed_scvi.obs = sn_missing.obs.copy()

        # Merge with SC minus the held-out cell -> final reference
        sc_plus_sn_scvi = sc.concat(
            [adata_sc, adata_sn_transformed_scvi], axis=0, merge="same"
        )
        # Save for deconvolution, reference and trained model.
        model.save(f"{args.output_path}/scvi_trained_model_{donor}", overwrite=True)
        save_bayesprism_references(
            sc_plus_sn_scvi, args.output_path, f"ref_{donor}_scviSN"
        )

        # ---- scvi_LSshift_SN (non-conditional) ----
        # clean up any scvi artifacts
        del model
        del adata_train_cond
        # Define the model save path
        model_save_path = f"{args.output_path}/scvi_notcond_trained_model_{donor}"
        adata_train_notcond = adata_train.copy()

        scvi.model.SCVI.setup_anndata(adata_train_notcond, batch_key=None)

        # Check if a trained model exists, training otherwise
        if os.path.exists(model_save_path):
            print(
                f"Loading pre-trained not conditional scVI model from {model_save_path}..."
            )
            model = scvi.model.SCVI.load(model_save_path, adata=adata_train_notcond)
            model.view_anndata_setup()
        else:
            # Training
            model = scvi.model.SCVI(
                adata_train_notcond, n_layers=2, n_latent=30, gene_likelihood="nb"
            )
            model.view_anndata_setup()
            model.train(early_stopping=True, early_stopping_patience=10)

        transformed_expr = transform_heldout_sn_to_mean_sc_VAE(
            model=model,
            sc_adata=adata_sc,
            sn_adata=adata_sn,
            sn_heldout_adata=sn_missing,
        )
        # build an AnnData
        adata_sn_transformed_scvi_notcond = sc.AnnData(X=transformed_expr)
        adata_sn_transformed_scvi_notcond.var_names = sn_missing.var_names
        adata_sn_transformed_scvi_notcond.obs = sn_missing.obs.copy()

        # Merge with SC minus the held-out cell -> final reference
        sc_plus_sn_scvi = sc.concat(
            [adata_sc, adata_sn_transformed_scvi_notcond], axis=0, merge="same"
        )

        # Save for deconvolution, reference and trained model.
        model.save(
            f"{args.output_path}/scvi_notcond_trained_model_{donor}", overwrite=True
        )
        save_bayesprism_references(
            sc_plus_sn_scvi, args.output_path, f"ref_{donor}_scvi_LSshift_SN"
        )

        # ---- DEG scvi (conditional) ----

        # 1) Filter SC minus hold-out + SN minus hold-out
        sc_filtered_scvi, sn_filtered_scvi = remove_diff_genes(
            sc_adata=adata_sc_fortrain, sn_adata=adata_sn, diff_genes=diff_genes
        )

        # 2) Subset the held-out SN cells to the same gene set
        common_genes_scvi = sc_filtered_scvi.var_names.intersection(
            sn_filtered_scvi.var_names
        )
        sn_missing_filtered_scvi = sn_missing[:, common_genes_scvi].copy()

        # 3) Create the AnnData that the model expects (SC + SN, DEGs removed)
        adata_train_deg_scvi = sc.concat([sc_filtered_scvi, sn_filtered_scvi], axis=0)
        adata_train_deg_scvi.obs_names_make_unique()

        scvi.model.SCVI.setup_anndata(adata_train_deg_scvi, batch_key="data_type")

        # 4) Load the conditional scVI (DEG-filtered) model
        deg_cond_model_path = f"{args.output_path}/scvi_NODEG_trained_model_{donor}"
        if os.path.exists(deg_cond_model_path):
            print(f"Loading deg conditional scVI model from: {deg_cond_model_path}")
            model_deg_cond = scvi.model.SCVI.load(
                deg_cond_model_path, adata=adata_train_deg_scvi
            )
        else:
            # Training
            model_deg_cond = scvi.model.SCVI(
                adata_train_deg_scvi,
                encode_covariates=True,
                deeply_inject_covariates=True,
                n_layers=2,
                n_latent=30,
                dispersion="gene-batch",
                gene_likelihood="nb",
            )
            model_deg_cond.view_anndata_setup()
            model_deg_cond.train(early_stopping=True, early_stopping_patience=10)

        # 5) Transform the held-out SN data
        transformed_expr_deg = model_deg_cond.get_normalized_expression(
            adata=sn_missing_filtered_scvi,
            transform_batch="single_cell",
            library_size=median_sc_lib,
            return_numpy=True,
        )

        # 6) Build an AnnData for the transformed held-out SN
        adata_sn_deg_cond_transform = sc.AnnData(X=transformed_expr_deg)
        adata_sn_deg_cond_transform.var_names = sn_missing_filtered_scvi.var_names
        adata_sn_deg_cond_transform.obs = sn_missing_filtered_scvi.obs.copy()

        # 7) Concatenate with the SC (DEG-filtered)
        sc_plus_sn_deg_cond_scvi = sc.concat(
            [sc_filtered_scvi, adata_sn_deg_cond_transform], axis=0, merge="same"
        )

        # 8) Save the reference
        save_bayesprism_references(
            sc_plus_sn_deg_cond_scvi, args.output_path, f"ref_{donor}_degScviSN"
        )
        model_deg_cond.save(deg_cond_model_path, overwrite=True)

        # ---- DEG scvi LS (not conditional) ----

        # Reuse sc_filtered_scvi, sn_filtered_scvi, and sn_missing_filtered_scvi from above

        # 1) Create the AnnData used for the non-conditional scVI model (DEG-filtered)
        adata_train_deg_notcond = sc.concat(
            [sc_filtered_scvi, sn_filtered_scvi], axis=0
        )
        adata_train_deg_notcond.obs_names_make_unique()

        scvi.model.SCVI.setup_anndata(adata_train_deg_notcond, batch_key=None)

        # 2) Load the non-conditional scVI (DEG-filtered) model
        deg_notcond_model_path = (
            f"{args.output_path}/scvi_NODEG_notcond_trained_model_{donor}"
        )
        if os.path.exists(deg_notcond_model_path):
            print(
                f"Loading deg non-conditional scVI model from: {deg_notcond_model_path}"
            )
            model_deg_notcond = scvi.model.SCVI.load(
                deg_notcond_model_path, adata=adata_train_deg_notcond
            )
        else:
            # Training
            model_deg_notcond = scvi.model.SCVI(
                adata_train_deg_scvi, n_layers=2, n_latent=30, gene_likelihood="nb"
            )
            model_deg_notcond.view_anndata_setup()
            model_deg_notcond.train(early_stopping=True, early_stopping_patience=10)

        # 3) Apply the local latent shift function
        df_transformed_deg_notcond = transform_heldout_sn_to_mean_sc_VAE(
            model=model_deg_notcond,
            sc_adata=sc_filtered_scvi,
            sn_adata=sn_filtered_scvi,
            sn_heldout_adata=sn_missing_filtered_scvi,
            k_neighbors=10,
        )

        # 4) Build the final AnnData
        adata_sn_deg_notcond_LS = sc.AnnData(X=df_transformed_deg_notcond.values)
        adata_sn_deg_notcond_LS.var_names = sn_missing_filtered_scvi.var_names
        adata_sn_deg_notcond_LS.obs = sn_missing_filtered_scvi.obs.copy()

        # 5) Concatenate with SC (DEG-filtered)
        sc_plus_sn_deg_notcond_LS = sc.concat(
            [sc_filtered_scvi, adata_sn_deg_notcond_LS], axis=0, merge="same"
        )

        # 6) Save the reference
        save_bayesprism_references(
            sc_plus_sn_deg_notcond_LS,
            args.output_path,
            f"ref_{donor}_degScviLSshift_SN",
        )
        model_deg_notcond.save(deg_notcond_model_path, overwrite=True)

        ### DEG-Filtered SN (intersect of 3 datsets) all cells are SN ####
        ###########################

        intersect_degs = pd.read_csv(
            os.path.join(os.getcwd(), "../data/intersect_3ds.csv"), index_col=0
        )
        print("this are the DEGs intersection:", intersect_degs)
        intersect_degs = {"all": pd.DataFrame(index=intersect_degs.values)}

        _, full_sn_3d = remove_diff_genes(
            sc_adata=adata_sn, sn_adata=adata_sn, diff_genes=intersect_degs
        )

        print("Intersection gene filtering of all SN ataset:")
        print("shape of sc anndata before filtering: ", adata_sn.shape)
        print("shape of combo anndata AFTER filtering: ", full_sn_3d.shape)

        save_bayesprism_references(
            full_sn_3d, args.output_path, f"ref_{donor}_degIntAllSN"
        )

        print("Saving the bulks and controls SN...")
        os.makedirs(args.output_path, exist_ok=True)

        if args.deconvolution_method == "bayesprism":
            # **Save BayesPrism Files**
            save_bayesprism_references(adata_sn, args.output_path, f"sn_raw_{donor}")
        else:
            raise ValueError(
                'deconvolution_methods must be "bayesprism" (others not supported yet.)'
            )

    print("\nDone generating per-donor references.")


if __name__ == "__main__":
    main()
