"""
This script performs single-nucleus to single-cell transcriptomic mapping and pseudobulk generation 
for adipose tissue datasets using scVI, PCA, and DEG filtering strategies. It trains conditional and 
non-conditional scVI models (with and without differentially expressed genes), applies various 
transformation methods to held-out snRNA-seq data, integrates it with single-cell data, and generates 
realistic and random pseudobulk mixtures. The resulting pseudobulks are saved for downstream analysis 
and benchmarking of deconvolution methods.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import anndata
import scanpy as sc
import scvi
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(1, "../../")
sys.path.insert(1, "../")
from src.helpers import (
    prepare_data,
    split_single_cell_data,
    pick_cells
)
from src.helpers import (
    downsample_cells_by_type,
    make_references,
    save_bayesprism_references,
    open_adipose_datasets_all,
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

# -----------------------------
# PARAMETERS
# -----------------------------
res_name = "ADP"
DATASET_NAME = res_name
# Set base paths (relative to the notebook’s location)
base_dir = ".."  # adjust as needed!
pseudos_save_path = f"{base_dir}/data/{res_name}/"
num_rand = 100
num_realistic = 100
num_cells = 1000

bulk_df, sc_adata, sn_adata, sn_missing = open_adipose_datasets_all(
    res_name="ADP", base_dir=base_dir
)

## The datasets are huge. We'll limit cells per type for training models.
sc_adata_train = downsample_cells_by_type(sc_adata, max_cells=1500)
sn_adata_train = downsample_cells_by_type(sn_adata, max_cells=1500)

""" SCVI Models """

## SCVI All Genes

model_save_path = (
    f"{base_dir}/data/{res_name}/scvi_allgenes_cond_trained_model_bulkcomparison"
)
model_cond_allgenes = scvi_train_model(
    model_save_path=model_save_path,
    sc_adata=sc_adata_train,
    sn_adata=sn_adata_train,
    conditional=True,
    remove_degs=False,
)

model_save_path = (
    f"{base_dir}/data/{res_name}/scvi_allgenes_notcond_trained_model_bulkcomparison"
)
model_notcond_allgenes = scvi_train_model(
    model_save_path=model_save_path,
    sc_adata=sc_adata_train,
    sn_adata=sn_adata_train,
    conditional=False,
    remove_degs=False,
)

## Differential Gene Expression (if needed)
print("Check if differential gene expression calcs. are needed...")
genes_save_path = f"{pseudos_save_path}degs.json"
if os.path.exists(genes_save_path):
    try:
        with open(genes_save_path, "r") as file:
            diff_genes_json = json.load(file)
        # Convert the JSON representation back into DataFrames
        diff_genes = {}
        for key, value in diff_genes_json.items():
            # Check if this value looks like a DataFrame stored in 'split' orientation
            if isinstance(value, dict) and {"index", "columns", "data"}.issubset(
                value.keys()
            ):
                diff_genes[key] = pd.DataFrame(**value)
            else:
                diff_genes[key] = value
    except json.JSONDecodeError:
        raise ValueError(
            "Previous DEG analysis not found properly run. check your deg.json file!"
        )
else:
    raise ValueError(
        "Previous DEG analysis not found. Please run process_bulks_train_models.py before this notebook!"
    )

flattened_index = [idx for df in diff_genes.values() for idx in df.index]
flattened_index = list(flattened_index)
all_de_genes_list = list(dict.fromkeys(flattened_index))
print("Total unique DEGs:", len(all_de_genes_list))

## And removing
sc_adata_nodeg = sc_adata[:, ~sc_adata.var_names.isin(all_de_genes_list)].copy()
sn_adata_nodeg = sn_adata[:, ~sn_adata.var_names.isin(all_de_genes_list)].copy()

sn_missing_nodeg = {}
sn_missing_nodeg[0] = sn_missing[0][
    :, ~sn_missing[0].var_names.isin(all_de_genes_list)
].copy()
sn_missing_nodeg[1] = sn_missing[1][
    :, ~sn_missing[1].var_names.isin(all_de_genes_list)
].copy()

sc_adata_train_nodeg = sc_adata_train[
    :, ~sc_adata_train.var_names.isin(all_de_genes_list)
].copy()
sn_adata_train_nodeg = sn_adata_train[
    :, ~sn_adata_train.var_names.isin(all_de_genes_list)
].copy()

print(sc_adata.shape)
print(sc_adata_nodeg.shape)

print(sn_adata.shape)
print(sn_adata_nodeg.shape)

print(sn_adata_train.shape)
print(sn_adata_train_nodeg.shape)

print(sc_adata_train.shape)
print(sc_adata_train_nodeg.shape)

print(sn_missing[0].shape)
print(sn_missing_nodeg[0].shape)

## And the next 2 models without DEG:
model_save_path = (
    f"{base_dir}/data/{res_name}/scvi_nodeg_cond_trained_model_bulkcomparison"
)
model_cond_nodeg = scvi_train_model(
    model_save_path=model_save_path,
    sc_adata=sc_adata_train_nodeg,
    sn_adata=sn_adata_train_nodeg,
    conditional=True,
    remove_degs=True,
    degs=all_de_genes_list,
)

model_save_path = (
    f"{base_dir}/data/{res_name}/scvi_nodeg_notcond_trained_model_bulkcomparison"
)
model_notcond_nodeg = scvi_train_model(
    model_save_path=model_save_path,
    sc_adata=sc_adata_train_nodeg,
    sn_adata=sn_adata_train_nodeg,
    conditional=False,
    remove_degs=True,
    degs=all_de_genes_list,
)

## Now the transforms:

median_sc_lib = calculate_median_library_size(sc_adata)

# scVI call with library_size
transformed_expr_0 = model_cond_allgenes.get_normalized_expression(
    adata=sn_missing[0],
    transform_batch="single_cell",
    library_size=median_sc_lib,  # python float
    return_numpy=True,
)
#  build an AnnData
adata_sn_transformed_scvi_0 = sc.AnnData(X=transformed_expr_0)
adata_sn_transformed_scvi_0.var_names = sn_missing[0].var_names
adata_sn_transformed_scvi_0.obs = sn_missing[0].obs.copy()

######################################################################

transformed_expr_1 = model_cond_allgenes.get_normalized_expression(
    adata=sn_missing[1],
    transform_batch="single_cell",
    library_size=median_sc_lib,  # python float
    return_numpy=True,
)
#  build an AnnData
adata_sn_transformed_scvi_1 = sc.AnnData(X=transformed_expr_1)
adata_sn_transformed_scvi_1.var_names = sn_missing[1].var_names
adata_sn_transformed_scvi_1.obs = sn_missing[1].obs.copy()

######################################################################

# Merge with SC minus the held-out cell -> final reference
scvi_cond_adata = sc.concat(
    [sc_adata, adata_sn_transformed_scvi_0], axis=0, merge="same"
)
scvi_cond_adata = sc.concat(
    [scvi_cond_adata, adata_sn_transformed_scvi_1], axis=0, merge="same"
)

median_sc_lib = calculate_median_library_size(sc_adata_nodeg)

######################################################################

# scVI call with library_size
transformed_expr_0 = model_cond_nodeg.get_normalized_expression(
    adata=sn_missing_nodeg[0],
    transform_batch="single_cell",
    library_size=median_sc_lib,  # python float
    return_numpy=True,
)
#  build an AnnData
adata_sn_transformed_scvi_0 = sc.AnnData(X=transformed_expr_0)
adata_sn_transformed_scvi_0.var_names = sn_missing_nodeg[0].var_names
adata_sn_transformed_scvi_0.obs = sn_missing_nodeg[0].obs.copy()

######################################################################

transformed_expr_1 = model_cond_nodeg.get_normalized_expression(
    adata=sn_missing_nodeg[1],
    transform_batch="single_cell",
    library_size=median_sc_lib,  # python float
    return_numpy=True,
)
#  build an AnnData
adata_sn_transformed_scvi_1 = sc.AnnData(X=transformed_expr_1)
adata_sn_transformed_scvi_1.var_names = sn_missing_nodeg[1].var_names
adata_sn_transformed_scvi_1.obs = sn_missing_nodeg[1].obs.copy()

######################################################################

# Merge with SC minus the held-out cell -> final reference
scvi_cond_nodeg_adata = sc.concat(
    [sc_adata_nodeg, adata_sn_transformed_scvi_0], axis=0, merge="same"
)
scvi_cond_nodeg_adata = sc.concat(
    [scvi_cond_nodeg_adata, adata_sn_transformed_scvi_1], axis=0, merge="same"
)

## And Non-Conditional Models with Latent Space transform:

transformed_expr_0 = transform_heldout_sn_to_mean_sc_VAE(
    model=model_notcond_allgenes,
    sc_adata=sc_adata,
    sn_adata=sn_adata,
    sn_heldout_adata=sn_missing[0],
)

# build an AnnData
adata_sn_transformed_scvi_notcond_0 = sc.AnnData(X=transformed_expr_0)
adata_sn_transformed_scvi_notcond_0.var_names = sn_missing[0].var_names
adata_sn_transformed_scvi_notcond_0.obs = sn_missing[0].obs.copy()

######################################################################

transformed_expr_1 = transform_heldout_sn_to_mean_sc_VAE(
    model=model_notcond_allgenes,
    sc_adata=sc_adata,
    sn_adata=sn_adata,
    sn_heldout_adata=sn_missing[1],
)

# build an AnnData
adata_sn_transformed_scvi_notcond_1 = sc.AnnData(X=transformed_expr_1)
adata_sn_transformed_scvi_notcond_1.var_names = sn_missing[1].var_names
adata_sn_transformed_scvi_notcond_1.obs = sn_missing[1].obs.copy()

######################################################################

# Merge with SC -> final
scvi_ls_adata = sc.concat(
    [sc_adata, adata_sn_transformed_scvi_notcond_0], axis=0, merge="same"
)
scvi_ls_adata = sc.concat(
    [scvi_ls_adata, adata_sn_transformed_scvi_notcond_1], axis=0, merge="same"
)


transformed_expr_0 = transform_heldout_sn_to_mean_sc_VAE(
    model=model_notcond_nodeg,
    sc_adata=sc_adata_nodeg,
    sn_adata=sn_adata_nodeg,
    sn_heldout_adata=sn_missing_nodeg[0],
)

# build an AnnData
adata_sn_transformed_scvi_notcond_nodeg_0 = sc.AnnData(X=transformed_expr_0)
adata_sn_transformed_scvi_notcond_nodeg_0.var_names = sn_missing_nodeg[0].var_names
adata_sn_transformed_scvi_notcond_nodeg_0.obs = sn_missing_nodeg[0].obs.copy()

######################################################################

transformed_expr_1 = transform_heldout_sn_to_mean_sc_VAE(
    model=model_notcond_nodeg,
    sc_adata=sc_adata_nodeg,
    sn_adata=sn_adata_nodeg,
    sn_heldout_adata=sn_missing_nodeg[1],
)

# build an AnnData
adata_sn_transformed_scvi_notcond_nodeg_1 = sc.AnnData(X=transformed_expr_1)
adata_sn_transformed_scvi_notcond_nodeg_1.var_names = sn_missing_nodeg[1].var_names
adata_sn_transformed_scvi_notcond_nodeg_1.obs = sn_missing_nodeg[1].obs.copy()

######################################################################

# Merge with SC -> final
scvi_ls_nodeg_adata = sc.concat(
    [sc_adata_nodeg, adata_sn_transformed_scvi_notcond_nodeg_0], axis=0, merge="same"
)
scvi_ls_nodeg_adata = sc.concat(
    [scvi_ls_nodeg_adata, adata_sn_transformed_scvi_notcond_nodeg_1],
    axis=0,
    merge="same",
)

## And make pseudobulks
scvi_ls_pseudos, scvi_ls_props = make_pseudobulks(
    scvi_ls_adata,
    {"random": num_rand, "realistic": num_realistic},
    num_cells=num_cells,
    noise=False,
    cell_types=scvi_ls_adata.obs.cell_types.unique(),
)


scvi_ls_nodeg_pseudos, scvi_ls_nodeg_props = make_pseudobulks(
    scvi_ls_nodeg_adata,
    {"random": num_rand, "realistic": num_realistic},
    num_cells=num_cells,
    noise=False,
    cell_types=scvi_ls_nodeg_adata.obs.cell_types.unique(),
)


scvi_cond_pseudos, scvi_cond_props = make_pseudobulks(
    scvi_cond_adata,
    {"random": num_rand, "realistic": num_realistic},
    num_cells=num_cells,
    noise=False,
    cell_types=scvi_cond_adata.obs.cell_types.unique(),
)

scvi_cond_nodeg_pseudos, scvi_cond_nodeg_props = make_pseudobulks(
    scvi_cond_nodeg_adata,
    {"random": num_rand, "realistic": num_realistic},
    num_cells=num_cells,
    noise=False,
    cell_types=scvi_cond_nodeg_adata.obs.cell_types.unique(),
)

""" Now PCA Transforms """
## First with all genes
sc_expr = sc_adata.to_df()
sc_celltypes = sc_adata.obs.cell_types

sn_expr = sn_adata.to_df()
sn_celltypes = sn_adata.obs.cell_types

sn_miss_expr = {}
sn_miss_celltypes = {}
sn_miss_expr[0] = sn_missing[0].to_df()
sn_miss_celltypes[0] = sn_missing[0].obs.cell_types

sn_miss_expr[1] = sn_missing[1].to_df()
sn_miss_celltypes[1] = sn_missing[1].obs.cell_types

df_transformed_sn_pca_0 = transform_heldout_sn_to_mean_sc_local(
    sc_data=sc_expr,
    sn_data=sn_expr,
    sn_heldout_data=sn_miss_expr[0],
    sc_celltype_labels=sc_celltypes,
    sn_celltype_labels=sn_celltypes,
    variance_threshold=0.75,
    k_neighbors=10,
)
transformed_sn_pca_0 = sc.AnnData(X=df_transformed_sn_pca_0.values)
transformed_sn_pca_0.obs["cell_types"] = sn_missing[0].obs["cell_types"].values
transformed_sn_pca_0.var_names = sn_missing[0].var_names

######################################################################

df_transformed_sn_pca_1 = transform_heldout_sn_to_mean_sc_local(
    sc_data=sc_expr,
    sn_data=sn_expr,
    sn_heldout_data=sn_miss_expr[1],
    sc_celltype_labels=sc_celltypes,
    sn_celltype_labels=sn_celltypes,
    variance_threshold=0.75,
    k_neighbors=10,
)
transformed_sn_pca_1 = sc.AnnData(X=df_transformed_sn_pca_1.values)
transformed_sn_pca_1.obs["cell_types"] = sn_missing[1].obs["cell_types"].values
transformed_sn_pca_1.var_names = sn_missing[1].var_names

######################################################################

trans = sc.concat([transformed_sn_pca_1, transformed_sn_pca_0])
pca_adata = sc.concat([sc_adata, trans])

## Now no DEG

sc_expr_nodeg = sc_adata_nodeg.to_df()
sc_celltypes = sc_adata_nodeg.obs.cell_types

sn_expr_nodeg = sn_adata_nodeg.to_df()
sn_celltypes = sn_adata_nodeg.obs.cell_types

sn_miss_expr_nodeg = {}
sn_miss_expr_nodeg[0] = sn_missing_nodeg[0].to_df()
sn_miss_celltypes[0] = sn_missing_nodeg[0].obs.cell_types

sn_miss_expr_nodeg[1] = sn_missing_nodeg[1].to_df()
sn_miss_celltypes[1] = sn_missing_nodeg[1].obs.cell_types

df_transformed_sn_pca_0 = transform_heldout_sn_to_mean_sc_local(
    sc_data=sc_expr_nodeg,
    sn_data=sn_expr_nodeg,
    sn_heldout_data=sn_miss_expr_nodeg[0],
    sc_celltype_labels=sc_celltypes,
    sn_celltype_labels=sn_celltypes,
    variance_threshold=0.75,
    k_neighbors=10,
)
transformed_sn_pca_0 = sc.AnnData(X=df_transformed_sn_pca_0.values)
transformed_sn_pca_0.obs["cell_types"] = sn_missing_nodeg[0].obs["cell_types"].values
transformed_sn_pca_0.var_names = sn_missing_nodeg[0].var_names

######################################################################
df_transformed_sn_pca_1 = transform_heldout_sn_to_mean_sc_local(
    sc_data=sc_expr_nodeg,
    sn_data=sn_expr_nodeg,
    sn_heldout_data=sn_miss_expr_nodeg[1],
    sc_celltype_labels=sc_celltypes,
    sn_celltype_labels=sn_celltypes,
    variance_threshold=0.75,
    k_neighbors=10,
)
transformed_sn_pca_1 = sc.AnnData(X=df_transformed_sn_pca_1.values)
transformed_sn_pca_1.obs["cell_types"] = sn_missing_nodeg[1].obs["cell_types"].values
transformed_sn_pca_1.var_names = sn_missing_nodeg[1].var_names

######################################################################

trans_nodeg = sc.concat([transformed_sn_pca_1, transformed_sn_pca_0])
pca_adata_nodeg = sc.concat([sc_adata_nodeg, trans_nodeg])

## and make pseudobulks:
pca_pseudos, pca_props = make_pseudobulks(
    pca_adata,
    {"random": num_rand, "realistic": num_realistic},
    num_cells=num_cells,
    noise=False,
    cell_types=pca_adata.obs.cell_types.unique(),
)

pca_nodeg_pseudos, pca_nodeg_props = make_pseudobulks(
    pca_adata_nodeg,
    {"random": num_rand, "realistic": num_realistic},
    num_cells=num_cells,
    noise=False,
    cell_types=pca_adata_nodeg.obs.cell_types.unique(),
)

""" Removal of DEG only transform now """

adata_nodeg = sc.concat([sn_missing_nodeg[0], sn_missing_nodeg[1]])
adata_nodeg = sc.concat([adata_nodeg, sc_adata_nodeg])

## and make pseudobulks
nodeg_pseudos, nodeg_props = make_pseudobulks(
    adata_nodeg,
    {"random": num_rand, "realistic": num_realistic},
    num_cells=num_cells,
    noise=False,
    cell_types=adata_nodeg.obs.cell_types.unique(),
)

## now the Raw S-Nuc

adata_rawsn = sc.concat([sn_missing[0], sn_missing[1]])
adata_rawsn = sc.concat([sc_adata, adata_rawsn])

## and make pseudobulks
rawsn_pseudos, rawsn_props = make_pseudobulks(
    adata_rawsn,
    {"random": num_rand, "realistic": num_realistic},
    num_cells=num_cells,
    noise=False,
    cell_types=adata_rawsn.obs.cell_types.unique(),
)

## Thecontrol now.

# single cell
allsc_pseudos, allsc_props = make_pseudobulks(
    sc_adata,
    {"random": num_rand, "realistic": num_realistic},
    num_cells=num_cells,
    noise=False,
    cell_types=sc_adata.obs.cell_types.unique(),
)

## Now saving all:

pseudobulk_dict = {
    "snRNA": pseudobulk_dict["snRNA"],
    "scVILS": pseudobulk_dict["scVILS"],
    "scVIcond": pseudobulk_dict["scVIcond"],
    "PCA": pseudobulk_dict["PCA"],
}

pseudobulk_nodeg_dict = {
    "scVILS (-DEG)": pseudobulk_nodeg_dict["scVILS (-DEG)"],
    "scVIcond (-DEG)": pseudobulk_nodeg_dict["scVIcond (-DEG)"],
    "PCA (-DEG)": pseudobulk_nodeg_dict["PCA (-DEG)"],
    "snRNA (-DEG)": pseudobulk_nodeg_dict["snRNA (-DEG)"],
}

# Make a directory to hold your saved objects
save_dir = f"{os.getcwd()}/../data/Real_ADP/bulks_for_comp/"
os.makedirs(save_dir, exist_ok=True)

# 1) Save real‐bulk DataFrame
bulk_df.to_pickle(os.path.join(save_dir, "bulk_df.pkl"))

# 2) Save each pseudobulk DataFrame
for name, df in pseudobulk_dict.items():
    # sanitize name for filename, e.g. replace spaces or slashes
    fname = name.replace(" ", "_").replace("/", "_")
    df.to_pickle(os.path.join(save_dir, f"{fname}_pseudobulk.pkl"))

# 3)Save “no‐DEG” versions too
real_df_nodeg.to_pickle(os.path.join(save_dir, "bulk_df_nodeg.pkl"))
for name, df in pseudobulk_nodeg_dict.items():
    fname = name.replace(" ", "_").replace("/", "_")
    df.to_pickle(os.path.join(save_dir, f"{fname}_pseudobulk_nodeg.pkl"))

# 4) Later, to load them back in one go:
with open(os.path.join(save_dir, "bulk_df.pkl"), "rb") as f:
    bulk_df = pickle.load(f)

pseudobulk_dict = {}
for p in os.listdir(save_dir):
    if p.endswith("_pseudobulk.pkl"):
        name = p.replace("_pseudobulk.pkl", "").replace("_", " ")
        with open(os.path.join(save_dir, p), "rb") as f:
            pseudobulk_dict[name] = pickle.load(f)
