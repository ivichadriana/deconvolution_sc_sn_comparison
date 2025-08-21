"""

The following script processes the bulks from real adipose tissue, does differential gene expression, and then trains needed models.

The bulk RNA‐seq data were generated from SAT samples (specifically, from 331 men in the METSIM cohort).
Please download [bulk RNA-seq here](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE135134).
For bulk RNA-seq gene name mapping, download: [Basic gene annotation file in GTF format](https://www.gencodegenes.org/human/).

"""

import os, re, glob, sys
import numpy as np
import scvi
import pandas as pd
import seaborn as sns
import json
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sys.path.insert(1, "../../")
sys.path.insert(1, "../")
sys.path.insert(1, "../../../")
from src.helpers import prepare_data, split_single_cell_data
from src.helpers import (
    pick_cells,
    make_references,
    get_normalized_expression_from_latent,
)
from src.helpers import transform_heldout_sn_to_mean_sc, calculate_median_library_size
from src.helpers import (
    make_pseudobulks,
    create_fixed_pseudobulk,
    downsample_cells_by_type,
)
from src.helpers import run_deseq2_for_cell_type, transform_heldout_sn_to_mean_sc_local
from src.helpers import (
    differential_expression_analysis,
    remove_diff_genes,
    scvi_train_model,
)
from src.helpers import (
    differential_expression_analysis_parallel,
    transform_heldout_sn_to_mean_sc_VAE,
)
from src.helpers import (
    save_cibersort,
    save_bayesprism_references,
    save_bayesprism_pseudobulks,
)

# -----------------------------
# PARAMETERS
# -----------------------------
res_name = "ADP"
# Set base paths (relative to the notebook’s location)
base_dir = ".."  # adjust as needed
# Define file paths
bulk_file = f"{base_dir}/data/{res_name}/GSE135134_METSIM_subcutaneousAdipose_RNAseq_TPMs_n434.txt.gz"
sn_sc_file = f"{base_dir}/data/{res_name}/sc_sn_{res_name}_allcells.h5ad"

# -----------------------------
# PROCESS BULKS
# -----------------------------

# 1. Read in the data
bulk_df = pd.read_csv(bulk_file, sep="\t", index_col=0, compression="gzip")
print("Bulk data shape:", bulk_df.shape)

adata = sc.read_h5ad(sn_sc_file)
print("Combined SN+SC data shape:", adata.shape)

# Path to the GTF file
## We use a mapping file (mapping_df) that links transcripts (e.g., ENST00000561901.1) to their corresponding genes (e.g., ZNF707).
gtf_path = f"{base_dir}/data/{res_name}/gencode.v47.basic.annotation.gtf.gz"

# Read the GTF file; skip header lines starting with "#"
gtf_df = pd.read_csv(gtf_path, sep="\t", comment="#", header=None)

# Filter for gene-level entries (feature column equals "gene")
gene_df = gtf_df[gtf_df[2] == "gene"].copy()


# Function to parse the attributes field of the GTF file
def parse_attributes(attr_string):
    attributes = {}
    for attribute in attr_string.split(";"):
        attribute = attribute.strip()
        if not attribute:
            continue
        key_value = attribute.split(" ", 1)
        if len(key_value) == 2:
            key, value = key_value
            # Remove quotes from the value
            attributes[key] = value.replace('"', "")
    return attributes


# Extract gene_id and gene_name from the attributes column (column 8)
gene_df["gene_id"] = gene_df[8].apply(lambda x: parse_attributes(x)["gene_id"])
gene_df["gene_name"] = gene_df[8].apply(lambda x: parse_attributes(x)["gene_name"])

# Remove version numbers from gene_id (e.g., convert ENSG00000000003.10 -> ENSG00000000003)
gene_df["gene_id"] = gene_df["gene_id"].str.split(".").str[0]

# Create a mapping DataFrame with unique gene_id and gene_name pairs
mapping_df = gene_df[["gene_id", "gene_name"]].drop_duplicates()

# Remove version numbers from bulk_df index for consistent merging
bulk_df.index = bulk_df.index.str.split(".").str[0]

# Merge the bulk RNA-seq DataFrame with the mapping on gene_id
bulk_df = bulk_df.merge(mapping_df, left_index=True, right_on="gene_id", how="left")

# # set gene_name as the new index if that is preferred
bulk_df = bulk_df.set_index("gene_name").T
bulk_df = bulk_df.loc[:, bulk_df.columns.notna()]
bulk_df = bulk_df.drop("gene_id", axis=0)
# Aggregate duplicate gene columns by summing them
bulk_df = bulk_df.groupby(bulk_df.columns, axis=1).sum()

# 2. Match genes between datasets
# Assume that bulk_df columns are gene names. Identify the common gene set.
common_genes = list(set(bulk_df.columns).intersection(set(adata.var_names)))
print("Number of common genes:", len(common_genes))

adata = adata[:, common_genes].copy()
bulk_df = bulk_df[common_genes].copy()

# and save bulks:
bulk_df.to_csv(f"{base_dir}/data/{res_name}/adipose_bulks_processed.csv")

# 3. Separate SN and SC data based on the 'data_type' column
sc_adata = adata[adata.obs["data_type"] == "single_cell"].copy()
sn_adata = adata[adata.obs["data_type"] == "single_nucleus"].copy()

del adata

## QC

print("Single-cell data shape:", sc_adata.shape)
print("Single-nucleus data shape:", sn_adata.shape)
print("Bulk data shape:", bulk_df.shape)

print(sc_adata.obs.cell_types.value_counts())
print(sn_adata.obs.cell_types.value_counts())

# -----------------------------
# TRAIN MODELS WITH ALL GENES
# -----------------------------

matching = np.intersect1d(
    sn_adata.obs.cell_types.unique(), sc_adata.obs.cell_types.unique()
)
nonmatching = np.setxor1d(
    sn_adata.obs.cell_types.unique(), sc_adata.obs.cell_types.unique()
)

sn_adata_match = sn_adata[sn_adata.obs.cell_types.isin(matching)].copy()

sn_missing = {}
sn_miss_adata = sn_adata[sn_adata.obs.cell_types.isin(nonmatching)].copy()
sn_missing[0] = sn_miss_adata[sn_miss_adata.obs.cell_types == nonmatching[0]].copy()
sn_missing[1] = sn_miss_adata[sn_miss_adata.obs.cell_types == nonmatching[1]].copy()

print(sn_adata_match.obs.cell_types.value_counts())

sn_adata = sn_adata_match.copy()

print(sn_miss_adata.obs.cell_types.value_counts())

sc_adata_train = downsample_cells_by_type(sc_adata, max_cells=1500)
sn_adata_train = downsample_cells_by_type(sn_adata, max_cells=1500)

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
adata = sc.concat([sc_adata, sn_adata])
de_change = {}

# -----------------------------
# DIFFERENTIAL GENE EXPRESSION
# -----------------------------
print("Check if differential gene expression calcs. are needed...")
genes_save_path = f"{base_dir}/data/{res_name}/degs.json"

if os.path.exists(genes_save_path):
    try:
        with open(genes_save_path, "r") as file:
            diff_genes_json = json.load(file)
        # Reconstruct DataFrames
        diff_genes = {}
        for key, value in diff_genes_json.items():
            if isinstance(value, dict) and {"index","columns","data"}.issubset(value.keys()):
                diff_genes[key] = pd.DataFrame(**value)
            else:
                # Fallback: ensure we have a DataFrame to avoid errors later
                diff_genes[key] = pd.DataFrame(value)
    except json.JSONDecodeError:
        print("Warning: JSON file is empty or corrupted. Recalculating diff_genes...")
        print("Creating pseudobulk of size 10 for SC and SN data for DGE analysis...")
        pseudo_sc_adata = create_fixed_pseudobulk(sc_adata, group_size=10)
        pseudo_sn_adata = create_fixed_pseudobulk(sn_adata, group_size=10)
        diff_genes = differential_expression_analysis_parallel(
            sn_adata=pseudo_sn_adata,
            sc_adata=pseudo_sc_adata,
            deseq_alpha=0.01,
            num_threads=4,
            n_cpus_per_thread=16,
        )
        diff_genes_json = {
            key: (df.to_dict(orient="split") if isinstance(df, pd.DataFrame) else df)
            for key, df in diff_genes.items()
        }
        # persist the repaired/recomputed data
        with open(genes_save_path, "w") as file:
            json.dump(diff_genes_json, file)
else:
    print("Not found previous diff. gene expr... calculating now!")
    print("Creating pseudobulk of size 10 for SC and SN data for DGE analysis...")
    pseudo_sc_adata = create_fixed_pseudobulk(sc_adata, group_size=10)
    pseudo_sn_adata = create_fixed_pseudobulk(sn_adata, group_size=10)
    diff_genes = differential_expression_analysis_parallel(
        sn_adata=pseudo_sn_adata,
        sc_adata=pseudo_sc_adata,
        deseq_alpha=0.01,
        num_threads=4,
        n_cpus_per_thread=16,
    )
    print("Found these many differentially expressed genes:")
    for key in diff_genes.keys():
        print(key, diff_genes[key].shape)

    diff_genes_json = {
        key: (df.to_dict(orient="split") if isinstance(df, pd.DataFrame) else df)
        for key, df in diff_genes.items()
    }
    with open(genes_save_path, "w") as file:
        json.dump(diff_genes_json, file)

flattened_index = [idx for df in diff_genes.values() for idx in df.index]
flattened_index = list(flattened_index)
all_de_genes_list = list(dict.fromkeys(flattened_index))
print("Total unique DEGs:", len(all_de_genes_list))

# -----------------------------
# TRAIN MODELS WITHOUT DIFF. EXPRESSED GENES
# -----------------------------

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

print("\n=== All models trained. ===")
