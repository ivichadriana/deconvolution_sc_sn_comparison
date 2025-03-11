# import the dependencies
import sys
# general imports
import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
import scipy as sp
from scipy.sparse import coo_matrix
import collections
from collections import Counter
import anndata as ad
import gzip
from scipy import sparse
import re
import shutil
from sklearn.decomposition import PCA
from sklearn.utils import resample
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import scanpy as sc
from scipy.sparse import issparse
from pydeseq2.default_inference import DefaultInference
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import os
import gc
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(1, '../../')
sys.path.insert(1, '../')
sys.path.insert(1, '../../../../../')

def remove_unassigned_cells(adata, dataset_name):
    """
    Removes cells with missing or unassigned cell types.

    Parameters:
    - adata: AnnData object
    - dataset_name: str, name of the dataset for logging

    Returns:
    - Cleaned AnnData object
    """

    # Ensure cell types are strings and remove leading/trailing spaces
    adata.obs["cell_types"] = adata.obs["cell_types"].astype(str).str.strip()

    # Define unwanted or unassigned cell types
    unwanted_types = ["Blood: Undefined", "NA", "SC & Eosinophil", "nan", "", "None"]

    # Count missing before removal
    initial_count = adata.n_obs
    missing_cells = adata.obs["cell_types"].isin(unwanted_types) | adata.obs["cell_types"].isna()

    # Remove unassigned/missing cells
    adata = adata[~missing_cells].copy()
    
    # Log the change
    removed_count = initial_count - adata.n_obs
    print(f"{dataset_name}: Removed {removed_count} unassigned/missing cells. Remaining: {adata.n_obs}")

    return adata

def transform_heldout_sn_to_mean_sc(
    sc_data: pd.DataFrame,
    sn_data: pd.DataFrame,
    sn_heldout_data: pd.DataFrame,
    sc_celltype_labels: pd.Series,
    sn_celltype_labels: pd.Series,
    heldout_label: str,
    variance_threshold: float = 0.90
):
    """
    Transforms the 'held-out' SN cell type (sn_heldout_data) to match the average
    PCA profile shift across overlapping SC vs SN cell types, then scales
    to match the median SC library size.

    Aligns columns across SC data, SN data, and the SN held‐out subset.
    Performs PCA using only the shared cell types (excluding the held‐out).
    Computes an average shift across overlapping cell types.
    Applies that composite shift to the held‐out cells.
    Exponentiates and rescales to match the median SC library size.

    Parameters
    ----------
    sc_data : pd.DataFrame
        Single-cell data, rows = cells, columns = genes (raw counts).
    sn_data : pd.DataFrame
        Single-nucleus data, rows = cells, columns = genes (raw counts).
    sn_heldout_data : pd.DataFrame
        Single-nucleus data for the held-out cell type, rows = cells, columns = genes (raw counts).
    sc_celltype_labels : pd.Series
        Cell-type labels for the rows of sc_data (index-aligned).
    sn_celltype_labels : pd.Series
        Cell-type labels for the rows of sn_data (index-aligned).
    heldout_label : str
        The label (cell type) that is missing in SC but present in SN (for sn_heldout_data).
    variance_threshold : float
        Fraction of variance to retain in PCA (e.g., 0.90).

    Returns
    -------
    pd.DataFrame
        The transformed SN-heldout data (rows = cells, columns = genes),
        with matched gene columns and library sizes akin to typical SC.
    """

    # 1) Align columns (genes) across SC, SN, and SN-heldout
    common_genes = sc_data.columns.intersection(sn_data.columns).intersection(sn_heldout_data.columns)
    sc_data_aligned = sc_data[common_genes]
    sn_data_aligned = sn_data[common_genes]
    sn_heldout_data_aligned = sn_heldout_data[common_genes]

    # 2) Identify overlapping cell types (excluding the held-out type)
    overlapping_types = np.intersect1d(sc_celltype_labels.unique(), sn_celltype_labels.unique())
    overlapping_types = overlapping_types[overlapping_types != heldout_label]

    # 3) Fit PCA on combined SC+SN, excluding the held-out SN
    keep_sn_mask = (sn_celltype_labels != heldout_label)
    combined_df = pd.concat([sc_data_aligned, sn_data_aligned.loc[keep_sn_mask]], axis=0)
    combined_log = np.log1p(combined_df)

    pca_full = PCA()
    pca_full.fit(combined_log)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.searchsorted(cumsum_var, variance_threshold) + 1
    pca = PCA(n_components=n_components)
    pca.fit(combined_log)

    # 4) Compute SHIFT VECTORS for each overlapping cell type
    #    SHIFT = mean(SC in PCA) - mean(SN in PCA)
    shift_vectors = []
    for ct in overlapping_types:
        sc_cells_ct = sc_data_aligned.loc[sc_celltype_labels == ct]
        sn_cells_ct = sn_data_aligned.loc[sn_celltype_labels == ct]

        if not sc_cells_ct.empty and not sn_cells_ct.empty:
            sc_log = np.log1p(sc_cells_ct)
            sn_log = np.log1p(sn_cells_ct)
            sc_pcs = pca.transform(sc_log)
            sn_pcs = pca.transform(sn_log)
            mean_sc_pca = sc_pcs.mean(axis=0)
            mean_sn_pca = sn_pcs.mean(axis=0)
            shift_vectors.append(mean_sc_pca - mean_sn_pca)

    # 5) Average these SHIFT vectors
    if len(shift_vectors) == 0:
        # fallback: do a simpler global SC-vs-SN shift
        print("No overlapping cell types found! Using global SC-SN shift.")
        all_sc_log = np.log1p(sc_data_aligned)
        all_sn_log = np.log1p(sn_data_aligned.loc[keep_sn_mask])
        composite_shift = (
            pca.transform(all_sc_log).mean(axis=0)
            - pca.transform(all_sn_log).mean(axis=0)
        )
    else:
        composite_shift = np.mean(shift_vectors, axis=0)

    # 6) Transform the HELD-OUT SN cells with that composite shift
    ho_log = np.log1p(sn_heldout_data_aligned)
    ho_pcs = pca.transform(ho_log)
    ho_pcs_shifted = ho_pcs + composite_shift
    ho_log_transformed = pca.inverse_transform(ho_pcs_shifted)

    # 7) Exponentiate to get back near "counts"
    ho_exp = np.expm1(ho_log_transformed)

    # 8) Scale each cell's library to match typical SC library size
    sc_totals = sc_data_aligned.sum(axis=1)
    median_sc_lib = np.median(sc_totals)
    for i in range(ho_exp.shape[0]):
        row_sum = ho_exp[i, :].sum()
        if row_sum > 0:
            ho_exp[i, :] *= (median_sc_lib / row_sum)

    # 9) Build final DataFrame with the same row index + gene columns
    transformed_df = pd.DataFrame(
        ho_exp,
        index=sn_heldout_data_aligned.index,
        columns=common_genes
    )
    return transformed_df



def filter_out_cell_markers(adata_sc, adata_sn, diff_genes, marker_threshold=50):
    """
    Removes differentially expressed genes that are also strong cell markers.

    Parameters
    ----------
    adata_sc : AnnData
        Single-cell AnnData object.
    adata_sn : AnnData
        Single-nucleus AnnData object.
    diff_genes : dict
        Dictionary of differentially expressed genes (DEGs) from DESeq2.
    marker_threshold : int, optional
        Number of top marker genes to consider per cell type, by default 50.

    Returns
    -------
    filtered_diff_genes : dict
        Differentially expressed genes with cell markers removed.
    """
    # Ensure data is logarithmized
    adata_sc = adata_sc.copy()
    adata_sn = adata_sn.copy()
    sc.pp.log1p(adata_sc)
    sc.pp.log1p(adata_sn)

    # Combine SC and SN data to get cell markers
    adata_combined = ad.concat([adata_sc, adata_sn])

    # Run Scanpy's `rank_genes_groups` to find top markers
    sc.tl.rank_genes_groups(adata_combined, groupby="cell_types", method="wilcoxon")

    # Extract top marker genes correctly
    marker_df = pd.DataFrame(adata_combined.uns["rank_genes_groups"]["names"])
    marker_genes = set(marker_df.iloc[:marker_threshold].values.flatten())

    print(f"Identified {len(marker_genes)} potential cell marker genes.")

    # Filter DEGs by removing markers
    filtered_diff_genes = {}
    for cell_type, genes_df in diff_genes.items():
        filtered_genes = genes_df[~genes_df.index.isin(marker_genes)]
        filtered_diff_genes[cell_type] = filtered_genes

    print("Filtered out marker genes from DEGs.")

    return filtered_diff_genes

def load_PNB_data(data_type: str, load_testing: bool = False):
    """
    Open the data of PNB. We have 1 SN and 1 SC from same patient, and 1 SC from another for testing.

    Parameters
    ----------
    data_type : str (either "single_nucleus" or "single_cell")

    Returns
    -------
    AnnData
        TheAnnData object.
    Metadata
        Pandas Dataframe.
    """
    res_name = "PNB"
    adata = []
    meta_data = []
    path = f"{os.getcwd()}/../data/{res_name}/"  # Path to original data.

    if data_type == "single_cell":
        if load_testing:
            # File paths
            h5_gz_file = Path(path, "GSM4186962_HTAPP-312-SMP-902_fresh-C4-T2_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186962_HTAPP-312-SMP-902_fresh-C4-T2_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186962_metadata_HTAPP-312-SMP-902_fresh-C4-T2_channel1.csv.gz")

            # Decompressing the h5.gz file
            with gzip.open(h5_gz_file, "rb") as f_in:
                with open(h5_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Importing the data
            sc_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
            sc_adata.var_names_make_unique()
            # Read the CSV file using pandas
            sc_metadata = pd.read_csv(csv_file)
            # clean up cell name to match obs in AnnData
            sc_metadata["cell"] = sc_metadata["Unnamed: 0"].apply(split_ID_2).astype(str)
            sc_metadata["cell"] = sc_metadata["cell"].apply(merge_strings)

            adata = sc_adata
            meta_data = sc_metadata

        else:
            # File paths
            h5_gz_file = Path(path, "GSM4186963_HTAPP-656-SMP-3481_fresh-T1_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186963_HTAPP-656-SMP-3481_fresh-T1_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186963_metadata_HTAPP-656-SMP-3481_fresh-T1_channel1.csv.gz")

            # Decompressing the h5.gz file
            with gzip.open(h5_gz_file, "rb") as f_in:
                with open(h5_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Importing the data
            sc_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
            sc_adata.var_names_make_unique()
            # Read the CSV file using pandas
            sc_metadata = pd.read_csv(csv_file)
            # clean up cell name to match obs in AnnData
            sc_metadata["cell"] = sc_metadata["Unnamed: 0"].apply(split_ID_2).astype(str)
            sc_metadata["cell"] = sc_metadata["cell"].apply(merge_strings)

            adata = sc_adata
            meta_data = sc_metadata

    elif data_type == "single_nucleus":
        # File paths
        h5_gz_file = Path(path, "GSM4186969_HTAPP-656-SMP-3481_TST_channel1_raw_gene_bc_matrices_h5.h5.gz")
        h5_file = Path(path, "GSM4186969_HTAPP-656-SMP-3481_TST_channel1_raw_gene_bc_matrices_h5.h5")
        csv_file = Path(path, "GSM4186969_metadata_HTAPP-656-SMP-3481_TST_channel1.csv.gz")

        # Decompressing the h5.gz file
        with gzip.open(h5_gz_file, "rb") as f_in:
            with open(h5_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Importing the data
        sn_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
        sn_adata.var_names_make_unique()
        # Read the CSV file using pandas
        sn_metadata = pd.read_csv(csv_file)
        # clean up cell name to match obs
        sn_metadata["cell"] = sn_metadata["Unnamed: 0"].apply(split_ID).astype(str)
        sn_metadata["cell"] = sn_metadata["cell"].apply(merge_strings)

        adata = sn_adata
        meta_data = sn_metadata

    else:
        print('Give valid data type: "single_cell" or "single_nucleus"')

    return adata, meta_data

def split_ID(row):
    """Takes in one row of DataFrame, returns the right cell name in metadata of MBC"""
    return row.split("-")[4]

def split_ID_2(row):
    """Takes in one row of DataFrame, returns the right cell name in metadata of MBC"""
    return row.split("-")[-1]

def merge_strings(row):
    """Takes in one row of DataFrame, returns the merged strings"""
    return row + "-1"

def assign_cell_types(adata: AnnData, cell_types_assign: np.array) -> AnnData:
    """
    Assign cell types from DataFrame in obs on the AnnData object and make sure the same cell-types are in both.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to assign cell types to.
    cell_types: Numpy Array
        The cell types to assign.

    Returns
    -------
    AnnData
        The AnnData object with cell types assigned in .obs["cell_types"].
    """
    # Check if the length matches:
    if len(cell_types_assign) != adata.X.shape[0]:
        raise AssertionError("Cell types do not match in Anndata and NumPy array")

    adata.obs["cell_types"] = cell_types_assign

    print(f"Done!\nCell Types in Data are: {adata.obs.cell_types.value_counts()}")

    return adata
    
def make_prop_table(adata: sc.AnnData, obs):
    """
    Makes proportion table from AnnData object's cell types.

    Parameters
    ----------
    adata: AnnData
    obs: str (name of obs with cell types)

    """
    num_cell_counter = Counter(adata.obs[obs])
    num_cells = []
    cell_types = []
    prop_cells = []
    tot_count = 0
    tot_prop = 0

    for cell in num_cell_counter:
        num_cells.append(num_cell_counter[cell])
        cell_types.append(cell)
        tot_count = tot_count + num_cell_counter[cell]

    for cell in num_cell_counter:
        proportion = num_cell_counter[cell] / tot_count
        prop_cells.append(proportion)
        tot_prop = tot_prop + proportion

    cell_types.append("Total")
    num_cells.append(tot_count)
    prop_cells.append(tot_prop)
    table = {"Cell_Types": cell_types, "Num_Cells": num_cells, "Prop_Cells": prop_cells}
    table = pd.DataFrame(table)
    return table

def load_MBC_data(data_type: str, load_testing: bool = False):
    """
    Open the data og MBC. We have 1 SN and 1 SC from same patient, and 1 SC from another for testing.

    Parameters
    ----------
    data_type : str (either "single_nucleus" or "single_cell")

    Returns
    -------
    AnnData
        TheAnnData object.
    Metadata
        Pandas Dataframe.
    """
    res_name = "MBC"
    adata = []
    meta_data = []
    path = f"{os.getcwd()}/../data/{res_name}/"  # Path to original data.

    if data_type == "single_cell":
        if load_testing:
            # File paths
            h5_gz_file = Path(path, "GSM4186973_HTAPP-285-SMP-751_fresh_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186973_HTAPP-285-SMP-751_fresh_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186973_metadata_HTAPP-285-SMP-751_fresh_channel1.csv.gz")

            # Decompressing the h5.gz file
            with gzip.open(h5_gz_file, "rb") as f_in:
                with open(h5_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Importing the data
            sc_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
            sc_adata.var_names_make_unique()
            # Read the CSV file using pandas
            sc_metadata = pd.read_csv(csv_file)
            # clean up cell name to match obs in AnnData
            sc_metadata["cell"] = sc_metadata["Unnamed: 0"].apply(split_ID).astype(str)
            sc_metadata["cell"] = sc_metadata["cell"].apply(merge_strings)

            adata = sc_adata
            meta_data = sc_metadata

        else:
            # File paths
            h5_gz_file = Path(path, "GSM4186974_HTAPP-963-SMP-4741_fresh_channel1_raw_gene_bc_matrices_h5.h5.gz")
            h5_file = Path(path, "GSM4186974_HTAPP-963-SMP-4741_fresh_channel1_raw_gene_bc_matrices_h5.h5")
            csv_file = Path(path, "GSM4186974_metadata_HTAPP-963-SMP-4741_fresh_channel1.csv.gz")

            # Decompressing the h5.gz file
            with gzip.open(h5_gz_file, "rb") as f_in:
                with open(h5_file, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Importing the data
            sc_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
            sc_adata.var_names_make_unique()
            # Read the CSV file using pandas
            sc_metadata = pd.read_csv(csv_file)
            # clean up cell name to match obs in AnnData
            sc_metadata["cell"] = sc_metadata["Unnamed: 0"].apply(split_ID).astype(str)
            sc_metadata["cell"] = sc_metadata["cell"].apply(merge_strings)

            adata = sc_adata
            meta_data = sc_metadata

    elif data_type == "single_nucleus":
        # File paths
        h5_gz_file = Path(path, "GSM4186980_HTAPP-963-SMP-4741_TST_channel1_raw_gene_bc_matrices_h5.h5.gz")
        h5_file = Path(path, "GSM4186980_HTAPP-963-SMP-4741_TST_channel1_raw_gene_bc_matrices_h5.h5")
        csv_file = Path(path, "GSM4186980_metadata_HTAPP-963-SMP-4741_TST_channel1.csv.gz")

        # Decompressing the h5.gz file
        with gzip.open(h5_gz_file, "rb") as f_in:
            with open(h5_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Importing the data
        sn_adata = sc.read_10x_h5(h5_file)  # Read the h5 file
        sn_adata.var_names_make_unique()
        # Read the CSV file using pandas
        sn_metadata = pd.read_csv(csv_file)
        # clean up cell name to match obs
        sn_metadata["cell"] = sn_metadata["Unnamed: 0"].apply(split_ID).astype(str)
        sn_metadata["cell"] = sn_metadata["cell"].apply(merge_strings)

        adata = sn_adata
        meta_data = sn_metadata

    else:
        print('Give valid data type: "single_cell" or "single_nucleus"')

    return adata, meta_data

def match_cell_types(adata1: AnnData, adata2: AnnData, cells_to_keep: np.array):
    """
    Match cell types in 2 AnnData objects according to cells in cells_to_keep.

    Parameters
    ----------
    adata1 : AnnData
        The AnnData object to match cell types.
    adata2 : AnnData
        The AnnData object to match cell types.
    cell_types_to_keep: Numpy Array
        The cell types to keep in both Anndatas

    Returns
    -------
    adata1:
        AnnData with same cell types.
    adata2:
        AnnData with same cell types.
    """
    # match cells:
    adata1 = adata1[adata1.obs["cell_types"].isin(cells_to_keep)].copy()
    adata2 = adata2[adata2.obs["cell_types"].isin(cells_to_keep)].copy()

    return adata1, adata2

def qc_check_references(sc_ref_path, sn_ref_path):
    """
    Validate the single-cell and single-nucleus references.
    """
    print("\nRunning QC on references...")

    # Load the references
    sc_ref = pd.read_csv(sc_ref_path, sep="\t", index_col=0)
    sn_ref = pd.read_csv(sn_ref_path, sep="\t", index_col=0)

    # Extract cell types
    sc_cell_types = set(sc_ref.columns[1:])
    sn_cell_types = set(sn_ref.columns[1:])

    # Validate matching cell types
    assert sc_cell_types == sn_cell_types, (
        f"Cell types in single-cell and single-nucleus references do not match!\n"
        f"SC cell types: {sc_cell_types}\n"
        f"SN cell types: {sn_cell_types}"
    )
    print("Cell types match between single-cell and single-nucleus references.")

    # Validate number of cells per cell type
    sc_cell_counts = sc_ref.iloc[:, 1:].notna().sum(axis=0)
    sn_cell_counts = sn_ref.iloc[:, 1:].notna().sum(axis=0)
    assert (sc_cell_counts == sn_cell_counts).all(), (
        "Number of cells per cell type in references do not match!"
    )
    print("Number of cells per cell type match between single-cell and single-nucleus references.")

def qc_check_pseudobulks(pseudobulk_path, proportions_path):
    """
    Validate the pseudobulks and proportions.
    """
    print("\nRunning QC on pseudobulks and proportions...")

    # Load pseudobulks and proportions
    pseudobulks = pd.read_csv(pseudobulk_path, sep="\t", index_col=0)
    proportions = pd.read_csv(proportions_path, sep="\t", index_col=0)

    # Validate proportions sum to ~1
    proportion_sums = proportions.sum(axis=1)
    assert np.allclose(proportion_sums, 1, atol=1e-3), "Proportions do not sum to 1 for some pseudobulks!"
    print("Proportions sum to 1 for all pseudobulks.")

def qc_check_cell_types_match(original_anndata_path, sc_ref_path, sn_ref_path, pseudobulk_path):
    """
    Ensure the cell types in the original AnnData match the outputs,
    considering only cell types shared across all outputs.
    """
    print("\nRunning QC to ensure cell types match original data...")

    # Load original AnnData and references
    adata = sc.read_h5ad(original_anndata_path)
    sc_ref = pd.read_csv(sc_ref_path, sep="\t", index_col=0)
    sn_ref = pd.read_csv(sn_ref_path, sep="\t", index_col=0)
    pseudobulks = pd.read_csv(pseudobulk_path, sep="\t", index_col=0)

    # Extract cell types and normalize them by removing any suffixes
    original_cell_types = set(adata.obs["cell_types"].unique())
    sc_cell_types = {col.split('.')[0] for col in sc_ref.columns[1:]}  # Remove numerical suffixes
    sn_cell_types = {col.split('.')[0] for col in sn_ref.columns[1:]}  # Remove numerical suffixes
    pseudobulk_cell_types = set(pseudobulks.columns[1:])  # Pseudobulk cell types don't have suffixes

    # Find common cell types across all datasets
    common_cell_types = original_cell_types.intersection(sc_cell_types, sn_cell_types, pseudobulk_cell_types)

    # Check for valid common cell types
    if not common_cell_types:
        print(f"Original cell types: {original_cell_types}")
        print(f"SC Reference cell types: {sc_cell_types}")
        print(f"SN Reference cell types: {sn_cell_types}")
        print(f"Pseudobulk cell types: {pseudobulk_cell_types}")
        raise AssertionError(
            "No common cell types found! Please check if the datasets were processed consistently."
        )

    # Validate that all datasets only have the shared cell types
    assert sc_cell_types.issubset(common_cell_types), (
        f"Single-cell reference contains unexpected cell types!\n"
        f"Expected: {common_cell_types}\n"
        f"Found: {sc_cell_types}"
    )
    assert sn_cell_types.issubset(common_cell_types), (
        f"Single-nucleus reference contains unexpected cell types!\n"
        f"Expected: {common_cell_types}\n"
        f"Found: {sn_cell_types}"
    )
    assert pseudobulk_cell_types.issubset(common_cell_types), (
        f"Pseudobulk cell types do not match expected cell types!\n"
        f"Expected: {common_cell_types}\n"
        f"Found: {pseudobulk_cell_types}"
    )

    print("Cell types match across original AnnData, references, and pseudobulks.")

def create_fixed_pseudobulk(adata, group_size=10):
    """
    Aggregates cells into pseudobulks of size `group_size` for each cell type.
    If the cell count is not divisible by `group_size`, leftover cells are dropped.
    
    Returns:
        AnnData where each row is a pseudobulk sample and columns are genes.
        The new AnnData has:
         - obs["cell_type"] to indicate which cell type the pseudobulk came from.
         - obs["group_id"] to indicate the pseudobulk group number within that cell type.
    """
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from scipy.sparse import issparse, csr_matrix
    
    all_pseudobulks = []
    obs_list = []
    var_names = adata.var_names
    
    # For each cell type, group the cells in sets of `group_size`
    for cell_type in adata.obs["cell_types"].unique():
        subset = adata[adata.obs["cell_types"] == cell_type]
        n_cells = subset.shape[0]
        
        # Number of complete groups
        n_groups = n_cells // group_size
        if n_groups == 0:
            continue  # skip if fewer than group_size cells for this type

        # Only keep the cells that form complete groups
        used_cells_count = n_groups * group_size
        subset = subset[:used_cells_count, :]
        
        # Convert to dense or keep it sparse
        mat = subset.X
        if issparse(mat):
            mat = mat.toarray()
        
        # Reshape so we can sum every 10 rows
        # (One approach: group them in consecutive blocks of group_size)
        mat_reshaped = mat.reshape(n_groups, group_size, mat.shape[1])
        
        # Sum across the group_size dimension => shape (n_groups, n_genes)
        pseudobulk_counts = mat_reshaped.sum(axis=1)
        
        # Create an obs DataFrame for these pseudobulks
        group_ids = [f"{cell_type}_group{i+1}" for i in range(n_groups)]
        obs_tmp = pd.DataFrame({
            "cell_types": cell_type,
            "group_id": group_ids
        }, index=group_ids)  # index must match the new row names
        
        # Store these counts to combine later
        all_pseudobulks.append(pseudobulk_counts)
        obs_list.append(obs_tmp)
    
    # If nothing was aggregated, return empty
    if not all_pseudobulks:
        print("No groups formed for any cell type.")
        return None
    
    # Concatenate everything
    big_matrix = np.vstack(all_pseudobulks)
    big_obs = pd.concat(obs_list, axis=0)
    big_obs.index.name = None  # Clean up
    
    # Create new AnnData
    pseudo_adata = AnnData(X=csr_matrix(big_matrix),  # keep it sparse to save memory
                           obs=big_obs,
                           var=pd.DataFrame(index=var_names))
    
    return pseudo_adata


def save_cibersort(pseudobulks_df=[], 
                        proportions_df=[], 
                        adata_sc_ref=[], 
                        sc_adata_filtered=[], 
                        adata_sn_ref=[], 
                        sn_adata_filtered=[]):
    # Save pseudobulks
    pseudobulks_df = pseudobulks_df.T  # Transpose to make genes rows and pseudobulks columns
    pseudobulks_df.index.name = "Gene"  # Set the index as "Gene"
    pseudobulks_df.to_csv(
        os.path.join(args.output_path, "pseudobulks.txt"),
        sep="\t", header=True, index=True, quoting=3  # Save with the gene names as row index
    )
    # Save proportions
    proportions_df.index.name = "Pseudobulk"  # Name the first column as Pseudobulk
    proportions_df.reset_index(inplace=True)  # Reset index to move it into the table
    proportions_df.to_csv(
        os.path.join(args.output_path, "proportions.txt"),
        sep="\t", header=True, index=False, quoting=3
    )
    # Save single-cell references
    sc_ref_df = pd.DataFrame(
        adata_sc_ref.X.toarray().T,  # Transpose to make genes rows and cells columns
        index=adata_sc_ref.var.index,  # Genes as row index
        columns=adata_sc_ref.obs['cell_types']  # Phenotypic labels (cell types)
    )
    sc_ref_df.insert(0, "Gene", sc_ref_df.index)  # Add gene column
    sc_ref_df.to_csv(
        os.path.join(args.output_path, "sc_reference.txt"),
        sep="\t", header=True, index=False, quoting=3
    )

    # And filtered:
    sc_ref_df_filtered = pd.DataFrame(
        sc_adata_filtered.X.toarray().T,  # Transpose to make genes rows and cells columns
        index=sc_adata_filtered.var.index,  # Genes as row index
        columns=sc_adata_filtered.obs['cell_types']  # Phenotypic labels (cell types)
    )
    sc_ref_df_filtered.insert(0, "Gene", sc_ref_df_filtered.index)  # Add gene column
    sc_ref_df_filtered.to_csv(
        os.path.join(args.output_path, "sc_reference_filtered.txt"),
        sep="\t", header=True, index=False, quoting=3
    )

    # Save single-nucleus reference
    sn_ref_df = pd.DataFrame(
        adata_sn_ref.X.toarray().T,  # Transpose to make genes rows and cells columns
        index=adata_sn_ref.var.index,  # Genes as row index
        columns=adata_sn_ref.obs['cell_types']  # Phenotypic labels (cell types)
    )
    sn_ref_df.insert(0, "Gene", sn_ref_df.index)  # Add gene column
    sn_ref_df.to_csv(
        os.path.join(args.output_path, "sn_reference.txt"),
        sep="\t", header=True, index=False, quoting=3
    )

    # And filtered:
    sn_ref_df_filtered = pd.DataFrame(
        sn_adata_filtered.X.toarray().T,  # Transpose to make genes rows and cells columns
        index=sn_adata_filtered.var.index,  # Genes as row index
        columns=sn_adata_filtered.obs['cell_types']  # Phenotypic labels (cell types)
    )
    sn_ref_df_filtered.insert(0, "Gene", sn_ref_df_filtered.index)  # Add gene column
    sn_ref_df_filtered.to_csv(
        os.path.join(args.output_path, "sn_reference_filtered.txt"),
        sep="\t", header=True, index=False, quoting=3
    )

def save_bayesprism_pseudobulks(pseudobulks_df, proportions_df, output_path):
    """
    Save pseudobulks for BayesPrism format:
    - Genes in rows
    - Samples in columns (indexed from 1 to n)
    """
    # Ensure genes are in rows and samples in columns
    pseudobulks_df = pseudobulks_df.T  # Transpose to get genes in rows, samples in columns
    print("Pseudobulks shape: ", pseudobulks_df.shape)

    # Assign sequential sample indices starting from 1
    sample_ids = range(1, pseudobulks_df.shape[1] + 1)
    pseudobulks_df.columns = sample_ids  # Columns should now be sample IDs
    pseudobulks_df.to_csv(os.path.join(output_path, "pseudobulks.csv"))

    # Ensure proportions_df has the correct sample index
    proportions_df.index = sample_ids  # Match proportions index to sample IDs
    proportions_df.to_csv(os.path.join(output_path, "proportions.csv"))

def save_bayesprism_references(adata, output_path, prefix):
    """
    Save BayesPrism reference files:
    1. `_signal.csv` - Gene expression matrix with genes in rows, cell IDs in columns.
    2. `_cell_state.csv` - Metadata table with cell_id, cell_type, cell_subtype, and tumor_state.
    """
    print(f"Saving BayesPrism reference files for {prefix}...")

    # Create _signal.csv (Gene expression matrix)
    if isinstance(adata.X, np.ndarray):
        tosave = adata.X.T
    else:
        tosave = adata.X.toarray().T
    signal_df = pd.DataFrame(
        tosave,  # Transpose to make genes rows and cells columns
        index=adata.var_names,  # Gene names as index
        columns=range(1, adata.n_obs + 1)  # Cell IDs as sequential numbers
    )
    signal_df.index.name = "gene_ids"
    signal_df.to_csv(os.path.join(output_path, f"{prefix}_signal.csv"))

    # Create _cell_state.csv (Metadata)
    cell_state_df = pd.DataFrame({
        "cell_id": range(1, adata.n_obs + 1),
        "cell_type": adata.obs["cell_types"].values,
        "cell_subtype": adata.obs["cell_types"].values,  # Same as cell_type
        "tumor_state": 0  # All cells are non-tumor (0)
    })
    cell_state_df.to_csv(os.path.join(output_path, f"{prefix}_cell_state.csv"), index=False)

def remove_diff_genes(sc_adata, sn_adata, diff_genes):
    """
    Removes differentially expressed genes from the AnnData objects.
    If no differentially expressed genes are found, return original data.
    """
    if not diff_genes:  # Handle the case where no genes are found
        print("No differentially expressed genes found. Skipping gene removal step.")
        return sc_adata, sn_adata

    diff_gene_set = list(set(np.concatenate([df.index.values for df in diff_genes.values()])))

    print("This is diff_gene_set completed:", diff_gene_set)

    print("These are the anndatas before filtering:")
    print(sc_adata.shape)
    print(sn_adata.shape)

    # Filter the AnnData objects
    sn_adata_filtered = sn_adata[:, ~np.isin(sn_adata.var_names, diff_gene_set)].copy()
    sc_adata_filtered = sc_adata[:, ~np.isin(sc_adata.var_names, diff_gene_set)].copy()

    print("These are the anndatas after filtering:")
    print(sc_adata_filtered.shape)
    print(sn_adata_filtered.shape)

    return sc_adata_filtered, sn_adata_filtered

def differential_expression_analysis_parallel(sn_adata, sc_adata, num_threads=4, n_cpus_per_thread=8, deseq_alpha=0.001):
    from multiprocessing import Pool

    common_cell_types = list(set(sn_adata.obs['cell_types']).intersection(sc_adata.obs['cell_types']))
    print(f"Running DESeq2 in parallel for {len(common_cell_types)} cell types...")

    tasks = []
    for ct in common_cell_types:
        # Subset once in the main process
        sn_subset = sn_adata[sn_adata.obs['cell_types'] == ct].copy()
        sc_subset = sc_adata[sc_adata.obs['cell_types'] == ct].copy()
        tasks.append((ct, sn_subset, sc_subset, n_cpus_per_thread, deseq_alpha))

    with Pool(num_threads) as p:
        results = p.starmap(run_deseq2_for_cell_type, tasks)

    diff_genes = {cell_type: res for cell_type, res in results if res is not None}

    return diff_genes

def differential_expression_analysis(sn_adata, sc_adata, deseq_alpha=0.001):
    """
    Perform differential gene expression analysis between corresponding cell types
    in single-nucleus (sn_adata) and single-cell (sc_adata) AnnData objects.

    Parameters:
    - sn_adata: AnnData object for single-nucleus data.
    - sc_adata: AnnData object for single-cell data.

    Returns:
    - diff_genes: Dictionary with cell types as keys and DataFrames of differentially
                  expressed genes as values.
    """

    # Ensure both AnnData objects have the same set of genes
    common_genes = sn_adata.var_names.intersection(sc_adata.var_names)
    sn_adata = sn_adata[:, common_genes]
    sc_adata = sc_adata[:, common_genes]

    # Initialize the dictionary to store results
    diff_genes = {}

    # Identify common cell types
    common_cell_types = set(sn_adata.obs['cell_types']).intersection(sc_adata.obs['cell_types'])

    for cell_type in common_cell_types:

        # Subset data for the current cell type
        sn_cells = sn_adata[sn_adata.obs['cell_types'] == cell_type]
        sc_cells = sc_adata[sc_adata.obs['cell_types'] == cell_type]
        print(f"single nucleus cells of cell type {cell_type}")
        print(sn_cells.shape)
        print(f"single cell cells of cell type {cell_type}")
        print(sc_cells.shape)

        # Convert sparse matrix to dense if necessary
        sn_X = sn_cells.X if issparse(sn_cells.X) else sn_cells.X.tocsc()
        sc_X = sc_cells.X if issparse(sc_cells.X) else sc_cells.X.tocsc()

        sn_df = pd.DataFrame(sn_X, columns=common_genes)
        sc_df = pd.DataFrame(sc_X, columns=common_genes)

        # Combine data into a single DataFrame
        combined_counts = pd.concat([sn_df, sc_df])

        # Create metadata DataFrame
        metadata = pd.DataFrame({
            'condition': ['sn'] * sn_cells.n_obs + ['sc'] * sc_cells.n_obs
        }, index=combined_counts.index)

        # Initialize and run DESeq2 analysis
        inference = DefaultInference(n_cpus=8)
        dds = DeseqDataSet(
            counts=combined_counts.astype(int),
            metadata=metadata,
            design_factors='condition', inference=inference
        )
        dds.deseq2()

        # Extract results
        deseq_stats = DeseqStats(dds, alpha=deseq_alpha)
        deseq_stats.summary()
        results_df = deseq_stats.results_df

        # Filter for significantly differentially expressed genes
        sig_genes = results_df[(results_df['padj'] < deseq_alpha) & (results_df['log2FoldChange'].abs() > 1)]

        # Store results in the dictionary
        diff_genes[cell_type] = sig_genes

    return diff_genes

def run_deseq2_for_cell_type(cell_type, sn_adata, sc_adata, n_cpus=8, deseq_alpha=0.001):
    """Runs DESeq2 for a single cell type without expression-based filtering."""
    print(f"Running DESeq2 for {cell_type}...")

    # Subset the data for the cell type
    sn_cells = sn_adata[sn_adata.obs['cell_types'] == cell_type]
    sc_cells = sc_adata[sc_adata.obs['cell_types'] == cell_type]

    print(f"For {cell_type}: sn_cells={sn_cells.shape}, sc_cells={sc_cells.shape}")

    # Skip if too few cells
    if sn_cells.shape[0] < 10 or sc_cells.shape[0] < 10:
        print(f"Skipping {cell_type}: Too few cells for DE analysis. sn_cells={sn_cells.shape[0]}, sc_cells={sc_cells.shape[0]}")
        return cell_type, None

    # Ensure both datasets have the same set of genes
    common_genes = sn_adata.var_names.intersection(sc_adata.var_names)
    print(f"Cell type {cell_type}: {len(common_genes)} common genes before filtering.")

    if len(common_genes) == 0:
        print(f"Skipping {cell_type}: No common genes found after intersection.")
        return cell_type, None

    sn_cells = sn_cells[:, common_genes]
    sc_cells = sc_cells[:, common_genes]

    # Convert sparse matrices to dense **only if necessary**
    sn_X = sn_cells.X.toarray() if issparse(sn_cells.X) else sn_cells.X
    sc_X = sc_cells.X.toarray() if issparse(sc_cells.X) else sc_cells.X

    # Ensure matrices are 2D
    if sn_X.ndim == 1:
        sn_X = sn_X.reshape(-1, len(common_genes))
    if sc_X.ndim == 1:
        sc_X = sc_X.reshape(-1, len(common_genes))

    print(f"After conversion: sn_X shape = {sn_X.shape}, sc_X shape = {sc_X.shape}")

    # Convert to DataFrames
    try:
        sn_df = pd.DataFrame(sn_X, index=sn_cells.obs_names, columns=common_genes)
        sc_df = pd.DataFrame(sc_X, index=sc_cells.obs_names, columns=common_genes)
    except ValueError as e:
        print(f"Error creating DataFrame for {cell_type}: {e}")
        print(f"sn_X shape: {sn_X.shape}, expected columns: {len(common_genes)}")
        return cell_type, None

    # Combine data into a single DataFrame
    combined_counts = pd.concat([sn_df, sc_df])

    # Metadata for DESeq2
    metadata = pd.DataFrame({'condition': ['sn'] * sn_cells.n_obs + ['sc'] * sc_cells.n_obs}, index=combined_counts.index)

    # Drop zero-variance genes (Retaining all others)
    combined_counts = combined_counts.loc[:, combined_counts.var(axis=0) > 0]
    print(f"Cell type {cell_type}: {combined_counts.shape[1]} genes after zero-variance filtering.")

    if combined_counts.shape[1] == 0:
        print(f"Skipping {cell_type}: No valid genes after zero-variance filtering.")
        return cell_type, None

    print(f"Running DESeq2 for cell type: {cell_type} with {combined_counts.shape[1]} genes.")

    # Run DESeq2
    try:
        inference = DefaultInference(n_cpus=n_cpus)
        dds = DeseqDataSet(counts=combined_counts.astype(int), metadata=metadata, design_factors="condition", inference=inference)
        dds.deseq2()
        deseq_stats = DeseqStats(dds, inference=inference, alpha=deseq_alpha)
        deseq_stats.summary()
        results_df = deseq_stats.results_df

        # Apply differential expression significance filtering
        sig_genes = results_df[(results_df['padj'] < deseq_alpha) & (results_df['log2FoldChange'].abs() > 1)]

        return cell_type, sig_genes

    except Exception as e:
        print(f"Error running DESeq2 for {cell_type}: {str(e)}")
        return cell_type, None

def make_pseudobulks(adata, pseudobulk_config, num_cells, noise, cell_types):

    adata = adata[adata.obs["cell_types"].isin(cell_types)]
    cell_types = adata.obs['cell_types'].unique()
    gene_ids = adata.var.index
    n = len(cell_types)
    pseudobulks = []
    proportions = []

    for prop_type, number_of_bulks in pseudobulk_config.items():
        for _ in range(number_of_bulks):
            if prop_type == 'random':
                prop_vector = np.random.dirichlet(np.ones(n))
                prop_vector = np.round(prop_vector, decimals=5)
                cell_counts = (prop_vector * num_cells).astype(int)
                while np.any(cell_counts == 0):
                    prop_vector = np.random.dirichlet(np.ones(n))
                    prop_vector = np.round(prop_vector, decimals=5)
                    cell_counts = (prop_vector * num_cells).astype(int)
            elif prop_type == 'realistic':
                cell_type_counts = adata.obs['cell_types'].value_counts(normalize=True)
                noise_ = np.random.normal(0, 0.01, cell_type_counts.shape)
                prop_vector = cell_type_counts[cell_types].values + noise_

                prop_vector = np.maximum(prop_vector, 0)
                prop_vector = prop_vector / prop_vector.sum()
                cell_counts = (prop_vector * num_cells).astype(int)
                while np.any(cell_counts == 0):
                    prop_vector = np.random.dirichlet(np.ones(n))
                    prop_vector = prop_vector / prop_vector.sum()
                    cell_counts = (prop_vector * num_cells).astype(int)
            else:
                raise ValueError("Unsupported prop_type.")

            sampled_cells = []
            for cell_type, count in zip(cell_types, cell_counts):
                sampled_cells.append(adata[adata.obs['cell_types'] == cell_type].X[
                    np.random.choice(
                        adata[adata.obs['cell_types'] == cell_type].shape[0],
                        count,
                        replace=len(adata[adata.obs['cell_types'] == cell_type]) < count),
                    :,].toarray())

            pseudobulk = np.sum(np.vstack(sampled_cells), axis=0).astype(float)
            if noise:
                pseudobulk += np.random.normal(0, 0.05, pseudobulk.shape)
                pseudobulk = np.clip(pseudobulk, 0, None)

            pseudobulks.append(pseudobulk)
            proportions.append(prop_vector)

    pseudobulks_df = pd.DataFrame(pseudobulks, columns=gene_ids)
    proportions_df = pd.DataFrame(proportions, columns=cell_types)

    return pseudobulks_df, proportions_df

def make_references(adata_sc_ref, adata_sn, max_cells_per_type=None, seed=42, cell_types=[]):
    """
    Create single-cell and single-nucleus references with an optional limit on the number of cells per type,
    ensuring reproducibility by using a fixed random seed. Cell types with fewer than `min_cells_per_type` are excluded.

    Parameters
    ----------
    adata_sc_ref : AnnData
        Single-cell reference dataset.
    adata_sn : AnnData
        Single-nucleus reference dataset.
    max_cells_per_type : int, optional
        Maximum number of cells to include per cell type. If None, no limit is applied.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    adata_sc_ref : AnnData
        Downsampled single-cell reference dataset.
    adata_sn_ref : AnnData
        Downsampled single-nucleus reference dataset.
    cell_types : list
        List of cell types included in the reference (i.e., those that passed the `min_cells_per_type` filter).
    """
    np.random.seed(seed)  # Set the random seed for reproducibility, same reference always

    adata_sc_ref = adata_sc_ref[adata_sc_ref.obs.cell_types.isin(cell_types)]
    adata_sn = adata_sn[adata_sn.obs.cell_types.isin(cell_types)]

    # Count the number of cells per cell type
    sc_cell_counts = adata_sc_ref.obs['cell_types'].value_counts()
    sn_cell_counts = adata_sn.obs['cell_types'].value_counts()

    # Get the minimum count per cell type
    min_counts = sc_cell_counts.combine(sn_cell_counts, min)

    # Apply max cell limit if specified
    if max_cells_per_type is not None:
        min_counts = min_counts.apply(lambda x: min(x, max_cells_per_type))

    # Downsample the data with reproducibility
    sc_indices = []
    sn_indices = []
    for cell_type, count in min_counts.items():
        # Select random but reproducible indices
        sc_available = adata_sc_ref.obs[adata_sc_ref.obs['cell_types'] == cell_type].index
        sn_available = adata_sn.obs[adata_sn.obs['cell_types'] == cell_type].index
        
        # Use random.choice with seed for reproducibility
        if len(sc_available) > count:
            sc_selected = np.random.choice(sc_available, int(count), replace=False)
        else:
            sc_selected = sc_available

        if len(sn_available) > count:
            sn_selected = np.random.choice(sn_available, int(count), replace=False)
        else:
            sn_selected = sn_available
            
        sc_indices.extend(sc_selected)
        sn_indices.extend(sn_selected)

    # Subset the AnnData objects
    adata_sc_ref = adata_sc_ref[sc_indices].copy()
    adata_sn_ref = adata_sn[sn_indices].copy()

    return adata_sc_ref, adata_sn_ref

def pick_cells(sc_adata_ref, sn_adata, sc_adata_pseudos, min_cells_per_type=75):
    """
    Selects cell types that are present in all three datasets (sc_adata_ref, sn_adata, sc_adata_pseudos)
    and have at least `min_cells_per_type` occurrences in each.

    Parameters
    ----------
    sc_adata_ref : AnnData
        Single-cell reference dataset.
    sn_adata : AnnData
        Single-nucleus reference dataset.
    sc_adata_pseudos : AnnData
        Single-cell pseudobulk dataset.
    min_cells_per_type : int
        Minimum number of cells required per cell type in each dataset.

    Returns
    -------
    valid_cell_types : list
        List of cell types that are present in all datasets and meet the minimum cell count requirement.
    """

    # Count occurrences of each cell type in all three datasets
    sc_ref_counts = sc_adata_ref.obs['cell_types'].value_counts()
    sn_counts = sn_adata.obs['cell_types'].value_counts()
    sc_pseudos_counts = sc_adata_pseudos.obs['cell_types'].value_counts()

    # Align counts so that all datasets have the same cell types
    sc_ref_counts, sn_counts = sc_ref_counts.align(sn_counts, fill_value=0)
    sn_counts, sc_pseudos_counts = sn_counts.align(sc_pseudos_counts, fill_value=0)

    # Find common cell types across all three datasets
    common_cell_types = sc_ref_counts.index.intersection(sn_counts.index).intersection(sc_pseudos_counts.index)

    # Keep only those cell types that meet the min_cells_per_type threshold in all datasets
    valid_cell_types = common_cell_types[
        (sc_ref_counts[common_cell_types] >= min_cells_per_type) &
        (sn_counts[common_cell_types] >= min_cells_per_type) &
        (sc_pseudos_counts[common_cell_types] >= min_cells_per_type)
    ].tolist()

    print("Valid cells found:", valid_cell_types)
    print(sc_ref_counts[common_cell_types])
    print(sc_pseudos_counts[common_cell_types])
    print(sn_counts[common_cell_types])
    assert len(valid_cell_types) > 3, "Less than 4 cell types found"

    return valid_cell_types

def split_single_cell_data(adata_sc, test_ratio=0.3, data_type=""):

    if data_type == "PNB":
        # Data already separated (Same patient for SC and SN references (Train))
        adata_ref = adata_sc[adata_sc.obs.batch == 'SCPCS000108']
        adata_pseudo = adata_sc[adata_sc.obs.batch == 'SCPCS000103']
    else:
        adata_ref = adata_sc[adata_sc.obs.deconvolution == 'reference']
        adata_pseudo = adata_sc[adata_sc.obs.deconvolution == 'pseudobulks']

    # Ensure unique observation names
    adata_pseudo.obs_names_make_unique()
    adata_ref.obs_names_make_unique()

    return adata_pseudo, adata_ref

def prepare_data(res_name, base_path):

    path_x = os.path.join(base_path, res_name)

    adata_path = os.path.join(path_x, f"sc_sn_{res_name}_processed.h5ad")
    print(adata_path)
    # Load the dataset in backed mode
    adata = sc.read_h5ad(adata_path, backed='r')

    # Load the AnnData object into memory
    adata = adata.to_memory()

    # Separate single-cell and single-nucleus data
    adata_sc = adata[adata.obs['data_type'] == 'single_cell'].copy()
    adata_sn = adata[adata.obs['data_type'] == 'single_nucleus'].copy()

    # # Additional filtering for PNB/RMS data
    if res_name == "PNB":
        # Filter SC for Adrenal gland tissue and specific participant IDs
        adata_sc = adata_sc[
            adata_sc.obs['batch'].isin(['SCPCS000108', 'SCPCS000103'])
        ].copy()

        # Filter SN for Adrenal gland tissue and specific participant IDs
        adata_sn = adata_sn[
            (adata_sn.obs['batch'].isin(['SCPCS000111']))
        ].copy()
        # # Filter SC for Adrenal gland tissue and specific participant IDs
        # adata_sc = adata_sc[
        #     adata_sc.obs['batch'].isin(['SCPCS000104', 'SCPCS000105'])
        # ].copy()

        # # Filter SN for Adrenal gland tissue and specific participant IDs
        # adata_sn = adata_sn[
        #     (adata_sn.obs['batch'].isin(['SCPCS000686']))
        # ].copy()

    # Ensure single-cell and single-nucleus have the same cell types
    common_cell_types = set(adata_sc.obs['cell_types']).intersection(set(adata_sn.obs['cell_types']))
    adata_sc = adata_sc[adata_sc.obs['cell_types'].isin(common_cell_types)].copy()
    adata_sn = adata_sn[adata_sn.obs['cell_types'].isin(common_cell_types)].copy()

    adata_sc.obs_names_make_unique()
    adata_sn.obs_names_make_unique()

    return adata_sc, adata_sn

def pick_best_datasets(sc_adatas, sn_adatas, min_cells_per_type=50, min_cell_types=4):
    """
    Selects 2 SC and 2 SN datasets that have the highest number of cells while 
    ensuring that at least 4 cell types meet the minimum required number of cells.

    Parameters
    ----------
    sc_adatas : dict
        Dictionary of single-cell AnnData objects.
    sn_adatas : dict
        Dictionary of single-nucleus AnnData objects.
    min_cells_per_type : int, optional
        Minimum number of cells per cell type required, by default 50.
    min_cell_types : int, optional
        Minimum number of different cell types that meet the threshold, by default 4.

    Returns
    -------
    selected_sc : list
        List of 2 selected SC dataset names.
    selected_sn : list
        List of 2 selected SN dataset names.
    """
    def filter_and_rank_adatas(adatas_dict):
        """
        Filters datasets that have at least `min_cell_types` with at least `min_cells_per_type` 
        and ranks them by total number of cells.
        """
        valid_datasets = []
        for name, adata in adatas_dict.items():
            cell_counts = adata.obs["cell_types"].value_counts()

            # Keep datasets that have at least `min_cell_types` passing the `min_cells_per_type`
            valid_types = cell_counts[cell_counts >= min_cells_per_type]
            if len(valid_types) >= min_cell_types:
                valid_datasets.append((name, adata.shape[0]))  # Store name and total cell count

        # Sort datasets by total number of cells (descending order)
        valid_datasets = sorted(valid_datasets, key=lambda x: x[1], reverse=True)

        return [name for name, _ in valid_datasets]  # Return only dataset names

    # Filter and rank SC and SN datasets
    best_sc_datasets = filter_and_rank_adatas(sc_adatas)
    best_sn_datasets = filter_and_rank_adatas(sn_adatas)

    # Select the top 2 SC and top 2 SN datasets
    selected_sc = best_sc_datasets[:2]
    selected_sn = best_sn_datasets[:2]

    print("Selected SC datasets:", selected_sc)
    print("Selected SN datasets:", selected_sn)

    return selected_sc, selected_sn