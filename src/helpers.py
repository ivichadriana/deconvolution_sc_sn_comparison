"""
Helper functions for processing and analyzing single-cell (scRNA-seq) and single-nucleus (snRNA-seq) data.

Includes utilities for:
- Loading and preprocessing raw and compressed datasets
- Cleaning and harmonizing metadata and cell type annotations
- Generating pseudobulks and reference matrices for deconvolution
- Quality control checks across data sources
- Model training (e.g., scVI) and evaluation metrics

Supports multiple datasets (PNB, OVC, MBC, Real_ADP) and formats outputs for tools like CIBERSORT and BayesPrism.

This script contains helper functions needed throughout the repository.
It should be in /src/ folder and imported as "from src.helpers import the_function"
"""

# Import the dependencies
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData, read_h5ad
import scipy as sp
import scvi
import tarfile
from scipy.sparse import coo_matrix
import collections
import torch
from collections import Counter
import anndata as ad
import gzip
from scipy import sparse
import re
import shutil
import matplotlib.collections as mcoll
from sklearn.utils import resample
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import json
import seaborn as sns
import scanpy as sc
from scipy.sparse import issparse
import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(1, "../../")
sys.path.insert(1, "../")
sys.path.insert(1, "../../../")


def plot_by_dataset(
    string_title="RMSE",
    string_column="RMSE",
    string_ylabel="RMSE",
    df_controls=None,
    dataset_order=None,
):

    # Map reference types to consistent colors
    ref_color_map = {
        "scRNA All (PosCtrl)": "#1f77b4",
        "snRNA All (NegCtrl)": "firebrick",
    }
    if string_column == "RMSE_All":
        plt.figure(figsize=(30, 15))
    else:
        plt.figure(figsize=(30, 13))

    ax = sns.boxplot(
        x="Dataset",
        y=string_column,
        hue="DisplayTransform",
        data=df_controls,
        palette=ref_color_map,
        linewidth=5,
        fliersize=10,
        order=dataset_order,
        hue_order=["scRNA All (PosCtrl)", "snRNA All (NegCtrl)"],
    )
    plt.xlabel("Dataset", fontsize=48, fontweight="bold")
    plt.ylabel(string_ylabel, fontsize=48, fontweight="bold")
    plt.xticks(rotation=0, fontsize=40)
    if string_column == "RMSE_All":
        plt.title(f"{string_title}", fontsize=62, fontweight="bold")
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4], fontsize=32)
        handles, _ = ax.get_legend_handles_labels()
        new_labels = ["scRNA-seq Reference", "snRNA-seq Reference"]
        ax.legend(
            handles=handles,
            labels=new_labels,
            title="Reference Type",
            title_fontproperties={"weight": "bold", "size": 42},
            fontsize=35,
            bbox_to_anchor=(1, -0.15),
        )
    else:
        plt.title(
            f"Deconvolution Performance by Dataset:\n{string_title}",
            fontsize=62,
            fontweight="bold",
        )
        plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=32)
        ax.legend_.remove()
    plt.tight_layout()
    plt.show()


def bootstrap_mean_ci(values, n_boot=5000, alpha=0.05, random_state=None):
    """
    Compute the mean and 95% bootstrap confidence interval (CI) of the mean.
    If all values are constant (as for controls), return that constant and CI equal to that value.
    """
    arr = pd.Series(values).dropna().values
    rng = np.random.default_rng(random_state)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    lower = np.percentile(means, 100 * (alpha / 2.0))
    upper = np.percentile(means, 100 * (1 - alpha / 2.0))
    return pd.Series({"mean": arr.mean(), "ci_lower": lower, "ci_upper": upper})


# Helper: aggregate mean & CI with bootstrap, ordering by 'order'
def aggregate_bootstrap(data, metric, order):
    # metric e.g. 'Pearson_NonRemoved'
    long = (
        data.groupby("DisplayTransform", observed=False)[metric]
        .apply(lambda x: bootstrap_mean_ci(x))
        .apply(pd.Series)
        .reset_index()
    )
    wide = long.pivot(index="DisplayTransform", columns="level_1", values=0)
    wide.columns.name = None
    wide = wide.rename(
        columns={
            "mean": f"{metric}_mean",
            "ci_lower": f"{metric}_ci_lower",
            "ci_upper": f"{metric}_ci_upper",
        }
    )
    # Reindex to ensure transforms appear in the correct display order
    return wide.reindex([t for t in order if t in wide.index])


## To plot the predicted vs. real proportions
def plot_proportions(
    results,
    ds,
    scenario,
    transform_mapping,
    display_order,
    dataset_cell_info,
    baseline_transforms,
    fig_x,
    fig_y,
):
    # 1) always use the same mapping & order
    tm, order = transform_mapping, display_order

    # 2) slice and label
    df = results[ds].copy()

    if "SampleID" in df.columns:
        df["SampleID"] = df["SampleID"].astype(str)
    df = filter_df(df, scenario)
    df["DisplayTransform"] = df["Transform"].map(tm)
    df["DisplayTransform"] = pd.Categorical(
        df["DisplayTransform"], categories=order, ordered=True
    )

    # 3) decide which transforms and marker style
    if scenario == "removed":
        col_order = [t for t in order if t in df["DisplayTransform"].unique()]
        marker_args = dict(s=100, alpha=0.8)
        height = 4
    else:
        col_order = order
        marker_args = dict(alpha=0.65)
        height = 8

    # 4) determine number of columns in the grid
    if scenario == "removed":
        col_wrap = 4
    else:
        col_wrap = 5 if baseline_transforms else 4

    if df.empty or len(col_order) == 0:
        print(f"[plot_proportions] No data for {ds} – {scenario}. Skipping.")
        return
    # 5) build the grid
    g = sns.FacetGrid(
        df,
        col="DisplayTransform",
        col_order=col_order,
        col_wrap=col_wrap,
        hue="CellType",
        hue_order=dataset_cell_info[ds]["cell_order"],
        palette=dataset_cell_info[ds]["palette"],
        height=height,
        aspect=1,
        legend_out=True,
    )
    g.map(sns.scatterplot, "TrueProp", "PredProp", **marker_args)

    # 6) annotate each facet
    for ax, trans in zip(g.axes.flatten(), col_order):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="both", labelsize=30)
        sub = df[df["DisplayTransform"] == trans]
        if len(sub) > 1:
            r, rmse = compute_metrics(sub["TrueProp"], sub["PredProp"])
            text = f"r = {r:.2f}\nRMSE = {rmse:.3f}"
        else:
            text = "r = NA\nRMSE = NA"
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            fontsize=40,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)

    # 7) titles and labels
    title_map = {
        "all": f"{ds} – All: Real vs. Predicted Proportions",
        "non_removed": f"{ds} – Non-Removed Cells: True vs. Predicted",
        "removed": f"{ds} Removed Cells: True vs. Predicted",
    }
    g.set_axis_labels("True Proportion", "Predicted Proportion", fontsize=45)
    g.fig.suptitle(title_map[scenario], y=1, fontsize=55, fontweight="bold")
    g.set_titles("{col_name}", fontsize=50)
    for ax in g.axes.flatten():
        ax.title.set_fontsize(55)

    # 8) legend styling
    g.add_legend(
        title="Cell Type:", fontsize=40, bbox_to_anchor=(1, 0.5), loc="center left"
    )
    leg = g._legend
    leg.set_title("Cell Type:")
    plt.setp(leg.get_title(), fontweight="bold", fontsize=45)
    for txt in leg.get_texts():
        txt.set_fontsize(45)
    for m in leg.findobj(match=mcoll.PathCollection):
        m.set_sizes([500])

    # 9) finalize
    g.fig.set_size_inches(fig_x, fig_y)
    plt.subplots_adjust(right=0.8)
    plt.tight_layout()
    plt.show()


def filter_df(df, scenario):
    ## To filter dataframe by scenario
    if scenario == "all":
        return df.copy()
    if scenario == "non_removed":
        controls = df[df["HoldoutCell"].isna()]
        holdout = df[df["HoldoutCell"].notna() & (df["CellType"] != df["HoldoutCell"])]
        return pd.concat([controls, holdout])
    if scenario == "removed":
        sub = df[df["HoldoutCell"].notna() & (df["CellType"] == df["HoldoutCell"])]
        return sub[~sub["Transform"].isin(["sc_raw", "sn_raw"])]
    raise ValueError(f"Unknown scenario: {scenario}")


def get_donor_views(adata_adipo_all, adata_sn_all, adata_neut, donor, seed=None):
    """
    Returns:
      adata_adipo (sampled to 95),
      adata_sn (this donor),
      sn_missing (pooled neutrophils + sampled adipocytes)
    Behavior is identical to the original inline code.
    """
    if seed is None:
        seed = 42
    adata_adipo = adata_adipo_all[adata_adipo_all.obs["batch"] == donor].copy()
    n_avail = adata_adipo.n_obs
    if n_avail < 95:
        raise ValueError(f"Not enough adipocytes in donor {donor}: {n_avail}")
    sc.pp.subsample(adata_adipo, n_obs=95, random_state=seed)

    adata_sn = adata_sn_all[adata_sn_all.obs["batch"] == donor].copy()

    # IMPORTANT: use the single pre-made 'adata_neut' (already a copy), don't re-copy per donor
    sn_missing = sc.concat([adata_neut, adata_adipo], axis=0, merge="same")
    return adata_adipo, adata_sn, sn_missing


def concat_and_save(adatas, output_path, name):
    # Merge and save reference anndatas
    ref = sc.concat(adatas, axis=0, merge="same")
    save_bayesprism_references(ref, output_path, name)


def set_all_seeds(seed: int = 42):
    import os, random, numpy as np, torch
    import scvi

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    torch.use_deterministic_algorithms(True)
    scvi.settings.seed(seed)


def compute_metrics(y_true, y_pred):
    """
    Compute RMSE and Pearson correlation (or NaN if fewer than 2 points).
    Returns (rmse, pearson).
    """
    rmse = root_mean_squared_error(y_true, y_pred)
    if len(y_true) < 2:
        pearson = np.nan
    else:
        pearson, _ = pearsonr(y_true, y_pred)
    return rmse, pearson


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
    missing_cells = (
        adata.obs["cell_types"].isin(unwanted_types) | adata.obs["cell_types"].isna()
    )

    # Remove unassigned/missing cells
    adata = adata[~missing_cells].copy()

    # Log the change
    removed_count = initial_count - adata.n_obs
    print(
        f"{dataset_name}: Removed {removed_count} unassigned/missing cells. Remaining: {adata.n_obs}"
    )

    return adata


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


def extract_tar_if_needed(tar_path: str, extract_to: str):
    """
    Extracts the given .tar file if it hasn't been extracted.

    Parameters
    ----------
    tar_path : str
        Path to the .tar file.
    extract_to : str
        Directory where data should be extracted.
    """
    # Check if the expected files already exist
    extracted_files = [
        f for f in os.listdir(extract_to) if f.endswith(".gz") or f.endswith(".h5")
    ]

    if extracted_files:
        print(f"Skipping extraction: Data already exists in {extract_to}.")
        return  # If files exist, no need to extract

    if os.path.exists(tar_path):
        print(f"Extracting {tar_path} to {extract_to}...")
        try:
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(path=extract_to)
            print("Extraction complete.")
        except Exception as e:
            print(f"Error extracting {tar_path}: {e}")
    else:
        print(f"ERROR: {tar_path} does not exist. Please check the file path.")


def load_PNB_data(data_type: str, load_testing: bool = False):
    """
    Open the data of PNB. We have 1 SN and 1 SC from same patient, and 1 SC from another for testing.

    Parameters
    ----------
    data_type : str (either "single_nucleus" or "single_cell")

    Returns
    -------
    AnnData
        The AnnData object.
    Metadata
        Pandas DataFrame.
    """
    res_name = "PNB"
    adata = []
    meta_data = []
    base_path = f"{os.getcwd()}/../data/{res_name}/"
    tar_file = (
        f"{os.getcwd()}/../data/{res_name}/GSE140819_RAW.tar"  # Path to the .tar file
    )

    # Ensure the tar file is extracted
    extract_tar_if_needed(tar_file, base_path)

    if data_type == "single_cell":
        if load_testing:
            # File paths
            h5_gz_file = Path(
                base_path,
                "GSM4186962_HTAPP-312-SMP-902_fresh-C4-T2_channel1_raw_gene_bc_matrices_h5.h5.gz",
            )
            h5_file = Path(
                base_path,
                "GSM4186962_HTAPP-312-SMP-902_fresh-C4-T2_channel1_raw_gene_bc_matrices_h5.h5",
            )
            csv_file = Path(
                base_path,
                "GSM4186962_metadata_HTAPP-312-SMP-902_fresh-C4-T2_channel1.csv.gz",
            )
        else:
            h5_gz_file = Path(
                base_path,
                "GSM4186963_HTAPP-656-SMP-3481_fresh-T1_channel1_raw_gene_bc_matrices_h5.h5.gz",
            )
            h5_file = Path(
                base_path,
                "GSM4186963_HTAPP-656-SMP-3481_fresh-T1_channel1_raw_gene_bc_matrices_h5.h5",
            )
            csv_file = Path(
                base_path,
                "GSM4186963_metadata_HTAPP-656-SMP-3481_fresh-T1_channel1.csv.gz",
            )

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
        h5_gz_file = Path(
            base_path,
            "GSM4186974_HTAPP-963-SMP-4741_fresh_channel1_raw_gene_bc_matrices_h5.h5.gz",
        )
        h5_file = Path(
            base_path,
            "GSM4186974_HTAPP-963-SMP-4741_fresh_channel1_raw_gene_bc_matrices_h5.h5",
        )
        csv_file = Path(
            base_path, "GSM4186974_metadata_HTAPP-963-SMP-4741_fresh_channel1.csv.gz"
        )

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


def load_OVC_data(data_type: str, load_testing: bool = False):
    """
    Open the data of OVC. We have 1 SN and 1 SC from same protocol (resection)
    and 1 SC from another for testing (ascites).

    Parameters
    ----------
    data_type : str (either "single_nucleus" or "single_cell")

    Returns
    -------
    AnnData
        The AnnData object.
    Metadata
        Pandas DataFrame.
    """
    res_name = "OVC"
    adata = []
    meta_data = []
    base_path = f"{os.getcwd()}/../data/{res_name}/"
    tar_file = (
        f"{os.getcwd()}/../data/{res_name}/GSE140819_RAW.tar"  # Path to the .tar file
    )

    # Ensure the tar file is extracted
    extract_tar_if_needed(tar_file, base_path)

    if data_type == "single_cell":
        if load_testing:
            # File paths
            h5_gz_file = Path(
                base_path,
                "GSM4186986_HTAPP-727-SMP-3781_fresh-CD45neg-T1_channel1_raw_gene_bc_matrices_h5.h5.gz",
            )
            h5_file = Path(
                base_path,
                "GSM4186986_HTAPP-727-SMP-3781_fresh-CD45neg-T1_channel1_raw_gene_bc_matrices_h5.h5",
            )
            csv_file = Path(
                base_path,
                "GSM4186986_metadata_HTAPP-727-SMP-3781_fresh-CD45neg-T1_channel1.csv.gz",
            )
        else:
            h5_gz_file = Path(
                base_path,
                "GSM4186985_HTAPP-624-SMP-3212_fresh_channel1_raw_feature_bc_matrix.h5.gz",
            )
            h5_file = Path(
                base_path,
                "GSM4186985_HTAPP-624-SMP-3212_fresh_channel1_raw_feature_bc_matrix.h5",
            )
            csv_file = Path(
                base_path,
                "GSM4186985_metadata_HTAPP-624-SMP-3212_fresh_channel1.csv.gz",
            )

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
        # if load_testing: ## this is the CST
        #     # File paths
        #     h5_gz_file = Path(base_path, "GSM4186987_HTAPP-316-SMP-991_CST_channel1_raw_gene_bc_matrices_h5.h5.gz")
        #     h5_file = Path(base_path, "GSM4186987_HTAPP-316-SMP-991_CST_channel1_raw_gene_bc_matrices_h5.h5")
        #     csv_file = Path(base_path, "GSM4186987_metadata_HTAPP-316-SMP-991_CST_channel1.csv.gz")
        # else: ## this is the TST
        h5_gz_file = Path(
            base_path,
            "GSM4186987_HTAPP-316-SMP-991_CST_channel1_raw_gene_bc_matrices_h5.h5.gz",
        )
        h5_file = Path(
            base_path,
            "GSM4186987_HTAPP-316-SMP-991_CST_channel1_raw_gene_bc_matrices_h5.h5",
        )
        csv_file = Path(
            base_path, "GSM4186987_metadata_HTAPP-316-SMP-991_CST_channel1.csv.gz"
        )

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
    Open the data of MBC. We have 1 SN and 1 SC from same patient, and 1 SC from another for testing.

    Parameters
    ----------
    data_type : str (either "single_nucleus" or "single_cell")

    Returns
    -------
    AnnData
        The AnnData object.
    Metadata
        Pandas DataFrame.
    """
    res_name = "MBC"
    adata = []
    meta_data = []
    base_path = f"{os.getcwd()}/../data/{res_name}/"  # Path to original data.
    tar_file = (
        f"{os.getcwd()}/../data/{res_name}/GSE140819_RAW.tar"  # Path to the .tar file
    )

    # Ensure the tar file is extracted
    extract_tar_if_needed(tar_file, base_path)

    if data_type == "single_cell":
        if load_testing:
            # File paths
            h5_gz_file = Path(
                base_path,
                "GSM4186973_HTAPP-285-SMP-751_fresh_channel1_raw_gene_bc_matrices_h5.h5.gz",
            )
            h5_file = Path(
                base_path,
                "GSM4186973_HTAPP-285-SMP-751_fresh_channel1_raw_gene_bc_matrices_h5.h5",
            )
            csv_file = Path(
                base_path, "GSM4186973_metadata_HTAPP-285-SMP-751_fresh_channel1.csv.gz"
            )
        else:
            h5_gz_file = Path(
                base_path,
                "GSM4186974_HTAPP-963-SMP-4741_fresh_channel1_raw_gene_bc_matrices_h5.h5.gz",
            )
            h5_file = Path(
                base_path,
                "GSM4186974_HTAPP-963-SMP-4741_fresh_channel1_raw_gene_bc_matrices_h5.h5",
            )
            csv_file = Path(
                base_path,
                "GSM4186974_metadata_HTAPP-963-SMP-4741_fresh_channel1.csv.gz",
            )

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

        h5_gz_file = Path(
            base_path,
            "GSM4186979_HTAPP-963-SMP-4741_CST_channel2_raw_gene_bc_matrices_h5.h5.gz",
        )
        h5_file = Path(
            base_path,
            "GSM4186979_HTAPP-963-SMP-4741_CST_channel2_raw_gene_bc_matrices_h5.h5",
        )
        csv_file = Path(
            base_path, "GSM4186979_metadata_HTAPP-963-SMP-4741_CST_channel2.csv.gz"
        )

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
    assert (
        sc_cell_counts == sn_cell_counts
    ).all(), "Number of cells per cell type in references do not match!"
    print(
        "Number of cells per cell type match between single-cell and single-nucleus references."
    )


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
    assert np.allclose(
        proportion_sums, 1, atol=1e-3
    ), "Proportions do not sum to 1 for some pseudobulks!"
    print("Proportions sum to 1 for all pseudobulks.")


def qc_check_cell_types_match(
    original_anndata_path, sc_ref_path, sn_ref_path, pseudobulk_path
):
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
    sc_cell_types = {
        col.split(".")[0] for col in sc_ref.columns[1:]
    }  # Remove numerical suffixes
    sn_cell_types = {
        col.split(".")[0] for col in sn_ref.columns[1:]
    }  # Remove numerical suffixes
    pseudobulk_cell_types = set(
        pseudobulks.columns[1:]
    )  # Pseudobulk cell types don't have suffixes

    # Find common cell types across all datasets
    common_cell_types = original_cell_types.intersection(
        sc_cell_types, sn_cell_types, pseudobulk_cell_types
    )

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


def save_cibersort(
    pseudobulks_df=[],
    proportions_df=[],
    adata_sc_ref=[],
    sc_adata_filtered=[],
    adata_sn_ref=[],
    sn_adata_filtered=[],
):
    # Save pseudobulks
    pseudobulks_df = (
        pseudobulks_df.T
    )  # Transpose to make genes rows and pseudobulks columns
    pseudobulks_df.index.name = "Gene"  # Set the index as "Gene"
    pseudobulks_df.to_csv(
        os.path.join(args.output_path, "pseudobulks.txt"),
        sep="\t",
        header=True,
        index=True,
        quoting=3,  # Save with the gene names as row index
    )
    # Save proportions
    proportions_df.index.name = "Pseudobulk"  # Name the first column as Pseudobulk
    proportions_df.reset_index(inplace=True)  # Reset index to move it into the table
    proportions_df.to_csv(
        os.path.join(args.output_path, "proportions.txt"),
        sep="\t",
        header=True,
        index=False,
        quoting=3,
    )
    # Save single-cell references
    sc_ref_df = pd.DataFrame(
        adata_sc_ref.X.toarray().T,  # Transpose to make genes rows and cells columns
        index=adata_sc_ref.var.index,  # Genes as row index
        columns=adata_sc_ref.obs["cell_types"],  # Phenotypic labels (cell types)
    )
    sc_ref_df.insert(0, "Gene", sc_ref_df.index)  # Add gene column
    sc_ref_df.to_csv(
        os.path.join(args.output_path, "sc_reference.txt"),
        sep="\t",
        header=True,
        index=False,
        quoting=3,
    )

    # And filtered:
    sc_ref_df_filtered = pd.DataFrame(
        sc_adata_filtered.X.toarray().T,  # Transpose to make genes rows and cells columns
        index=sc_adata_filtered.var.index,  # Genes as row index
        columns=sc_adata_filtered.obs["cell_types"],  # Phenotypic labels (cell types)
    )
    sc_ref_df_filtered.insert(0, "Gene", sc_ref_df_filtered.index)  # Add gene column
    sc_ref_df_filtered.to_csv(
        os.path.join(args.output_path, "sc_reference_filtered.txt"),
        sep="\t",
        header=True,
        index=False,
        quoting=3,
    )

    # Save single-nucleus reference
    sn_ref_df = pd.DataFrame(
        adata_sn_ref.X.toarray().T,  # Transpose to make genes rows and cells columns
        index=adata_sn_ref.var.index,  # Genes as row index
        columns=adata_sn_ref.obs["cell_types"],  # Phenotypic labels (cell types)
    )
    sn_ref_df.insert(0, "Gene", sn_ref_df.index)  # Add gene column
    sn_ref_df.to_csv(
        os.path.join(args.output_path, "sn_reference.txt"),
        sep="\t",
        header=True,
        index=False,
        quoting=3,
    )

    # And filtered:
    sn_ref_df_filtered = pd.DataFrame(
        sn_adata_filtered.X.toarray().T,  # Transpose to make genes rows and cells columns
        index=sn_adata_filtered.var.index,  # Genes as row index
        columns=sn_adata_filtered.obs["cell_types"],  # Phenotypic labels (cell types)
    )
    sn_ref_df_filtered.insert(0, "Gene", sn_ref_df_filtered.index)  # Add gene column
    sn_ref_df_filtered.to_csv(
        os.path.join(args.output_path, "sn_reference_filtered.txt"),
        sep="\t",
        header=True,
        index=False,
        quoting=3,
    )


def save_bayesprism_pseudobulks(pseudobulks_df, proportions_df, output_path):
    """
    Save pseudobulks for BayesPrism format:
    - Genes in rows
    - Samples in columns (indexed from 1 to n)
    """
    # Ensure genes are in rows and samples in columns
    pseudobulks_df = (
        pseudobulks_df.T
    )  # Transpose to get genes in rows, samples in columns
    print("Pseudobulks shape: ", pseudobulks_df.shape)

    # Assign sequential sample indices starting from 1
    sample_ids = range(1, pseudobulks_df.shape[1] + 1)
    pseudobulks_df.columns = sample_ids  # Columns should now be sample IDs
    pseudobulks_df.to_csv(os.path.join(output_path, "pseudobulks.csv"))

    # Ensure proportions_df has the correct sample index
    proportions_df.index = sample_ids  # Match proportions index to sample IDs
    proportions_df.to_csv(os.path.join(output_path, "proportions.csv"))


def save_bayesprism_realbulks(bulks_df, output_path):
    """
    Save pseudobulks for BayesPrism format:
    - Genes in rows
    - Samples in columns (indexed from 1 to n)
    """
    # Ensure genes are in rows and samples in columns
    bulks_df = bulks_df.T  # Transpose to get genes in rows, samples in columns
    print("Bulks shape: ", bulks_df.shape)

    # Assign sequential sample indices starting from 1
    sample_ids = range(1, bulks_df.shape[1] + 1)
    bulks_df.columns = sample_ids  # Columns should now be sample IDs
    bulks_df.to_csv(os.path.join(output_path, "processed_bulks.csv"))


def save_comparison_pseudobulks(pseudobulks_df, proportions_df, output_path, name):
    """
    Save pseudobulks for later comparison with real bulks

    """
    # Ensure order
    print("Pseudobulks shape: ", pseudobulks_df.shape)

    # Assign sequential sample indices starting from 1
    pseudobulks_df.to_csv(os.path.join(output_path, f"pseudobulks_{name}.csv"))

    # Ensure proportions_df has the correct sample index
    proportions_df.index = pseudobulks_df.index  # Match proportions index to sample IDs
    proportions_df.to_csv(os.path.join(output_path, f"proportions_{name}.csv"))


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
        columns=range(1, adata.n_obs + 1),  # Cell IDs as sequential numbers
    )
    signal_df.index.name = "gene_ids"
    signal_df.to_csv(os.path.join(output_path, f"{prefix}_signal.csv"))

    # Create _cell_state.csv (Metadata)
    cell_state_df = pd.DataFrame(
        {
            "cell_id": range(1, adata.n_obs + 1),
            "cell_type": adata.obs["cell_types"].values,
            "cell_subtype": adata.obs["cell_types"].values,  # Same as cell_type
            "tumor_state": 0,  # All cells are non-tumor (0)
        }
    )
    cell_state_df.to_csv(
        os.path.join(output_path, f"{prefix}_cell_state.csv"), index=False
    )


def make_pseudobulks(adata, pseudobulk_config, num_cells, noise, cell_types):

    adata = adata[adata.obs["cell_types"].isin(cell_types)]
    cell_types = adata.obs["cell_types"].unique()
    gene_ids = adata.var.index
    n = len(cell_types)
    pseudobulks = []
    proportions = []

    for prop_type, number_of_bulks in pseudobulk_config.items():
        for _ in range(number_of_bulks):
            if prop_type == "random":
                prop_vector = np.random.dirichlet(np.ones(n))
                prop_vector = np.round(prop_vector, decimals=5)
                cell_counts = (prop_vector * num_cells).astype(int)
                while np.any(cell_counts == 0):
                    prop_vector = np.random.dirichlet(np.ones(n))
                    prop_vector = np.round(prop_vector, decimals=5)
                    cell_counts = (prop_vector * num_cells).astype(int)
            elif prop_type == "realistic":
                cell_type_counts = adata.obs["cell_types"].value_counts(normalize=True)
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
                sampled_cells.append(
                    adata[adata.obs["cell_types"] == cell_type]
                    .X[
                        np.random.choice(
                            adata[adata.obs["cell_types"] == cell_type].shape[0],
                            count,
                            replace=len(adata[adata.obs["cell_types"] == cell_type])
                            < count,
                        ),
                        :,
                    ]
                    .toarray()
                )

            pseudobulk = np.sum(np.vstack(sampled_cells), axis=0).astype(float)
            if noise:
                pseudobulk += np.random.normal(0, 0.05, pseudobulk.shape)
                pseudobulk = np.clip(pseudobulk, 0, None)

            pseudobulks.append(pseudobulk)
            proportions.append(prop_vector)

    pseudobulks_df = pd.DataFrame(pseudobulks, columns=gene_ids)
    proportions_df = pd.DataFrame(proportions, columns=cell_types)

    return pseudobulks_df, proportions_df


def make_references(
    adata_sc_ref, adata_sn, max_cells_per_type=None, seed=42, cell_types=[]
):
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
    np.random.seed(
        seed
    )  # Set the random seed for reproducibility, same reference always

    adata_sc_ref = adata_sc_ref[adata_sc_ref.obs.cell_types.isin(cell_types)]
    adata_sn = adata_sn[adata_sn.obs.cell_types.isin(cell_types)]

    # Count the number of cells per cell type
    sc_cell_counts = adata_sc_ref.obs["cell_types"].value_counts()
    sn_cell_counts = adata_sn.obs["cell_types"].value_counts()

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
        sc_available = adata_sc_ref.obs[
            adata_sc_ref.obs["cell_types"] == cell_type
        ].index
        sn_available = adata_sn.obs[adata_sn.obs["cell_types"] == cell_type].index

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


def pick_cells(
    sc_adata_ref,
    sn_adata,
    sc_adata_pseudos,
    min_cells_per_type=50,
    check_min_cells=True,
):
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
    sc_ref_counts = sc_adata_ref.obs["cell_types"].value_counts()
    sn_counts = sn_adata.obs["cell_types"].value_counts()
    sc_pseudos_counts = sc_adata_pseudos.obs["cell_types"].value_counts()

    # Align counts so that all datasets have the same cell types
    sc_ref_counts, sn_counts = sc_ref_counts.align(sn_counts, fill_value=0)
    sn_counts, sc_pseudos_counts = sn_counts.align(sc_pseudos_counts, fill_value=0)

    # Find common cell types across all three datasets
    common_cell_types = sc_ref_counts.index.intersection(sn_counts.index).intersection(
        sc_pseudos_counts.index
    )

    # Keep only those cell types that meet the min_cells_per_type threshold in all datasets
    valid_cell_types = common_cell_types[
        (sc_ref_counts[common_cell_types] >= min_cells_per_type)
        & (sn_counts[common_cell_types] >= min_cells_per_type)
        & (sc_pseudos_counts[common_cell_types] >= min_cells_per_type)
    ].tolist()

    print("Valid cells found:", valid_cell_types)
    print(sc_ref_counts[common_cell_types])
    print(sc_pseudos_counts[common_cell_types])
    print(sn_counts[common_cell_types])
    if check_min_cells:
        assert len(valid_cell_types) > 3, "Less than 4 cell types found"

    return valid_cell_types


def split_single_cell_data(adata_sc, test_ratio=0.3, data_type=""):

    if data_type == "PNB":
        # Data already separated (Same patient for SC and SN references (Train))
        adata_ref = adata_sc[adata_sc.obs.batch == "SCPCS000108"]
        adata_pseudo = adata_sc[adata_sc.obs.batch == "SCPCS000103"]
    else:
        adata_ref = adata_sc[adata_sc.obs.deconvolution == "reference"]
        adata_pseudo = adata_sc[adata_sc.obs.deconvolution == "pseudobulks"]

    # Ensure unique observation names
    adata_pseudo.obs_names_make_unique()
    adata_ref.obs_names_make_unique()

    return adata_pseudo, adata_ref


def prepare_data(res_name, base_path):

    path_x = os.path.join(base_path, res_name)

    adata_path = os.path.join(path_x, f"sc_sn_{res_name}_processed.h5ad")
    print(adata_path)
    # Load the dataset in backed mode
    adata = sc.read_h5ad(adata_path, backed="r")

    # Load the AnnData object into memory
    adata = adata.to_memory()

    # Separate single-cell and single-nucleus data
    adata_sc = adata[adata.obs["data_type"] == "single_cell"].copy()
    adata_sn = adata[adata.obs["data_type"] == "single_nucleus"].copy()

    # Ensure single-cell and single-nucleus have the same cell types
    common_cell_types = set(adata_sc.obs["cell_types"]).intersection(
        set(adata_sn.obs["cell_types"])
    )
    adata_sc = adata_sc[adata_sc.obs["cell_types"].isin(common_cell_types)].copy()
    adata_sn = adata_sn[adata_sn.obs["cell_types"].isin(common_cell_types)].copy()

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
                valid_datasets.append(
                    (name, adata.shape[0])
                )  # Store name and total cell count

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


def downsample_cells_by_type(adata, max_cells=1500, cell_type_key="cell_types"):
    """
    Downsamples an AnnData object so that each unique cell type (as given by
    the cell_type_key column in adata.obs) has at most max_cells cells.

    Parameters
    ----------
    adata : AnnData
        The input AnnData object.
    max_cells : int, optional
        Maximum number of cells to retain per cell type (default is 1500).
    cell_type_key : str, optional
        The key in adata.obs that defines the cell types (default is "cell_types").

    Returns
    -------
    AnnData
        A new AnnData object containing the downsampled data.
    """
    # List to store indices of cells to keep
    selected_indices = []

    # Loop over each unique cell type
    for cell_type in adata.obs[cell_type_key].unique():
        # Get the indices of cells for the current cell type
        cell_indices = adata.obs.index[adata.obs[cell_type_key] == cell_type]

        # If there are more than max_cells, sample max_cells randomly
        if len(cell_indices) > max_cells:
            sampled = np.random.choice(cell_indices, size=max_cells, replace=False)
        else:
            sampled = cell_indices
        selected_indices.extend(sampled)

    # Return a new AnnData object with the selected cells
    return adata[selected_indices].copy()


def compute_metrics(y_true, y_pred):
    """Compute Pearson correlation and RMSE between two arrays."""
    rmse = root_mean_squared_error(y_true, y_pred)
    if len(y_true) < 2:
        pearson = np.nan
    else:
        pearson, _ = pearsonr(y_true, y_pred)
    return pearson, rmse


def plot_controls(
    string_title="RMSE",
    string_column="RMSE",
    string_ylabel="RMSE",
    df_controls={},
    control_colors={},
    dataset_palette={},
):

    # Create the boxplot for controls using our control colors.
    plt.figure(figsize=(18, 12))
    sns.boxplot(
        x="DisplayTransform",
        y=string_column,
        data=df_controls,
        palette=control_colors,
        linewidth=7,
        fliersize=10,
    )
    plt.title(
        f"Deconvolution Performance:\n{string_title}", fontsize=56, fontweight="bold"
    )
    plt.xlabel("Reference Data Type", fontsize=50, fontweight="bold")
    plt.ylabel(string_ylabel, fontsize=48, fontweight="bold")
    plt.xticks([0, 1], ["scRNA Ref.", "snRNA Ref."], fontsize=55)
    if string_column == "RMSE_All":
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4], fontsize=45)
    else:
        plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=45)
    plt.tight_layout()
    plt.show()

    # Create the boxplot for controls using our control colors.
    fig, ax = plt.subplots(figsize=(18, 12))
    sns.boxplot(
        x="DisplayTransform",
        y=string_column,
        hue="Dataset",
        data=df_controls,
        palette=dataset_palette,  # dict mapping each dataset name → color
        linewidth=7,
        fliersize=10,
        ax=ax,
        dodge=True,  # separate positions for each hue
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles,
        labels=labels,
        title="Dataset",
        title_fontsize=36,  # larger title
        fontsize=32,  # larger text
        loc="upper left",
        bbox_to_anchor=(1.02, 1),  # move it outside the main plot
    )

    plt.title(
        f"Deconvolution Performance:\n{string_title}", fontsize=56, fontweight="bold"
    )
    plt.xlabel("Reference Data Type", fontsize=50, fontweight="bold")
    plt.ylabel(string_ylabel, fontsize=48, fontweight="bold")
    if string_column == "RMSE_All":
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4], fontsize=45)
    else:
        plt.yticks([-1, -0.5, 0, 0.5, 1], fontsize=45)
    plt.xticks([0, 1], ["scRNA Ref.", "snRNA Ref."], fontsize=55)
    plt.tight_layout()
    plt.show()


# --- Functions for computing metrics ---
def compute_average_expression(matrix):
    """
    Compute the average expression vector (per gene) from a DataFrame
    where rows = genes and columns = cells.
    """
    return matrix.mean(axis=1).values.flatten()


def compute_cosine_similarity(matrix1, matrix2):
    """
    Compute cosine similarity between the average gene expression vectors
    of two DataFrames (genes as rows, cells as columns).
    """
    vec1 = compute_average_expression(matrix1)
    vec2 = compute_average_expression(matrix2)
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def compute_pearson_similarity(matrix1, matrix2):
    """
    Compute Pearson correlation between the average gene expression vectors
    of two DataFrames (genes as rows, cells as columns).
    """
    vec1 = compute_average_expression(matrix1)
    vec2 = compute_average_expression(matrix2)
    corr, _ = pearsonr(vec1, vec2)
    return corr


def scvi_train_model(
    model_save_path,
    sc_adata,
    sn_adata,
    heldout_label=None,
    conditional=True,
    remove_degs=False,
    degs=[],
):
    """
    Example function that:
    1) Concatenates SC + SN excluding the 'heldout_label' if specified
    2) Optionally removes DEGs
    3) Trains scVI
        - conditional => batch_key='data_type'
        - not conditional => batch_key=None
    4) Returns the trained model
    """

    # 1) Concatenate
    combined_adata = sc.concat([sc_adata, sn_adata], axis=0, merge="same")
    if heldout_label is not None:
        combined_adata = combined_adata[
            combined_adata.obs["cell_types"] != heldout_label
        ].copy()

    # 2) Remove DEGs if needed
    if remove_degs and len(degs) > 0:
        # Keep only genes not in `degs`
        keep_genes = [g for g in combined_adata.var_names if g not in degs]
        combined_adata = combined_adata[:, keep_genes].copy()

    if conditional:
        scvi.model.SCVI.setup_anndata(combined_adata, batch_key="data_type")
    else:
        scvi.model.SCVI.setup_anndata(combined_adata, batch_key=None)

    # 3) scVI setup
    # Check if a trained model exists, training otherwise
    if os.path.exists(model_save_path):
        print(f"Loading pre-trained scVI model from {model_save_path}...")
        model = scvi.model.SCVI.load(model_save_path, adata=combined_adata)
    else:
        if conditional:
            model = scvi.model.SCVI(
                combined_adata,
                encode_covariates=True,
                deeply_inject_covariates=True,
                n_layers=2,
                n_latent=30,
                dispersion="gene-batch",
                gene_likelihood="nb",
            )
        else:
            model = scvi.model.SCVI(
                combined_adata, n_layers=2, n_latent=30, gene_likelihood="nb"
            )
        # 4) Train
        model.view_anndata_setup()
        model.train(early_stopping=True, early_stopping_patience=10)
        model.save(model_save_path, overwrite=True)

    return model


def open_adipose_datasets_all(res_name="Real_ADP", base_dir=".."):
    """Open datasets and return them processed and in order"""

    # Define file paths
    bulk_file = f"{base_dir}/data/{res_name}/adipose_bulks_processed.csv"
    sn_sc_file = f"{base_dir}/data/{res_name}/sc_sn_{res_name}_allcells.h5ad"

    # 1. Read in the data
    bulk_df = pd.read_csv(bulk_file, index_col=0)
    print("Bulk data shape:", bulk_df.shape)

    adata = sc.read_h5ad(sn_sc_file)
    print("Combined SN+SC data shape:", adata.shape)

    # Match genes between datasets
    # Assume that bulk_df columns are gene names. Identify the common gene set.
    common_genes = list(set(bulk_df.columns).intersection(set(adata.var_names)))
    print("Number of common genes:", len(common_genes))

    adata = adata[:, common_genes].copy()
    bulk_df = bulk_df[common_genes].copy()

    # 3. Separate SN and SC data based on the 'data_type' column
    sc_adata = adata[adata.obs["data_type"] == "single_cell"].copy()
    sn_adata = adata[adata.obs["data_type"] == "single_nucleus"].copy()

    del adata

    print("Single-cell data shape:", sc_adata.shape)
    print("Single-nucleus data shape:", sn_adata.shape)
    print("Bulk data shape:", bulk_df.shape)
    print("Cell Type counts:")
    print(sc_adata.obs.cell_types.value_counts())
    print(sn_adata.obs.cell_types.value_counts())

    matching = np.intersect1d(
        sn_adata.obs.cell_types.unique(), sc_adata.obs.cell_types.unique()
    )
    matching

    nonmatching = np.setxor1d(
        sn_adata.obs.cell_types.unique(), sc_adata.obs.cell_types.unique()
    )
    nonmatching

    sn_adata_match = sn_adata[sn_adata.obs.cell_types.isin(matching)].copy()

    sn_missing = {}
    sn_miss_adata = sn_adata[sn_adata.obs.cell_types.isin(nonmatching)].copy()
    sn_missing[0] = sn_miss_adata[sn_miss_adata.obs.cell_types == nonmatching[0]].copy()
    sn_missing[1] = sn_miss_adata[sn_miss_adata.obs.cell_types == nonmatching[1]].copy()

    print(sn_adata_match.obs.cell_types.value_counts())

    sn_adata = sn_adata_match.copy()

    print(sn_miss_adata.obs.cell_types.value_counts())

    return bulk_df, sc_adata, sn_adata, sn_missing
