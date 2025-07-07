"""
Helper functions for transforming single-nucleus (snRNA-seq) data into single-cell (scRNA-seq)-like expression profiles.

Includes methods based on:
- scVI latent space shifting
- PCA-based global and local transformations
- Library size normalization and log/exp transformations

Used for benchmarking and improving cross-modality integration and deconvolution tasks by simulating scRNA-like profiles from snRNA data.

It should be in /src/ folder and imported as "from src.transforms import the_function"

"""

# import the dependencies
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
from typing import Optional
import torch
from collections import Counter
import anndata as ad
import gzip
from scipy import sparse
import re
import shutil
from sklearn.utils import resample
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pickle
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


def transform_heldout_sn_to_mean_sc_VAE(
    model: scvi.model.SCVI,
    sc_adata: sc.AnnData,
    sn_adata: sc.AnnData,
    sn_heldout_adata: sc.AnnData,
    heldout_label: Optional[str] = None,
    k_neighbors: int = 10,
):
    """
    Shift single-nucleus (SN) held-out cells into a single-cell (SC)-like expression space using a local neighbor-based
    transformation in the scVI latent space.

    This function aims to transform the held-out SN cells so that their latent representations resemble those of SC cells.
    It does so by:

    1. **Gene Alignment**: Subsets each AnnData object to the set of genes common to sc_adata, sn_adata, and sn_heldout_adata.
    2. **Latent Representation**: Uses the provided scVI model to obtain latent embeddings (z_sc, z_sn, and z_ho) for the SC data,
       the non-held-out SN data, and the held-out SN data, respectively.
    3. **Local Shift Computation**:
       - Computes a shift vector for each non-held-out SN cell as (mean latent SC) – (latent SN).
         The mean latent SC is the centroid (mean) of the SC embeddings.
       - For each held-out SN cell, finds its k nearest neighbors (in latent space) among the non-held-out SN embeddings
         and averages their shift vectors. This local shift is then added to the held-out cell’s latent representation,
         yielding a “SC-like” latent position.
    4. **Decode to Expression**: After updating the held-out SN cells’ latent representations, it decodes these shifted
       embeddings back to gene expression space with a library size scaled to the median of the SC data.
    5. **Output**: Returns a DataFrame of the transformed expression matrix (rows = held-out SN cells, columns = genes).

    Parameters
    ----------
    model : scvi.model.SCVI
        A trained scVI model (non-conditional). It should be fit on SC plus SN minus the held-out cell type.
    sc_adata : sc.AnnData
        Single-cell AnnData, excluding the held-out cell type. Used to define the SC latent centroid and library size.
    sn_adata : sc.AnnData
        Single-nucleus AnnData, excluding the held-out cell type. Used to build the neighbor graph and compute local shifts.
    sn_heldout_adata : sc.AnnData
        The held-out single-nucleus cells, which will be transformed into a SC-like latent space.
    heldout_label : str
        The label/name of the held-out cell type. Not currently used inside the function,
        but kept for record or future reference in 'sn_heldout_adata.obs'.
    k_neighbors : int, default=10
        The number of nearest neighbors to use for local shift averaging.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the shifted and decoded gene expression for the held-out SN cells. Each row corresponds
        to one held-out cell, and columns are the common genes shared by SC, SN, and the held-out subset.

    Notes
    -----
    - This procedure is designed to emulate a local “shift” from SN to SC in the scVI latent space, preserving
      local structure among the SN cells while anchoring to the global SC centroid.
    - The scVI model should not have been trained on the held-out cell type (i.e., that cell type’s SN cells
      were excluded from model training) to avoid data leakage and simulate a real missing-cell scenario.
    """

    # 1) Align genes
    common_genes = (
        set(sc_adata.var_names)
        .intersection(sn_adata.var_names)
        .intersection(sn_heldout_adata.var_names)
    )
    sc_adata = sc_adata[:, sorted(common_genes)].copy()
    sn_adata = sn_adata[:, sorted(common_genes)].copy()
    sn_heldout_adata = sn_heldout_adata[:, sorted(common_genes)].copy()

    sc_celltype_labels = sc_adata.obs["cell_types"].astype(str)
    sn_celltype_labels = sn_adata.obs["cell_types"].astype(str)

    # Identify overlapping cell types (excluding the held-out type)
    if heldout_label:
        overlapping_types = np.intersect1d(
            sc_celltype_labels.unique(), sn_celltype_labels.unique()
        )
        overlapping_types = overlapping_types[overlapping_types != heldout_label]

        sc_data_aligned = (
            sc_adata[sc_adata.obs.cell_types.isin(overlapping_types)].copy().to_df()
        )
        sn_data_aligned = (
            sn_adata[sn_adata.obs.cell_types.isin(overlapping_types)].copy().to_df()
        )

    else:
        sn_data_aligned = sn_adata
        sc_data_aligned = sc_adata

    # 2) Get latent representations
    z_sc = model.get_latent_representation(sc_adata)
    z_sn = model.get_latent_representation(sn_adata)

    mean_sc_latent = z_sc.mean(axis=0)
    sn_shifts = mean_sc_latent - z_sn  # shape: (#sn_cells, latent_dim)

    z_ho = model.get_latent_representation(sn_heldout_adata)

    # 3) Build kNN on sn latent space
    knn = NearestNeighbors(n_neighbors=k_neighbors).fit(z_sn)
    z_ho_shifted = np.zeros_like(z_ho)

    for i in range(z_ho.shape[0]):
        _, neigh_idx = knn.kneighbors(z_ho[i, :].reshape(1, -1))
        local_shift = sn_shifts[neigh_idx[0]].mean(axis=0)
        z_ho_shifted[i, :] = z_ho[i, :] + local_shift

    # 4) Put z_ho_shifted in sn_heldout_adata.obsm
    sn_heldout_adata.obsm["_scvi_latent"] = z_ho_shifted

    # 5) Determine library size from sc_adata
    sc_totals = np.array(sc_adata.X.sum(axis=1)).flatten()
    median_sc_lib = np.median(sc_totals)

    transformed_expr = get_normalized_expression_from_latent(
        model,
        z_ho_shifted,
        library_size=median_sc_lib,
        gene_list=list(common_genes),
        return_numpy=True,
    )

    # 7) Return as a DataFrame with the same shape as the # of cells in sn_heldout_adata
    #    and columns = common_genes
    df_out = pd.DataFrame(
        transformed_expr, index=sn_heldout_adata.obs_names, columns=list(common_genes)
    )
    return df_out


def get_normalized_expression_from_latent(
    model: scvi.model.SCVI,
    latent: np.ndarray | torch.Tensor,
    library_size: float | None = 1.0,
    gene_list: list[str] | None = None,
    n_samples: int = 1,
    return_mean: bool = True,
    return_numpy: bool = True,
):
    """
    Decode expression from a user-provided latent representation using scvi-tools 1.2.x,
    which requires calling model.module.decoder(x, dispersion, library).

    Parameters
    ----------
    model : scvi.model.SCVI
        A trained scVI model. Typically configured with `dispersion="gene-batch"` or similar.
    latent : np.ndarray or torch.Tensor
        The (n_cells, n_latent) latent representation to decode. This could be your
        custom-shifted latent for hold-out cells.
    library_size : float or None, default=1.0
        If not None, each cell's px_scale is multiplied by this value.
        By default=1.0, all cells have library=1.0. If you prefer no uniform scaling,
        pass None and then do any scaling afterwards, or pass a different float to enforce
        that uniform sum.
    gene_list : list of str, optional
        If provided, only return these genes. Must exist in `model.adata.var_names`.
    n_samples : int, default=1
        Number of times to sample from the decoder. If > 1 and `return_mean=True`,
        the output is the average expression across samples.
    return_mean : bool, default=True
        If True and `n_samples>1`, returns the mean. If False and n_samples>1,
        returns a 3D (n_samples, n_cells, n_genes) array or tensor.
    return_numpy : bool, default=True
        If True, returns a NumPy array; otherwise returns a pandas DataFrame
        (with shape `(n_cells, n_genes)` or `(n_samples, n_cells, n_genes)`).

    Returns
    -------
    np.ndarray or pd.DataFrame
        Decoded expression from the latent codes. If `return_mean=True`, shape is (n_cells, n_genes).
        If `return_mean=False and n_samples>1`, shape is (n_samples, n_cells, n_genes).

    Notes
    -----
    - This function bypasses the encoder stage. It only decodes from the provided `latent` matrix.
    - scVI's `dispersion` mode is inferred from the model itself, e.g. `model.module.dispersion`.
    - The shape of `latent` must be `(n_cells, n_latent)`.
    """
    device = next(model.module.decoder.parameters()).device
    # 1) Ensure 'latent' is on the correct device, shape=(n_cells, n_latent)
    if not torch.is_tensor(latent):
        latent_tensor = torch.tensor(latent, dtype=torch.float32, device=device)
    else:
        latent_tensor = latent.to(device)

    # 2) Build library tensor of shape (n_cells, 1).
    #    If library_size=1.0 => library=1 for all cells.
    #    If None => you could revert to library=1 anyway, or pass 0 => but None means "no uniform scaling."
    n_cells = latent_tensor.shape[0]
    if library_size is not None:
        library_tensor = torch.full(
            (n_cells, 1), library_size, dtype=torch.float32, device=device
        )
    else:
        # default to all ones if you want "no uniform scaling"
        library_tensor = torch.ones((n_cells, 1), dtype=torch.float32, device=device)

    # 3) Fetch the model's dispersion string. e.g. "gene-batch", "gene", etc.
    dispersion_mode = model.module.dispersion
    print("dispersion mode: ", dispersion_mode)

    # 4) We'll accumulate multiple samples of px_scale if n_samples>1
    samples_list = []
    model.module.decoder.eval()  # turn off dropout, etc.
    with torch.no_grad():
        for _ in range(n_samples):
            # scvi-tools 1.2.x expects DecoderSCVI.forward(x, dispersion, library)
            # in that exact positional order:
            if len(model.module.decoder.px_decoder.n_cat_list) > 0:
                # For each expected categorical input, create a dummy tensor.
                # Assuming each categorical covariate is of size (n_cells, 1)
                dummy_cats = [
                    torch.zeros(
                        (latent_tensor.size(0), 1), dtype=torch.long, device=device
                    )
                    for _ in model.module.decoder.px_decoder.n_cat_list
                ]
                decoder_out = model.module.decoder(
                    dispersion_mode, latent_tensor, library_tensor, *dummy_cats
                )
            else:
                decoder_out = model.module.decoder(
                    dispersion_mode, latent_tensor, library_tensor
                )

            # decoder_out is typically a dict with "px_scale", "px_r", etc.
            px_scale = decoder_out[0]
            samples_list.append(px_scale.cpu())

    # 5) If multiple samples, handle them
    if n_samples > 1:
        stacked = torch.stack(
            samples_list, dim=0
        )  # shape: (n_samples, n_cells, n_genes)
        if return_mean:
            expr_tensor = stacked.mean(dim=0)  # shape: (n_cells, n_genes)
        else:
            expr_tensor = stacked  # shape: (n_samples, n_cells, n_genes)
    else:
        expr_tensor = samples_list[0]  # shape: (n_cells, n_genes)

    # 6) Subset genes if needed
    var_names = model.adata.var_names
    if gene_list is not None:
        gene_indices = [i for i, g in enumerate(var_names) if g in gene_list]
        if expr_tensor.ndim == 2:
            expr_tensor = expr_tensor[:, gene_indices]
        elif expr_tensor.ndim == 3:
            expr_tensor = expr_tensor[..., gene_indices]
        var_names = var_names[gene_indices]

    # 7) Return as numpy or DataFrame
    expr_np = expr_tensor.numpy()
    if return_numpy:
        return expr_np
    else:
        # If shape is (n_cells, n_genes) => DataFrame with gene columns
        # If shape is (n_samples, n_cells, n_genes), we'd have to decide indexing.
        # Usually you only do DataFrame for 2D.
        if expr_np.ndim == 2:
            return pd.DataFrame(expr_np, columns=var_names)
        else:
            # We'll just return a 3D np.array if shape is 3D
            # or create a multi-index. It's simpler to keep it as np.
            raise ValueError(
                "DataFrame output for 3D arrays is ambiguous. "
                "Set return_mean=True or return_numpy=True."
            )


def transform_heldout_sn_to_mean_sc_local(
    sc_data: pd.DataFrame,
    sn_data: pd.DataFrame,
    sn_heldout_data: pd.DataFrame,
    sc_celltype_labels: pd.Series,
    sn_celltype_labels: pd.Series,
    heldout_label: Optional[str] = None,
    variance_threshold: float = 0.75,
    k_neighbors: int = 10,
):
    """
    Local PCA-based transformation for the 'held-out' SN cell type.

    Unlike the original method that used a single “composite shift,”
    this version does a local kNN in PCA space for each held-out cell.

    Steps:
    1) Align columns (genes) across SC, SN, and SN-heldout.
    2) Fit PCA on combined SC + SN (excluding the held-out label).
    3) For each overlapping SN cell, SHIFT_i = (mean(SC PCA) - sn_pcs[i]).
    4) For each held-out SN cell:
       - transform to PCA coords
       - find k nearest neighbors among sn_pcs
       - average SHIFT among those neighbors
       - apply SHIFT -> invert transform -> exp -> clamp -> library-size scale
    5) Return the resulting DataFrame with “SC-like” expression.

    Parameters
    ----------
    sc_data : pd.DataFrame
        Single-cell data, rows = cells, columns = genes (raw counts).
    sn_data : pd.DataFrame
        Single-nucleus data, rows = cells, columns = genes (raw counts),
        for all SN cell types EXCEPT the held-out, plus the overlapping types.
    sn_heldout_data : pd.DataFrame
        Single-nucleus data for the held-out cell type, rows = cells, columns = genes (raw counts).
    sc_celltype_labels : pd.Series
        Cell-type labels for sc_data’s rows.
    sn_celltype_labels : pd.Series
        Cell-type labels for sn_data’s rows.
    heldout_label : str
        The SN cell type that is missing from SC but present in SN (sn_heldout_data).
    variance_threshold : float
        Fraction of variance to retain in PCA (e.g. 0.90).
    k_neighbors : int
        Number of neighbors for local averaging of SHIFT in PCA space.

    Returns
    -------
    pd.DataFrame
        The transformed SN-heldout data (rows = cells, columns = genes),
        with expression scaled similarly to SC.
    """

    # 1) Align columns
    common_genes = sc_data.columns.intersection(sn_data.columns).intersection(
        sn_heldout_data.columns
    )
    sc_data_aligned = sc_data[common_genes]
    sn_data_aligned = sn_data[common_genes]
    sn_heldout_aligned = sn_heldout_data[common_genes]

    # 2) Identify overlapping cell types (excluding the held-out type)
    if heldout_label:

        overlapping_types = np.intersect1d(
            sc_celltype_labels.unique(), sn_celltype_labels.unique()
        )
        overlapping_types = overlapping_types[overlapping_types != heldout_label]

        # 3) Fit PCA on combined SC + SN (overlapping), excluding the held-out SN if available
        keep_sn_mask = sn_celltype_labels != heldout_label
        combined_df = pd.concat(
            [sc_data_aligned, sn_data_aligned.loc[keep_sn_mask]], axis=0
        )

    else:
        combined_df = pd.concat([sc_data_aligned, sn_data_aligned], axis=0)

    # log1p + scale
    combined_log = np.log1p(combined_df)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_log)

    # find # of comps for variance_threshold
    # do PCA with chosen comps
    pca = PCA(n_components=variance_threshold)
    pca.fit(combined_scaled)

    # transform SC
    sc_log = np.log1p(sc_data_aligned)
    sc_scaled = scaler.transform(sc_log)
    sc_pcs = pca.transform(sc_scaled)

    # transform SN (overlapping)
    if heldout_label:
        sn_overlap = sn_data_aligned.loc[keep_sn_mask]
        sn_log = np.log1p(sn_overlap)
    else:
        sn_log = np.log1p(sn_data_aligned)

    sn_scaled = scaler.transform(sn_log)
    sn_pcs = pca.transform(sn_scaled)

    # 4) SHIFT vectors for each overlapping SN cell
    #    SHIFT_i = (mean(SC in PCA) - sn_pcs[i]).
    mean_sc_pca = sc_pcs.mean(axis=0)
    sn_shifts = mean_sc_pca - sn_pcs  # shape = (n_SN, n_comps)

    # 5) local transform for held-out SN cells
    #    (a) transform to PCA, (b) find KNN among sn_pcs, (c) average SHIFT, (d) apply SHIFT
    ho_log = np.log1p(sn_heldout_aligned)
    ho_scaled = scaler.transform(ho_log)
    ho_pcs = pca.transform(ho_scaled)

    # build knn on sn_pcs
    knn = NearestNeighbors(n_neighbors=k_neighbors).fit(sn_pcs)
    ho_pcs_shifted = np.zeros_like(ho_pcs)

    for i in range(ho_pcs.shape[0]):
        this_cell = ho_pcs[i, :].reshape(1, -1)
        # find neighbors
        _, neigh_idx = knn.kneighbors(this_cell)
        # average SHIFT
        local_shift = sn_shifts[neigh_idx[0], :].mean(axis=0)
        ho_pcs_shifted[i, :] = ho_pcs[i, :] + local_shift

    # invert transform: pca -> scale -> expm1
    ho_log_transformed = pca.inverse_transform(ho_pcs_shifted)
    ho_unscaled = scaler.inverse_transform(ho_log_transformed)
    ho_counts = np.expm1(ho_unscaled)
    ho_counts[ho_counts < 0] = 0

    # 6) Scale each cell’s library to match median SC library size
    sc_totals = sc_data_aligned.sum(axis=1)  # sum per cell
    median_sc_lib = np.median(sc_totals)
    for i in range(ho_counts.shape[0]):
        row_sum = ho_counts[i, :].sum()
        if row_sum > 0:
            ho_counts[i, :] *= median_sc_lib / row_sum

    # 7) Build final DataFrame
    transformed_df = pd.DataFrame(
        ho_counts, index=sn_heldout_aligned.index, columns=common_genes
    )
    return transformed_df


def transform_heldout_sn_to_mean_sc(
    sc_data: pd.DataFrame,
    sn_data: pd.DataFrame,
    sn_heldout_data: pd.DataFrame,
    sc_celltype_labels: pd.Series,
    sn_celltype_labels: pd.Series,
    heldout_label: Optional[str] = None,
    variance_threshold: float = 0.90,
):
    """
    Transforms the 'held-out' SN cell type (sn_heldout_data) to match the average
    PCA profile shift across overlapping SC vs SN cell types, then scales
    to match the median SC library size.

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
    ## to ensure that only genes present in all three DataFrames are used. This step prevents misalignment in subsequent transforms.
    common_genes = sc_data.columns.intersection(sn_data.columns).intersection(
        sn_heldout_data.columns
    )
    sc_data_aligned = sc_data[common_genes]
    sn_data_aligned = sn_data[common_genes]
    sn_heldout_data_aligned = sn_heldout_data[common_genes]

    # 2) Identify overlapping cell types (excluding the held-out type)
    if heldout_label:
        # We gather the cell types shared by both SC and SN.
        overlapping_types = np.intersect1d(
            sc_celltype_labels.unique(), sn_celltype_labels.unique()
        )
        # We exclude the heldout type so it won’t affect the PCA transformation.
        overlapping_types = overlapping_types[overlapping_types != heldout_label]

        # 3) Fit PCA on combined SC+SN, excluding the held-out SN
        ## We remove the heldout single‐nucleus cells from the PCA training set (i.e., “SN except heldout”),
        # so PCA is only learning from the types SC and SN have in common
        keep_sn_mask = sn_celltype_labels != heldout_label
        combined_df = pd.concat(
            [sc_data_aligned, sn_data_aligned.loc[keep_sn_mask]], axis=0
        )

    else:
        combined_df = pd.concat([sc_data_aligned, sn_data_aligned], axis=0)

    # Apply log1p and standard scaling
    combined_log = np.log1p(combined_df)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_log)

    # Fit
    # PCA with the chosen variance threshold.
    # ensuring we only keep enough components to reach the variance_threshold
    pca = PCA(n_components=variance_threshold)
    pca.fit(combined_scaled)

    # 4) Compute SHIFT VECTORS for each overlapping cell type
    shift_vectors = []
    for ct in overlapping_types:
        sc_cells_ct = sc_data_aligned.loc[sc_celltype_labels == ct]
        sn_cells_ct = sn_data_aligned.loc[sn_celltype_labels == ct]

        if not sc_cells_ct.empty and not sn_cells_ct.empty:
            sc_scaled = scaler.transform(np.log1p(sc_cells_ct))
            sn_scaled = scaler.transform(np.log1p(sn_cells_ct))
            sc_pcs = pca.transform(sc_scaled)
            sn_pcs = pca.transform(sn_scaled)
            shift_vectors.append(np.mean(sc_pcs, axis=0) - np.mean(sn_pcs, axis=0))

    # 5) Compute the composite shift vector
    if len(shift_vectors) == 0:
        print("No overlapping cell types found! Using global SC-SN shift.")
        all_sc_scaled = scaler.transform(np.log1p(sc_data_aligned))
        all_sn_scaled = scaler.transform(np.log1p(sn_data_aligned.loc[keep_sn_mask]))
        composite_shift = np.mean(pca.transform(all_sc_scaled), axis=0) - np.mean(
            pca.transform(all_sn_scaled), axis=0
        )
    else:
        composite_shift = np.mean(shift_vectors, axis=0)

    # 6) Transform the HELD-OUT SN cells with that composite shift
    ho_scaled = scaler.transform(np.log1p(sn_heldout_data_aligned))
    ho_pcs = pca.transform(ho_scaled)
    ho_pcs_shifted = ho_pcs + composite_shift
    ho_log_transformed = pca.inverse_transform(ho_pcs_shifted)

    # 7) Exponentiate (count scale) and clip negatives
    # we revert the log1p transformation with expm1, then clamp negative values to zero.
    ho_exp = np.expm1(ho_log_transformed)
    ho_exp[ho_exp < 0] = 0  # Ensure no negatives

    # 8) Scale each cell's library to match the median SC library size
    # each cell in the transformed set is scaled to match the “typical” (median) SC library size.
    median_sc_lib = np.median(sc_data_aligned.sum(axis=1))
    for i in range(ho_exp.shape[0]):
        row_sum = ho_exp[i, :].sum()
        if row_sum > 0:
            ho_exp[i, :] *= median_sc_lib / row_sum

    # 9) Convert to DataFrame
    transformed_df = pd.DataFrame(
        ho_exp, index=sn_heldout_data_aligned.index, columns=common_genes
    )

    return transformed_df


def calculate_median_library_size(adata):
    """Returns median library size of given AnnData"""

    # Summation across genes => returns total counts per cell
    totals = adata.X.sum(axis=1)  # shape: (n_cells,) or maybe a sparse (n_cells,1)

    # 1) Convert to a dense 1D numpy array
    if issparse(totals):
        # For a (n_cells,1) sparse matrix, do:
        totals = totals.A.flatten()
    else:
        # Sometimes sum returns a numpy array with shape (n_cells,1)
        totals = np.array(totals).flatten()  # ensure 1D

    # 2) Now the median is a single float
    median_lib = np.median(totals)
    print("Median library size:", median_lib, type(median_lib))
    return median_lib
