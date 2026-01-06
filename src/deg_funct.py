"""
Differential Expression Analysis for Single-Cell and Single-Nucleus RNA-seq

This module provides tools to perform DESeq2-based differential gene expression (DGE) analysis
between single-cell (SC) and single-nucleus (SN) RNA-seq data using PyDESeq2.

Features:
- Pseudobulk creation for SC and SN data
- Parallelized DGE analysis by cell type
- Removal of DEGs from AnnData objects
- Support for loading/saving DEGs in JSON format
- Integration of DEGs across datasets (e.g., PBMC, MBC, ADP)

It should be in /src/ folder and imported as "from src.deg_funct import the_function"

"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix
from multiprocessing import Pool
from anndata import AnnData
from pydeseq2.default_inference import DefaultInference
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from typing import Dict, Iterable, Optional, Set, Union


def _to_str_set(x: Union[Iterable, pd.Index, pd.Series, pd.DataFrame]) -> Set[str]:
    """
    Robustly coerce a variety of inputs (iterable of genes, pandas Index/Series,
    or a DataFrame with genes as index) into a set[str] of gene names.
    """
    if x is None:
        return set()
    if isinstance(x, pd.DataFrame):
        return set(map(str, x.index))
    if isinstance(x, (pd.Index, pd.Series)):
        return set(map(str, x.values))
    try:
        return set(map(str, x))
    except TypeError:
        raise TypeError(
            "Unsupported type for gene collection; pass an iterable/Index/Series/DataFrame."
        )


def select_random_control_genes(
    adata_sc_subset,
    adata_sn_subset,
    diff_genes: Dict[str, Union[pd.DataFrame, pd.Index, pd.Series, Iterable[str]]],
    intersect3ds_genes: Union[Iterable[str], pd.Index, pd.Series, pd.DataFrame],
    size: Optional[int] = None,
    seed: Optional[int] = 42,
    require_presence_in_both: bool = True,
    return_as: str = "dict",
):
    """
    Build a size-matched random gene set that EXCLUDES:
      1) all genes listed as DE between cell types (union over `diff_genes`)
      2) all genes contained in the 'intersect_3ds' list

    Parameters
    ----------
    adata_sc_subset : anndata.AnnData
        SC AnnData (typically SC minus the held-out cell type).
    adata_sn_subset : anndata.AnnData
        SN AnnData (typically SN for the held-out cell type, or SN minus held-out for PCA prep).
    diff_genes : dict[str, (DataFrame|Index|Series|Iterable[str])]
        Mapping from cell_type (or any key) -> collection whose *index/values* are DEG gene names.
        Example: what you currently pass into `remove_diff_genes`.
    intersect3ds_genes : (Iterable|Index|Series|DataFrame)
        The intersect_3ds gene list to exclude (e.g., loaded from `intersect_3ds.csv`).
    size : int or None, default None
        Number of genes to sample. If None, uses the size of the *union* of DEGs in `diff_genes`.
    seed : int or None
        Seed for reproducible sampling (NumPy Generator).
    require_presence_in_both : bool, default True
        If True, only sample from genes present in BOTH adata_sc_subset and adata_sn_subset.
        If False, sample from the SC gene universe only.
    return_as : {"dict","index","list"}, default "dict"
        - "dict": return {"random": DataFrame(index=random_genes)} suitable for `remove_diff_genes`
        - "index": return a pandas.Index of gene names
        - "list":  return a Python list of gene names

    Returns
    -------
    One of:
        dict[str, pd.DataFrame] | pd.Index | list[str]

    Raises
    ------
    ValueError if there are no eligible candidate genes to sample.
    """
    # --- define the gene universe we can draw from ---
    sc_genes = set(map(str, adata_sc_subset.var_names))
    if require_presence_in_both:
        sn_genes = set(map(str, adata_sn_subset.var_names))
        gene_universe = sc_genes & sn_genes
    else:
        gene_universe = sc_genes

    # --- union of all DE genes provided in `diff_genes` ---
    deg_union: Set[str] = set()
    for df_like in (diff_genes or {}).values():
        deg_union |= _to_str_set(df_like)

    # --- intersect_3ds as a set[str] ---
    intersect3ds_set = _to_str_set(intersect3ds_genes)

    # --- candidates are genes in the universe minus exclusions ---
    excluded = deg_union | intersect3ds_set
    candidates = sorted(gene_universe - excluded)

    if size is None:
        size = len(deg_union)

    if len(candidates) == 0:
        raise ValueError(
            "No eligible non-DEG, non-intersect3ds genes available to sample from. "
            "Relax constraints or check inputs."
        )

    # Cap size if candidate pool is smaller; warn noiselessly via print (you can switch to logging)
    k = min(size, len(candidates))
    if k < size:
        print(
            f"[select_random_control_genes] Requested {size} genes but only {len(candidates)} "
            f"eligible; sampling {k}."
        )

    rng = np.random.default_rng(seed)
    picked = rng.choice(candidates, size=k, replace=False)

    if return_as == "dict":
        return {"random": pd.DataFrame(index=pd.Index(picked.astype(str), name=None))}
    if return_as == "index":
        return pd.Index(picked.astype(str))
    if return_as == "list":
        return picked.astype(str).tolist()

    raise ValueError("`return_as` must be one of {'dict','index','list'}.")


def load_gene_list(csv_path):
    """
    Robustly read a CSV that may or may not have a header OR an index col.
    Returns List[str] without NaNs / ints.
    """
    df = pd.read_csv(csv_path, dtype=str)
    if df.shape[1] == 1:
        series = df.iloc[:, 0]
    else:
        series = df.iloc[:, 1]
    return series.dropna().tolist()


def remove_diff_genes(sc_adata, sn_adata, diff_genes):
    """
    Removes differentially expressed genes from the AnnData objects.
    If no differentially expressed genes are found, return original data.
    """
    if not diff_genes:
        print("No differentially expressed genes found. Skipping gene removal step.")
        return sc_adata, sn_adata

    # --- Build a normalized set[str] of DEGs (works for DF, Index/list) ---
    diff_gene_set = set()
    for v in diff_genes.values():
        if hasattr(v, "index"):  # DataFrame or Series with gene names in index
            diff_gene_set |= set(map(str, v.index))
        else:  # list-like of gene names
            diff_gene_set |= set(map(str, v))

    if len(diff_gene_set) == 0:
        print("DEG collection is empty. Skipping gene removal step.")
        return sc_adata, sn_adata

    # --- Ensure var_names are compared as strings ---
    sc_genes = sc_adata.var_names.astype(str)
    sn_genes = sn_adata.var_names.astype(str)

    # --- Filter ---
    mask_sc = ~sc_genes.isin(diff_gene_set)
    mask_sn = ~sn_genes.isin(diff_gene_set)

    sc_adata_filtered = sc_adata[:, mask_sc].copy()
    sn_adata_filtered = sn_adata[:, mask_sn].copy()

    # very light sanity log
    print(
        f"[remove_diff_genes] removed: SC={int((~mask_sc).sum())}, SN={int((~mask_sn).sum())}"
    )

    return sc_adata_filtered, sn_adata_filtered


def _coerce_to_df_of_genes(val) -> pd.DataFrame:
    """
    Return a DataFrame with genes in the index from a variety of inputs:
      - split-orient dict {'index','columns','data'}
      - existing DataFrame
      - list of genes
      - dict of {gene: ...}
    """
    if isinstance(val, pd.DataFrame):
        df = val.copy()
        df.index = df.index.astype(str)
        return df

    if isinstance(val, dict) and {"index", "columns", "data"}.issubset(val.keys()):
        df = pd.DataFrame(**val)
        df.index = df.index.astype(str)
        return df

    # list-like: treat as just gene names
    if isinstance(val, list):
        return pd.DataFrame(index=pd.Index(map(str, val)))

    # dict-like (not split): keys are genes
    if isinstance(val, dict):
        return pd.DataFrame(index=pd.Index(map(str, val.keys())))

    # last resort: try to iterate
    try:
        return pd.DataFrame(index=pd.Index(map(str, val)))
    except Exception:
        return pd.DataFrame(index=pd.Index([], dtype=str))


def differential_expression_analysis_parallel(
    sn_adata, sc_adata, num_threads=4, n_cpus_per_thread=8, deseq_alpha=0.001
):
    from multiprocessing import Pool

    common_cell_types = list(
        set(sn_adata.obs["cell_types"]).intersection(sc_adata.obs["cell_types"])
    )
    print(f"Running DESeq2 in parallel for {len(common_cell_types)} cell types...")

    tasks = []
    for ct in common_cell_types:
        # Subset once in the main process
        sn_subset = sn_adata[sn_adata.obs["cell_types"] == ct].copy()
        sc_subset = sc_adata[sc_adata.obs["cell_types"] == ct].copy()
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
    common_cell_types = set(sn_adata.obs["cell_types"]).intersection(
        sc_adata.obs["cell_types"]
    )

    for cell_type in common_cell_types:

        # Subset data for the current cell type
        sn_cells = sn_adata[sn_adata.obs["cell_types"] == cell_type]
        sc_cells = sc_adata[sc_adata.obs["cell_types"] == cell_type]
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
        metadata = pd.DataFrame(
            {"condition": ["sn"] * sn_cells.n_obs + ["sc"] * sc_cells.n_obs},
            index=combined_counts.index,
        )

        # Initialize and run DESeq2 analysis
        inference = DefaultInference(n_cpus=8)
        dds = DeseqDataSet(
            counts=combined_counts.astype(int),
            metadata=metadata,
            design_factors="condition",
            inference=inference,
        )
        dds.deseq2()

        # Extract results
        deseq_stats = DeseqStats(
            dds, contrast=["condition", "sn", "sc"], alpha=deseq_alpha
        )
        deseq_stats.summary()
        results_df = deseq_stats.results_df

        # Filter for significantly differentially expressed genes
        sig_genes = results_df[
            (results_df["padj"] < deseq_alpha)
            & (results_df["log2FoldChange"].abs() > 1)
        ]

        # Store results in the dictionary
        diff_genes[cell_type] = sig_genes

    return diff_genes


def run_deseq2_for_cell_type(
    cell_type, sn_adata, sc_adata, n_cpus=8, deseq_alpha=0.001
):
    """Runs DESeq2 for a single cell type without expression-based filtering."""
    print(f"Running DESeq2 for {cell_type}...")

    # Subset the data for the cell type
    sn_cells = sn_adata[sn_adata.obs["cell_types"] == cell_type]
    sc_cells = sc_adata[sc_adata.obs["cell_types"] == cell_type]

    print(f"For {cell_type}: sn_cells={sn_cells.shape}, sc_cells={sc_cells.shape}")

    # Skip if too few cells
    if sn_cells.shape[0] < 3 or sc_cells.shape[0] < 3:
        print(
            f"Skipping {cell_type}: Too few cells for DE analysis. sn_cells={sn_cells.shape[0]}, sc_cells={sc_cells.shape[0]}"
        )
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
    metadata = pd.DataFrame(
        {"condition": ["sn"] * sn_cells.n_obs + ["sc"] * sc_cells.n_obs},
        index=combined_counts.index,
    )

    # Drop zero-variance genes (Retaining all others)
    combined_counts = combined_counts.loc[:, combined_counts.var(axis=0) > 0]
    print(
        f"Cell type {cell_type}: {combined_counts.shape[1]} genes after zero-variance filtering."
    )

    if combined_counts.shape[1] == 0:
        print(f"Skipping {cell_type}: No valid genes after zero-variance filtering.")
        return cell_type, None

    print(
        f"Running DESeq2 for cell type: {cell_type} with {combined_counts.shape[1]} genes."
    )

    # Run DESeq2
    try:
        inference = DefaultInference(n_cpus=n_cpus)
        dds = DeseqDataSet(
            counts=combined_counts.astype(int),
            metadata=metadata,
            design_factors="condition",
            inference=inference,
        )
        dds.deseq2()
        deseq_stats = DeseqStats(
            dds,
            contrast=["condition", "sn", "sc"],  # or ["condition", "sc", "sn"]
            alpha=deseq_alpha,
        )
        deseq_stats.summary()
        results_df = deseq_stats.results_df

        # Apply differential expression significance filtering
        sig_genes = results_df[
            (results_df["padj"] < deseq_alpha)
            & (results_df["log2FoldChange"].abs() > 1)
        ]

        return cell_type, sig_genes

    except Exception as e:
        print(f"Error running DESeq2 for {cell_type}: {str(e)}")
        return cell_type, None


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
        mat_reshaped = mat.reshape(n_groups, group_size, mat.shape[1])

        # Sum across the group_size dimension => shape (n_groups, n_genes)
        pseudobulk_counts = mat_reshaped.sum(axis=1)

        # Create an obs DataFrame for these pseudobulks
        group_ids = [f"{cell_type}_group{i+1}" for i in range(n_groups)]
        obs_tmp = pd.DataFrame(
            {"cell_types": cell_type, "group_id": group_ids}, index=group_ids
        )  # index must match the new row names

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
    pseudo_adata = AnnData(
        X=csr_matrix(big_matrix),  # keep it sparse to save memory
        obs=big_obs,
        var=pd.DataFrame(index=var_names),
    )

    return pseudo_adata


def load_others_degs(res_name: str, base_path: str):
    """
    Load DEG files for the datasets other than 'res_name' and return:
      { "others": df }
    where 'df' is a pandas DataFrame whose index holds the union of all genes
    from the other datasets.

    If no genes are found, raises a ValueError to indicate something is wrong.

    This function is compatible with remove_diff_genes(...),
    which expects { cell_type -> pd.DataFrame }.
    """

    ALL_DATASETS = ["PBMC", "MBC", "ADP"]
    other_ds = [ds for ds in ALL_DATASETS if ds != res_name]
    union_genes = set()

    for ds in other_ds:
        base_path_dataset = os.path.join(base_path, ds)
        degs_file = os.path.join(base_path_dataset, "degs.json")
        if not os.path.exists(degs_file):
            print(f"[Warning] Missing DEG file for {ds}: {degs_file}. Skipping.")
            continue

        with open(degs_file, "r") as f:
            degs_json = json.load(f)

        # degs_json is a dict keyed by cell type.
        # Each value may be:
        #   - a dict in 'split' format => DataFrame of DEGs (df.index = gene names),
        #   - a list of gene names,
        #   - or a dict of gene->pvalue/etc.
        for cell_type, val in degs_json.items():
            if isinstance(val, dict) and {"index", "columns", "data"}.issubset(
                val.keys()
            ):
                df = pd.DataFrame(**val)
                union_genes.update(df.index)
            elif isinstance(val, list):
                union_genes.update(val)
            elif isinstance(val, dict):
                union_genes.update(val.keys())
            else:
                print(
                    f"[Warning] Unexpected DEG format for cell type {cell_type} in dataset {ds}."
                )

    # If no genes found, raise an error rather than returning empty
    if len(union_genes) == 0:
        raise ValueError(
            f"No DEGs found in the other datasets for '{res_name}'"
            "This indicates an unexpected condition or missing data."
        )

    # Otherwise, create a DataFrame with these genes in the index
    df_others = pd.DataFrame(index=sorted(union_genes))

    # Return as a dict with a single key "others"
    return {"others": df_others}


def load_or_calc_degs(
    output_path, adata_sc_ref, adata_sn_ref, deseq_alpha, patient_id=False
):
    genes_save_path = (
        f"{output_path}/degs_{patient_id}.json"
        if patient_id
        else f"{output_path}/degs.json"
    )

    # 1) Try load
    if os.path.exists(genes_save_path):
        try:
            with open(genes_save_path, "r") as file:
                raw = json.load(file)
        except json.JSONDecodeError:
            raw = None

        if raw is not None:
            # Coerce every entry into a DataFrame with genes in the index
            diff_genes = {str(k): _coerce_to_df_of_genes(v) for k, v in raw.items()}
            # Re-save canonically (split) so future runs are consistent
            diff_genes_json = {
                k: v.to_dict(orient="split") for k, v in diff_genes.items()
            }
            with open(genes_save_path, "w") as f:
                json.dump(diff_genes_json, f)
            return diff_genes

        # fallthrough → recompute if unreadable

    # 2) Compute fresh (pseudobulk) if not found or unreadable
    print("Not found previous diff. gene expr... calculating now!")
    print("Creating pseudobulk of size 10 for SC and SN data for DGE analysis...")
    pseudo_sc_adata = create_fixed_pseudobulk(adata_sc_ref, group_size=10)
    pseudo_sn_adata = create_fixed_pseudobulk(adata_sn_ref, group_size=10)

    diff_genes = differential_expression_analysis_parallel(
        sn_adata=pseudo_sn_adata,
        sc_adata=pseudo_sc_adata,
        deseq_alpha=deseq_alpha,
        num_threads=4,  # matches SLURM --ntasks
        n_cpus_per_thread=16,  # matches SLURM --cpus-per-task
    )

    # Coerce every entry to DataFrame (even fresh results) for uniformity
    diff_genes = {str(k): _coerce_to_df_of_genes(v) for k, v in diff_genes.items()}

    print("Found these many differentially expressed genes:")
    for key, df in diff_genes.items():
        print(key, df.shape)

    # Save canonically as split
    diff_genes_json = {k: v.to_dict(orient="split") for k, v in diff_genes.items()}
    with open(genes_save_path, "w") as file:
        json.dump(diff_genes_json, file)

    return diff_genes
