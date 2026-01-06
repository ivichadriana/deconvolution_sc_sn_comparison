#!/usr/bin/env python
"""
Aggregate deconvolution results (simulation / real / per-donor).

Usage examples
--------------
# simulation (ground truth available)
python process_results.py --dataset=PBMC --simulation

# real bulk deconvolution
python process_results.py --dataset=Real_ADP

# per-donor references
python process_results.py --dataset=Real_ADP --perdonor

"""
from __future__ import annotations
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error

# --------------------------------------------------------------------
#  Arguments
# --------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Aggregate outputs")
parser.add_argument(
    "--dataset",
    required=True,
    help="Dataset name (folder under ../data/deconvolution/)",
)
parser.add_argument(
    "--simulation",
    action="store_true",
    help="Synthetic pseudobulks with ground-truth proportions.",
)
parser.add_argument(
    "--perdonor",
    action="store_true",
    help="One-reference-per-donor experiment (no ground truth).",
)
parser.add_argument(
    "--method",
    required=True,
    help="Method name: BayesPrism, DWLS, or SCDC",
)

args = parser.parse_args()
method = args.method

# --------------------------------------------------------------------
# Settings (add your transform name here!)
# --------------------------------------------------------------------
valid_transforms = {
    "rawSN",
    "pcaSN",
    "degSN",
    "degRandSN",
    "degIntSN",
    "degOtherSN",
    "degPCA_SN",
    "degpcaSN",
    "scviSN",
    "scvi_LSshift_SN",
    "degScviSN",
    "degScviLSshift_SN",
    "degIntAllSN",
}

# -------- mutually-exclusive flags --------
if args.simulation and args.perdonor:
    sys.exit("ERROR: Use only one of --simulation or --perdonor.")

mode = "simulation" if args.simulation else ("perdonor" if args.perdonor else "real")
print(f"Processing {args.dataset!r}  |  mode = {mode}")

BASE_DIR = os.getcwd()
DATA_PATH = os.path.join(BASE_DIR, "data", "deconvolution", args.dataset)
RESULTS_PATH = os.path.join(BASE_DIR, "results", args.dataset)
os.makedirs(RESULTS_PATH, exist_ok=True)

# --------------------------------------------------------------------
# 1) Ground-truth (simulation only) (same for all methods)
# --------------------------------------------------------------------
if mode == "simulation":
    gt_file = os.path.join(DATA_PATH, "proportions.csv")
    if not os.path.exists(gt_file):
        sys.exit(f"Missing ground-truth proportions at {gt_file}")
    true_props_master = pd.read_csv(gt_file, index_col=0).sort_index()
    print("Loaded ground truth:", true_props_master.shape)
else:
    true_props_master = None

# --------------------------------------------------------------------
# 2) Collect deconvolution outputs
# --------------------------------------------------------------------
bp_files = sorted(glob.glob(os.path.join(RESULTS_PATH, f"*_{method}_proportions.csv")))
print(f"Found {len(bp_files)} {method} files")

# --------------------------------------------------------------------
# 3) Donor list (per-donor only) (same for all methods)
# --------------------------------------------------------------------
if mode == "perdonor":
    donors_csv = os.path.join(DATA_PATH, "donors.csv")
    if not os.path.exists(donors_csv):
        sys.exit(f"Cannot find donors.csv at {donors_csv}")
    with open(donors_csv) as fh:
        donor_list = [d.strip() for d in fh.readline().split(",") if d.strip()]
else:
    donor_list = []


# --------------------------------------------------------------------
# 4) Filename parser
# --------------------------------------------------------------------
def parse_bp_filename(path: str, mode: str, method: str):
    fname = os.path.basename(path)
    core = re.sub(rf"_{method}_proportions\.csv$", "", fname)

    # ---------- simulation ----------
    if mode == "simulation":
        if core in {"sc_raw", "sn_raw", "degIntAllSN"}:
            return None, core
        if core.startswith("ref_"):
            tail = core[4:]
            for t in valid_transforms:
                if tail.endswith("_" + t):
                    return None, t
        return None, None

    # ------------ real bulk ----------
    if mode == "real":
        # ----- special sc/sn-raw references -----
        if core in {"sc_raw_real", "sn_raw_real"}:
            return None, core  # transform = sc_raw_real / sn_raw_real

        # “all-donors” versions, e.g. sc_raw_alldonors_alldonors
        if core.startswith(("sc_raw_", "sn_raw_")):
            parts = core.split("_")  # ['sc','raw', ...]
            transform = "_".join(parts[:2])  # 'sc_raw' or 'sn_raw'
            return None, transform

        # ----- aggregate references -----
        # pattern:  ref_real_<TRANSFORM>
        if core.startswith("ref_real_"):
            trans = core[9:]  # strip 'ref_real_'
            if trans in valid_transforms:
                return None, trans
            # lowercase variant (degpcaSN)
            if trans.lower() in valid_transforms:
                return None, trans.lower()
        # anything else (incl. ref_<DONOR>_*) is ignored
        return None, None
    # ---------- per-donor ----------
    if mode == "perdonor" and core.startswith("ref_"):
        # tail looks like "Hs_SAT_01-1_degSN" or "Hs_SAT_266-1_pcaSN"
        tail = core[4:]

        # Find which donor this tail starts with
        donor = None
        transform = None
        for d in donor_list:
            prefix = d + "_"
            if tail.startswith(prefix):
                donor = d
                transform = tail[len(prefix) :]  # everything after "<donor>_"
                break

        # If we didn't match any donor prefix, ignore this file
        if donor is None or transform is None:
            return None, None

        # Keep only known transforms (case-insensitive)
        if transform in valid_transforms or transform.lower() in valid_transforms:
            transform = (
                transform if transform in valid_transforms else transform.lower()
            )
            return donor, transform

        # Unknown transform => skip
        return None, None

    return None, None


# --------------------------------------------------------------------
# 5) Build evaluation DataFrame
# --------------------------------------------------------------------
eval_records = []

for fp in bp_files:
    donor, trans = parse_bp_filename(fp, mode=mode, method=method)
    if trans is None:
        print("  » skip (unrecognised):", os.path.basename(fp))
        continue

    print(f"  Loading {os.path.basename(fp)} : donor={donor}, transform={trans}")
    pred = pd.read_csv(fp, sep="\t", index_col=0)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    # ground-truth or placeholder
    if mode == "simulation":
        # harmonise sample IDs: cast to string, then strip leading "X" if present
        pred.index = pred.index.astype(str).str.lstrip("X")
        true_props_master.index = true_props_master.index.astype(str).str.lstrip("X")

        # align only overlapping
        common_idx = pred.index.intersection(true_props_master.index)
        common_cols = pred.columns.intersection(true_props_master.columns)
        pred = pred.loc[common_idx, common_cols]
        truth = true_props_master.loc[common_idx, common_cols]
    else:
        truth = pred.copy()

    # reindex 1..N
    pred.index = truth.index = range(1, len(pred) + 1)

    # melt + merge
    pl = (
        pred.reset_index()
        .melt(id_vars="index", var_name="CellType", value_name="PredProp")
        .rename(columns={"index": "SampleID"})
    )
    tl = (
        truth.reset_index()
        .melt(id_vars="index", var_name="CellType", value_name="TrueProp")
        .rename(columns={"index": "SampleID"})
    )

    # -------------------------------------------------------------
    # Derive the held-out cell type from the filename, if any
    # -------------------------------------------------------------
    holdout = "None"
    core = os.path.basename(fp).replace(f"_{method}_proportions.csv", "")
    if core.startswith("ref_") and trans in valid_transforms:
        # ref_<CELLTYPE>_<TRANSFORM>
        tail = core[4:]  # strip "ref_"
        cell_part = tail[: -(len(trans) + 1)]  # drop "_<TRANSFORM>"
        holdout = cell_part.replace("_", " ")  # restore spaces

    merged = pl.merge(tl, on=["SampleID", "CellType"])
    merged["Transform"] = trans
    merged["HoldoutCell"] = holdout
    merged["Donor"] = donor or "None"
    eval_records.append(merged)

eval_df = pd.concat(eval_records, ignore_index=True) if eval_records else pd.DataFrame()
print("eval_df:", eval_df.shape)

# --------------------------------------------------------------------
# 6) Per-sample metrics
# --------------------------------------------------------------------
corr_rows = []
for (t, samp), g in eval_df.groupby(["Transform", "SampleID"]):
    if g.shape[0] < 2:
        continue
    pr = pearsonr(g["TrueProp"], g["PredProp"])[0]
    rmse = root_mean_squared_error(g["TrueProp"], g["PredProp"])
    corr_rows.append({"Transform": t, "SampleID": samp, "Pearson": pr, "RMSE": rmse})
corr_df = pd.DataFrame(corr_rows)
print("corr_df:", corr_df.shape)

# --------------------------------------------------------------------
# 7) Save
# --------------------------------------------------------------------
tag = "_perdonor" if mode == "perdonor" else ""
eval_out = os.path.join(RESULTS_PATH, f"results{tag}_{method}_{args.dataset}.csv")
corr_out = os.path.join(RESULTS_PATH, f"evaluation{tag}_{method}_{args.dataset}.csv")
eval_df.to_csv(eval_out, index=False)
corr_df.to_csv(corr_out, index=False)
print("Saved: ", eval_out)
print("Saved: ", corr_out)
