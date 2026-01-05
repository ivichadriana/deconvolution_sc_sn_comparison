"""
Reference dimension checker for BayesPrism-ready outputs.

What it checks
--------------
1) For each reference pair:
   - {name}_signal.csv and {name}_cell_state.csv both exist.
   - #cells in signal == #rows in cell_state (auto-detects if signal is transposed).
   - gene_count > 0 and cell_count > 0.

2) For each "scope" (e.g., ref_<CELLTYPE>_*, ref_real_*, ref_<DONOR>_*):
   - Every DEG-filtered reference (name contains "deg") has
     gene_count <= max gene_count among NON-DEG variants within the same scope
     (rawSN, pcaSN, scviSN, scvi_LSshift_SN).
"""

import argparse
import sys
from pathlib import Path
import re
import pandas as pd

# Known variant suffixes we expect from your pipelines.
NON_DEG_VARIANTS = {
    "rawSN",
    "pcaSN",
    "scviSN",
    "scvi_LSshift_SN",
}
DEG_VARIANTS = {
    "degSN",
    "degOtherSN",
    "degIntSN",
    "degRandSN",
    "degPCA_SN",
    "degScviSN",
    "degScviLSshift_SN",
    "degIntAllSN",  # whole-SN filtered variants
}

# Regex to grab the trailing variant piece from names like:
#  - ref_<celltype>_rawSN
#  - ref_<donor>_degScviLSshift_SN
#  - ref_real_degPCA_SN
VARIANT_RE = re.compile(
    r"(.*)_(rawSN|pcaSN|scviSN|scvi_LSshift_SN|degSN|degOtherSN|degIntSN|degRandSN|degPCA_SN|degScviSN|degScviLSshift_SN|degIntAllSN)$"
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output_path", required=True, help="Folder where references were written"
    )
    ap.add_argument(
        "--fail_on_warning",
        action="store_true",
        help="Exit non-zero on warnings (default only errors are fatal)",
    )
    return ap.parse_args()


def load_pair(signal_path: Path, state_path: Path):
    """Load signal and cell_state; return (genes, cells, orientation, df_signal.index/columns)."""
    sig = pd.read_csv(signal_path, index_col=0)
    st = pd.read_csv(state_path, index_col=0)

    # Try to match cells along columns first (common layout: genes x cells)
    if sig.shape[1] == st.shape[0]:
        genes = sig.shape[0]
        cells = sig.shape[1]
        orient = "genes x cells"
    elif sig.shape[0] == st.shape[0]:
        # Transposed layout
        genes = sig.shape[1]
        cells = sig.shape[0]
        orient = "cells x genes (transposed)"
        sig = sig.T  # normalize to genes x cells in-memory for checks that need it
    else:
        raise ValueError(
            f"Cell count mismatch: signal shape={sig.shape}, cell_state rows={st.shape[0]}"
        )
    return genes, cells, orient


def split_scope_and_variant(ref_name: str):
    """
    Given a base ref name without the trailing '_signal/_cell_state' (e.g. 'ref_real_degSN'),
    return (scope, variant, is_deg).
    Scope keeps everything before the variant so that all variants of the same donor/celltype group together.
    """
    m = VARIANT_RE.match(ref_name)
    if not m:
        # Non-standard references like 'sc_raw', 'sn_raw', 'sc_raw_real', 'sn_raw_real'
        return ref_name, None, False
    scope, variant = m.group(1), m.group(2)
    is_deg = (variant in DEG_VARIANTS) or ("deg" in variant.lower())
    return scope, variant, is_deg


def main():
    args = parse_args()
    outdir = Path(args.output_path)
    if not outdir.exists():
        print(f"[ERROR] Output folder not found: {outdir}")
        sys.exit(2)

    # Discover all *_signal.csv files and pair with *_cell_state.csv
    signal_files = sorted(outdir.glob("**/*_signal.csv"))
    if not signal_files:
        print(f"[ERROR] No *_signal.csv files found under {outdir}")
        sys.exit(2)

    rows = []
    errors = []
    warnings = []

    # We’ll compute per-scope baseline (max non-DEG gene count)
    scope2_nondeg_max_genes = {}

    # First pass: read, basic dimension checks, and compute baseline per scope
    for sig in signal_files:
        name = sig.stem.replace("_signal", "")
        state = sig.with_name(name + "_cell_state.csv")
        if not state.exists():
            errors.append(f"[MISSING] {name}: missing {state.name}")
            continue

        try:
            genes, cells, orient = load_pair(sig, state)
        except Exception as e:
            errors.append(f"[BAD_SHAPE] {name}: {e}")
            continue

        scope, variant, is_deg = split_scope_and_variant(name)

        rows.append(
            {
                "name": name,
                "scope": scope,
                "variant": variant or "",
                "is_deg": is_deg,
                "genes": genes,
                "cells": cells,
                "orientation": orient,
                "signal_path": str(sig),
                "cell_state_path": str(state),
            }
        )

        if genes <= 0 or cells <= 0:
            errors.append(f"[EMPTY] {name}: genes={genes}, cells={cells}")

        # Track non-DEG baseline per scope
        if variant in NON_DEG_VARIANTS:
            scope2_nondeg_max_genes[scope] = max(
                scope2_nondeg_max_genes.get(scope, 0), genes
            )

    # Second pass: enforce DEG ≤ baseline for each scope (when baseline exists)
    for r in rows:
        if r["is_deg"]:
            base = scope2_nondeg_max_genes.get(r["scope"], None)
            if base is None:
                warnings.append(
                    f"[NO_BASELINE] {r['name']}: no non-DEG baseline found in scope '{r['scope']}' "
                    f"to compare gene count ({r['genes']})."
                )
            else:
                if r["genes"] > base:
                    errors.append(
                        f"[DEG_TOO_LARGE] {r['name']}: genes={r['genes']} > baseline_nonDEG_genes={base} "
                        f"(scope='{r['scope']}')."
                    )

    # Pretty print summary table
    if rows:
        df = pd.DataFrame(rows).sort_values(by=["scope", "is_deg", "variant", "name"])
        print("\n=== Reference dimension summary ===")
        with pd.option_context("display.max_rows", None, "display.width", 140):
            print(
                df[
                    [
                        "name",
                        "scope",
                        "variant",
                        "is_deg",
                        "genes",
                        "cells",
                        "orientation",
                    ]
                ].to_string(index=False)
            )
    else:
        print("[ERROR] No valid reference pairs to report.")

    # Print diagnostics
    if warnings:
        print("\n--- Warnings ---")
        for w in warnings:
            print(w)
    if errors:
        print("\n--- Errors ---")
        for e in errors:
            print(e)

    if errors or (warnings and args.fail_on_warning):
        sys.exit(1)
    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
