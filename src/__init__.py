import logging
from importlib.metadata import version
from rich.console import Console
from rich.logging import RichHandler

# # Import helper functions
# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Nice logging outputs
console = Console(force_terminal=True)
if console.is_jupyter is True:
    console.is_jupyter = False
ch = RichHandler(show_path=False, console=console, show_time=False)
formatter = logging.Formatter("Deconvolution Comparison SN vs. SC: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Prevent double outputs
logger.propagate = False

from .helpers import pick_best_datasets, prepare_data, split_single_cell_data, pick_cells, make_pseudobulks, make_references
from .helpers import run_deseq2_for_cell_type, differential_expression_analysis, differential_expression_analysis_parallel, remove_diff_genes
from .helpers import save_cibersort, save_bayesprism_pseudobulks, save_bayesprism_references, load_MBC_data, make_prop_table, remove_unassigned_cells
from .helpers import qc_check_references, qc_check_pseudobulks, qc_check_cell_types_match, match_cell_types, assign_cell_types
from .helpers import split_ID, split_ID_2, merge_strings, load_PNB_data, filter_out_cell_markers, create_fixed_pseudobulk, transform_heldout_sn_to_mean_sc

__all__ = [
    "pick_best_datasets", 
    "prepare_data",
    "split_single_cell_data",
    "pick_cells",
    "make_references",
    "make_pseudobulks",
    "run_deseq2_for_cell_type",
    "differential_expression_analysis",
    "differential_expression_analysis_parallel",
    "save_cibersort",
    "save_bayesprism_references",
    "save_bayesprism_pseudobulks",
    "match_cell_types",
    "load_MBC_data", 
    "make_prop_table",
    "assign_cell_types",
    "split_ID",
    "split_ID_2",
    "merge_strings",
    "load_PNB_data",
    "filter_out_cell_markers",
    "create_fixed_pseudobulk",
    "transform_heldout_sn_to_mean_sc",
    "remove_unassigned_cells"
]
