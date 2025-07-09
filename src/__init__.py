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

from .helpers import *
from .deg_funct import *
from .transforms import *