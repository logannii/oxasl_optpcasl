"""
OXASL_OPTPCASL

Python library for optimizing multi-PLD pCASL acquisitions
Ported to Python from MATLAB
"""

__author__ = "Joseph Woods"
__copyright__ = "Copyright 2019 University of Nottingham"
__credits__ = ["Joseph Wood, Martin Craig"]
__maintainer__ = "Martin Craig"
__email__ = "martin.craig@eng.ox.ac.uk"

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from oxasl_optpcasl import cost, kinetic_model, main, optimize, scan, structures

__all__ = [
    "__version__",
    "cost",
    "kinetic_model",
    "main",
    "optimize",
    "scan",
    "structures",
]
