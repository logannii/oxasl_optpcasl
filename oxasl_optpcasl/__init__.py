"""
OXASL_OPTPCASL

Python library for optimizing multi-PLD pCASL acquisitions
Ported to Python from MATLAB
"""

from types import SimpleNamespace

__author__ = "Joseph Woods"
__copyright__ = "Copyright 2019 University of Nottingham"
__credits__ = ["Joseph Wood, Martin Craig"]
__maintainer__ = "Martin Craig"
__email__ = "martin.craig@eng.ox.ac.uk"

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from . import cost, kinetic_model, optimize, scan, structures
from .cost import ATTCost, CBFCost, DOptimalCost, LOptimalCost
from .kinetic_model import BuxtonPcasl
from .optimize import Optimizer
from .scan import (
    FixedLDPcaslProtocol,
    Hadamard,
    HadamardFreeLunch,
    HadamardMultiLd,
    HadamardSingleLd,
    HadamardT1Decay,
    MultiPLDPcaslMultiLD,
    MultiPLDPcaslVarLD,
)
from .structures import ATTDist, Limits, PhysParams, ScanParams

ASLParams = PhysParams
ASLScan = ScanParams
VAR_MULTI_PCASL = "var_multi_pCASL"


class ASLScanCompat(ScanParams):
    """Compatibility wrapper for the historical ASLScan API used by examples/tests."""

    def __init__(self, protocol=None, **kwargs):
        self.protocol = protocol or VAR_MULTI_PCASL
        kwargs.setdefault("duration", 300)
        kwargs.setdefault("npld", 6)
        kwargs.setdefault("readout", 0.5)
        kwargs.setdefault("noise", 0.0013)
        kwargs.setdefault("ld", [1.8])
        if "slices" in kwargs and "nslices" not in kwargs:
            kwargs["nslices"] = kwargs.pop("slices")
        kwargs.setdefault("nslices", 1)
        kwargs.setdefault("slicedt", 0.0)
        if "lds" in kwargs and "ld" not in kwargs:
            kwargs["ld"] = kwargs.pop("lds")
        super().__init__(**kwargs)
        self.slices = self.nslices


ASLScan = ASLScanCompat


class _CompatOptimizer:
    """Compatibility wrapper over the current optimizer implementation."""

    def __init__(self, params, scan, att_dist, lims, ld_lims=None, cost_model=None):
        self.params = params
        self.scan = scan
        self.att_dist = att_dist
        self.lims = lims
        self.ld_lims = ld_lims or Limits(0.1, 1.8, 0.025, name="LD")
        self.cost_model = cost_model
        self.kinetic_model = BuxtonPcasl(params)
        self.protocol = self._build_protocol()

    def _build_protocol(self):
        scan_params = self.scan
        # Default to the historical fixed-LD multi-PLD protocol unless the caller
        # explicitly requests a variable-LD protocol via a dedicated flag.
        if getattr(scan_params, "optimize_ld", False):
            protocol_class = MultiPLDPcaslVarLD
        else:
            protocol_class = FixedLDPcaslProtocol

        return protocol_class(
            self.kinetic_model,
            scan_params,
            self.att_dist,
            self.lims,
            self.ld_lims,
        )

    def optimize(self, reps=1):
        optimizer = Optimizer(self.protocol, self.cost_model)
        output = optimizer.optimize(self.protocol.initial_params(), reps=reps)
        result = SimpleNamespace(**output)
        result.plds = output["plds"]
        result.lds = output.get("lds")
        result.best_cost = output["best_cost"]
        result.params = output["params"]
        result.cov_optimized = self.protocol.cov(output["params"])
        if (
            getattr(result.cov_optimized, "ndim", 0) == 5
            and result.cov_optimized.shape[0] == 1
        ):
            result.cov_optimized = result.cov_optimized[0]
        return result


class DOptimal(_CompatOptimizer):
    def __init__(self, params, scan, att_dist, lims, ld_lims=None):
        super().__init__(
            params,
            scan,
            att_dist,
            lims,
            ld_lims=ld_lims,
            cost_model=DOptimalCost(),
        )


class LOptimal(_CompatOptimizer):
    def __init__(self, A, params, scan, att_dist, lims, ld_lims=None):
        super().__init__(
            params,
            scan,
            att_dist,
            lims,
            ld_lims=ld_lims,
            cost_model=LOptimalCost(A),
        )


LOptimal = LOptimal


def __getattr__(name):
    if name == "main":
        from . import main as main_module

        return main_module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "ASLParams",
    "ASLScan",
    "ATTDist",
    "ATTCost",
    "BuxtonPcasl",
    "CBFCost",
    "DOptimal",
    "DOptimalCost",
    "FixedLDPcaslProtocol",
    "LOptimal",
    "LOptimalCost",
    "Limits",
    "PhysParams",
    "ScanParams",
    "VAR_MULTI_PCASL",
    "cost",
    "kinetic_model",
    "main",
    "optimize",
    "scan",
    "structures",
]
