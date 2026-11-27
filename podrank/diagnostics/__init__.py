"""Post-R1 diagnostics: calibration, posterior predictive checks, forward split."""

from __future__ import annotations

from podrank.diagnostics.calibration import (
    CalibrationReport,
    ReliabilityBin,
    predictive_quality,
)
from podrank.diagnostics.forward import ForwardSplitReport, ForwardWindow, forward_split
from podrank.diagnostics.ppc import PpcGroup, PpcReport, PpcRow, posterior_predictive
from podrank.diagnostics.tuning import (
    SigmaBetaCVResult,
    SigmaBetaCVSelection,
    SigmaBetaResult,
    SigmaBetaSelection,
    log_evidence,
    tune_sigma_beta,
    tune_sigma_beta_cv,
)

__all__ = [
    "CalibrationReport",
    "ForwardSplitReport",
    "ForwardWindow",
    "PpcGroup",
    "SigmaBetaCVResult",
    "SigmaBetaCVSelection",
    "SigmaBetaResult",
    "SigmaBetaSelection",
    "PpcReport",
    "PpcRow",
    "ReliabilityBin",
    "forward_split",
    "log_evidence",
    "posterior_predictive",
    "predictive_quality",
    "tune_sigma_beta",
    "tune_sigma_beta_cv",
]
