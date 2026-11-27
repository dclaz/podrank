"""Human-readable R1 diagnostics and optional plots."""

from __future__ import annotations

from podrank.report.markdown import (
    calibration_markdown,
    forward_markdown,
    ppc_markdown,
)

__all__ = ["calibration_markdown", "forward_markdown", "ppc_markdown"]
