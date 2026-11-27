"""Optional matplotlib plots for R1 diagnostics."""

from __future__ import annotations

from typing import Any

from podrank.diagnostics.calibration import CalibrationReport
from podrank.diagnostics.ppc import PpcReport


def reliability_plot(report: CalibrationReport) -> Any:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    predicted = [bin.predicted for bin in report.reliability]
    observed = [bin.observed for bin in report.reliability]
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.plot(predicted, observed, marker="o")
    axis.set_xlabel("Predicted top probability")
    axis.set_ylabel("Observed frequency")
    axis.set_title("Reliability")
    return figure


def ppc_plot(report: PpcReport) -> Any:
    import matplotlib.pyplot as plt

    rows = [row for row in report.rows if row.group == "agent"]
    figure, axis = plt.subplots()
    axis.bar([row.key for row in rows], [row.z for row in rows])
    axis.axhline(3.0, color="red", linestyle="--")
    axis.axhline(-3.0, color="red", linestyle="--")
    axis.set_ylabel("z")
    axis.set_title("Agent posterior predictive z")
    axis.tick_params(axis="x", rotation=45)
    return figure
