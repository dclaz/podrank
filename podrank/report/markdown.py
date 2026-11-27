"""Markdown renderers for R1 diagnostics."""

from __future__ import annotations

from podrank.diagnostics.calibration import CalibrationReport
from podrank.diagnostics.forward import ForwardSplitReport
from podrank.diagnostics.ppc import PpcReport


def _value(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def calibration_markdown(report: CalibrationReport) -> str:
    lines = [
        "# Calibration",
        "",
        f"Scored **{report.n_games}** decisive games in **{report.n_clusters}** clusters "
        f"({report.n_excluded} excluded).",
        "",
        "| Model | Log-loss | Brier | Cluster log-loss | Cluster Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in report.log_loss:
        lines.append(
            f"| {name} | {_value(report.log_loss[name])} | {_value(report.brier[name])} | "
            f"{_value(report.cluster_log_loss.get(name))} | "
            f"{_value(report.cluster_brier.get(name))} |"
        )
    lines.extend(
        ["", "## Reliability", "", "| Bin | Count | Predicted | Observed |", "|---|---:|---:|---:|"]
    )
    for bin in report.reliability:
        lines.append(
            f"| {bin.lo:.1f}–{bin.hi:.1f} | {bin.count} | "
            f"{bin.predicted:.3f} | {bin.observed:.3f} |"
        )
    return "\n".join(lines) + "\n"


def ppc_markdown(report: PpcReport) -> str:
    lines = [
        "# Posterior predictive checks",
        "",
        f"Scored **{report.n_games}** decisive games ({report.n_excluded} excluded).",
        "",
        "| Group | Key | Observed | Expected | z | Flag |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in report.rows:
        lines.append(
            f"| {row.group} | {row.key} | {row.observed} | {row.expected:.3f} | "
            f"{row.z:.3f} | {'**yes**' if row.flagged else ''} |"
        )
    return "\n".join(lines) + "\n"


def forward_markdown(report: ForwardSplitReport) -> str:
    lines = [
        "# Forward split",
        "",
        "| Window | Train clusters | Scored | Excluded | Full log-loss | Best baseline |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for window in report.windows:
        best = min(
            (
                name
                for name in window.log_loss
                if name != "full" and window.log_loss[name] is not None
            ),
            key=lambda name: window.log_loss[name] or 0.0,
            default=None,
        )
        lines.append(
            f"| {window.index} | {window.train_clusters} | {window.scored_games} | "
            f"{window.excluded_games} | {_value(window.log_loss.get('full'))} | "
            f"{best} {_value(window.log_loss.get(best) if best else None)} |"
        )
    lines.extend(
        [
            "",
            f"Pooled full log-loss: **{_value(report.full_log_loss)}**; "
            f"best baseline: **{report.best_baseline}**; "
            f"difference: **{_value(report.mean_difference)}** "
            f"(SE {_value(report.standard_error)}); not worse: **{report.not_worse}**.",
        ]
    )
    return "\n".join(lines) + "\n"
