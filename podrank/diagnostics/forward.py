"""Forward-split and held-out-cluster evaluation (R1 diagnostics, §10.1/S3)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.data.table import GameTable, normalize_game_row
from podrank.diagnostics.calibration import CalibrationReport, predictive_quality
from podrank.rating.fit import fit
from podrank.types import GameRow


@dataclass(frozen=True, slots=True)
class ForwardWindow:
    index: int
    train_clusters: int
    train_games: int
    scored_games: int
    excluded_games: int
    log_loss: dict[str, float | None]
    brier: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class ForwardSplitReport:
    windows: tuple[ForwardWindow, ...]
    full_log_loss: float | None
    best_baseline: str | None
    mean_difference: float | None
    standard_error: float | None
    not_worse: bool | None


def _cluster_key(row: GameRow) -> tuple[str, str]:
    return ("cluster", row.cluster_id) if row.cluster_id else ("game", row.game_id)


def _log_loss(report: CalibrationReport, name: str) -> float | None:
    return report.log_loss.get(name)


def forward_split(
    games: GameTable,
    agents: AgentRegistry,
    decks: DeckRegistry,
    *,
    n_windows: int = 5,
    n_burn: int = 20,
    **fit_kwargs: Any,
) -> ForwardSplitReport:
    """Fit on earlier clusters and score each later window once."""

    if isinstance(n_windows, bool) or not isinstance(n_windows, int) or n_windows < 2:
        raise ValueError("n_windows must be an integer >= 2")
    if isinstance(n_burn, bool) or not isinstance(n_burn, int) or n_burn < 0:
        raise ValueError("n_burn must be a non-negative integer")
    records = list(games)
    rows = [normalize_game_row(record) for record in records]
    by_cluster: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        by_cluster.setdefault(_cluster_key(row), []).append(index)
    times = {
        key: min(rows[index].played_at for index in indices) for key, indices in by_cluster.items()
    }
    ordered = sorted(by_cluster, key=lambda key: (times[key], key))
    if len(ordered) < n_windows:
        raise ValueError("n_windows exceeds the number of clusters")
    windows = [ordered[index::n_windows] for index in range(n_windows)]
    # Re-chunk contiguously rather than round-robin so earlier windows stay earlier.
    size = len(ordered) // n_windows
    remainder = len(ordered) % n_windows
    windows = []
    start = 0
    for index in range(n_windows):
        length = size + (1 if index < remainder else 0)
        windows.append(ordered[start : start + length])
        start += length

    training_clusters: list[tuple[str, str]] = list(windows[0])
    training_rows = {key for key in training_clusters}
    reports: list[ForwardWindow] = []
    cluster_values: dict[str, list[float]] = {}
    for window_index in range(1, n_windows):
        current = windows[window_index]
        burn = 0
        train_agents = {
            agent
            for key in training_rows
            for index in by_cluster[key]
            for agent in rows[index].agents
        }
        train_decks = {
            deck for key in training_rows for index in by_cluster[key] for deck in rows[index].decks
        }
        for key in current:
            if burn >= n_burn:
                break
            new_agent = any(
                agent not in train_agents
                for index in by_cluster[key]
                for agent in rows[index].agents
            )
            new_deck = any(
                deck not in train_decks for index in by_cluster[key] for deck in rows[index].decks
            )
            if new_agent or new_deck:
                training_rows.add(key)
                for index in by_cluster[key]:
                    train_agents.update(rows[index].agents)
                    train_decks.update(rows[index].decks)
                burn += 1
        train_records = [records[index] for key in training_rows for index in by_cluster[key]]
        post = fit(GameTable.from_records(train_records), agents, decks, **fit_kwargs)
        scoring_records = [records[index] for key in current for index in by_cluster[key]]
        report = predictive_quality(post, scoring_records)
        for name, values in report.cluster_values.items():
            cluster_values.setdefault(name, []).extend(values)
        reports.append(
            ForwardWindow(
                index=window_index,
                train_clusters=len(training_rows),
                train_games=len(train_records),
                scored_games=report.n_games,
                excluded_games=report.n_excluded,
                log_loss={name: _log_loss(report, name) for name in report.log_loss},
                brier={name: value for name, value in report.brier.items()},
            )
        )
        training_rows.update(current)
    if not reports:
        return ForwardSplitReport((), None, None, None, None, None)
    baseline_names = ("uniform", "seat", "theta", "theta_deck")
    means = {
        name: float(np.mean(cluster_values[name]))
        for name in baseline_names
        if cluster_values.get(name)
    }
    best = min(means, key=means.__getitem__) if means else None
    full_values = np.asarray(cluster_values.get("full", []), dtype=np.float64)
    full_mean = float(full_values.mean()) if full_values.size else None
    difference = None
    standard_error = None
    not_worse = None
    if best is not None and full_values.size:
        baseline_values = np.asarray(cluster_values[best], dtype=np.float64)
        if baseline_values.size == full_values.size and baseline_values.size > 1:
            deltas = full_values - baseline_values
            difference = float(deltas.mean())
            standard_error = float(deltas.std(ddof=1) / math.sqrt(deltas.size))
            not_worse = difference <= 2.0 * standard_error
    return ForwardSplitReport(
        windows=tuple(reports),
        full_log_loss=full_mean,
        best_baseline=best,
        mean_difference=difference,
        standard_error=standard_error,
        not_worse=not_worse,
    )


__all__ = ["ForwardSplitReport", "ForwardWindow", "forward_split"]
