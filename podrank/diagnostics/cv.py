"""Whole-cluster K-fold evaluation (R1 diagnostics, S3)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.data.table import GameTable, normalize_game_row
from podrank.diagnostics.calibration import predictive_quality
from podrank.rating.fit import fit


@dataclass(frozen=True, slots=True)
class CVFold:
    index: int
    train_clusters: int
    scored_games: int
    excluded_games: int
    log_loss: dict[str, float | None]
    brier: dict[str, float | None]


@dataclass(frozen=True, slots=True)
class ClusterCVReport:
    folds: tuple[CVFold, ...]
    full_log_loss: float | None
    best_baseline: str | None
    mean_difference: float | None
    standard_error: float | None
    not_worse: bool | None


def cluster_cv(
    games: GameTable,
    agents: AgentRegistry,
    decks: DeckRegistry,
    *,
    folds: int = 5,
    **fit_kwargs: Any,
) -> ClusterCVReport:
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    records = list(games)
    rows = [normalize_game_row(record) for record in records]
    keys: list[tuple[str, str]] = []
    indices: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = ("cluster", row.cluster_id) if row.cluster_id else ("game", row.game_id)
        if key not in indices:
            keys.append(key)
        indices.setdefault(key, []).append(index)
    if len(keys) < folds:
        raise ValueError("folds exceeds the number of clusters")
    fold_of = {key: position % folds for position, key in enumerate(sorted(keys))}
    reports: list[CVFold] = []
    cluster_values: dict[str, list[float]] = {}
    for fold in range(folds):
        train = [records[index] for key in keys if fold_of[key] != fold for index in indices[key]]
        test = [records[index] for key in keys if fold_of[key] == fold for index in indices[key]]
        post = fit(GameTable.from_records(train), agents, decks, **fit_kwargs)
        report = predictive_quality(post, test)
        for name, values in report.cluster_values.items():
            cluster_values.setdefault(name, []).extend(values)
        reports.append(
            CVFold(
                index=fold,
                train_clusters=len(keys) - sum(1 for key in keys if fold_of[key] == fold),
                scored_games=report.n_games,
                excluded_games=report.n_excluded,
                log_loss=dict(report.log_loss),
                brier=dict(report.brier),
            )
        )
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
    return ClusterCVReport(
        folds=tuple(reports),
        full_log_loss=full_mean,
        best_baseline=best,
        mean_difference=difference,
        standard_error=standard_error,
        not_worse=not_worse,
    )


__all__ = ["CVFold", "ClusterCVReport", "cluster_cv"]
