"""Sigma-beta selection by Laplace evidence (R1 diagnostics, S4)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.data.table import GameTable
from podrank.rating.fit import fit
from podrank.rating.likelihood import objective
from podrank.rating.spec import ModelSpec


@dataclass(frozen=True, slots=True)
class SigmaBetaResult:
    sigma_beta: float
    log_evidence: float


@dataclass(frozen=True, slots=True)
class SigmaBetaSelection:
    results: tuple[SigmaBetaResult, ...]
    best: SigmaBetaResult


def log_evidence(post: Any) -> float:
    """Laplace log evidence at the posterior mode."""

    precision = np.asarray(post.precision, dtype=np.float64)
    hessian = np.asarray(post.data_hessian, dtype=np.float64) + np.diag(precision)
    factor = np.linalg.cholesky(hessian)
    logdet = 2.0 * float(np.sum(np.log(np.diag(factor))))
    parameter_count = hessian.shape[0]
    value = objective(post.mode, post.patterns, post.precision)
    prior_normalisation = -0.5 * float(np.sum(np.log(2.0 * math.pi / precision)))
    return float(
        -float(value)
        + prior_normalisation
        + 0.5 * parameter_count * math.log(2.0 * math.pi)
        - 0.5 * logdet
    )


def tune_sigma_beta(
    games: GameTable,
    agents: AgentRegistry,
    decks: DeckRegistry,
    *,
    grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    **fit_kwargs: Any,
) -> SigmaBetaSelection:
    if not grid:
        raise ValueError("grid must not be empty")
    results: list[SigmaBetaResult] = []
    base_spec: ModelSpec = fit_kwargs.pop("spec", None) or ModelSpec()
    for sigma in sorted(set(grid)):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma_beta grid values must be finite and > 0")
        tuned = ModelSpec(
            swap_effect=base_spec.swap_effect,
            allow_pooled_variants=base_spec.allow_pooled_variants,
            weak_ratio=base_spec.weak_ratio,
            tol_eig=base_spec.tol_eig,
            tol_alias=base_spec.tol_alias,
            sigma_beta=sigma,
            max_params=base_spec.max_params,
            max_dense_bytes=base_spec.max_dense_bytes,
        )
        post = fit(games, agents, decks, tuned, **fit_kwargs)
        results.append(SigmaBetaResult(sigma, log_evidence(post)))
    best = max(results, key=lambda item: item.log_evidence)
    return SigmaBetaSelection(tuple(results), best)


__all__ = [
    "SigmaBetaCVResult",
    "SigmaBetaCVSelection",
    "SigmaBetaResult",
    "SigmaBetaSelection",
    "log_evidence",
    "tune_sigma_beta",
    "tune_sigma_beta_cv",
]


@dataclass(frozen=True, slots=True)
class SigmaBetaCVResult:
    sigma_beta: float
    log_loss: float


@dataclass(frozen=True, slots=True)
class SigmaBetaCVSelection:
    results: tuple[SigmaBetaCVResult, ...]
    best: SigmaBetaCVResult


def tune_sigma_beta_cv(
    games: GameTable,
    agents: AgentRegistry,
    decks: DeckRegistry,
    *,
    grid: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0),
    folds: int = 5,
    **fit_kwargs: Any,
) -> SigmaBetaCVSelection:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if not grid:
        raise ValueError("grid must not be empty")
    records = list(games)
    from podrank.data.table import normalize_game_row

    rows = [normalize_game_row(record) for record in records]
    keys: list[tuple[str, str]] = []
    indices: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = ("cluster", row.cluster_id) if row.cluster_id else ("game", row.game_id)
        if key not in indices:
            keys.append(key)
        indices.setdefault(key, []).append(index)
    fold_of = {key: position % folds for position, key in enumerate(sorted(keys))}
    base_spec: ModelSpec = fit_kwargs.pop("spec", None) or ModelSpec()
    results: list[SigmaBetaCVResult] = []
    for sigma in sorted(set(grid)):
        spec = ModelSpec(
            swap_effect=base_spec.swap_effect,
            allow_pooled_variants=base_spec.allow_pooled_variants,
            weak_ratio=base_spec.weak_ratio,
            tol_eig=base_spec.tol_eig,
            tol_alias=base_spec.tol_alias,
            sigma_beta=sigma,
            max_params=base_spec.max_params,
            max_dense_bytes=base_spec.max_dense_bytes,
        )
        losses: list[float] = []
        for fold in range(folds):
            train = [
                records[index] for key in keys if fold_of[key] != fold for index in indices[key]
            ]
            test = [
                records[index] for key in keys if fold_of[key] == fold for index in indices[key]
            ]
            post = fit(GameTable.from_records(train), agents, decks, spec, **fit_kwargs)
            from podrank.diagnostics.calibration import predictive_quality

            report = predictive_quality(post, test)
            values = report.cluster_values.get("full", ())
            if values:
                losses.append(float(np.mean(values)))
        if losses:
            results.append(SigmaBetaCVResult(sigma, float(np.mean(losses))))
    if not results:
        raise ValueError("no fold produced a scored cluster")
    best = min(results, key=lambda item: item.log_loss)
    return SigmaBetaCVSelection(tuple(results), best)
