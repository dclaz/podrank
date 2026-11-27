"""Decisive-game predictive quality and reliability (R1 diagnostics, §10.1)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from podrank.data.table import GameTable, normalize_game_row
from podrank.rating.posterior import Posterior
from podrank.types import GameRow

Baseline = Literal["uniform", "seat", "theta", "theta_deck", "full"]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    predicted: float
    observed: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    n_games: int
    n_clusters: int
    n_excluded: int
    log_loss: dict[str, float | None]
    brier: dict[str, float | None]
    cluster_log_loss: dict[str, float | None]
    cluster_brier: dict[str, float | None]
    reliability: tuple[ReliabilityBin, ...]


def _parameter(post: Posterior, name: str) -> float | None:
    position = post.parameter_index.positions.get(name)
    return None if position is None else float(post.mode[position])


def _utilities(post: Posterior, row: GameRow, baseline: Baseline) -> np.ndarray | None:
    if baseline == "uniform":
        return np.zeros(4, dtype=np.float64)
    values = np.zeros(4, dtype=np.float64)
    for seat, agent in enumerate(row.agents):
        theta = _parameter(post, f"theta[{agent}]")
        if theta is None:
            return None
        deck = row.decks[seat]
        if baseline in {"full", "theta_deck"}:
            record = post._decks.by_id.get(deck)  # noqa: SLF001
            if record is None:
                return None
            mu = _parameter(post, f"mu[{record.bracket}]")
            delta = _parameter(post, f"delta[{deck}]")
            if mu is None or delta is None:
                return None
            values[seat] += mu + delta
        if baseline == "full":
            seat_effect = _parameter(post, f"seat[{seat}]")
            if seat_effect is None:
                return None
            values[seat] += seat_effect
            if post.spec.swap_effect:
                eta = _parameter(post, "eta") or 0.0
                values[seat] += eta * row.swap_counts[seat]
            if any(row.swap_counts) and not post.spec.swap_effect:
                return None
        if baseline == "seat":
            seat_effect = _parameter(post, f"seat[{seat}]")
            if seat_effect is None:
                return None
            values[seat] = seat_effect
        elif baseline == "theta":
            values[seat] = theta
        elif baseline in {"theta_deck", "full"}:
            values[seat] += theta
    return values


def _probabilities(utilities: np.ndarray) -> np.ndarray:
    shifted = utilities - utilities.max()
    weights = np.exp(shifted)
    return np.asarray(weights / weights.sum(), dtype=np.float64)


def _brier(probabilities: np.ndarray, winner: int) -> float:
    target = np.zeros(4, dtype=np.float64)
    target[winner] = 1.0
    return float(np.sum((probabilities - target) ** 2))


def _cluster_key(row: GameRow) -> tuple[str, str]:
    return ("cluster", row.cluster_id) if row.cluster_id else ("game", row.game_id)


def predictive_quality(
    post: Posterior,
    games: GameTable | Iterable[Mapping[str, object]],
    *,
    reliability_bins: int = 10,
) -> CalibrationReport:
    """Score decisive games with the fitted model and baselines."""

    if reliability_bins <= 0:
        raise ValueError("reliability_bins must be positive")
    baselines: tuple[Baseline, ...] = ("uniform", "seat", "theta", "theta_deck", "full")
    losses: dict[str, list[float]] = {name: [] for name in baselines}
    briers: dict[str, list[float]] = {name: [] for name in baselines}
    cluster_games: dict[tuple[str, str], list[dict[str, tuple[float, float]]]] = {}
    reliability: list[tuple[float, bool]] = []
    excluded = 0
    for row in (normalize_game_row(record) for record in games):
        if row.outcome != "win" or row.winner_seat is None:
            continue
        scores: dict[str, tuple[float, float]] = {}
        for baseline in baselines:
            utilities = _utilities(post, row, baseline)
            if utilities is None:
                if baseline == "full":
                    excluded += 1
                continue
            probabilities = _probabilities(utilities)
            scores[baseline] = (
                -math.log(float(probabilities[row.winner_seat])),
                _brier(probabilities, row.winner_seat),
            )
        if "full" not in scores:
            continue
        for score_name, (loss, brier) in scores.items():
            losses[score_name].append(loss)
            briers[score_name].append(brier)
        cluster_games.setdefault(_cluster_key(row), []).append(scores)
        full_utilities = _utilities(post, row, "full")
        assert full_utilities is not None
        probabilities = _probabilities(full_utilities)
        top = int(np.argmax(probabilities))
        reliability.append((float(probabilities[top]), top == row.winner_seat))
    cluster_loss: dict[str, list[float]] = {name: [] for name in baselines}
    cluster_brier: dict[str, list[float]] = {name: [] for name in baselines}
    for entries in cluster_games.values():
        for name in baselines:
            if all(name in entry for entry in entries):
                cluster_loss[name].append(float(np.mean([entry[name][0] for entry in entries])))
                cluster_brier[name].append(float(np.mean([entry[name][1] for entry in entries])))
    bins: list[ReliabilityBin] = []
    if reliability:
        edges = np.linspace(0.0, 1.0, reliability_bins + 1)
        probabilities = np.asarray([item[0] for item in reliability])
        correct = np.asarray([item[1] for item in reliability], dtype=np.float64)
        for index in range(reliability_bins):
            lo, hi = float(edges[index]), float(edges[index + 1])
            upper = probabilities <= hi if index == reliability_bins - 1 else probabilities < hi
            mask = (probabilities >= lo) & upper
            count = int(mask.sum())
            if count:
                bins.append(
                    ReliabilityBin(
                        lo=lo,
                        hi=hi,
                        count=count,
                        predicted=float(probabilities[mask].mean()),
                        observed=float(correct[mask].mean()),
                    )
                )
    return CalibrationReport(
        n_games=len(losses["full"]),
        n_clusters=len(cluster_games),
        n_excluded=excluded,
        log_loss={
            name: (float(np.mean(values)) if values else None) for name, values in losses.items()
        },
        brier={
            name: (float(np.mean(values)) if values else None) for name, values in briers.items()
        },
        cluster_log_loss={
            name: (float(np.mean(values)) if values else None)
            for name, values in cluster_loss.items()
        },
        cluster_brier={
            name: (float(np.mean(values)) if values else None)
            for name, values in cluster_brier.items()
        },
        reliability=tuple(bins),
    )


__all__ = ["CalibrationReport", "ReliabilityBin", "predictive_quality"]
