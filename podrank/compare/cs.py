"""Two-sided confidence sequence for R2 (E6, F13)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from podrank.compare.pods import Design

_GRID_STEP = 0.0025
_RANGES: dict[Design, tuple[float, float]] = {
    "duplicate": (-0.25, 0.75),
    "paired": (-1.0, 1.0),
}
_G = 16


@dataclass(frozen=True, slots=True)
class ConfidenceSequence:
    lo: float
    hi: float
    cs_conflict: bool


def grid_points(design: Design) -> tuple[float, ...]:
    x_min, x_max = _RANGES[design]
    count = round((x_max - x_min) / _GRID_STEP)
    return tuple(x_min + index * _GRID_STEP for index in range(1, count))


def _excluded(
    scores: Sequence[float], m: float, lam_bound: float, threshold: float, *, upper: bool
) -> bool:
    values = np.asarray(scores, dtype=np.float64)
    lams = lam_bound * np.arange(1, _G + 1, dtype=np.float64) / _G
    if upper:
        factors = 1.0 + lams[None, :] * (values[:, None] - m)
    else:
        factors = 1.0 + lams[None, :] * (m - values[:, None])
    if np.any(factors <= 0):
        raise ValueError("confidence-sequence factors must stay positive")
    log_wealth = np.cumsum(np.log(factors), axis=0)
    maximum = np.max(log_wealth, axis=1)
    log_mixture = maximum + np.log(np.mean(np.exp(log_wealth - maximum[:, None]), axis=1))
    return bool(np.max(log_mixture) >= math.log(threshold))


def confidence_sequence(
    x_up: Sequence[float], x_low: Sequence[float], *, design: Design, cs_alpha: float
) -> ConfidenceSequence | None:
    if design not in _RANGES:
        raise ValueError("design must be 'duplicate' or 'paired'")
    if len(x_up) != len(x_low):
        raise ValueError("x_up and x_low must have the same length")
    if not x_up:
        return None
    if isinstance(cs_alpha, bool) or not isinstance(cs_alpha, (int, float)):
        raise ValueError("cs_alpha must lie strictly in (0, 1)")
    if not 0 < float(cs_alpha) < 1:
        raise ValueError("cs_alpha must lie strictly in (0, 1)")
    x_min, x_max = _RANGES[design]
    threshold = 2.0 / float(cs_alpha)
    lo = x_min
    hi = x_max
    for m in grid_points(design):
        if _excluded(x_up, m, 0.5 / (m - x_min), threshold, upper=True):
            lo = max(lo, m)
        if _excluded(x_low, m, 0.5 / (x_max - m), threshold, upper=False):
            hi = min(hi, m)
    return ConfidenceSequence(lo=lo, hi=hi, cs_conflict=lo >= hi)


__all__ = ["ConfidenceSequence", "confidence_sequence", "grid_points"]
