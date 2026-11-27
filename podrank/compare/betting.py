"""Fixed-mixture betting wealth for R2 (E1, E3, F3)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from podrank.compare.pods import Design

_DEFAULT_GRID = 16
_RANGES: dict[Design, tuple[float, float]] = {
    "duplicate": (-0.25, 0.75),
    "paired": (-1.0, 1.0),
}


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be a finite number")
    return numeric


@dataclass(frozen=True, slots=True)
class Superiority:
    """Upper claim mu > margin, lower claim mu < 0."""

    margin: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "margin", _finite(self.margin, "margin"))


@dataclass(frozen=True, slots=True)
class NonInferiority:
    """Both directions claim mu > -margin."""

    margin: float

    def __post_init__(self) -> None:
        margin = _finite(self.margin, "margin")
        if margin <= 0:
            raise ValueError("NonInferiority margin must be > 0")
        object.__setattr__(self, "margin", margin)


type Hypothesis = Superiority | NonInferiority


@dataclass(frozen=True, slots=True)
class Thresholds:
    design: Design
    x_min: float
    x_max: float
    m_low: float
    m_up: float


def thresholds(hypothesis: Hypothesis, design: Design) -> Thresholds:
    if design not in _RANGES:
        raise ValueError("design must be 'duplicate' or 'paired'")
    x_min, x_max = _RANGES[design]
    if isinstance(hypothesis, Superiority):
        m_up = hypothesis.margin
        m_low = 0.0
    else:
        m_up = -hypothesis.margin
        m_low = -hypothesis.margin
    if not (x_min < m_low <= m_up < x_max):
        raise ValueError(f"invalid thresholds: require {x_min} < m_low <= m_up < {x_max}")
    return Thresholds(design=design, x_min=x_min, x_max=x_max, m_low=m_low, m_up=m_up)


@dataclass(frozen=True, slots=True)
class Wealth:
    """A fixed mixture of nonnegative betting factors."""

    design: Design
    hypothesis: Hypothesis
    alpha_up: float = 0.05
    alpha_low: float = 0.05
    grid: int = _DEFAULT_GRID
    log_wealth_up: np.ndarray = field(init=False, repr=False)
    log_wealth_low: np.ndarray = field(init=False, repr=False)
    lam_up: np.ndarray = field(init=False, repr=False)
    lam_low: np.ndarray = field(init=False, repr=False)
    thresholds: Thresholds = field(init=False)

    def __post_init__(self) -> None:
        resolved = thresholds(self.hypothesis, self.design)
        object.__setattr__(self, "thresholds", resolved)
        for name in ("alpha_up", "alpha_low"):
            alpha = _finite(getattr(self, name), name)
            if not 0 < alpha < 1:
                raise ValueError(f"{name} must lie strictly in (0, 1)")
            object.__setattr__(self, name, alpha)
        if isinstance(self.grid, bool) or not isinstance(self.grid, int) or self.grid <= 0:
            raise ValueError("grid must be a positive integer")
        lam_up_bound = 0.5 / (resolved.m_up - resolved.x_min)
        lam_low_bound = 0.5 / (resolved.x_max - resolved.m_low)
        up = lam_up_bound * np.arange(1, self.grid + 1, dtype=np.float64) / self.grid
        low = lam_low_bound * np.arange(1, self.grid + 1, dtype=np.float64) / self.grid
        object.__setattr__(self, "lam_up", up)
        object.__setattr__(self, "lam_low", low)
        object.__setattr__(self, "log_wealth_up", np.zeros(self.grid, dtype=np.float64))
        object.__setattr__(self, "log_wealth_low", np.zeros(self.grid, dtype=np.float64))

    @property
    def threshold_up(self) -> float:
        return 1.0 / self.alpha_up

    @property
    def threshold_low(self) -> float:
        return 1.0 / self.alpha_low

    def factors_up(self) -> np.ndarray:
        return np.exp(self.log_wealth_up)

    def factors_low(self) -> np.ndarray:
        return np.exp(self.log_wealth_low)

    def mixture_up(self) -> float:
        return _mixture(self.log_wealth_up)

    def mixture_low(self) -> float:
        return _mixture(self.log_wealth_low)

    def crossed_up(self) -> bool:
        return _log_mixture(self.log_wealth_up) >= math.log(self.threshold_up)

    def crossed_low(self) -> bool:
        return _log_mixture(self.log_wealth_low) >= math.log(self.threshold_low)

    def update(self, x_up: float, x_low: float) -> None:
        upper_factors = 1.0 + self.lam_up * (x_up - self.thresholds.m_up)
        lower_factors = 1.0 + self.lam_low * (self.thresholds.m_low - x_low)
        if np.any(upper_factors <= 0) or np.any(lower_factors <= 0):
            raise ValueError("betting factors must stay positive")
        np.add(self.log_wealth_up, np.log(upper_factors), out=self.log_wealth_up)
        np.add(self.log_wealth_low, np.log(lower_factors), out=self.log_wealth_low)


def _log_mixture(log_wealth: np.ndarray) -> float:
    maximum = float(np.max(log_wealth))
    return maximum + float(np.log(np.mean(np.exp(log_wealth - maximum))))


def _mixture(log_wealth: np.ndarray) -> float:
    return float(np.exp(_log_mixture(log_wealth)))


__all__ = [
    "Hypothesis",
    "NonInferiority",
    "Superiority",
    "Thresholds",
    "Wealth",
    "thresholds",
]
