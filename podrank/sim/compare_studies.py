"""S5/S11 study helpers for the R2 comparison core.

The runners use the frozen cmp_null/cmp_effect worlds and, for S11, the SC4
interaction world with an independently enumerated target. Results are recorded
in ADR 0013.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from podrank.compare.betting import Hypothesis, NonInferiority, Superiority, thresholds
from podrank.compare.pods import Design
from podrank.sim.compare_worlds import CompareWorld, reference_compare_world

FailureMode = Literal["none", "random", "outcome"]


@dataclass(frozen=True, slots=True)
class S5Cell:
    name: str
    hypothesis: Hypothesis
    mean: float
    design: Design
    rho: float
    failure: FailureMode
    check_above: bool = True
    check_below: bool = True
    world: CompareWorld | None = None
    theta_a: float = 0.0


@dataclass(frozen=True, slots=True)
class S5Result:
    cell: S5Cell
    replications: int
    false_above: int
    false_below: int
    alpha: float
    mc_se: float


@dataclass(frozen=True, slots=True)
class S11Result:
    cell: S5Cell
    replications: int
    blocks: int
    mean: float
    truth: float
    bias: float
    mc_se: float
    flagged: bool


def _outcomes(
    rng: np.random.Generator,
    n_blocks: int,
    p_a: float,
    p_b: float,
    q: float,
    rho: float,
    failure: FailureMode,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return duplicate A points and paired (A, B) points."""

    def draw(role_seed: float) -> np.ndarray:
        u = rng.random((n_blocks, 4))
        if rho > 0:
            shared = rng.random(n_blocks)
            mask = rng.random(n_blocks) < rho
            u[mask] = shared[mask, None]
        return u

    if p_b == 0.0:
        u = draw(0.0)
        points = np.where(u < p_a, 1.0, np.where(u < p_a + q, 0.25, 0.0))
        if failure == "random":
            points = np.where(rng.random((n_blocks, 4)) < 0.02, np.nan, points)
        elif failure == "outcome":
            probability = np.where(points == 1.0, 0.10, 0.002)
            points = np.where(rng.random((n_blocks, 4)) < probability, np.nan, points)
        return points, None

    a = draw(0.0)
    b = draw(1.0)
    points_a = np.where(a < p_a, 1.0, np.where(a < p_a + q, 0.25, 0.0))
    points_b = np.where(b < p_b, 1.0, np.where(b < p_b + q, 0.25, 0.0))
    if failure == "random":
        mask_a = rng.random((n_blocks, 4)) < 0.02
        mask_b = rng.random((n_blocks, 4)) < 0.02
        points_a = np.where(mask_a, np.nan, points_a)
        points_b = np.where(mask_b, np.nan, points_b)
    elif failure == "outcome":
        prob_a = np.where(points_a == 1.0, 0.10, 0.002)
        prob_b = np.where(points_b == 1.0, 0.10, 0.002)
        points_a = np.where(rng.random((n_blocks, 4)) < prob_a, np.nan, points_a)
        points_b = np.where(rng.random((n_blocks, 4)) < prob_b, np.nan, points_b)
    return points_a, points_b


def _world_outcomes(
    rng: np.random.Generator,
    world: CompareWorld,
    theta_a: float,
    *,
    design: Design,
    rho: float,
    failure: FailureMode,
    blocks: int,
    interaction_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = world.draw_rate + world.cap_rate
    if design == "duplicate":
        if interaction_index is None:
            p, _unused = world.block_probabilities(theta_a, design, blocks, rng)
        else:
            p, _unused = world.interaction_block_probabilities(
                theta_a, design, interaction_index, blocks, rng
            )
        u = rng.random((blocks, 4))
        if rho > 0:
            shared = rng.random(blocks)
            mask = rng.random(blocks) < rho
            u[mask] = shared[mask, None]
        points = np.where(u < p, 1.0, np.where(u < p + q, 0.25, 0.0))
        if failure == "random":
            points = np.where(rng.random((blocks, 4)) < 0.02, np.nan, points)
        elif failure == "outcome":
            probability = np.where(points == 1.0, 0.10, 0.002)
            points = np.where(rng.random((blocks, 4)) < probability, np.nan, points)
        up = np.where(np.isnan(points), 0.0, points)
        low = np.where(np.isnan(points), 1.0, points)
        return (
            up.mean(axis=1) - 0.25,
            low.mean(axis=1) - 0.25,
            np.any(np.isnan(points), axis=1),
        )
    if interaction_index is None:
        pa, pb = world.block_probabilities(theta_a, design, blocks, rng)
    else:
        pa, pb = world.interaction_block_probabilities(
            theta_a, design, interaction_index, blocks, rng
        )
    assert pb is not None
    ua = rng.random((blocks, 4))
    ub = rng.random((blocks, 4))
    if rho > 0:
        shared_a = rng.random(blocks)
        shared_b = rng.random(blocks)
        mask = rng.random(blocks) < rho
        ua[mask] = shared_a[mask, None]
        ub[mask] = shared_b[mask, None]
    points_a = np.where(ua < pa, 1.0, np.where(ua < pa + q, 0.25, 0.0))
    points_b = np.where(ub < pb, 1.0, np.where(ub < pb + q, 0.25, 0.0))
    if failure == "random":
        points_a = np.where(rng.random((blocks, 4)) < 0.02, np.nan, points_a)
        points_b = np.where(rng.random((blocks, 4)) < 0.02, np.nan, points_b)
    elif failure == "outcome":
        points_a = np.where(
            rng.random((blocks, 4)) < np.where(points_a == 1.0, 0.10, 0.002), np.nan, points_a
        )
        points_b = np.where(
            rng.random((blocks, 4)) < np.where(points_b == 1.0, 0.10, 0.002), np.nan, points_b
        )
    up_a = np.where(np.isnan(points_a), 0.0, points_a)
    low_a = np.where(np.isnan(points_a), 1.0, points_a)
    up_b = np.where(np.isnan(points_b), 1.0, points_b)
    low_b = np.where(np.isnan(points_b), 0.0, points_b)
    return (
        0.25 * (up_a - up_b).sum(axis=1),
        0.25 * (low_a - low_b).sum(axis=1),
        np.any(np.isnan(points_a) | np.isnan(points_b), axis=1),
    )


def block_scores(
    rng: np.random.Generator,
    *,
    design: Design,
    mean: float,
    rho: float,
    failure: FailureMode,
    blocks: int,
    world: CompareWorld | None = None,
    theta_a: float = 0.0,
    interaction_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if world is not None:
        return _world_outcomes(
            rng,
            world,
            theta_a,
            design=design,
            rho=rho,
            failure=failure,
            blocks=blocks,
            interaction_index=interaction_index,
        )
    q = 0.05
    if design == "duplicate":
        p_a = mean + 0.25 - 0.25 * q
        points_a, _ = _outcomes(rng, blocks, p_a, 0.0, q, rho, failure)
        up = np.where(np.isnan(points_a), 0.0, points_a)
        low = np.where(np.isnan(points_a), 1.0, points_a)
        return (
            up.mean(axis=1) - 0.25,
            low.mean(axis=1) - 0.25,
            np.any(np.isnan(points_a), axis=1),
        )
    p_b = (1.0 - q - mean) / 2.0
    p_a = p_b + mean
    points_a, points_b = _outcomes(rng, blocks, p_a, p_b, q, rho, failure)
    assert points_b is not None
    up_a = np.where(np.isnan(points_a), 0.0, points_a)
    low_a = np.where(np.isnan(points_a), 1.0, points_a)
    up_b = np.where(np.isnan(points_b), 1.0, points_b)
    low_b = np.where(np.isnan(points_b), 0.0, points_b)
    up = 0.25 * (up_a - up_b).sum(axis=1)
    low = 0.25 * (low_a - low_b).sum(axis=1)
    return up, low, np.any(np.isnan(points_a) | np.isnan(points_b), axis=1)


def _crossing(
    scores: np.ndarray, lam_bound: float, threshold: float, center: float, *, upper: bool
) -> int | None:
    lams = lam_bound * np.arange(1, 17, dtype=np.float64) / 16.0
    if upper:
        factors = 1.0 + (scores[:, None] - center) * lams[None, :]
    else:
        factors = 1.0 + (center - scores[:, None]) * lams[None, :]
    if np.any(factors <= 0):
        raise ValueError("betting factors must stay positive")
    log_wealth = np.cumsum(np.log(factors), axis=0)
    maximum = np.max(log_wealth, axis=1)
    mixture = maximum + np.log(np.mean(np.exp(log_wealth - maximum[:, None]), axis=1))
    hits = np.flatnonzero(mixture >= math.log(threshold))
    return int(hits[0]) if hits.size else None


def run_s5_cell(
    cell: S5Cell, *, replications: int, blocks: int, seed: int, alpha: float = 0.05
) -> S5Result:
    resolved = thresholds(cell.hypothesis, cell.design)
    false_above = false_below = 0
    rng = np.random.default_rng(seed)
    for _ in range(replications):
        up, low, failed = block_scores(
            rng,
            design=cell.design,
            mean=cell.mean,
            rho=cell.rho,
            failure=cell.failure,
            blocks=blocks,
            world=cell.world,
            theta_a=cell.theta_a,
        )
        first_failure_rule = None
        cumulative = np.cumsum(failed)
        index = np.arange(1, blocks + 1)
        active = index >= 50
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = cumulative / index
        trigger = np.flatnonzero(active & (fraction > 0.02))
        if trigger.size:
            first_failure_rule = int(trigger[0])
        up_hit = (
            _crossing(
                up,
                0.5 / (resolved.m_up - resolved.x_min),
                1.0 / alpha,
                resolved.m_up,
                upper=True,
            )
            if cell.check_above
            else None
        )
        low_hit = (
            _crossing(
                low,
                0.5 / (resolved.x_max - resolved.m_low),
                1.0 / alpha,
                resolved.m_low,
                upper=False,
            )
            if cell.check_below
            else None
        )
        if first_failure_rule is not None:
            up_hit = up_hit if up_hit is not None and up_hit <= first_failure_rule else None
            low_hit = low_hit if low_hit is not None and low_hit <= first_failure_rule else None
        if up_hit is not None and low_hit is not None:
            if up_hit < low_hit:
                false_above += 1
            elif low_hit < up_hit:
                false_below += 1
        elif up_hit is not None:
            false_above += 1
        elif low_hit is not None:
            false_below += 1
    se = math.sqrt(alpha * (1 - alpha) / replications)
    return S5Result(cell, replications, false_above, false_below, alpha, se)


def s5_cells() -> tuple[S5Cell, ...]:
    cells: list[S5Cell] = []
    hypotheses = (
        ("superiority_0", Superiority(0.0), "superiority_0", (True, True)),
        ("superiority_002", Superiority(0.02), "superiority_002", (True, False)),
        ("noninferiority_002", NonInferiority(0.02), "noninferiority_002", (True, True)),
    )
    world = reference_compare_world("cmp_null")
    by_label = {cell.label: cell for cell in world.cells}
    for name, hypothesis, label, directions in hypotheses:
        target = by_label[label]
        for design in ("duplicate", "paired"):
            theta = target.theta_a(design)
            for rho in (0.0, 0.3, 0.6, 1.0):
                for failure in ("none", "random", "outcome"):
                    cells.append(
                        S5Cell(
                            f"{name}:{design}:rho{rho}:{failure}",
                            hypothesis,
                            target.mean,
                            design,
                            rho,
                            failure,
                            check_above=directions[0],
                            check_below=directions[1],
                            world=world,
                            theta_a=theta,
                        )
                    )
    return tuple(cells)


def run_s11_cell(
    *,
    design: Design,
    mean: float,
    rho: float,
    replications: int,
    blocks: int,
    seed: int,
    cell_index: int = 0,
) -> S11Result:
    world = reference_compare_world("cmp_effect")
    target = next(cell for cell in world.cells if abs(cell.mean - mean) < 1e-15)
    theta = target.theta_a(design)
    truth = world.interaction_target(theta, design, cell_index)
    rng = np.random.default_rng(seed)
    means = np.empty(replications, dtype=np.float64)
    for index in range(replications):
        up, _low, _failed = block_scores(
            rng,
            design=design,
            mean=mean,
            rho=rho,
            failure="none",
            blocks=blocks,
            world=world,
            theta_a=theta,
            interaction_index=cell_index,
        )
        means[index] = float(up.mean())
    estimate = float(means.mean())
    se = float(means.std(ddof=1) / math.sqrt(replications))
    bias = estimate - truth
    return S11Result(
        S5Cell(f"s11:{design}:rho{rho}:mu{mean}", Superiority(0.0), mean, design, rho, "none"),
        replications,
        blocks,
        estimate,
        truth,
        bias,
        se,
        abs(bias) > 2 * se,
    )


__all__ = [
    "FailureMode",
    "S5Cell",
    "S5Result",
    "S11Result",
    "block_scores",
    "run_s5_cell",
    "run_s11_cell",
    "s5_cells",
]
