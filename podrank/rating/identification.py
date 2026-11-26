"""Identification diagnostics and contrast statuses (D1–D5)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import cho_solve
from scipy.stats import norm as normal_distribution

from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.rating.design import ParameterIndex, PatternTable
from podrank.rating.likelihood import likelihood_hess
from podrank.rating.spec import ModelSpec
from podrank.types import GameRow

ContrastStatus = Literal["zero", "unrankable", "disconnected", "aliased", "weak", "identified"]


@dataclass(frozen=True, slots=True)
class Interval:
    lo: float
    hi: float
    method: str
    labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnchorUnsupported:
    anchor_id: str


@dataclass(frozen=True, slots=True)
class ComponentsReport:
    anchor_id: str
    anchor_component: tuple[str, ...]
    other_components: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class EligibilityRow:
    agent_id: str
    contrast_status: ContrastStatus
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class UnrankableReport:
    agents: tuple[str, ...]
    decks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ContrastResult:
    status: ContrastStatus
    value: float | None
    data_share: float | None
    aliasing_residual: float | None
    prior_dependent_value: float | None
    interval: Interval | None
    labels: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AgentContrastRow:
    agent_id: str
    status: ContrastStatus
    value: float | None
    data_share: float | None
    aliasing_residual: float | None
    prior_dependent_value: float | None
    interval: Interval | None
    labels: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SeatContrastRow:
    seat: int
    status: ContrastStatus
    value: float | None
    data_share: float | None
    aliasing_residual: float | None
    prior_dependent_value: float | None
    interval: Interval | None
    labels: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeckContrastRow:
    deck_id: str
    status: ContrastStatus
    value: float | None
    data_share: float | None
    aliasing_residual: float | None
    prior_dependent_value: float | None
    interval: Interval | None
    labels: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeckNuisanceReport:
    baseline_status: Literal["available", "unavailable"]
    reference_deck_ids: tuple[str, ...]
    rows: tuple[DeckContrastRow, ...]


@dataclass(frozen=True, slots=True)
class EffectNotModelled:
    reason: str = "swap_effect is off"


@dataclass(frozen=True, slots=True)
class WeakDirection:
    eigenvalue: float
    parameter_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IdentificationReport:
    components: ComponentsReport | AnchorUnsupported
    weak_directions: tuple[WeakDirection, ...]
    n_gauge: int


@dataclass(frozen=True, slots=True)
class ContrastContext:
    mode: np.ndarray
    lower_factor: np.ndarray
    precision: np.ndarray
    row_space: np.ndarray
    index: ParameterIndex
    spec: ModelSpec


def _component_map(
    supported_agents: Sequence[str], likelihood_rows: Sequence[GameRow]
) -> dict[str, str]:
    parent = {agent: agent for agent in supported_agents}

    def find(agent: str) -> str:
        while parent[agent] != agent:
            parent[agent] = parent[parent[agent]]
            agent = parent[agent]
        return agent

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in likelihood_rows:
        distinct = sorted(set(row.agents))
        if len(distinct) > 1:
            first = distinct[0]
            for other in distinct[1:]:
                union(first, other)
    return {agent: find(agent) for agent in supported_agents}


def build_components(
    agents: AgentRegistry,
    supported_agents: Iterable[str],
    likelihood_rows: Sequence[GameRow],
    anchor_id: str,
) -> tuple[ComponentsReport | AnchorUnsupported, Mapping[str, str]]:
    supported = tuple(sorted(set(supported_agents)))
    roots = _component_map(supported, likelihood_rows)
    if anchor_id not in supported:
        return AnchorUnsupported(anchor_id=anchor_id), roots
    anchor_root = roots[anchor_id]
    groups: dict[str, list[str]] = {}
    for agent in supported:
        groups.setdefault(roots[agent], []).append(agent)
    anchor_component = tuple(sorted(groups.pop(anchor_root)))
    other_components = tuple(
        sorted((tuple(sorted(group)) for group in groups.values()), key=lambda item: item[0])
    )
    return (
        ComponentsReport(
            anchor_id=anchor_id,
            anchor_component=anchor_component,
            other_components=other_components,
        ),
        roots,
    )


def row_space_basis(patterns: PatternTable, tol_eig: float) -> np.ndarray:
    """Return an orthonormal basis of the within-game difference row space."""

    n_parameters = len(patterns.index)
    compact = patterns.compact
    if compact is not None:
        reduced_size = compact.reduced_size
        gram_flat = np.zeros(reduced_size**2, dtype=np.float64)
        beta_start = patterns.index.beta_start
        seat_start = patterns.index.reduced_seat_start
        eta_index = patterns.index.reduced_eta_index
        batch_size = 100_000
        for start in range(0, compact.n_patterns, batch_size):
            stop = min(start + batch_size, compact.n_patterns)
            agents = compact.agent_indices[start:stop]
            decks = compact.deck_indices[start:stop]
            swaps = compact.swap_counts[start:stop] if compact.swap_counts is not None else None
            ones = np.ones(stop - start, dtype=np.float64)
            for seat in range(1, 4):
                positions = [
                    agents[:, seat].astype(np.int64),
                    agents[:, 0].astype(np.int64),
                    (beta_start + decks[:, seat]).astype(np.int64),
                    (beta_start + decks[:, 0]).astype(np.int64),
                    np.full(stop - start, seat_start + seat, dtype=np.int64),
                    np.full(stop - start, seat_start, dtype=np.int64),
                ]
                coefficients = [ones, -ones, ones, -ones, ones, -ones]
                if eta_index is not None:
                    assert swaps is not None
                    positions.append(np.full(stop - start, eta_index, dtype=np.int64))
                    coefficients.append((swaps[:, seat] - swaps[:, 0]).astype(np.float64))
                active = len(positions)
                for first in range(active):
                    for second in range(first, active):
                        values = coefficients[first] * coefficients[second]
                        flat = positions[first] * reduced_size + positions[second]
                        gram_flat += np.bincount(flat, weights=values, minlength=reduced_size**2)
                        if first != second:
                            flat_transposed = positions[second] * reduced_size + positions[first]
                            gram_flat += np.bincount(
                                flat_transposed, weights=values, minlength=reduced_size**2
                            )
        gram_reduced = gram_flat.reshape(reduced_size, reduced_size)
        gram = compact.reduction @ gram_reduced @ compact.reduction.T
    else:
        gram = np.zeros((n_parameters, n_parameters), dtype=np.float64)
        for pattern in patterns.patterns:
            base = pattern.design[0]
            for seat in range(1, 4):
                difference = pattern.design[seat] - base
                gram += np.outer(difference, difference)
    if not np.any(gram):
        return np.zeros((n_parameters, 0), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    threshold = tol_eig * float(eigenvalues[-1])
    keep = eigenvalues > threshold
    return np.asarray(eigenvectors[:, keep], dtype=np.float64)


def null_space_basis(row_space: np.ndarray) -> np.ndarray:
    """Return an orthonormal basis for the orthogonal complement of ``row_space``."""

    if row_space.shape[1] == 0:
        return np.eye(row_space.shape[0])
    if row_space.shape[1] == row_space.shape[0]:
        return np.zeros((row_space.shape[0], 0), dtype=np.float64)
    left, _, _ = np.linalg.svd(row_space, full_matrices=True)
    return np.asarray(left[:, row_space.shape[1] :], dtype=np.float64)


def aliasing_residual(vector: np.ndarray, row_space: np.ndarray) -> float | None:
    length = float(np.linalg.norm(vector))
    if length == 0:
        return None
    projection = row_space @ (row_space.T @ vector)
    return float(np.linalg.norm(vector - projection) / length)


def _top_loadings(
    vector: np.ndarray, row_space: np.ndarray, index: ParameterIndex, limit: int = 3
) -> tuple[str, ...]:
    residual = vector - row_space @ (row_space.T @ vector)
    if not np.any(residual):
        return ()
    order = np.argsort(-np.abs(residual))
    selected = [index.names[int(position)] for position in order[:limit]]
    return tuple(selected)


def _data_share_and_variance(
    vector: np.ndarray, context: ContrastContext
) -> tuple[float | None, float, float]:
    try:
        solved = cho_solve((context.lower_factor, True), vector, check_finite=True)
    except Exception:
        return None, 0.0, 0.0
    numerator = float(np.dot(vector, solved))
    denominator = float(np.dot(vector, vector / context.precision))
    if denominator <= 0:
        return None, 0.0, 0.0
    numerator = max(0.0, numerator)
    return 1.0 - numerator / denominator, numerator, denominator


def evaluate_contrast(
    vector: np.ndarray,
    context: ContrastContext,
    *,
    labels: tuple[str, ...] = (),
    include_withheld: bool = False,
) -> ContrastResult:
    """Evaluate D3–D5 for one full-parameter contrast vector."""

    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (len(context.index),):
        raise ValueError("contrast vector has the wrong length")
    if not np.all(np.isfinite(vector)):
        raise ValueError("contrast vector must be finite")
    length = float(np.linalg.norm(vector))
    if length == 0:
        return ContrastResult(
            status="zero",
            value=0.0,
            data_share=None,
            aliasing_residual=None,
            prior_dependent_value=None,
            interval=Interval(0.0, 0.0, "exact", labels),
            labels=labels,
            reasons=(),
        )

    residual = aliasing_residual(vector, context.row_space)
    assert residual is not None
    if residual > context.spec.tol_alias:
        loadings = _top_loadings(vector, context.row_space, context.index)
        return ContrastResult(
            status="aliased",
            value=None,
            data_share=None,
            aliasing_residual=residual,
            prior_dependent_value=(
                float(np.dot(vector, context.mode)) if include_withheld else None
            ),
            interval=None,
            labels=labels,
            reasons=("aliased", *loadings),
        )

    data_share, variance, _ = _data_share_and_variance(vector, context)
    status: ContrastStatus
    if data_share is not None and data_share < 0.5:
        status = "weak"
    else:
        status = "identified"
    value = float(np.dot(vector, context.mode))
    interval = None
    if status in {"weak", "identified"}:
        sd = math.sqrt(max(0.0, variance))
        z_value = float(normal_distribution.ppf(0.975))
        interval = Interval(value - z_value * sd, value + z_value * sd, "laplace", labels)
    return ContrastResult(
        status=status,
        value=value,
        data_share=data_share,
        aliasing_residual=residual,
        prior_dependent_value=None,
        interval=interval,
        labels=labels,
        reasons=(),
    )


def agent_vector(index: ParameterIndex, agent_id: str) -> np.ndarray:
    result = np.zeros(len(index), dtype=np.float64)
    result[index.index(f"theta[{agent_id}]")] = 1.0
    return result


def seat_vector(index: ParameterIndex, seat: int) -> np.ndarray:
    result = np.zeros(len(index), dtype=np.float64)
    result[index.index(f"seat[{seat}]")] = 1.0
    return result


def beta_vector(
    index: ParameterIndex, deck_id: str, decks: DeckRegistry, coefficient: float = 1.0
) -> np.ndarray:
    result = np.zeros(len(index), dtype=np.float64)
    bracket = decks.by_id[deck_id].bracket
    result[index.index(f"mu[{bracket}]")] += coefficient
    result[index.index(f"delta[{deck_id}]")] += coefficient
    return result


def build_eligibility(
    agents: AgentRegistry,
    index: ParameterIndex,
    context: ContrastContext,
    components: ComponentsReport | AnchorUnsupported,
    component_of: Mapping[str, str],
) -> tuple[EligibilityRow, ...] | AnchorUnsupported:
    if isinstance(components, AnchorUnsupported):
        return components
    anchor = components.anchor_id
    anchor_component = set(components.anchor_component)
    rows: list[EligibilityRow] = []
    for agent_id in sorted(agents.by_id):
        if f"theta[{agent_id}]" not in index.positions:
            rows.append(
                EligibilityRow(
                    agent_id=agent_id,
                    contrast_status="unrankable",
                    eligible=False,
                    reasons=("unrankable",),
                )
            )
            continue
        if agent_id == anchor:
            rows.append(
                EligibilityRow(
                    agent_id=agent_id,
                    contrast_status="zero",
                    eligible=True,
                    reasons=(),
                )
            )
            continue
        if agent_id not in anchor_component:
            rows.append(
                EligibilityRow(
                    agent_id=agent_id,
                    contrast_status="disconnected",
                    eligible=False,
                    reasons=("disconnected",),
                )
            )
            continue
        vector = agent_vector(index, agent_id) - agent_vector(index, anchor)
        result = evaluate_contrast(vector, context)
        status = result.status
        rows.append(
            EligibilityRow(
                agent_id=agent_id,
                contrast_status=status,
                eligible=status in {"zero", "weak", "identified"},
                reasons=result.reasons,
            )
        )
    return tuple(rows)


def _gauge_vectors(index: ParameterIndex, decks: DeckRegistry) -> list[np.ndarray]:
    gauges: list[np.ndarray] = []
    for block_name in ("theta", "seat", "mu"):
        block = index.block(block_name)
        if np.any(block):
            gauges.append(block)
    for bracket in index.supported_brackets:
        vector = np.zeros(len(index), dtype=np.float64)
        vector[index.index(f"mu[{bracket}]")] = 1.0
        for deck_id in index.supported_decks:
            if decks.by_id[deck_id].bracket == bracket:
                vector[index.index(f"delta[{deck_id}]")] = -1.0
        gauges.append(vector)
    return gauges


def _weak_directions(
    patterns: PatternTable,
    context: ContrastContext,
    decks: DeckRegistry,
    spec: ModelSpec,
    data_hessian: np.ndarray | None,
) -> tuple[tuple[WeakDirection, ...], int]:
    if data_hessian is None:
        data_hessian = likelihood_hess(context.mode, patterns)
    scale = 1.0 / np.sqrt(context.precision)
    whitened = data_hessian * np.outer(scale, scale)
    gauges = _gauge_vectors(context.index, decks)
    if gauges:
        mapped = np.column_stack([np.sqrt(context.precision) * gauge for gauge in gauges])
        left, singular, _ = np.linalg.svd(mapped, full_matrices=True)
        tolerance = max(float(singular[0]) * 1e-12, 1e-15) if singular.size else 0.0
        rank = int(np.sum(singular > tolerance))
        complement = left[:, rank:]
    else:
        complement = np.eye(len(context.index))
    if complement.shape[1] == 0:
        return (), len(gauges)
    reduced = complement.T @ whitened @ complement
    eigenvalues, eigenvectors = np.linalg.eigh(reduced)
    directions: list[WeakDirection] = []
    for eigenvalue, eigenvector in zip(eigenvalues, eigenvectors.T, strict=True):
        if float(eigenvalue) >= spec.weak_ratio:
            continue
        full = (complement @ eigenvector) / np.sqrt(context.precision)
        length = float(np.linalg.norm(full))
        if length == 0:
            continue
        order = np.argsort(-np.abs(full / length))
        names = tuple(context.index.names[int(position)] for position in order[:3])
        directions.append(WeakDirection(eigenvalue=float(eigenvalue), parameter_names=names))
    directions.sort(key=lambda item: item.eigenvalue)
    return tuple(directions), len(gauges)


def build_identification_report(
    patterns: PatternTable,
    context: ContrastContext,
    components: ComponentsReport | AnchorUnsupported,
    decks: DeckRegistry,
    data_hessian: np.ndarray | None = None,
) -> IdentificationReport:
    directions, n_gauge = _weak_directions(patterns, context, decks, context.spec, data_hessian)
    return IdentificationReport(
        components=components,
        weak_directions=directions,
        n_gauge=n_gauge,
    )


__all__ = [
    "AgentContrastRow",
    "AnchorUnsupported",
    "ComponentsReport",
    "ContrastContext",
    "ContrastResult",
    "ContrastStatus",
    "DeckContrastRow",
    "DeckNuisanceReport",
    "EffectNotModelled",
    "EligibilityRow",
    "IdentificationReport",
    "Interval",
    "SeatContrastRow",
    "UnrankableReport",
    "WeakDirection",
    "agent_vector",
    "aliasing_residual",
    "beta_vector",
    "build_components",
    "build_eligibility",
    "build_identification_report",
    "evaluate_contrast",
    "null_space_basis",
    "row_space_basis",
    "seat_vector",
]
