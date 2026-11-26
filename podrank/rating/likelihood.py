"""Luce likelihood, priors and the penalised objective (L1)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from podrank.rating.design import CompactPatterns, ParameterIndex, PatternTable

_COMPACT_BATCH_PATTERNS = 100_000


def _softmax(utilities: np.ndarray) -> np.ndarray:
    maximum = float(np.max(utilities))
    shifted = utilities - maximum
    denominator = float(np.sum(np.exp(shifted)))
    return np.asarray(np.exp(shifted) / denominator, dtype=np.float64)


def _log_softmax(utilities: np.ndarray) -> np.ndarray:
    maximum = float(np.max(utilities))
    shifted = utilities - maximum
    log_denominator = maximum + math.log(float(np.sum(np.exp(shifted))))
    return np.asarray(utilities - log_denominator, dtype=np.float64)


def _softmax_rows(utilities: np.ndarray) -> np.ndarray:
    maximum = np.max(utilities, axis=1, keepdims=True)
    shifted = utilities - maximum
    denominator = np.sum(np.exp(shifted), axis=1, keepdims=True)
    return np.asarray(np.exp(shifted) / denominator, dtype=np.float64)


def _log_softmax_rows(utilities: np.ndarray) -> np.ndarray:
    maximum = np.max(utilities, axis=1, keepdims=True)
    shifted = utilities - maximum
    log_denominator = maximum + np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return np.asarray(utilities - log_denominator, dtype=np.float64)


def pattern_grad_u(probabilities: np.ndarray, target: np.ndarray) -> np.ndarray:
    weight = float(np.sum(target))
    return np.asarray(weight * probabilities - target, dtype=np.float64)


def pattern_hess_u(probabilities: np.ndarray, target: np.ndarray) -> np.ndarray:
    weight = float(np.sum(target))
    return np.asarray(
        weight * (np.diag(probabilities) - np.outer(probabilities, probabilities)),
        dtype=np.float64,
    )


def _dense_nll(psi: np.ndarray, table: PatternTable) -> float:
    vector = np.asarray(psi, dtype=np.float64)
    total = 0.0
    for pattern in table.patterns:
        utilities = pattern.design @ vector
        log_probabilities = _log_softmax(utilities)
        total += -float(np.dot(pattern.target, log_probabilities))
    return total


def _dense_grad(psi: np.ndarray, table: PatternTable) -> np.ndarray:
    vector = np.asarray(psi, dtype=np.float64)
    gradient = np.zeros(len(table.index), dtype=np.float64)
    for pattern in table.patterns:
        probabilities = _softmax(pattern.design @ vector)
        gradient += pattern.design.T @ pattern_grad_u(probabilities, pattern.target)
    return gradient


def _dense_hess(psi: np.ndarray, table: PatternTable) -> np.ndarray:
    vector = np.asarray(psi, dtype=np.float64)
    hessian = np.zeros((len(table.index), len(table.index)), dtype=np.float64)
    for pattern in table.patterns:
        probabilities = _softmax(pattern.design @ vector)
        hessian += pattern.design.T @ pattern_hess_u(probabilities, pattern.target) @ pattern.design
    return hessian


def _compact_utilities(
    psi: np.ndarray, table: PatternTable, compact: CompactPatterns, start: int, stop: int
) -> np.ndarray:
    agents = compact.agent_indices[start:stop]
    decks = compact.deck_indices[start:stop]
    theta_start = table.index.block_slices["theta"][0]
    utilities = (
        psi[theta_start + agents]
        + psi[compact.deck_mu_positions[decks]]
        + psi[compact.deck_delta_positions[decks]]
        + psi[compact.seat_positions]
    )
    if compact.eta_position is not None:
        assert compact.swap_counts is not None
        utilities = utilities + psi[compact.eta_position] * compact.swap_counts[start:stop]
    return np.asarray(utilities, dtype=np.float64)


def _compact_accumulate_gradient(
    g_red: np.ndarray,
    probabilities: np.ndarray,
    target: np.ndarray,
    compact: CompactPatterns,
    index: ParameterIndex,
    start: int,
    stop: int,
) -> None:
    weight = target.sum(axis=1).astype(np.float64)
    g_u = weight[:, None] * probabilities - target
    agents = compact.agent_indices[start:stop]
    decks = compact.deck_indices[start:stop]
    reduced_size = compact.reduced_size
    theta_counts = np.bincount(agents.ravel(), weights=g_u.ravel(), minlength=reduced_size)
    beta_counts = np.bincount(
        (index.beta_start + decks).ravel(),
        weights=g_u.ravel(),
        minlength=reduced_size,
    )
    g_red += theta_counts
    g_red += beta_counts
    g_red[index.reduced_seat_start : index.reduced_seat_start + 4] += g_u.sum(axis=0)
    if compact.eta_position is not None:
        assert compact.swap_counts is not None
        eta_index = index.reduced_eta_index
        assert eta_index is not None
        g_red[eta_index] += float(np.sum(g_u * compact.swap_counts[start:stop]))


def _compact_accumulate_hessian(
    hessian_flat: np.ndarray,
    probabilities: np.ndarray,
    target: np.ndarray,
    compact: CompactPatterns,
    index: ParameterIndex,
    start: int,
    stop: int,
) -> None:
    reduced_size = compact.reduced_size
    weight = target.sum(axis=1).astype(np.float64)
    agents = compact.agent_indices[start:stop]
    decks = compact.deck_indices[start:stop]
    beta_start = index.beta_start
    seat_start = index.reduced_seat_start
    eta_index = index.reduced_eta_index
    swaps = compact.swap_counts[start:stop] if compact.swap_counts is not None else None
    ones = np.ones(stop - start, dtype=np.float64)
    pair_list = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for left, right in pair_list:
        pair_weight = weight * probabilities[:, left] * probabilities[:, right]
        positions = [
            agents[:, left].astype(np.int64),
            agents[:, right].astype(np.int64),
            (beta_start + decks[:, left]).astype(np.int64),
            (beta_start + decks[:, right]).astype(np.int64),
            np.full(stop - start, seat_start + left, dtype=np.int64),
            np.full(stop - start, seat_start + right, dtype=np.int64),
        ]
        coefficients = [ones, -ones, ones, -ones, ones, -ones]
        if eta_index is not None:
            assert swaps is not None
            positions.append(np.full(stop - start, eta_index, dtype=np.int64))
            coefficients.append((swaps[:, left] - swaps[:, right]).astype(np.float64))
        active = len(positions)
        for first in range(active):
            for second in range(first, active):
                values = pair_weight * coefficients[first] * coefficients[second]
                flat = positions[first] * reduced_size + positions[second]
                hessian_flat += np.bincount(flat, weights=values, minlength=reduced_size**2)
                if first != second:
                    flat_transposed = positions[second] * reduced_size + positions[first]
                    hessian_flat += np.bincount(
                        flat_transposed, weights=values, minlength=reduced_size**2
                    )


@dataclass(frozen=True, slots=True)
class CompactEvaluation:
    objective: float
    gradient: np.ndarray
    hessian: np.ndarray
    utilities: np.ndarray | None
    probabilities: np.ndarray | None


def compact_evaluate(
    psi: np.ndarray,
    table: PatternTable,
    precision: np.ndarray,
    *,
    cache: bool = False,
) -> CompactEvaluation:
    """Fused objective, gradient and Hessian in reduced likelihood coordinates."""

    compact = table.compact
    if compact is None:
        raise ValueError("compact_evaluate requires a compact pattern table")
    vector = np.asarray(psi, dtype=np.float64)
    reduced_size = compact.reduced_size
    n_parameters = len(table.index)
    g_red = np.zeros(reduced_size, dtype=np.float64)
    hessian_flat = np.zeros(reduced_size**2, dtype=np.float64)
    utilities_cache = np.empty((compact.n_patterns, 4), dtype=np.float64) if cache else None
    probabilities_cache = np.empty((compact.n_patterns, 4), dtype=np.float64) if cache else None
    nll = 0.0
    for start in range(0, compact.n_patterns, _COMPACT_BATCH_PATTERNS):
        stop = min(start + _COMPACT_BATCH_PATTERNS, compact.n_patterns)
        utilities = _compact_utilities(vector, table, compact, start, stop)
        probabilities = _softmax_rows(utilities)
        target = compact.targets[start:stop]
        log_probabilities = _log_softmax_rows(utilities)
        nll += -float(np.sum(target * log_probabilities))
        _compact_accumulate_gradient(
            g_red, probabilities, target, compact, table.index, start, stop
        )
        _compact_accumulate_hessian(
            hessian_flat, probabilities, target, compact, table.index, start, stop
        )
        if utilities_cache is not None and probabilities_cache is not None:
            utilities_cache[start:stop] = utilities
            probabilities_cache[start:stop] = probabilities

    gradient = compact.reduction @ g_red
    hessian_red = hessian_flat.reshape(reduced_size, reduced_size)
    hessian = compact.reduction @ hessian_red @ compact.reduction.T
    objective_value = nll + prior_penalty(vector, precision)
    gradient = gradient + precision * vector
    hessian = hessian + np.diag(precision)
    assert gradient.shape == (n_parameters,)
    return CompactEvaluation(
        objective=objective_value,
        gradient=np.asarray(gradient, dtype=np.float64),
        hessian=np.asarray(hessian, dtype=np.float64),
        utilities=utilities_cache,
        probabilities=probabilities_cache,
    )


def compact_likelihood_nll(psi: np.ndarray, table: PatternTable) -> float:
    compact = table.compact
    if compact is None:
        raise ValueError("compact_likelihood_nll requires a compact pattern table")
    vector = np.asarray(psi, dtype=np.float64)
    total = 0.0
    for start in range(0, compact.n_patterns, _COMPACT_BATCH_PATTERNS):
        stop = min(start + _COMPACT_BATCH_PATTERNS, compact.n_patterns)
        utilities = _compact_utilities(vector, table, compact, start, stop)
        log_probabilities = _log_softmax_rows(utilities)
        total += -float(np.sum(compact.targets[start:stop] * log_probabilities))
    return total


def compact_likelihood_grad(psi: np.ndarray, table: PatternTable) -> np.ndarray:
    compact = table.compact
    if compact is None:
        raise ValueError("compact_likelihood_grad requires a compact pattern table")
    vector = np.asarray(psi, dtype=np.float64)
    reduced_size = compact.reduced_size
    g_red = np.zeros(reduced_size, dtype=np.float64)
    for start in range(0, compact.n_patterns, _COMPACT_BATCH_PATTERNS):
        stop = min(start + _COMPACT_BATCH_PATTERNS, compact.n_patterns)
        utilities = _compact_utilities(vector, table, compact, start, stop)
        probabilities = _softmax_rows(utilities)
        _compact_accumulate_gradient(
            g_red,
            probabilities,
            compact.targets[start:stop],
            compact,
            table.index,
            start,
            stop,
        )
    return np.asarray(compact.reduction @ g_red, dtype=np.float64)


def compact_likelihood_hess(psi: np.ndarray, table: PatternTable) -> np.ndarray:
    compact = table.compact
    if compact is None:
        raise ValueError("compact_likelihood_hess requires a compact pattern table")
    vector = np.asarray(psi, dtype=np.float64)
    reduced_size = compact.reduced_size
    hessian_flat = np.zeros(reduced_size**2, dtype=np.float64)
    for start in range(0, compact.n_patterns, _COMPACT_BATCH_PATTERNS):
        stop = min(start + _COMPACT_BATCH_PATTERNS, compact.n_patterns)
        utilities = _compact_utilities(vector, table, compact, start, stop)
        probabilities = _softmax_rows(utilities)
        _compact_accumulate_hessian(
            hessian_flat,
            probabilities,
            compact.targets[start:stop],
            compact,
            table.index,
            start,
            stop,
        )
    hessian_red = hessian_flat.reshape(reduced_size, reduced_size)
    return np.asarray(compact.reduction @ hessian_red @ compact.reduction.T, dtype=np.float64)


def likelihood_nll(psi: np.ndarray, table: PatternTable) -> float:
    if table.compact is not None:
        return compact_likelihood_nll(psi, table)
    return _dense_nll(psi, table)


def likelihood_grad(psi: np.ndarray, table: PatternTable) -> np.ndarray:
    if table.compact is not None:
        return compact_likelihood_grad(psi, table)
    return _dense_grad(psi, table)


def likelihood_hess(psi: np.ndarray, table: PatternTable) -> np.ndarray:
    if table.compact is not None:
        return compact_likelihood_hess(psi, table)
    return _dense_hess(psi, table)


def prior_penalty(psi: np.ndarray, precision: np.ndarray) -> float:
    vector = np.asarray(psi, dtype=np.float64)
    weights = np.asarray(precision, dtype=np.float64)
    return 0.5 * float(np.dot(vector, weights * vector))


def objective(psi: np.ndarray, table: PatternTable, precision: np.ndarray) -> float:
    return likelihood_nll(psi, table) + prior_penalty(psi, precision)


def objective_grad(psi: np.ndarray, table: PatternTable, precision: np.ndarray) -> np.ndarray:
    vector = np.asarray(psi, dtype=np.float64)
    return np.asarray(likelihood_grad(vector, table) + precision * vector, dtype=np.float64)


def objective_hess(psi: np.ndarray, table: PatternTable, precision: np.ndarray) -> np.ndarray:
    vector = np.asarray(psi, dtype=np.float64)
    return np.asarray(likelihood_hess(vector, table) + np.diag(precision), dtype=np.float64)


def _dense_objective_delta(
    psi: np.ndarray,
    step: np.ndarray,
    table: PatternTable,
    precision: np.ndarray,
) -> float:
    vector = np.asarray(psi, dtype=np.float64)
    direction = np.asarray(step, dtype=np.float64)
    terms: list[float] = []
    for pattern in table.patterns:
        utilities = pattern.design @ vector
        increments = pattern.design @ direction
        probabilities = _softmax(utilities)
        relative = increments - increments[0]
        if float(np.max(np.abs(relative))) <= 0.5:
            log_ratio = math.log1p(
                math.fsum(
                    float(probabilities[seat]) * math.expm1(float(relative[seat]))
                    for seat in range(4)
                )
            )
        else:
            maximum = float(np.max(utilities + increments))
            shifted = utilities + increments - maximum
            numerator = maximum + math.log(float(np.sum(np.exp(shifted))))
            maximum0 = float(np.max(utilities))
            shifted0 = utilities - maximum0
            denominator = maximum0 + math.log(float(np.sum(np.exp(shifted0))))
            log_ratio = numerator - denominator
        for seat in range(4):
            target = int(pattern.target[seat])
            if target:
                terms.append(float(target) * (log_ratio - float(relative[seat])))
    ridge = math.fsum(
        float(precision[index])
        * float(direction[index])
        * (float(vector[index]) + float(direction[index]) / 2.0)
        for index in range(len(vector))
    )
    return math.fsum(terms) + ridge


def _compact_objective_delta(
    psi: np.ndarray,
    step: np.ndarray,
    table: PatternTable,
    precision: np.ndarray,
    utilities_cache: np.ndarray | None,
    probabilities_cache: np.ndarray | None,
) -> float:
    compact = table.compact
    if compact is None:
        raise ValueError("compact delta requires a compact pattern table")
    vector = np.asarray(psi, dtype=np.float64)
    direction = np.asarray(step, dtype=np.float64)
    if utilities_cache is not None and probabilities_cache is not None:
        return _compact_delta_from_cache(
            vector,
            direction,
            table,
            precision,
            utilities_cache,
            probabilities_cache,
        )
    terms_sum = 0.0
    for start in range(0, compact.n_patterns, _COMPACT_BATCH_PATTERNS):
        stop = min(start + _COMPACT_BATCH_PATTERNS, compact.n_patterns)
        utilities = _compact_utilities(vector, table, compact, start, stop)
        probabilities = _softmax_rows(utilities)
        terms_sum += _compact_delta_batch(
            direction, table, compact, utilities, probabilities, start, stop
        )
    ridge = math.fsum(
        float(precision[index])
        * float(direction[index])
        * (float(vector[index]) + float(direction[index]) / 2.0)
        for index in range(len(vector))
    )
    return terms_sum + ridge


def _compact_delta_from_cache(
    vector: np.ndarray,
    direction: np.ndarray,
    table: PatternTable,
    precision: np.ndarray,
    utilities: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    compact = table.compact
    if compact is None:
        raise ValueError("compact delta requires a compact pattern table")
    terms_sum = 0.0
    for start in range(0, compact.n_patterns, _COMPACT_BATCH_PATTERNS):
        stop = min(start + _COMPACT_BATCH_PATTERNS, compact.n_patterns)
        terms_sum += _compact_delta_batch(
            direction,
            table,
            compact,
            utilities[start:stop],
            probabilities[start:stop],
            start,
            stop,
        )
    ridge = math.fsum(
        float(precision[index])
        * float(direction[index])
        * (float(vector[index]) + float(direction[index]) / 2.0)
        for index in range(len(direction))
    )
    return terms_sum + ridge


def _compact_delta_batch(
    direction: np.ndarray,
    table: PatternTable,
    compact: CompactPatterns,
    utilities: np.ndarray,
    probabilities: np.ndarray,
    start: int,
    stop: int,
) -> float:
    agents = compact.agent_indices[start:stop]
    decks = compact.deck_indices[start:stop]
    theta_start = table.index.block_slices["theta"][0]
    increments = (
        direction[theta_start + agents]
        + direction[compact.deck_mu_positions[decks]]
        + direction[compact.deck_delta_positions[decks]]
        + direction[compact.seat_positions]
    )
    if compact.eta_position is not None:
        assert compact.swap_counts is not None
        increments = increments + direction[compact.eta_position] * compact.swap_counts[start:stop]
    relative = increments - increments[:, :1]
    maximum = np.max(np.abs(relative), axis=1)
    log_ratio = np.empty(stop - start, dtype=np.float64)
    small = maximum <= 0.5
    if np.any(small):
        log_ratio[small] = np.log1p(
            np.sum(probabilities[small] * np.expm1(relative[small]), axis=1)
        )
    large = ~small
    if np.any(large):
        util_large = utilities[large]
        rel_large = relative[large]
        maximum_shift = np.max(util_large + rel_large, axis=1, keepdims=True)
        numerator = maximum_shift + np.log(
            np.sum(np.exp(util_large + rel_large - maximum_shift), axis=1, keepdims=True)
        )
        maximum0 = np.max(util_large, axis=1, keepdims=True)
        denominator = maximum0 + np.log(
            np.sum(np.exp(util_large - maximum0), axis=1, keepdims=True)
        )
        log_ratio[large] = (numerator - denominator)[:, 0]
    target = compact.targets[start:stop]
    terms = target * (log_ratio[:, None] - relative)
    return float(np.sum(terms))


def objective_delta(
    psi: np.ndarray,
    step: np.ndarray,
    table: PatternTable,
    precision: np.ndarray,
    *,
    utilities_cache: np.ndarray | None = None,
    probabilities_cache: np.ndarray | None = None,
) -> float:
    if table.compact is not None:
        return _compact_objective_delta(
            psi,
            step,
            table,
            precision,
            utilities_cache,
            probabilities_cache,
        )
    return _dense_objective_delta(psi, step, table, precision)


__all__ = [
    "CompactEvaluation",
    "compact_evaluate",
    "compact_likelihood_grad",
    "compact_likelihood_hess",
    "compact_likelihood_nll",
    "likelihood_grad",
    "likelihood_hess",
    "likelihood_nll",
    "objective",
    "objective_delta",
    "objective_grad",
    "objective_hess",
    "pattern_grad_u",
    "pattern_hess_u",
    "prior_penalty",
]
