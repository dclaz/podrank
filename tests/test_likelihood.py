"""Compressed likelihood and objective fixtures (F1, F5, F10)."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from podrank.data.table import normalize_game_row
from podrank.rating.design import (
    ParameterIndex,
    Pattern,
    PatternTable,
    build_patterns,
    design_row,
)
from podrank.rating.identification import (
    ContrastContext,
    aliasing_residual,
    build_components,
    build_identification_report,
    row_space_basis,
)
from podrank.rating.likelihood import (
    likelihood_grad,
    likelihood_hess,
    likelihood_nll,
    objective,
    objective_delta,
    objective_grad,
    objective_hess,
    pattern_grad_u,
    pattern_hess_u,
    prior_penalty,
)
from podrank.rating.spec import ModelSpec
from tests._fixtures import make_game, registries


def _identity_index(count: int = 4) -> ParameterIndex:
    names = tuple(f"u{i}" for i in range(count))
    return ParameterIndex(
        names=names,
        positions=MappingProxyType({name: index for index, name in enumerate(names)}),
        supported_agents=(),
        supported_decks=(),
        supported_brackets=(),
        has_eta=False,
        block_slices=MappingProxyType({}),
    )


def _table(design: np.ndarray, target: list[int]) -> PatternTable:
    index = _identity_index(design.shape[1])
    key = (("a", "b", "c", "d"), ("d1", "d2", "d3", "d4"), (0, 0, 0, 0))
    pattern = Pattern(
        key=key,
        design=design,
        target=np.asarray(target, dtype=np.int64),
        weight=sum(target),
    )
    return PatternTable(
        patterns=(pattern,), index=index, cluster_counts=MappingProxyType({}), n_games=sum(target)
    )


def test_f1_one_decisive_game() -> None:
    utilities = np.array([0.5, 0.0, 0.0, -0.5])
    table = _table(np.eye(4), [1, 0, 0, 0])
    probabilities = np.array([0.387456, 0.235004, 0.235004, 0.142537])

    nll = likelihood_nll(utilities, table)
    gradient = likelihood_grad(utilities, table)
    hessian = likelihood_hess(utilities, table)

    assert nll == pytest.approx(-np.log(probabilities[0]), rel=1e-5, abs=1e-6)
    assert gradient == pytest.approx(np.array([-0.612544, 0.235004, 0.235004, 0.142537]), abs=2e-6)
    assert hessian[0] == pytest.approx(
        np.array([0.237334, -0.091054, -0.091054, -0.055227]), abs=2e-6
    )


def test_f5_compressed_target_mass_and_double_weighting_is_wrong() -> None:
    table = _table(np.eye(4), [1, 1, 0, 0])
    psi = np.zeros(4)
    probabilities = np.full(4, 0.25)

    nll = likelihood_nll(psi, table)
    gradient = likelihood_grad(psi, table)
    hessian = likelihood_hess(psi, table)

    assert nll == pytest.approx(2 * -np.log(0.25), rel=0, abs=1e-12)
    assert gradient == pytest.approx(np.array([-0.5, -0.5, 0.5, 0.5]), abs=0)
    assert np.diag(hessian) == pytest.approx(np.full(4, 0.375), abs=0)
    assert hessian[0, 1] == pytest.approx(-0.125, abs=0)

    wrong = 2.0 * (probabilities - np.array([1, 1, 0, 0]))
    assert not np.allclose(wrong, gradient)

    weight = float(table.patterns[0].weight)
    assert np.allclose(
        pattern_grad_u(probabilities, table.patterns[0].target),
        weight * probabilities - table.patterns[0].target,
    )
    assert np.allclose(
        pattern_hess_u(probabilities, table.patterns[0].target),
        weight * (np.diag(probabilities) - np.outer(probabilities, probabilities)),
    )


def test_f10_gauge_shift_changes_only_the_prior_penalty() -> None:
    table = _table(np.eye(2), [1, 0])
    precision = np.array([1 / 9, 1 / 9])
    before = np.array([0.5, 0.0])
    after = before + 1.0

    assert likelihood_nll(after, table) == pytest.approx(likelihood_nll(before, table), abs=1e-14)
    assert likelihood_grad(after, table) == pytest.approx(likelihood_grad(before, table), abs=1e-14)
    assert prior_penalty(before, precision) == pytest.approx(0.0138888888888889, rel=1e-14)
    assert prior_penalty(after, precision) == pytest.approx(0.1805555555555556, rel=1e-14)
    change = objective(after, table, precision) - objective(before, table, precision)
    assert change == pytest.approx(1 / 6, rel=1e-14)


def test_gradient_and_hessian_match_finite_differences() -> None:
    agents, decks = registries()
    rows = [
        normalize_game_row(make_game("g1")),
        normalize_game_row(make_game("g2", winner_seat=1, cluster_id="b1")),
        normalize_game_row(
            make_game("g3", agents=("H", "A", "B", "C"), winner_seat=2, cluster_id="b1")
        ),
    ]
    index = ParameterIndex.build(
        agents,
        decks,
        ["A", "B", "C", "H"],
        ["d1", "d2", "d3", "d4"],
        swap_effect=False,
    )
    table = build_patterns(rows, index, decks, ModelSpec())
    precision = np.full(len(index), 1 / 9)
    psi = np.linspace(-0.4, 0.5, len(index))

    analytic_grad = objective_grad(psi, table, precision)
    numeric_grad = np.zeros_like(analytic_grad)
    step = 1e-6
    for position in range(len(index)):
        delta = np.zeros_like(psi)
        delta[position] = step
        numeric_grad[position] = (
            objective(psi + delta, table, precision) - objective(psi - delta, table, precision)
        ) / (2 * step)
    assert np.allclose(analytic_grad, numeric_grad, rtol=1e-6, atol=1e-8)

    analytic_hess = objective_hess(psi, table, precision)
    numeric_hess = np.zeros_like(analytic_hess)
    for position in range(len(index)):
        delta = np.zeros_like(psi)
        delta[position] = step
        numeric_hess[:, position] = (
            objective_grad(psi + delta, table, precision)
            - objective_grad(psi - delta, table, precision)
        ) / (2 * step)
    assert np.allclose(analytic_hess, numeric_hess, rtol=1e-6, atol=1e-7)
    assert np.allclose(analytic_hess, analytic_hess.T)


def test_stable_objective_delta_matches_the_direct_difference() -> None:
    table = _table(np.eye(4), [1, 1, 0, 0])
    precision = np.full(4, 1 / 9)
    psi = np.array([0.01, -0.02, 0.03, -0.04])
    direction = np.array([1e-6, -2e-6, 3e-6, -4e-6])

    assert objective_delta(psi, direction, table, precision) == pytest.approx(
        objective(psi + direction, table, precision) - objective(psi, table, precision),
        rel=1e-10,
        abs=1e-14,
    )


def _compact_and_dense_tables():
    agents, decks = registries()
    rows = [
        normalize_game_row(make_game("g1")),
        normalize_game_row(make_game("g2", winner_seat=1, cluster_id="b1")),
        normalize_game_row(
            make_game("g3", agents=("H", "A", "B", "C"), winner_seat=2, cluster_id="b1")
        ),
    ]
    index = ParameterIndex.build(
        agents,
        decks,
        ["A", "B", "C", "H"],
        ["d1", "d2", "d3", "d4"],
        swap_effect=False,
    )
    compact = build_patterns(rows, index, decks, ModelSpec())
    patterns = []
    for position in range(compact.compact.n_patterns):
        agent_ids = tuple(
            index.supported_agents[int(value)] for value in compact.compact.agent_indices[position]
        )
        deck_ids = tuple(
            index.supported_decks[int(value)] for value in compact.compact.deck_indices[position]
        )
        swaps = (0, 0, 0, 0)
        target = compact.compact.targets[position]
        patterns.append(
            Pattern(
                key=(agent_ids, deck_ids, swaps),
                design=design_row(agent_ids, deck_ids, swaps, index, decks),
                target=target,
                weight=int(target.sum()),
            )
        )
    dense = PatternTable(
        index=index,
        patterns=tuple(patterns),
        cluster_counts=compact.cluster_counts,
        n_games=compact.n_games,
    )
    return compact, dense, agents, decks, rows


def test_compact_likelihood_matches_the_dense_reference() -> None:
    compact, dense, _, _, _ = _compact_and_dense_tables()
    precision = np.full(len(compact.index), 1 / 9)
    psi = np.linspace(-0.35, 0.4, len(compact.index))
    step = np.linspace(1e-6, -4e-6, len(compact.index))

    assert likelihood_nll(psi, compact) == pytest.approx(likelihood_nll(psi, dense), abs=1e-12)
    assert np.allclose(likelihood_grad(psi, compact), likelihood_grad(psi, dense), atol=1e-12)
    assert np.allclose(likelihood_hess(psi, compact), likelihood_hess(psi, dense), atol=1e-12)
    assert objective_delta(psi, step, compact, precision) == pytest.approx(
        objective_delta(psi, step, dense, precision), rel=1e-12, abs=1e-14
    )


def test_compact_d4_and_d2_match_the_dense_reference() -> None:
    compact, dense, agents, decks, rows = _compact_and_dense_tables()
    dense_space = row_space_basis(dense, ModelSpec().tol_eig)
    compact_space = row_space_basis(compact, ModelSpec().tol_eig)
    for left, right in (("A", "H"), ("B", "H"), ("A", "B")):
        vector = np.zeros(len(compact.index))
        vector[compact.index.index(f"theta[{left}]")] = 1.0
        vector[compact.index.index(f"theta[{right}]")] -= 1.0
        assert aliasing_residual(vector, compact_space) == pytest.approx(
            aliasing_residual(vector, dense_space), abs=1e-12
        )

    psi = np.linspace(-0.2, 0.3, len(compact.index))
    precision = np.full(len(compact.index), 1 / 9)
    components, _ = build_components(agents, compact.index.supported_agents, rows, anchor_id="H")
    dense_context = ContrastContext(
        mode=psi,
        lower_factor=np.eye(len(compact.index)),
        precision=precision,
        row_space=dense_space,
        index=compact.index,
        spec=ModelSpec(),
    )
    compact_context = ContrastContext(
        mode=psi,
        lower_factor=np.eye(len(compact.index)),
        precision=precision,
        row_space=compact_space,
        index=compact.index,
        spec=ModelSpec(),
    )
    dense_report = build_identification_report(
        dense,
        dense_context,
        components,
        decks,
        likelihood_hess(psi, dense),
    )
    compact_report = build_identification_report(
        compact,
        compact_context,
        components,
        decks,
        likelihood_hess(psi, compact),
    )
    assert [item.eigenvalue for item in compact_report.weak_directions] == pytest.approx(
        [item.eigenvalue for item in dense_report.weak_directions], rel=1e-12, abs=1e-12
    )
    assert compact_report.n_gauge == dense_report.n_gauge


def test_compact_cluster_counts_match_the_mapping() -> None:
    compact, _, _, _, _ = _compact_and_dense_tables()
    data = compact.compact
    reconstructed = {
        (
            int(pattern),
            int(winner),
            data.cluster_keys[int(cluster)],
        ): int(count)
        for pattern, winner, cluster, count in zip(
            data.cluster_patterns,
            data.cluster_winner_seats,
            data.cluster_indices,
            data.cluster_counts,
            strict=True,
        )
    }
    assert sum(reconstructed.values()) == compact.n_games
    assert len(data.cluster_keys) == len({key for (_, _, key) in reconstructed})
