"""Newton solver boundaries (Appendix C.2, F20b)."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from types import MappingProxyType

import numpy as np
import pytest

from podrank.data.table import normalize_game_row
from podrank.errors import ConvergenceError
from podrank.rating.design import ParameterIndex, Pattern, PatternTable, design_row
from podrank.rating.likelihood import (
    objective,
    objective_delta,
    objective_grad,
    objective_hess,
)
from podrank.rating.solve import newton
from tests._fixtures import make_game, registries


def test_one_full_step_on_a_quadratic() -> None:
    result = newton(
        lambda x: 0.5 * float(x[0] ** 2),
        lambda x: np.array([x[0]]),
        lambda x: np.array([[1.0]]),
        np.array([1.0]),
    )

    assert result.accepted_steps == 1
    assert result.mode == pytest.approx(np.array([0.0]))
    assert result.decrement_sq == 0.0
    assert result.lower_factor == pytest.approx(np.array([[1.0]]))


def test_shifted_objective_uses_the_stable_delta_callback() -> None:
    def objective(x: np.ndarray) -> float:
        return 1_250_000.0 + 0.5 * float(x[0] ** 2)

    def delta(x: np.ndarray, step: np.ndarray) -> float:
        h = float(step[0])
        return h * (float(x[0]) + h / 2)

    result = newton(
        objective,
        lambda x: np.array([x[0]]),
        lambda x: np.array([[1.0]]),
        np.array([1e-5]),
        objective_delta=delta,
    )

    assert result.accepted_steps == 1
    assert float(result.mode[0]) == pytest.approx(0.0, abs=1e-20)
    assert result.decrement_sq == pytest.approx(0.0, abs=1e-30)


def test_exponential_limit_raises_after_fifty_steps() -> None:
    def objective(x: np.ndarray) -> float:
        return math.exp(float(x[0]) + 100.0)

    with pytest.raises(ConvergenceError):
        newton(
            objective,
            lambda x: np.array([math.exp(float(x[0]) + 100.0)]),
            lambda x: np.array([[math.exp(float(x[0]) + 100.0)]]),
            np.array([0.0]),
        )


def test_f20b_luce_solver_and_stable_delta() -> None:
    agents, decks = registries()
    row = normalize_game_row(
        make_game(
            "g1",
            agents=("H", "A", "B", "C"),
            decks=("d1", "d2", "d3", "d4"),
            winner_seat=0,
        )
    )
    index = ParameterIndex.build(
        agents,
        decks,
        ["A", "B", "C", "H"],
        ["d1", "d2", "d3", "d4"],
        swap_effect=False,
    )
    design = design_row(row.agents, row.decks, row.swap_counts, index, decks)
    table = PatternTable(
        patterns=(
            Pattern(
                key=(row.agents, row.decks, row.swap_counts),
                design=design,
                target=np.array([1_000_000, 1_000_000, 0, 0], dtype=np.int64),
                weight=2_000_000,
            ),
        ),
        index=index,
        cluster_counts=MappingProxyType({}),
        n_games=2_000_000,
    )
    precision = np.zeros(len(index))
    for name in index.names:
        if name.startswith("theta["):
            precision[index.index(name)] = 1 / 9
        elif name.startswith("seat["):
            precision[index.index(name)] = 4.0
        elif name.startswith("mu["):
            precision[index.index(name)] = 1.0
        elif name.startswith("delta["):
            precision[index.index(name)] = 4.0

    solution = newton(
        lambda psi: objective(psi, table, precision),
        lambda psi: objective_grad(psi, table, precision),
        lambda psi: objective_hess(psi, table, precision),
        np.zeros(len(index)),
        objective_delta=lambda psi, step: objective_delta(psi, step, table, precision),
        decrement_scale=2_000_000,
    )
    assert solution.decrement_sq <= 2e-14

    step = np.zeros(len(index))
    increments = [1e-8, -2e-8, 3e-8, -4e-8]
    for seat, value in enumerate(increments):
        step[index.index(f"seat[{seat}]")] = value
    actual = objective_delta(np.zeros(len(index)), step, table, precision)

    with localcontext() as context:
        context.prec = 80
        q = [Decimal(value) for value in increments]
        q0 = q[0]
        relative = [value - q0 for value in q]
        sum_exp = sum(Decimal("0.25") * value.exp() for value in relative)
        log_ratio = sum_exp.ln()
        targets = [Decimal(1_000_000), Decimal(1_000_000), Decimal(0), Decimal(0)]
        likelihood = Decimal(2_000_000) * log_ratio - sum(
            target * value for target, value in zip(targets, relative, strict=True)
        )
        ridge = sum(
            Decimal(str(precision[index.index(f"seat[{seat}]")])) * Decimal(str(value)) ** 2 / 2
            for seat, value in enumerate(increments)
        )
        expected = float(likelihood + ridge)

    assert actual == pytest.approx(expected, rel=1e-10, abs=1e-12)
