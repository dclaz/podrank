"""ModelSpec and solver error paths."""

from __future__ import annotations

import numpy as np
import pytest

from podrank.errors import ConvergenceError
from podrank.rating.solve import newton
from podrank.rating.spec import ModelSpec


@pytest.mark.parametrize(
    "kwargs",
    [
        {"swap_effect": 1},
        {"allow_pooled_variants": 1},
        {"weak_ratio": 0},
        {"weak_ratio": float("nan")},
        {"tol_eig": 0},
        {"tol_eig": 1},
        {"tol_alias": -1},
        {"tol_alias": 1},
        {"max_params": 0},
        {"max_params": True},
        {"max_dense_bytes": 0},
        {"max_dense_bytes": True},
    ],
)
def test_model_spec_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ModelSpec(**kwargs)  # type: ignore[arg-type]


def test_solver_rejects_bad_inputs() -> None:
    identity = lambda x: np.eye(1)  # noqa: E731
    gradient = lambda x: np.array([x[0]])  # noqa: E731
    objective = lambda x: 0.5 * float(x[0] ** 2)  # noqa: E731

    with pytest.raises(ValueError, match="x0"):
        newton(objective, gradient, identity, np.array([[1.0]]))
    with pytest.raises(ValueError, match="x0"):
        newton(objective, gradient, identity, np.array([float("nan")]))
    with pytest.raises(ValueError, match="decrement_scale"):
        newton(objective, gradient, identity, np.array([1.0]), decrement_scale=0)
    with pytest.raises(ValueError, match="decrement_scale"):
        newton(objective, gradient, identity, np.array([1.0]), decrement_scale=float("nan"))


def test_solver_rejects_nonfinite_derivatives_and_bad_hessian() -> None:
    objective = lambda x: 0.5 * float(x[0] ** 2)  # noqa: E731
    with pytest.raises(ConvergenceError, match="non-finite"):
        newton(
            objective,
            lambda x: np.array([float("nan")]),
            lambda x: np.eye(1),
            np.array([1.0]),
        )
    with pytest.raises(ConvergenceError, match="factorisation"):
        newton(
            objective,
            lambda x: np.array([x[0]]),
            lambda x: np.zeros((1, 1)),
            np.array([1.0]),
        )


def test_solver_armijo_exhaustion() -> None:
    with pytest.raises(ConvergenceError, match="Armijo"):
        newton(
            lambda x: -float(x[0]),
            lambda x: np.array([1.0]),
            lambda x: np.eye(1),
            np.array([0.0]),
        )
