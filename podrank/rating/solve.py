"""Damped Newton solver with Cholesky steps (Appendix C.2)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from podrank.errors import ConvergenceError

Objective = Callable[[np.ndarray], float]
Gradient = Callable[[np.ndarray], np.ndarray]
Hessian = Callable[[np.ndarray], np.ndarray]
Delta = Callable[[np.ndarray, np.ndarray], float]
Evaluate = Callable[[np.ndarray], tuple[float, np.ndarray, np.ndarray]]


@dataclass(frozen=True, slots=True)
class SolveResult:
    """The returned mode and its lower Cholesky factor."""

    mode: np.ndarray
    lower_factor: np.ndarray
    accepted_steps: int
    decrement_sq: float
    backtracks: int = 0
    data_hessian: np.ndarray | None = None


def _lower_factor(hessian: np.ndarray) -> np.ndarray:
    try:
        factor, _ = cho_factor(hessian, lower=True, check_finite=True)
    except (LinAlgError, ValueError) as exc:
        raise ConvergenceError("Hessian factorisation failed") from exc
    return np.tril(factor)


def _solve(factor: np.ndarray, vector: np.ndarray) -> np.ndarray:
    try:
        solution = cho_solve((factor, True), vector, check_finite=True)
    except (LinAlgError, ValueError) as exc:
        raise ConvergenceError("Cholesky solve failed") from exc
    return np.asarray(solution, dtype=np.float64)


def newton(
    objective: Objective,
    grad: Gradient,
    hess: Hessian,
    x0: np.ndarray,
    *,
    objective_delta: Delta | None = None,
    decrement_scale: float = 1.0,
    evaluate: Evaluate | None = None,
) -> SolveResult:
    """Minimise a smooth objective with damped Newton and Armijo backtracking."""

    psi = np.asarray(x0, dtype=np.float64).copy()
    if psi.ndim != 1 or not np.all(np.isfinite(psi)):
        raise ValueError("x0 must be a finite one-dimensional vector")
    scale = float(decrement_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("decrement_scale must be finite and > 0")

    accepted = 0
    backtracks = 0
    while True:
        if evaluate is not None:
            _, gradient_value, hessian_value = evaluate(psi)
            gradient = np.asarray(gradient_value, dtype=np.float64)
            hessian = np.asarray(hessian_value, dtype=np.float64)
        else:
            gradient = np.asarray(grad(psi), dtype=np.float64)
            hessian = np.asarray(hess(psi), dtype=np.float64)
        if not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
            raise ConvergenceError("non-finite derivative at the current iterate")
        factor = _lower_factor(hessian)
        direction = _solve(factor, -gradient)
        decrement_sq = -float(np.dot(gradient, direction))
        if decrement_sq < 0 and abs(decrement_sq) <= 1e-20:
            decrement_sq = 0.0
        if not math.isfinite(decrement_sq) or decrement_sq < 0:
            raise ConvergenceError("invalid Newton decrement")

        if decrement_sq / scale <= 1e-20:
            return SolveResult(
                mode=psi.copy(),
                lower_factor=factor,
                accepted_steps=accepted,
                decrement_sq=decrement_sq,
                backtracks=backtracks,
            )
        if accepted >= 50:
            raise ConvergenceError("no convergence after 50 accepted steps")

        slope = float(np.dot(gradient, direction))
        accepted_trial = False
        for exponent in range(41):
            step_size = 2.0**-exponent
            trial = psi + step_size * direction
            if not np.all(np.isfinite(trial)):
                continue
            try:
                trial_objective = float(objective(trial))
            except (FloatingPointError, OverflowError):
                continue
            if not math.isfinite(trial_objective):
                continue
            if objective_delta is not None:
                delta = float(objective_delta(psi, step_size * direction))
            else:
                delta = trial_objective - float(objective(psi))
            if not math.isfinite(delta):
                continue
            if delta <= 1e-4 * step_size * slope:
                psi = trial
                accepted += 1
                backtracks += exponent
                accepted_trial = True
                break
        if not accepted_trial:
            backtracks += 40
            raise ConvergenceError("Armijo backtracking exhausted")


__all__ = ["SolveResult", "newton"]
