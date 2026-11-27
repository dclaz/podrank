"""R1 fitting pipeline: I0, design, likelihood, Newton and posterior."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping

import numpy as np

from podrank._version import __version__
from podrank.data.inclusion import Inclusion, run_pipeline
from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.data.table import GameTable
from podrank.errors import ModelTooLargeError, SwapsNotModelledError
from podrank.rating.design import ParameterIndex, PatternTable, build_patterns
from podrank.rating.identification import (
    ContrastContext,
    build_components,
    build_eligibility,
    null_space_basis,
    row_space_basis,
)
from podrank.rating.likelihood import (
    compact_evaluate,
    likelihood_hess,
    objective,
    objective_delta,
    objective_grad,
    objective_hess,
)
from podrank.rating.posterior import Posterior, Provenance, build_outcome_rates
from podrank.rating.solve import SolveResult, newton
from podrank.rating.spec import ModelSpec
from podrank.types import ClusterKey


def _precision_vector(index: ParameterIndex, spec: ModelSpec) -> np.ndarray:
    precision = np.zeros(len(index), dtype=np.float64)
    for name in index.names:
        if name.startswith("theta["):
            precision[index.index(name)] = 1.0 / 9.0
        elif name.startswith("seat["):
            precision[index.index(name)] = 4.0
        elif name.startswith("mu["):
            precision[index.index(name)] = 1.0
        elif name.startswith("delta["):
            precision[index.index(name)] = 1.0 / (spec.sigma_beta**2)
        elif name == "eta":
            precision[index.index(name)] = 100.0
        else:  # pragma: no cover - guarded by ParameterIndex
            raise AssertionError(f"unknown parameter {name!r}")
    return precision


def _warm_start(index: ParameterIndex, values: Mapping[str, float] | None) -> np.ndarray:
    result = np.zeros(len(index), dtype=np.float64)
    if values is None:
        return result
    for name, value in values.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"warm_start[{name!r}] must be finite")
        if name in index.positions:
            result[index.positions[name]] = numeric
    return result


def _memory_preflight(parameter_count: int, spec: ModelSpec) -> None:
    if parameter_count > spec.max_params:
        raise ModelTooLargeError(
            f"model has {parameter_count} parameters; max_params is {spec.max_params}"
        )
    estimate = 4 * 8 * parameter_count**2
    if estimate > spec.max_dense_bytes:
        raise ModelTooLargeError(
            f"estimated dense workspace is {estimate} bytes; "
            f"max_dense_bytes is {spec.max_dense_bytes}"
        )


_DEFAULT_SPEC = ModelSpec()
_DEFAULT_INCLUSION = Inclusion.default()


def _validate_cluster_strata(
    cluster_strata: Mapping[str, Mapping[ClusterKey, str]] | None,
    frame: Collection[ClusterKey],
) -> None:
    if cluster_strata is None:
        return
    if not isinstance(cluster_strata, Mapping):
        raise ValueError("cluster_strata must be a mapping or None")
    reserved = {"suite", "engine_version"}
    for column, values in cluster_strata.items():
        if not isinstance(column, str) or not column:
            raise ValueError("cluster_strata column names must be non-empty strings")
        if column in reserved:
            raise ValueError(f"cluster_strata column {column!r} is reserved")
        if not isinstance(values, Mapping):
            raise ValueError(f"cluster_strata[{column!r}] must be a mapping")
        missing = set(frame) - set(values)
        if missing:
            raise ValueError(f"cluster_strata[{column!r}] is missing frame keys")
        for _key, label in values.items():
            if not isinstance(label, str) or not label:
                raise ValueError(f"cluster_strata[{column!r}] labels must be non-empty strings")


def solve_patterns(
    patterns: PatternTable,
    precision: np.ndarray,
    initial: np.ndarray,
    *,
    decrement_scale: float,
) -> SolveResult:
    """Solve one compressed likelihood table from a supplied start point."""

    cache: dict[str, object] = {
        "psi": None,
        "utilities": None,
        "probabilities": None,
    }
    prior_diagonal = np.diag(precision)
    last_data_hessian: np.ndarray | None = None

    def evaluate(psi: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        nonlocal last_data_hessian
        evaluation = compact_evaluate(psi, patterns, precision, cache=True)
        last_data_hessian = evaluation.hessian - prior_diagonal
        cache["psi"] = psi.copy()
        cache["utilities"] = evaluation.utilities
        cache["probabilities"] = evaluation.probabilities
        return evaluation.objective, evaluation.gradient, evaluation.hessian

    def delta(psi: np.ndarray, step: np.ndarray) -> float:
        cached_psi = cache["psi"]
        if isinstance(cached_psi, np.ndarray) and np.array_equal(psi, cached_psi):
            utilities = cache["utilities"]
            probabilities = cache["probabilities"]
            assert isinstance(utilities, np.ndarray)
            assert isinstance(probabilities, np.ndarray)
            return objective_delta(
                psi,
                step,
                patterns,
                precision,
                utilities_cache=utilities,
                probabilities_cache=probabilities,
            )
        return objective_delta(psi, step, patterns, precision)

    result = newton(
        lambda psi: objective(psi, patterns, precision),
        lambda psi: objective_grad(psi, patterns, precision),
        lambda psi: objective_hess(psi, patterns, precision),
        initial,
        objective_delta=delta,
        decrement_scale=decrement_scale,
        evaluate=evaluate,
    )
    return SolveResult(
        mode=result.mode,
        lower_factor=result.lower_factor,
        accepted_steps=result.accepted_steps,
        decrement_sq=result.decrement_sq,
        backtracks=result.backtracks,
        data_hessian=last_data_hessian,
    )


def fit(
    games: GameTable,
    agents: AgentRegistry,
    decks: DeckRegistry,
    spec: ModelSpec = _DEFAULT_SPEC,
    inclusion: Inclusion = _DEFAULT_INCLUSION,
    split_policy: Mapping[str, Collection[str]] | None = None,
    cluster_strata: Mapping[str, Mapping[ClusterKey, str]] | None = None,
    *,
    warm_start: Mapping[str, float] | None = None,
    tag_schema: Mapping[str, object] | None = None,
) -> Posterior:
    """Fit the R1 Luce model and return a :class:`Posterior`."""

    pipeline = run_pipeline(
        games,
        agents,
        decks,
        inclusion=inclusion,
        split_policy=split_policy,
        tag_schema=tag_schema,
        validate_only=False,
        strict=True,
    )

    nonzero_swaps = any(
        any(count != 0 for count in row.swap_counts) for row in pipeline.eligible_rows
    )
    pooled_variants = False
    if nonzero_swaps and not spec.swap_effect:
        if not spec.allow_pooled_variants:
            raise SwapsNotModelledError(
                "nonzero swap counts require swap_effect=True or allow_pooled_variants=True"
            )
        pooled_variants = True

    index = ParameterIndex.build(
        agents,
        decks,
        pipeline.supported_agents,
        pipeline.supported_decks,
        swap_effect=spec.swap_effect,
    )
    _memory_preflight(len(index), spec)
    patterns = build_patterns(pipeline.likelihood_rows, index, decks, spec)
    precision = _precision_vector(index, spec)
    initial = _warm_start(index, warm_start)
    cluster_values: dict[ClusterKey, list[str | None]] = {}
    for row in pipeline.likelihood_rows:
        key: ClusterKey = ("cluster", row.cluster_id) if row.cluster_id else ("game", row.game_id)
        values = cluster_values.get(key)
        if values is None:
            cluster_values[key] = [row.suite, row.engine_version]
        else:
            if values[0] != row.suite:
                values[0] = None
            if values[1] != row.engine_version:
                values[1] = None
    frame_keys = tuple(sorted(cluster_values))
    _validate_cluster_strata(cluster_strata, frame_keys)
    cluster_table = {key: (cluster_values[key][0], cluster_values[key][1]) for key in frame_keys}
    stratum_crossings: dict[str, tuple[int, ...]] = {}
    for column, position in (("suite", 0), ("engine_version", 1)):
        indices = tuple(
            index for index, key in enumerate(frame_keys) if cluster_values[key][position] is None
        )
        if indices:
            stratum_crossings[column] = indices
    retained_strata = (
        {}
        if cluster_strata is None
        else {column: dict(values) for column, values in cluster_strata.items()}
    )
    outcome_rates = build_outcome_rates(pipeline.eligible_rows)

    solution = solve_patterns(
        patterns,
        precision,
        initial,
        decrement_scale=max(1.0, float(len(pipeline.likelihood_rows))),
    )

    row_space = row_space_basis(patterns, spec.tol_eig)
    null_space = null_space_basis(row_space)
    data_hessian = solution.data_hessian
    if data_hessian is None:
        data_hessian = likelihood_hess(solution.mode, patterns)
    context = ContrastContext(
        mode=solution.mode,
        lower_factor=solution.lower_factor,
        precision=precision,
        row_space=row_space,
        index=index,
        spec=spec,
    )
    anchor_id = agents.anchors[0].agent_id
    components, component_of = build_components(
        agents,
        index.supported_agents,
        pipeline.likelihood_rows,
        anchor_id,
    )
    eligibility = build_eligibility(agents, index, context, components, component_of)
    provenance = Provenance(
        package_version=__version__,
        data_hash=pipeline.data_hash,
        agent_snapshot_id=agents.snapshot_id,
        deck_snapshot_id=decks.snapshot_id,
    )
    return Posterior(
        mode=solution.mode,
        lower_factor=solution.lower_factor,
        parameter_index=index,
        spec=spec,
        inclusion=inclusion,
        _counts=pipeline.counts,
        _agents=agents,
        _decks=decks,
        data_hash=pipeline.data_hash,
        patterns=patterns,
        precision=precision,
        row_space=row_space,
        null_space=null_space,
        data_hessian=data_hessian,
        _components=components,
        component_of=component_of,
        _eligibility=eligibility,
        pooled_variants=pooled_variants,
        _provenance=provenance,
        _outcome_rates=outcome_rates,
        cluster_table=cluster_table,
        cluster_strata=retained_strata,
        stratum_crossings=stratum_crossings,
    )


__all__ = ["fit"]
