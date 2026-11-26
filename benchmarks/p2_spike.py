"""Reproducible P2 performance spike on a synthetic perf_large-shaped world.

Run one measurement per fresh process, for example:

    uv run --locked python benchmarks/p2_spike.py
    uv run --locked python benchmarks/p2_spike.py --clusters 1000 --sdwp-contexts 1000

This script deliberately uses synthetic data because the P3 simulator/world
files do not exist yet. It reports cold-fit components separately from I0 and
D2/D4, as required by IMPLEMENTATION_PLAN.md §12.

Run it on the named 16-core reference machine for release decisions. Local WSL2
or laptop results are provisional: run five fresh processes, pin the relevant
BLAS thread settings before import, and report median/min/max rather than the
fastest run.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import resource
import time

import numpy as np
import scipy

from podrank import (
    AgentRegistry,
    DeckRegistry,
    Field,
    FieldContext,
    GameTable,
    Inclusion,
    ModelSpec,
)
from podrank.data.inclusion import run_pipeline
from podrank.rating.design import ParameterIndex, build_patterns
from podrank.rating.fit import _precision_vector
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
from podrank.rating.solve import newton


def _rows(args: argparse.Namespace, agents: AgentRegistry):
    rng = np.random.default_rng(args.seed)
    rows = []
    for cluster in range(args.clusters):
        agent_indices = rng.choice(args.agents, size=4, replace=False)
        bracket = int(rng.integers(0, args.brackets))
        deck_indices = rng.choice(args.decks_per_bracket, size=4, replace=False)
        deck_ids = [f"deck_b{bracket}_{int(index):02d}" for index in deck_indices]
        winner = int(rng.integers(0, 4))
        for rotation in range(4):
            seats = [None] * 4
            for position, agent_index in enumerate(agent_indices):
                seats[(position + rotation) % 4] = f"agent_{int(agent_index):02d}"
            rows.append(
                {
                    "game_id": f"c{cluster:07d}r{rotation}",
                    "played_at": f"2026-01-01T00:00:{cluster % 60:02d}Z",
                    "agents": seats,
                    "decks": deck_ids,
                    "policy_versions": [agents.by_id[agent].policy_version for agent in seats],
                    "swap_counts": [0, 0, 0, 0],
                    "outcome": "win",
                    "winner_seat": winner,
                    "suite": "train",
                    "cluster_id": f"c{cluster:07d}",
                    "engine_version": "e1",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--brackets", type=int, default=5)
    parser.add_argument("--decks-per-bracket", type=int, default=28)
    parser.add_argument("--clusters", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sdwp-contexts", type=int, default=10_000)
    args = parser.parse_args()

    started = time.perf_counter()
    agent_rows = [
        {
            "agent_id": f"agent_{index:02d}",
            "kind": "heuristic" if index == 0 else "checkpoint",
            "policy_version": f"a{index}v1",
            "is_anchor": index == 0,
        }
        for index in range(args.agents)
    ]
    deck_rows = [
        {
            "deck_id": f"deck_b{bracket}_{index:02d}",
            "bracket": bracket,
            "split": "train",
            "deck_version": f"b{bracket}_{index}v1",
        }
        for bracket in range(args.brackets)
        for index in range(args.decks_per_bracket)
    ]
    agents = AgentRegistry.from_records(agent_rows)
    decks = DeckRegistry.from_records(deck_rows)
    generation_started = time.perf_counter()
    rows = _rows(args, agents)
    generation_seconds = time.perf_counter() - generation_started
    table_started = time.perf_counter()
    table = GameTable.from_records(rows)
    table_seconds = time.perf_counter() - table_started
    del rows
    gc.collect()
    rss_after_input = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    i0_started = time.perf_counter()
    pipeline = run_pipeline(table, agents, decks, inclusion=Inclusion())
    i0_seconds = time.perf_counter() - i0_started
    index = ParameterIndex.build(
        agents,
        decks,
        pipeline.supported_agents,
        pipeline.supported_decks,
        swap_effect=False,
    )
    compression_started = time.perf_counter()
    spec = ModelSpec()
    patterns = build_patterns(pipeline.likelihood_rows, index, decks, spec)
    compression_seconds = time.perf_counter() - compression_started
    compact = patterns.compact
    assert compact is not None
    pattern_bytes = sum(
        array.nbytes
        for array in (
            compact.agent_indices,
            compact.deck_indices,
            compact.targets,
            compact.deck_mu_positions,
            compact.deck_delta_positions,
            compact.seat_positions,
            compact.reduction,
            compact.cluster_patterns,
            compact.cluster_winner_seats,
            compact.cluster_indices,
            compact.cluster_counts,
        )
    )
    precision = _precision_vector(index)
    cache: dict[str, object] = {"psi": None, "utilities": None, "probabilities": None}

    def evaluate(psi: np.ndarray):
        evaluation = compact_evaluate(psi, patterns, precision, cache=True)
        cache["psi"] = psi.copy()
        cache["utilities"] = evaluation.utilities
        cache["probabilities"] = evaluation.probabilities
        return evaluation.objective, evaluation.gradient, evaluation.hessian

    def delta(psi: np.ndarray, step: np.ndarray) -> float:
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

    solver_started = time.perf_counter()
    solution = newton(
        lambda psi: objective(psi, patterns, precision),
        lambda psi: objective_grad(psi, patterns, precision),
        lambda psi: objective_hess(psi, patterns, precision),
        np.zeros(len(index)),
        objective_delta=delta,
        decrement_scale=max(1.0, float(len(pipeline.likelihood_rows))),
        evaluate=evaluate,
    )
    solver_seconds = time.perf_counter() - solver_started
    d4_started = time.perf_counter()
    row_space = row_space_basis(patterns, spec.tol_eig)
    null_space = null_space_basis(row_space)
    d4_seconds = time.perf_counter() - d4_started
    d2_started = time.perf_counter()
    data_hessian = likelihood_hess(solution.mode, patterns)
    d2_seconds = time.perf_counter() - d2_started

    sdwp_seconds = None
    if args.sdwp_contexts:
        rng = np.random.default_rng(args.seed + 1)
        contexts = [
            FieldContext(
                decks=tuple(
                    f"deck_b{int(rng.integers(0, args.brackets))}_"
                    f"{int(rng.integers(0, args.decks_per_bracket)):02d}"
                    for _ in range(4)
                ),
                swap_counts=(0, 0, 0, 0),
                opponents=tuple(f"agent_{int(rng.integers(0, 3)):02d}" for _ in range(3)),
                focal_seat=int(rng.integers(0, 4)),
                weight=1.0,
            )
            for _ in range(args.sdwp_contexts)
        ]
        field = Field.from_contexts("perf_spike", 1, contexts)
        components, component_of = build_components(
            agents, index.supported_agents, pipeline.likelihood_rows, "agent_00"
        )
        contrast_context = ContrastContext(
            mode=solution.mode,
            lower_factor=solution.lower_factor,
            precision=precision,
            row_space=row_space,
            index=index,
            spec=spec,
        )
        eligibility = build_eligibility(agents, index, contrast_context, components, component_of)
        posterior = Posterior(
            mode=solution.mode,
            lower_factor=solution.lower_factor,
            parameter_index=index,
            spec=spec,
            inclusion=Inclusion(),
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
            pooled_variants=False,
            _provenance=Provenance(
                package_version="spike",
                data_hash=pipeline.data_hash,
                agent_snapshot_id=agents.snapshot_id,
                deck_snapshot_id=decks.snapshot_id,
            ),
            _outcome_rates=build_outcome_rates(pipeline.eligible_rows),
            cluster_table={},
            cluster_strata={},
            stratum_crossings={},
        )
        sdwp_started = time.perf_counter()
        posterior.sdwp(field)
        sdwp_seconds = time.perf_counter() - sdwp_started

    total_seconds = time.perf_counter() - started
    print(
        json.dumps(
            {
                "games": len(table),
                "patterns": len(patterns),
                "parameters": len(index),
                "reduced_parameters": compact.reduced_size,
                "rank_d4": int(row_space.shape[1]),
                "nullity_d4": int(null_space.shape[1]),
                "accepted_steps": solution.accepted_steps,
                "backtracks": solution.backtracks,
                "decrement_sq": solution.decrement_sq,
                "timings_s": {
                    "generation": generation_seconds,
                    "game_table": table_seconds,
                    "i0": i0_seconds,
                    "compression": compression_seconds,
                    "newton": solver_seconds,
                    "cold_fit_design_compression_newton_cholesky": (
                        compression_seconds + solver_seconds
                    ),
                    "d4": d4_seconds,
                    "d2": d2_seconds,
                    "sdwp": sdwp_seconds,
                    "total_process": total_seconds,
                },
                "pattern_array_mib": pattern_bytes / 2**20,
                "rss_after_input_mib": rss_after_input,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "peak_rss_includes_game_table_and_generation": True,
                "environment": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "cpu_count": os.cpu_count(),
                    "cpu_affinity": (
                        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
                    ),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                    "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                    "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
