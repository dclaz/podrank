"""SC1/SC2 statistical study harness.

The releases use this module to run the predeclared coverage/emission/bias
study over the frozen reference worlds. It is a release-review tool: a target
miss is reported for review, never converted into a test tolerance change.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np

from podrank.data import GameTable
from podrank.errors import ConvergenceError, NoRankableEvidence
from podrank.rating import fit
from podrank.rating.field import Field
from podrank.rating.identification import AnchorUnsupported
from podrank.sim.scenarios import standard_rows
from podrank.sim.world import SimWorld, reference_world
from podrank.uncertainty import cluster_bootstrap

StudyKind = Literal["log_worth", "sdwp"]


@dataclass(frozen=True, slots=True)
class StudyQuantity:
    key: tuple[str, ...]
    kind: StudyKind
    truth: float
    n_scored: int
    n_emitted: int
    n_hits: int
    coverage: float | None
    mean_bias: float | None


@dataclass(frozen=True, slots=True)
class StudyReport:
    scenario: str
    world: str
    replications: int
    bootstrap_replicates: int
    failed_replications: int
    quantities: tuple[StudyQuantity, ...]

    @property
    def n_scored(self) -> int:
        return sum(quantity.n_scored for quantity in self.quantities)

    @property
    def n_emitted(self) -> int:
        return sum(quantity.n_emitted for quantity in self.quantities)

    @property
    def emission_rate(self) -> float:
        return self.n_emitted / self.n_scored if self.n_scored else 0.0

    @property
    def pooled_coverage(self) -> float | None:
        scored = self.n_emitted
        if not scored:
            return None
        hits = sum(quantity.n_hits for quantity in self.quantities)
        return hits / scored

    def low_coverage(self, threshold: float = 0.90) -> tuple[tuple[str, ...], ...]:
        return tuple(
            quantity.key
            for quantity in self.quantities
            if quantity.coverage is not None and quantity.coverage < threshold
        )

    def target_misses(
        self,
        *,
        coverage_range: tuple[float, float] = (0.93, 0.97),
        min_emission: float = 0.99,
        max_log_worth_bias: float = 0.05,
        max_sdwp_bias: float = 0.01,
    ) -> tuple[str, ...]:
        misses: list[str] = []
        coverage = self.pooled_coverage
        if coverage is None or not coverage_range[0] <= coverage <= coverage_range[1]:
            misses.append(f"pooled coverage {coverage!r} outside {coverage_range}")
        if self.emission_rate < min_emission:
            misses.append(f"emission rate {self.emission_rate:.4f} < {min_emission}")
        for quantity in self.quantities:
            if quantity.mean_bias is None:
                continue
            limit = max_log_worth_bias if quantity.kind == "log_worth" else max_sdwp_bias
            if abs(quantity.mean_bias) > limit:
                misses.append(f"{quantity.key} bias {quantity.mean_bias:.4f} exceeds {limit}")
        return tuple(misses)


@dataclass(slots=True)
class _Accumulator:
    kind: StudyKind
    n_scored: int = 0
    n_emitted: int = 0
    n_hits: int = 0
    bias_sum: float = 0.0
    truth_sum: float = 0.0


def scoring_contrasts(world: SimWorld) -> tuple[tuple[str, str, str], ...]:
    """Return the predeclared §11.3 contrast scoring list, deduplicated."""

    ordered = sorted(world.agent_ids, key=lambda agent: world.theta[agent])
    median = ordered[(len(ordered) - 1) // 2]
    candidates = [
        ("top_vs_second", ordered[-1], ordered[-2]),
        ("top_vs_median", ordered[-1], median),
        ("median_vs_bottom", median, ordered[0]),
    ]
    candidates.extend(
        (f"vs_anchor:{agent}", agent, "agent_00")
        for agent in world.agent_ids
        if agent != "agent_00"
    )
    unique: dict[tuple[str, str], str] = {}
    for label, left, right in candidates:
        unique.setdefault((left, right), label)
    return tuple((label, left, right) for (left, right), label in unique.items())


def truth_sdwp(
    world: SimWorld,
    field: Field,
    agent_id: str,
    *,
    deck_effects: dict[str, float] | None = None,
) -> float:
    """Compute SDWP at the generating parameters, independently of the fit."""

    theta = world.theta
    mu = world.bracket_effects
    delta = world.deck_effects if deck_effects is None else deck_effects
    seats = world.seats
    decks = world.deck_registry()
    total = 0.0
    for context in field.contexts:
        non_focal = [seat for seat in range(4) if seat != context.focal_seat]
        agents = [agent_id] * 4
        for opponent, seat in zip(context.opponents, non_focal, strict=True):
            agents[seat] = opponent
        utilities = np.empty(4, dtype=np.float64)
        for seat in range(4):
            deck = context.decks[seat]
            utilities[seat] = (
                theta[agents[seat]] + mu[decks.by_id[deck].bracket] + delta[deck] + seats[seat]
            )
        utilities -= utilities.max()
        probabilities = np.exp(utilities)
        probabilities /= probabilities.sum()
        total += context.weight * float(probabilities[context.focal_seat])
    return total


def redraw_deck_effects(world: SimWorld, seed: int, replication: int) -> dict[str, float]:
    """Draw the SC1/SC2 replication's true δ from N(0, 0.5²)."""

    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(1, replication)))
    )
    return {deck: float(rng.normal(0.0, 0.5)) for deck in world.deck_ids}


def replication_seed(seed: int, replication: int) -> int:
    state = np.random.SeedSequence(seed, spawn_key=(0, replication)).generate_state(1)
    return int(state[0])


def _default_field_contexts(world: SimWorld) -> int:
    return 24 if world.agents <= 6 else 400


@dataclass(frozen=True, slots=True)
class _StudyJob:
    world_name: str
    seed: int
    replication: int
    rho_div: float | None
    redraw_delta: bool
    field_contexts: int
    bootstrap_replicates: int
    min_clusters: int
    min_entity_clusters: int


StudyUpdate = tuple[int, int, int, float, float]


def _process_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context()


def _replication_job(
    job: _StudyJob,
) -> tuple[int, dict[tuple[str, ...], StudyUpdate], bool]:
    world = reference_world(job.world_name)
    field = world.field(job.field_contexts)
    contrasts = scoring_contrasts(world)
    run_seed = replication_seed(job.seed, job.replication)
    deck_effects = (
        redraw_deck_effects(world, job.seed, job.replication) if job.redraw_delta else None
    )
    rows = standard_rows(
        world,
        run_seed=run_seed,
        deck_effects=deck_effects,
        rho_div=job.rho_div,
    )
    try:
        post = fit(
            GameTable.from_records(rows),
            world.agent_registry(),
            world.deck_registry(),
        )
    except (ConvergenceError, NoRankableEvidence):
        return job.replication, {}, True
    boot = cluster_bootstrap(
        post,
        n=job.bootstrap_replicates,
        seed=run_seed,
        min_clusters=job.min_clusters,
        min_entity_clusters=job.min_entity_clusters,
    )
    updates: dict[tuple[str, ...], StudyUpdate] = {}
    for _, left, right in contrasts:
        result = boot.contrast(left, right)
        if result.status not in {"weak", "identified"} or result.value is None:
            continue
        truth = world.theta[left] - world.theta[right]
        emitted = 0
        hits = 0
        if result.interval is not None:
            emitted = 1
            hits = int(result.interval.lo <= truth <= result.interval.hi)
        updates[("contrast", left, right)] = (
            1,
            emitted,
            hits,
            result.value - truth,
            truth,
        )
    sdwp_report = boot.sdwp(field)
    if isinstance(sdwp_report, AnchorUnsupported):
        return job.replication, updates, True
    truths = {
        agent: truth_sdwp(world, field, agent, deck_effects=deck_effects)
        for agent in world.agent_ids
    }
    for row in sdwp_report.rows:
        truth = truths[row.agent_id]
        emitted = 0
        hits = 0
        if row.interval is not None:
            emitted = 1
            hits = int(row.interval.lo <= truth <= row.interval.hi)
        updates[("sdwp", row.agent_id)] = (1, emitted, hits, row.sdwp - truth, truth)
    return job.replication, updates, False


def run_study(
    scenario: str,
    *,
    world_name: str = "standard",
    replications: int,
    bootstrap_replicates: int = 100,
    seed: int = 303,
    rho_div: float | None = None,
    redraw_delta: bool = True,
    field_contexts: int | None = None,
    min_clusters: int = 50,
    min_entity_clusters: int = 10,
    workers: int = 1,
) -> StudyReport:
    """Run one SC1/SC2-style coverage, emission and bias study.

    Replications are independent. ``workers > 1`` spreads them over a process
    pool; each worker keeps its bootstrap serial. One BLAS thread per worker is
    a reference-run requirement, configured before Python starts.
    """

    if replications < 1:
        raise ValueError("replications must be positive")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    world = reference_world(world_name)
    resolved_field_contexts = (
        _default_field_contexts(world) if field_contexts is None else field_contexts
    )
    contrasts = scoring_contrasts(world)
    accumulators: dict[tuple[str, ...], _Accumulator] = {
        ("contrast", left, right): _Accumulator("log_worth") for _, left, right in contrasts
    }
    accumulators.update((("sdwp", agent), _Accumulator("sdwp")) for agent in world.agent_ids)

    jobs = [
        _StudyJob(
            world_name=world_name,
            seed=seed,
            replication=replication,
            rho_div=rho_div,
            redraw_delta=redraw_delta,
            field_contexts=resolved_field_contexts,
            bootstrap_replicates=bootstrap_replicates,
            min_clusters=min_clusters,
            min_entity_clusters=min_entity_clusters,
        )
        for replication in range(replications)
    ]
    if workers == 1:
        results = [_replication_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, replications),
            mp_context=_process_context(),
        ) as executor:
            results = list(executor.map(_replication_job, jobs))

    failed = 0
    for _, updates, failed_flag in results:
        if failed_flag:
            failed += 1
        for key, (scored, emitted, hits, bias, truth) in updates.items():
            accumulator = accumulators[key]
            accumulator.n_scored += scored
            accumulator.n_emitted += emitted
            accumulator.n_hits += hits
            accumulator.bias_sum += bias
            accumulator.truth_sum += truth

    quantities: list[StudyQuantity] = []
    for key in sorted(accumulators):
        accumulator = accumulators[key]
        scored = accumulator.n_scored
        emitted = accumulator.n_emitted
        quantities.append(
            StudyQuantity(
                key=key,
                kind=accumulator.kind,
                truth=accumulator.truth_sum / scored if scored else 0.0,
                n_scored=scored,
                n_emitted=emitted,
                n_hits=accumulator.n_hits,
                coverage=accumulator.n_hits / emitted if emitted else None,
                mean_bias=accumulator.bias_sum / scored if scored else None,
            )
        )
    return StudyReport(
        scenario=scenario,
        world=world_name,
        replications=replications,
        bootstrap_replicates=bootstrap_replicates,
        failed_replications=failed,
        quantities=tuple(quantities),
    )


__all__ = [
    "StudyKind",
    "StudyQuantity",
    "StudyReport",
    "redraw_deck_effects",
    "replication_seed",
    "run_study",
    "scoring_contrasts",
    "truth_sdwp",
]
