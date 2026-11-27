"""Cluster bootstrap for R1 posterior statistics (B1–B6)."""

from __future__ import annotations

import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from podrank.errors import ConvergenceError
from podrank.rating.design import CompactPatterns, ParameterIndex, PatternTable
from podrank.rating.field import Field
from podrank.rating.fit import solve_patterns
from podrank.rating.identification import (
    AgentContrastRow,
    AnchorUnsupported,
    ContrastResult,
    DeckContrastRow,
    DeckNuisanceReport,
    EffectNotModelled,
    Interval,
    SeatContrastRow,
    agent_vector,
    beta_vector,
    seat_vector,
)
from podrank.rating.posterior import Posterior, SdwpReport, SdwpRow
from podrank.types import ClusterKey

DiagnosticKey = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class QuantityDiagnostics:
    n_converged: int
    n_zero_support: int | None
    n_used: int


@dataclass
class BootstrapResult:
    posterior: Posterior
    n: int
    seed: int
    strata: str | None
    replicate_weights: np.ndarray
    replicate_modes: np.ndarray
    failed_fraction: float
    min_clusters: int
    min_entity_clusters: int
    frame_keys: tuple[ClusterKey, ...]
    agent_clusters: Mapping[str, frozenset[int]]
    deck_clusters: Mapping[str, frozenset[int]]
    _diagnostics: dict[DiagnosticKey, QuantityDiagnostics] = field(default_factory=dict)

    @property
    def n_converged(self) -> int:
        return int(self.replicate_modes.shape[0])

    @property
    def n_failed(self) -> int:
        return self.n - self.n_converged

    def diagnostics(self) -> Mapping[DiagnosticKey, QuantityDiagnostics]:
        return dict(self._diagnostics)

    def _record(
        self, key: DiagnosticKey, n_zero_support: int | None, n_used: int
    ) -> QuantityDiagnostics:
        diagnostic = QuantityDiagnostics(
            n_converged=self.n_converged,
            n_zero_support=n_zero_support,
            n_used=n_used,
        )
        self._diagnostics[key] = diagnostic
        return diagnostic

    def _gate(
        self,
        key: DiagnosticKey,
        support_sets: Sequence[frozenset[int]],
    ) -> tuple[bool, tuple[str, ...], int | None]:
        if len(self.frame_keys) < self.min_clusters:
            return False, ("insufficient_clusters",), None
        if any(len(support) < self.min_entity_clusters for support in support_sets):
            return False, ("insufficient_clusters",), None
        if self.failed_fraction > 0.05:
            return False, ("unstable_bootstrap",), None
        zero_support = np.zeros(self.replicate_weights.shape[0], dtype=bool)
        for support in support_sets:
            if not support:
                zero_support[:] = True
                break
            indices = np.fromiter(support, dtype=np.int64, count=len(support))
            zero_support |= ~np.any(self.replicate_weights[:, indices] != 0, axis=1)
        n_zero = int(np.count_nonzero(zero_support))
        if n_zero / self.n_converged > 0.05:
            return False, ("insufficient_clusters",), n_zero
        return True, (), n_zero

    def _interval(
        self,
        key: DiagnosticKey,
        values: np.ndarray,
        support_sets: Sequence[frozenset[int]],
        labels: tuple[str, ...],
    ) -> tuple[Interval | None, tuple[str, ...], QuantityDiagnostics]:
        passed, gate_labels, n_zero = self._gate(key, support_sets)
        if not passed:
            return None, (*labels, *gate_labels), self._record(key, n_zero, 0)
        if values.size == 0:
            return None, (*labels, "unstable_bootstrap"), self._record(key, n_zero, 0)
        lo, hi = np.quantile(values, [0.025, 0.975], method="linear")
        result_labels = list(labels)
        if self.failed_fraction > 0.01:
            result_labels.append("unstable_bootstrap")
        interval = Interval(
            lo=float(lo),
            hi=float(hi),
            method="cluster_bootstrap",
            labels=tuple(result_labels),
        )
        self._record(key, n_zero, int(values.size))
        return interval, tuple(result_labels), self._diagnostics[key]

    def _support_for_contrast(self, vector: np.ndarray) -> list[frozenset[int]]:
        index = self.posterior.parameter_index
        supports: list[frozenset[int]] = []
        theta_start, theta_stop = index.block_slices["theta"]
        for position in range(theta_start, theta_stop):
            if vector[position] != 0:
                agent_id = index.supported_agents[position - theta_start]
                supports.append(self.agent_clusters[agent_id])
        delta_start, delta_stop = index.block_slices["delta"]
        for position in range(delta_start, delta_stop):
            if vector[position] != 0:
                deck_id = index.supported_decks[position - delta_start]
                supports.append(self.deck_clusters[deck_id])
        return supports

    def _support_for_agents_and_decks(
        self, agent_ids: Sequence[str], deck_ids: Sequence[str]
    ) -> list[frozenset[int]]:
        return [
            *(self.agent_clusters[agent_id] for agent_id in agent_ids),
            *(self.deck_clusters[deck_id] for deck_id in deck_ids),
        ]

    def contrast(self, a: str, b: str, *, include_withheld: bool = False) -> ContrastResult:
        original = self.posterior.contrast(a, b, include_withheld=include_withheld)
        key: DiagnosticKey = ("contrast", a, b)
        if original.status == "zero":
            self._record(key, None, 0)
            return original
        if original.status not in {"weak", "identified"}:
            self._record(key, None, 0)
            return original
        vector = agent_vector(self.posterior.parameter_index, a) - agent_vector(
            self.posterior.parameter_index, b
        )
        values = self.replicate_modes @ vector
        interval, labels, _ = self._interval(
            key,
            values,
            self._support_for_contrast(vector),
            original.labels,
        )
        return replace(original, interval=interval, labels=labels)

    def contrast_vector(self, vector: np.ndarray) -> ContrastResult:
        original = self.posterior.contrast_vector(vector)
        if original.status == "zero":
            self._record(("contrast_vector",), None, 0)
            return original
        if original.status not in {"weak", "identified"}:
            self._record(("contrast_vector",), None, 0)
            return original
        values = self.replicate_modes @ np.asarray(vector, dtype=np.float64)
        interval, labels, _ = self._interval(
            ("contrast_vector",),
            values,
            self._support_for_contrast(np.asarray(vector, dtype=np.float64)),
            original.labels,
        )
        return replace(original, interval=interval, labels=labels)

    def log_worth(
        self, *, include_withheld: bool = False
    ) -> tuple[AgentContrastRow, ...] | AnchorUnsupported:
        original = self.posterior.log_worth(include_withheld=include_withheld)
        if isinstance(original, AnchorUnsupported):
            return original
        anchor = self.posterior.components()
        if isinstance(anchor, AnchorUnsupported):
            return anchor
        rows: list[AgentContrastRow] = []
        for row in original:
            key: DiagnosticKey = ("log_worth", row.agent_id)
            if row.status == "zero":
                self._record(key, None, 0)
                rows.append(row)
                continue
            if row.status not in {"weak", "identified"}:
                self._record(key, None, 0)
                rows.append(row)
                continue
            vector = agent_vector(self.posterior.parameter_index, row.agent_id) - agent_vector(
                self.posterior.parameter_index, anchor.anchor_id
            )
            values = self.replicate_modes @ vector
            interval, labels, _ = self._interval(
                key,
                values,
                self._support_for_contrast(vector),
                row.labels,
            )
            rows.append(replace(row, interval=interval, labels=labels))
        return tuple(rows)

    def seats(self) -> tuple[SeatContrastRow, ...]:
        original = self.posterior.seats()
        mean = (
            sum(
                (seat_vector(self.posterior.parameter_index, seat) for seat in range(4)),
                np.zeros(len(self.posterior.parameter_index)),
            )
            / 4.0
        )
        rows: list[SeatContrastRow] = []
        for row in original:
            key: DiagnosticKey = ("seat", row.seat)
            if row.status not in {"zero", "weak", "identified"}:
                self._record(key, None, 0)
                rows.append(row)
                continue
            if row.status == "zero":
                self._record(key, None, 0)
                rows.append(row)
                continue
            vector = seat_vector(self.posterior.parameter_index, row.seat) - mean
            values = self.replicate_modes @ vector
            interval, labels, _ = self._interval(key, values, (), row.labels)
            rows.append(replace(row, interval=interval, labels=labels))
        return tuple(rows)

    def decks(self) -> DeckNuisanceReport:
        original = self.posterior.decks()
        if original.baseline_status == "unavailable":
            return original
        mean_reference = sum(
            (
                beta_vector(self.posterior.parameter_index, deck_id, self.posterior._decks)
                for deck_id in original.reference_deck_ids
            ),
            np.zeros(len(self.posterior.parameter_index)),
        ) / len(original.reference_deck_ids)
        rows: list[DeckContrastRow] = []
        for row in original.rows:
            key: DiagnosticKey = ("deck", row.deck_id)
            if row.status not in {"zero", "weak", "identified"}:
                self._record(key, None, 0)
                rows.append(row)
                continue
            if row.status == "zero":
                self._record(key, None, 0)
                rows.append(row)
                continue
            vector = (
                beta_vector(
                    self.posterior.parameter_index,
                    row.deck_id,
                    self.posterior._decks,
                )
                - mean_reference
            )
            values = self.replicate_modes @ vector
            interval, labels, _ = self._interval(
                key, values, self._support_for_contrast(vector), row.labels
            )
            rows.append(replace(row, interval=interval, labels=labels))
        return DeckNuisanceReport(
            baseline_status=original.baseline_status,
            reference_deck_ids=original.reference_deck_ids,
            rows=tuple(rows),
        )

    def swap_effect(self) -> ContrastResult | EffectNotModelled:
        original = self.posterior.swap_effect()
        if isinstance(original, EffectNotModelled):
            return original
        key: DiagnosticKey = ("swap_effect",)
        if original.status not in {"zero", "weak", "identified"}:
            self._record(key, None, 0)
            return original
        if original.status == "zero":
            self._record(key, None, 0)
            return original
        vector = np.zeros(len(self.posterior.parameter_index))
        vector[self.posterior.parameter_index.index("eta")] = 1.0
        values = self.replicate_modes @ vector
        interval, labels, _ = self._interval(key, values, (), original.labels)
        return replace(original, interval=interval, labels=labels)

    def sdwp(self, field: Field) -> SdwpReport | AnchorUnsupported:
        original = self.posterior.sdwp(field)
        if isinstance(original, AnchorUnsupported):
            return original
        common_agents: set[str] = set()
        common_decks: set[str] = set()
        for context in field.contexts:
            common_agents.update(context.opponents)
            common_decks.update(context.decks)
        agent_order = [row.agent_id for row in original.rows]
        values_matrix = self.posterior._sdwp_matrix(field, agent_order, self.replicate_modes)
        rows: list[SdwpRow] = []
        for index, row in enumerate(original.rows):
            key: DiagnosticKey = ("sdwp", original.field_hash, row.agent_id)
            interval, labels, _ = self._interval(
                key,
                values_matrix[:, index],
                self._support_for_agents_and_decks(
                    sorted({row.agent_id} | common_agents), sorted(common_decks)
                ),
                row.labels,
            )
            rows.append(replace(row, interval=interval, labels=labels))
        return replace(original, rows=tuple(rows))


def _replicate_targets(compact: CompactPatterns, weights: np.ndarray) -> np.ndarray:
    flat = _target_flat(compact)
    return _replicate_targets_from_flat(compact, weights, flat, compact.n_patterns * 4)


def _target_flat(compact: CompactPatterns) -> np.ndarray:
    return compact.cluster_patterns.astype(np.int64) * 4 + compact.cluster_winner_seats.astype(
        np.int64
    )


def _replicate_targets_from_flat(
    compact: CompactPatterns, weights: np.ndarray, flat: np.ndarray, size: int
) -> np.ndarray:
    values = weights[compact.cluster_indices] * compact.cluster_counts
    return (
        np.bincount(flat, weights=values, minlength=size)
        .astype(np.int64)
        .reshape(compact.n_patterns, 4)
    )


def _refit_weights(post: Posterior, weights: np.ndarray) -> np.ndarray:
    """Refit one replicate from exact cluster weights."""

    compact = post.patterns.compact
    if compact is None:
        raise ValueError("cluster bootstrap requires compact pattern state")
    targets = _replicate_targets(compact, np.asarray(weights, dtype=np.int64))
    replicate_table = replace(
        post.patterns,
        compact=replace(compact, targets=targets),
    )
    solution = solve_patterns(
        replicate_table,
        post.precision,
        post.mode,
        decrement_scale=max(1.0, float(int(targets.sum()))),
    )
    return solution.mode


def _support_maps(
    posterior: Posterior,
    frame_keys: tuple[ClusterKey, ...],
) -> tuple[dict[str, frozenset[int]], dict[str, frozenset[int]]]:
    compact = posterior.patterns.compact
    if compact is None:
        raise ValueError("cluster bootstrap requires compact pattern state")
    frame_index = {key: index for index, key in enumerate(frame_keys)}
    cluster_to_frame = np.array([frame_index[key] for key in compact.cluster_keys], dtype=np.int64)
    frame_count = len(frame_keys)
    frames = cluster_to_frame[compact.cluster_indices.astype(np.int64)]
    patterns = compact.cluster_patterns.astype(np.int64)
    agents = posterior.parameter_index.supported_agents
    decks = posterior.parameter_index.supported_decks

    def entity_sets(entity_indices: np.ndarray, count: int) -> list[frozenset[int]]:
        flat = (entity_indices.astype(np.int64) * frame_count + frames[:, None]).ravel()
        unique = np.unique(flat)
        entity = unique // frame_count
        frame = unique % frame_count
        order = np.argsort(entity, kind="stable")
        entity = entity[order]
        frame = frame[order]
        boundaries = np.searchsorted(entity, np.arange(count + 1))
        return [
            frozenset(frame[boundaries[index] : boundaries[index + 1]].tolist())
            for index in range(count)
        ]

    agent_sets = entity_sets(compact.agent_indices[patterns], len(agents))
    deck_sets = entity_sets(compact.deck_indices[patterns], len(decks))
    return (
        {agent: agent_sets[index] for index, agent in enumerate(agents)},
        {deck: deck_sets[index] for index, deck in enumerate(decks)},
    )


def _stratum_groups(
    posterior: Posterior, strata: str | None, frame_keys: tuple[ClusterKey, ...]
) -> list[list[int]] | None:
    if strata is None:
        return None
    labels: Mapping[ClusterKey, str | None]
    if strata in {"suite", "engine_version"}:
        crossing = posterior.crossing_keys(strata)
        if crossing:
            raise ValueError(f"stratum {strata!r} crosses clusters {crossing!r}")
        column = 0 if strata == "suite" else 1
        labels = {key: posterior.cluster_table[key][column] for key in frame_keys}
    elif strata in posterior.cluster_strata:
        labels = posterior.cluster_strata[strata]
    else:
        raise ValueError(f"unknown cluster stratum column {strata!r}")
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(frame_keys):
        label = labels[key]
        if label is None:
            raise ValueError(f"stratum {strata!r} has no label for {key!r}")
        groups.setdefault(str(label), []).append(index)
    return [groups[label] for label in sorted(groups)]


@dataclass(frozen=True, slots=True)
class _WorkerState:
    patterns: PatternTable
    precision: np.ndarray
    mode: np.ndarray
    seed: int
    frame_size: int
    groups: list[list[int]] | None
    target_flat: np.ndarray
    target_size: int


_WORKER_STATE: _WorkerState | None = None


def _make_worker_state(
    post: Posterior,
    seed: int,
    frame_size: int,
    groups: list[list[int]] | None,
) -> _WorkerState:
    index = post.parameter_index
    compact = post.patterns.compact
    if compact is None:
        raise ValueError("cluster bootstrap requires compact pattern state")
    picklable_index = ParameterIndex(
        names=index.names,
        positions=dict(index.positions),
        supported_agents=index.supported_agents,
        supported_decks=index.supported_decks,
        supported_brackets=index.supported_brackets,
        has_eta=index.has_eta,
        block_slices=dict(index.block_slices),
    )
    return _WorkerState(
        patterns=PatternTable(
            index=picklable_index,
            compact=compact,
            patterns=(),
            cluster_counts=dict(post.patterns.cluster_counts),
            n_games=post.patterns.n_games,
        ),
        precision=post.precision,
        mode=post.mode,
        seed=seed,
        frame_size=frame_size,
        groups=groups,
        target_flat=_target_flat(compact),
        target_size=compact.n_patterns * 4,
    )


def _sample_weights(state: _WorkerState, replicate: int) -> np.ndarray:
    rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(state.seed, spawn_key=(replicate,)))
    )
    weights = np.zeros(state.frame_size, dtype=np.int64)
    if state.groups is None:
        return np.bincount(
            rng.integers(0, state.frame_size, size=state.frame_size),
            minlength=state.frame_size,
        ).astype(np.int64)
    for group in state.groups:
        indices = np.asarray(group, dtype=np.int64)
        draws = rng.integers(0, len(indices), size=len(indices))
        weights += np.bincount(indices[draws], minlength=state.frame_size)
    return weights


def _run_replicate(state: _WorkerState, replicate: int) -> tuple[np.ndarray, np.ndarray]:
    weights = _sample_weights(state, replicate)
    compact = state.patterns.compact
    if compact is None:
        raise ValueError("cluster bootstrap requires compact pattern state")
    targets = _replicate_targets_from_flat(compact, weights, state.target_flat, state.target_size)
    replicate_table = replace(
        state.patterns,
        compact=replace(compact, targets=targets),
    )
    solution = solve_patterns(
        replicate_table,
        state.precision,
        state.mode,
        decrement_scale=max(1.0, float(int(targets.sum()))),
    )
    return weights, solution.mode


def _init_worker(state: _WorkerState) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state


def _process_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context()


def _worker_task(replicate: int) -> tuple[np.ndarray, np.ndarray | None]:
    if _WORKER_STATE is None:  # pragma: no cover - pool initializer invariant
        raise RuntimeError("bootstrap worker was not initialised")
    try:
        return _run_replicate(_WORKER_STATE, replicate)
    except ConvergenceError:
        weights = _sample_weights(_WORKER_STATE, replicate)
        return weights, None


def cluster_bootstrap(
    post: Posterior,
    n: int = 400,
    *,
    seed: int,
    strata: str | None = None,
    min_clusters: int = 50,
    min_entity_clusters: int = 10,
    workers: int = 1,
) -> BootstrapResult:
    """B1–B6 cluster bootstrap over likelihood-bearing clusters."""

    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if min_clusters <= 0 or min_entity_clusters <= 0:
        raise ValueError("cluster thresholds must be positive")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    compact = post.patterns.compact
    if compact is None:
        raise ValueError("cluster bootstrap requires compact pattern state")
    frame_keys = tuple(sorted(post.cluster_table))
    if len(frame_keys) != len(compact.cluster_keys) or set(frame_keys) != set(compact.cluster_keys):
        raise ValueError("posterior cluster frame does not match compact state")
    groups = _stratum_groups(post, strata, frame_keys)
    agent_clusters, deck_clusters = _support_maps(post, frame_keys)
    frame_size = len(frame_keys)
    state = _make_worker_state(post, seed, frame_size, groups)
    if workers == 1:
        results: list[tuple[np.ndarray, np.ndarray | None]] = []
        for replicate in range(n):
            try:
                results.append(_run_replicate(state, replicate))
            except ConvergenceError:
                results.append((_sample_weights(state, replicate), None))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_process_context(),
            initializer=_init_worker,
            initargs=(state,),
        ) as executor:
            results = list(executor.map(_worker_task, range(n)))
    weights_list: list[np.ndarray] = []
    modes_list: list[np.ndarray] = []
    failed = 0
    for weights, mode in results:
        if mode is None:
            failed += 1
            continue
        weights_list.append(weights)
        modes_list.append(mode)
    modes = (
        np.stack(modes_list)
        if modes_list
        else np.empty((0, len(post.parameter_index)), dtype=np.float64)
    )
    weights_array = (
        np.stack(weights_list) if weights_list else np.empty((0, frame_size), dtype=np.int64)
    )
    return BootstrapResult(
        posterior=post,
        n=n,
        seed=seed,
        strata=strata,
        replicate_weights=weights_array,
        replicate_modes=modes,
        failed_fraction=failed / n,
        min_clusters=min_clusters,
        min_entity_clusters=min_entity_clusters,
        frame_keys=frame_keys,
        agent_clusters=agent_clusters,
        deck_clusters=deck_clusters,
    )


__all__ = [
    "BootstrapResult",
    "DiagnosticKey",
    "QuantityDiagnostics",
    "cluster_bootstrap",
]
