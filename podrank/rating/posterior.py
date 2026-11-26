"""R1 posterior result and reporting methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
from scipy.linalg import solve_triangular

from podrank._version import __version__
from podrank.data.inclusion import Inclusion
from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.errors import (
    PersistenceError,
    SwapsNotModelledError,
    UnknownEntityError,
    UnsupportedFieldError,
    UnsupportedPredictionError,
)
from podrank.rating._archive import read_posterior_archive, write_posterior_archive
from podrank.rating.design import (
    CompactPatterns,
    ParameterIndex,
    Pattern,
    PatternTable,
    design_row,
)
from podrank.rating.field import Field, FieldContext
from podrank.rating.identification import (
    AgentContrastRow,
    AnchorUnsupported,
    ComponentsReport,
    ContrastContext,
    ContrastResult,
    DeckContrastRow,
    DeckNuisanceReport,
    EffectNotModelled,
    EligibilityRow,
    IdentificationReport,
    Interval,
    SeatContrastRow,
    UnrankableReport,
    agent_vector,
    aliasing_residual,
    beta_vector,
    build_identification_report,
    evaluate_contrast,
    seat_vector,
)
from podrank.rating.predict import PredictionInput, PredictionRow, coerce_prediction_input
from podrank.rating.spec import ModelSpec
from podrank.types import ClusterKey, CountsReport, EntityCounts, GameRow, OutcomeCounts

SdwpMethod = Literal["map", "laplace"]


@dataclass(frozen=True, slots=True)
class Provenance:
    package_version: str
    data_hash: str
    agent_snapshot_id: str
    deck_snapshot_id: str
    field_hash: str | None = None


@dataclass(frozen=True, slots=True)
class OutcomeRateRow:
    agent_id: str
    suite: str
    counts: OutcomeCounts
    draw_rate: float
    cap_rate: float
    failed_rate: float


@dataclass(frozen=True, slots=True)
class OutcomeRatesReport:
    rows: tuple[OutcomeRateRow, ...]


@dataclass(frozen=True, slots=True)
class SdwpRow:
    agent_id: str
    sdwp: float
    contrast_status: str
    field_label: str
    failing_contexts: tuple[int, ...]
    labels: tuple[str, ...]
    interval: Interval | None
    sdwp_posterior_mean: float | None = None


@dataclass(frozen=True, slots=True)
class SdwpReport:
    field_id: str
    version: int
    field_hash: str
    context_version_fingerprint: str
    method: SdwpMethod
    provenance: Provenance
    rows: tuple[SdwpRow, ...]


def _probabilities(utilities: np.ndarray) -> np.ndarray:
    maximum = float(np.max(utilities))
    shifted = utilities - maximum
    weights = np.exp(shifted)
    return np.asarray(weights / float(np.sum(weights)), dtype=np.float64)


def _probabilities_rows(utilities: np.ndarray) -> np.ndarray:
    maximum = np.max(utilities, axis=1, keepdims=True)
    shifted = utilities - maximum
    weights = np.exp(shifted)
    return np.asarray(weights / np.sum(weights, axis=1, keepdims=True), dtype=np.float64)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def build_outcome_rates(rows: Sequence[GameRow]) -> OutcomeRatesReport:
    """Aggregate integer outcome counts once by agent and suite."""

    counts: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        for agent_id in set(row.agents):
            bucket = counts.setdefault((agent_id, row.suite), [0, 0, 0, 0])
            if row.outcome == "win":
                bucket[0] += 1
            elif row.outcome == "draw":
                bucket[1] += 1
            elif row.outcome == "cap":
                bucket[2] += 1
            else:
                bucket[3] += 1
    result: list[OutcomeRateRow] = []
    for (agent_id, suite), values in sorted(counts.items()):
        outcome_counts = OutcomeCounts(
            win=values[0], draw=values[1], cap=values[2], failed=values[3]
        )
        total = outcome_counts.total
        result.append(
            OutcomeRateRow(
                agent_id=agent_id,
                suite=suite,
                counts=outcome_counts,
                draw_rate=_rate(outcome_counts.draw, total),
                cap_rate=_rate(outcome_counts.cap, total),
                failed_rate=_rate(outcome_counts.failed, total),
            )
        )
    return OutcomeRatesReport(rows=tuple(result))


@dataclass(frozen=True, slots=True, eq=False)
class Posterior:
    """Fitted R1 posterior and its reporting surface."""

    mode: np.ndarray
    lower_factor: np.ndarray
    parameter_index: ParameterIndex
    spec: ModelSpec
    inclusion: Inclusion
    _counts: CountsReport
    _agents: AgentRegistry
    _decks: DeckRegistry
    data_hash: str
    patterns: PatternTable
    precision: np.ndarray
    row_space: np.ndarray
    null_space: np.ndarray
    data_hessian: np.ndarray
    _components: ComponentsReport | AnchorUnsupported
    component_of: Mapping[str, str]
    _eligibility: tuple[EligibilityRow, ...] | AnchorUnsupported
    pooled_variants: bool
    _provenance: Provenance
    _outcome_rates: OutcomeRatesReport
    cluster_table: Mapping[ClusterKey, tuple[str | None, str | None]]
    cluster_strata: Mapping[str, Mapping[ClusterKey, str]]
    stratum_crossings: Mapping[str, tuple[int, ...]]

    @property
    def _labels(self) -> tuple[str, ...]:
        return ("pooled_variants",) if self.pooled_variants else ()

    def counts(self) -> CountsReport:
        return self._counts

    def components(self) -> ComponentsReport | AnchorUnsupported:
        return self._components

    def headline_eligibility(self) -> tuple[EligibilityRow, ...] | AnchorUnsupported:
        return self._eligibility

    def provenance(self) -> Provenance:
        return self._provenance

    def _context(self) -> ContrastContext:
        return ContrastContext(
            mode=self.mode,
            lower_factor=self.lower_factor,
            precision=self.precision,
            row_space=self.row_space,
            index=self.parameter_index,
            spec=self.spec,
        )

    def log_worth(
        self, *, include_withheld: bool = False
    ) -> tuple[AgentContrastRow, ...] | AnchorUnsupported:
        if isinstance(self._components, AnchorUnsupported):
            return self._components
        anchor = self._components.anchor_id
        anchor_component = set(self._components.anchor_component)
        context = self._context()
        rows: list[AgentContrastRow] = []
        for agent_id in sorted(self._agents.by_id):
            if f"theta[{agent_id}]" not in self.parameter_index.positions:
                rows.append(
                    AgentContrastRow(
                        agent_id=agent_id,
                        status="unrankable",
                        value=None,
                        data_share=None,
                        aliasing_residual=None,
                        prior_dependent_value=None,
                        interval=None,
                        labels=self._labels,
                        reasons=("unrankable",),
                    )
                )
                continue
            if agent_id == anchor:
                result = evaluate_contrast(
                    np.zeros(len(self.parameter_index)),
                    context,
                    labels=self._labels,
                )
                rows.append(self._agent_row(agent_id, result))
                continue
            if agent_id not in anchor_component:
                vector = agent_vector(self.parameter_index, agent_id) - agent_vector(
                    self.parameter_index, anchor
                )
                rows.append(
                    AgentContrastRow(
                        agent_id=agent_id,
                        status="disconnected",
                        value=None,
                        data_share=None,
                        aliasing_residual=None,
                        prior_dependent_value=(
                            float(np.dot(vector, self.mode)) if include_withheld else None
                        ),
                        interval=None,
                        labels=self._labels,
                        reasons=("disconnected",),
                    )
                )
                continue
            vector = agent_vector(self.parameter_index, agent_id) - agent_vector(
                self.parameter_index, anchor
            )
            result = evaluate_contrast(
                vector,
                context,
                labels=self._labels,
                include_withheld=include_withheld,
            )
            rows.append(self._agent_row(agent_id, result))
        return tuple(rows)

    @staticmethod
    def _agent_row(agent_id: str, result: ContrastResult) -> AgentContrastRow:
        return AgentContrastRow(
            agent_id=agent_id,
            status=result.status,
            value=result.value,
            data_share=result.data_share,
            aliasing_residual=result.aliasing_residual,
            prior_dependent_value=result.prior_dependent_value,
            interval=result.interval,
            labels=result.labels,
            reasons=result.reasons,
        )

    def contrast(self, a: str, b: str, *, include_withheld: bool = False) -> ContrastResult:
        if a not in self._agents.by_id:
            raise UnknownEntityError(f"unknown agent {a!r}")
        if b not in self._agents.by_id:
            raise UnknownEntityError(f"unknown agent {b!r}")
        supported = set(self.parameter_index.supported_agents)
        context = self._context()
        if a not in supported or b not in supported:
            return ContrastResult(
                status="unrankable",
                value=None,
                data_share=None,
                aliasing_residual=None,
                prior_dependent_value=None,
                interval=None,
                labels=self._labels,
                reasons=("unrankable",),
            )
        if a == b:
            return evaluate_contrast(
                np.zeros(len(self.parameter_index)), context, labels=self._labels
            )
        if self.component_of.get(a) != self.component_of.get(b):
            vector = agent_vector(self.parameter_index, a) - agent_vector(self.parameter_index, b)
            return ContrastResult(
                status="disconnected",
                value=None,
                data_share=None,
                aliasing_residual=None,
                prior_dependent_value=(
                    float(np.dot(vector, self.mode)) if include_withheld else None
                ),
                interval=None,
                labels=self._labels,
                reasons=("disconnected",),
            )
        vector = agent_vector(self.parameter_index, a) - agent_vector(self.parameter_index, b)
        return evaluate_contrast(
            vector,
            context,
            labels=self._labels,
            include_withheld=include_withheld,
        )

    def contrast_vector(self, vector: np.ndarray) -> ContrastResult:
        return evaluate_contrast(
            np.asarray(vector, dtype=np.float64), self._context(), labels=self._labels
        )

    def seats(self) -> tuple[SeatContrastRow, ...]:
        context = self._context()
        mean = (
            sum(
                (seat_vector(self.parameter_index, seat) for seat in range(4)),
                np.zeros(len(self.parameter_index)),
            )
            / 4.0
        )
        rows: list[SeatContrastRow] = []
        for seat in range(4):
            result = evaluate_contrast(
                seat_vector(self.parameter_index, seat) - mean,
                context,
                labels=self._labels,
            )
            rows.append(
                SeatContrastRow(
                    seat=seat,
                    status=result.status,
                    value=result.value,
                    data_share=result.data_share,
                    aliasing_residual=result.aliasing_residual,
                    prior_dependent_value=result.prior_dependent_value,
                    interval=result.interval,
                    labels=result.labels,
                    reasons=result.reasons,
                )
            )
        return tuple(rows)

    def decks(self) -> DeckNuisanceReport:
        reference = tuple(
            deck_id
            for deck_id in self.parameter_index.supported_decks
            if self._decks.by_id[deck_id].split == "train"
        )
        if not reference:
            return DeckNuisanceReport(baseline_status="unavailable", reference_deck_ids=(), rows=())
        context = self._context()
        mean_reference = sum(
            (beta_vector(self.parameter_index, deck_id, self._decks) for deck_id in reference),
            np.zeros(len(self.parameter_index)),
        ) / len(reference)
        rows: list[DeckContrastRow] = []
        for deck_id in self.parameter_index.supported_decks:
            vector = beta_vector(self.parameter_index, deck_id, self._decks) - mean_reference
            result = evaluate_contrast(
                vector,
                context,
                labels=(*self._labels, "prior_dependent_split"),
            )
            rows.append(
                DeckContrastRow(
                    deck_id=deck_id,
                    status=result.status,
                    value=result.value,
                    data_share=result.data_share,
                    aliasing_residual=result.aliasing_residual,
                    prior_dependent_value=result.prior_dependent_value,
                    interval=result.interval,
                    labels=result.labels,
                    reasons=result.reasons,
                )
            )
        return DeckNuisanceReport(
            baseline_status="available",
            reference_deck_ids=reference,
            rows=tuple(rows),
        )

    def swap_effect(self) -> ContrastResult | EffectNotModelled:
        if not self.spec.swap_effect:
            return EffectNotModelled()
        vector = np.zeros(len(self.parameter_index))
        vector[self.parameter_index.index("eta")] = 1.0
        return evaluate_contrast(vector, self._context(), labels=self._labels)

    def identification_report(self) -> IdentificationReport:
        return build_identification_report(
            self.patterns,
            self._context(),
            self._components,
            self._decks,
            self.data_hessian,
        )

    def unrankable(self) -> UnrankableReport:
        supported_agents = set(self.parameter_index.supported_agents)
        supported_decks = set(self.parameter_index.supported_decks)
        return UnrankableReport(
            agents=tuple(sorted(set(self._agents.by_id) - supported_agents)),
            decks=tuple(sorted(set(self._decks.by_id) - supported_decks)),
        )

    def outcome_rates(self) -> OutcomeRatesReport:
        return self._outcome_rates

    def crossing_keys(self, column: str) -> tuple[ClusterKey, ...]:
        """Resolve a stratum column's crossing frame indices to tagged keys."""

        if column not in {"suite", "engine_version"} and column not in self.cluster_strata:
            raise ValueError(f"unknown cluster stratum column {column!r}")
        frame = tuple(sorted(self.cluster_table))
        indices = self.stratum_crossings.get(column, ())
        try:
            return tuple(frame[index] for index in indices)
        except IndexError as exc:
            raise PersistenceError(
                f"cluster stratum {column!r} has an out-of-range crossing index"
            ) from exc

    def sample(self, n: int, seed: int) -> np.ndarray:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        standard = rng.standard_normal((n, len(self.parameter_index)))
        solved = solve_triangular(
            self.lower_factor, standard.T, lower=True, trans="T", check_finite=True
        )
        return np.asarray(self.mode + solved.T, dtype=np.float64)

    def _context_design(self, context: FieldContext, focal_agent: str) -> np.ndarray:
        non_focal = [seat for seat in range(4) if seat != context.focal_seat]
        agents = [focal_agent] * 4
        for opponent, seat in zip(context.opponents, non_focal, strict=True):
            agents[seat] = opponent
        swaps = context.swap_counts if self.spec.swap_effect else (0, 0, 0, 0)
        return design_row(
            (agents[0], agents[1], agents[2], agents[3]),
            context.decks,
            swaps,
            self.parameter_index,
            self._decks,
        )

    def _sdwp_map(self, field: Field, agent_id: str) -> tuple[float, tuple[int, ...]]:
        total = 0.0
        failing: list[int] = []
        for index, context in enumerate(field.contexts):
            design = self._context_design(context, agent_id)
            utilities = design @ self.mode
            probability = float(_probabilities(utilities)[context.focal_seat])
            total += context.weight * probability
            focal_row = design[context.focal_seat]
            for seat in range(4):
                if seat == context.focal_seat:
                    continue
                residual = aliasing_residual(design[seat] - focal_row, self.row_space)
                if residual is not None and residual > self.spec.tol_alias:
                    failing.append(index)
                    break
        return total, tuple(failing)

    def _sdwp_compact(
        self, field: Field, agent_ids: Sequence[str]
    ) -> list[tuple[str, float, tuple[int, ...]]]:
        compact = self.patterns.compact
        if compact is None:
            raise ValueError("compact SDWP requires a compact pattern table")
        deck_lookup = {
            deck_id: position
            for position, deck_id in enumerate(self.parameter_index.supported_decks)
        }
        agent_lookup = {
            agent_id: position
            for position, agent_id in enumerate(self.parameter_index.supported_agents)
        }
        context_count = len(field.contexts)
        focal_seats = np.array([context.focal_seat for context in field.contexts], dtype=np.int64)
        deck_indices = np.array(
            [[deck_lookup[deck_id] for deck_id in context.decks] for context in field.contexts],
            dtype=np.int64,
        )
        opponent_indices = np.array(
            [
                [agent_lookup[opponent] for opponent in context.opponents]
                for context in field.contexts
            ],
            dtype=np.int64,
        )
        weights = np.array([context.weight for context in field.contexts], dtype=np.float64)
        swaps = (
            np.array([context.swap_counts for context in field.contexts], dtype=np.int64)
            if self.spec.swap_effect
            else None
        )
        all_seats = np.arange(4, dtype=np.int64)
        non_focal_seats = np.empty((context_count, 3), dtype=np.int64)
        for row in range(context_count):
            non_focal_seats[row] = all_seats[all_seats != focal_seats[row]]
        row_indices = np.arange(context_count, dtype=np.int64)
        theta_start = self.parameter_index.block_slices["theta"][0]
        eta_position = compact.eta_position
        base_projections: list[np.ndarray] = []
        if self.null_space.shape[1] > 0:
            for offset in range(3):
                seats = non_focal_seats[:, offset]
                base_positions = np.column_stack(
                    [
                        compact.deck_mu_positions[deck_indices[row_indices, seats]],
                        compact.deck_mu_positions[deck_indices[row_indices, focal_seats]],
                        compact.deck_delta_positions[deck_indices[row_indices, seats]],
                        compact.deck_delta_positions[deck_indices[row_indices, focal_seats]],
                        compact.seat_positions[seats],
                        compact.seat_positions[focal_seats],
                    ]
                ).astype(np.int64)
                base_coefficients = np.column_stack(
                    [
                        np.ones(context_count),
                        -np.ones(context_count),
                        np.ones(context_count),
                        -np.ones(context_count),
                        np.ones(context_count),
                        -np.ones(context_count),
                    ]
                )
                if swaps is not None and eta_position is not None:
                    base_positions = np.column_stack(
                        [base_positions, np.full(context_count, eta_position, dtype=np.int64)]
                    )
                    base_coefficients = np.column_stack(
                        [
                            base_coefficients,
                            (swaps[row_indices, seats] - swaps[row_indices, focal_seats]).astype(
                                np.float64
                            ),
                        ]
                    )
                selected = self.null_space[base_positions]
                base_projections.append(
                    np.einsum("bkr,bk->br", selected, base_coefficients, optimize=True)
                )

        output: list[tuple[str, float, tuple[int, ...]]] = []
        for agent_id in agent_ids:
            agent_index = agent_lookup[agent_id]
            agent_indices = np.empty((context_count, 4), dtype=np.int64)
            agent_indices[row_indices, focal_seats] = agent_index
            agent_indices[row_indices[:, None], non_focal_seats] = opponent_indices
            utilities = (
                self.mode[theta_start + agent_indices]
                + self.mode[compact.deck_mu_positions[deck_indices]]
                + self.mode[compact.deck_delta_positions[deck_indices]]
                + self.mode[compact.seat_positions]
            )
            if swaps is not None and eta_position is not None:
                utilities = utilities + self.mode[eta_position] * swaps
            probabilities = _probabilities_rows(utilities)
            total = float(np.dot(weights, probabilities[row_indices, focal_seats]))
            failing_flags = np.zeros(context_count, dtype=bool)
            if self.null_space.shape[1] > 0:
                for offset in range(3):
                    seats = non_focal_seats[:, offset]
                    theta_opponent = self.null_space[
                        theta_start + agent_indices[row_indices, seats]
                    ]
                    theta_focal = self.null_space[
                        theta_start + agent_indices[row_indices, focal_seats]
                    ]
                    projected = base_projections[offset] + theta_opponent - theta_focal
                    residual = np.sqrt(np.sum(projected**2, axis=1))
                    failing_flags[np.flatnonzero(residual > self.spec.tol_alias)] = True
            output.append((agent_id, total, tuple(np.flatnonzero(failing_flags).tolist())))
        return output

    def _sdwp_for_draw(self, field: Field, agent_id: str, psi: np.ndarray) -> float:
        total = 0.0
        for context in field.contexts:
            design = self._context_design(context, agent_id)
            total += context.weight * float(_probabilities(design @ psi)[context.focal_seat])
        return total

    def _sdwp_matrix(self, field: Field, agent_ids: Sequence[str], modes: np.ndarray) -> np.ndarray:
        """Return SDWP for every ``(mode, agent)`` pair on the compact path.

        The result has shape ``(len(modes), len(agent_ids))``. It is the batched
        equivalent of calling ``_sdwp_for_draw`` for each mode and agent.
        """

        compact = self.patterns.compact
        if compact is None:
            raise ValueError("compact SDWP requires a compact pattern table")
        modes = np.asarray(modes, dtype=np.float64)
        if modes.ndim != 2 or modes.shape[1] != len(self.parameter_index):
            raise ValueError("modes have the wrong shape")
        if not agent_ids:
            return np.zeros((modes.shape[0], 0), dtype=np.float64)
        deck_lookup = {
            deck_id: position
            for position, deck_id in enumerate(self.parameter_index.supported_decks)
        }
        agent_lookup = {
            agent_id: position
            for position, agent_id in enumerate(self.parameter_index.supported_agents)
        }
        context_count = len(field.contexts)
        focal_seats = np.array([context.focal_seat for context in field.contexts], dtype=np.int64)
        deck_indices = np.array(
            [[deck_lookup[deck_id] for deck_id in context.decks] for context in field.contexts],
            dtype=np.int64,
        )
        opponent_indices = np.array(
            [
                [agent_lookup[opponent] for opponent in context.opponents]
                for context in field.contexts
            ],
            dtype=np.int64,
        )
        weights = np.array([context.weight for context in field.contexts], dtype=np.float64)
        swaps = (
            np.array([context.swap_counts for context in field.contexts], dtype=np.int64)
            if self.spec.swap_effect
            else None
        )
        all_seats = np.arange(4, dtype=np.int64)
        non_focal_seats = np.empty((context_count, 3), dtype=np.int64)
        for row in range(context_count):
            non_focal_seats[row] = all_seats[all_seats != focal_seats[row]]
        row_indices = np.arange(context_count, dtype=np.int64)
        theta_start = self.parameter_index.block_slices["theta"][0]
        eta_position = compact.eta_position
        seat_effects = modes[:, compact.seat_positions]
        mu = modes[:, compact.deck_mu_positions[deck_indices]]
        delta = modes[:, compact.deck_delta_positions[deck_indices]]
        output = np.empty((modes.shape[0], len(agent_ids)), dtype=np.float64)
        for column, agent_id in enumerate(agent_ids):
            try:
                agent_index = agent_lookup[agent_id]
            except KeyError as error:
                raise UnknownEntityError(f"unknown agent {agent_id!r}") from error
            agent_indices = np.empty((context_count, 4), dtype=np.int64)
            agent_indices[row_indices, focal_seats] = agent_index
            agent_indices[row_indices[:, None], non_focal_seats] = opponent_indices
            theta = modes[:, theta_start + agent_indices]
            utilities = theta + mu + delta + seat_effects[:, None, :]
            if swaps is not None and eta_position is not None:
                utilities = utilities + modes[:, eta_position][:, None, None] * swaps[None, :, :]
            probabilities = _probabilities_rows(utilities.reshape(-1, 4)).reshape(
                modes.shape[0], context_count, 4
            )
            output[:, column] = np.dot(probabilities[:, row_indices, focal_seats], weights)
        return output

    def sdwp(
        self,
        field: Field,
        *,
        method: SdwpMethod = "map",
        n_draws: int = 1000,
        seed: int | None = None,
    ) -> SdwpReport | AnchorUnsupported:
        if method not in {"map", "laplace"}:
            raise ValueError("method must be 'map' or 'laplace'")
        if isinstance(n_draws, bool) or not isinstance(n_draws, int) or n_draws <= 0:
            raise ValueError("n_draws must be a positive integer")
        if method == "laplace" and seed is None:
            raise ValueError("method='laplace' requires an explicit seed")
        if isinstance(self._components, AnchorUnsupported):
            return self._components

        supported_decks = set(self.parameter_index.supported_decks)
        supported_agents = set(self.parameter_index.supported_agents)
        anchor_component = set(self._components.anchor_component)
        for context in field.contexts:
            for deck_id in context.decks:
                if deck_id not in supported_decks:
                    raise UnsupportedFieldError(f"deck {deck_id!r} has no likelihood-used games")
            for opponent in context.opponents:
                if opponent not in supported_agents or opponent not in anchor_component:
                    raise UnsupportedFieldError(
                        f"opponent {opponent!r} is not rankable in the anchor component"
                    )
            if not self.spec.swap_effect and any(context.swap_counts):
                raise SwapsNotModelledError(
                    "field context has nonzero swap counts while the swap effect is off"
                )

        fingerprint = field.context_version_fingerprint(self._agents, self._decks)
        eligibility = self._eligibility
        if isinstance(eligibility, AnchorUnsupported):
            return eligibility
        eligible = [row for row in eligibility if row.eligible]
        if self.patterns.compact is not None:
            preliminary = self._sdwp_compact(field, [row.agent_id for row in eligible])
        else:
            preliminary = []
            for row in eligible:
                value, failing = self._sdwp_map(field, row.agent_id)
                preliminary.append((row.agent_id, value, failing))
        preliminary.sort(key=lambda item: (-item[1], item[0]))

        statuses = {row.agent_id: row.contrast_status for row in eligible}
        rows: list[SdwpRow] = []
        if method == "map":
            for agent_id, value, failing in preliminary:
                label = "prior_dependent_field" if failing else "field_identified"
                rows.append(
                    SdwpRow(
                        agent_id=agent_id,
                        sdwp=value,
                        contrast_status=statuses[agent_id],
                        field_label=label,
                        failing_contexts=failing,
                        labels=self._labels,
                        interval=None,
                    )
                )
        else:
            assert seed is not None
            draws = self.sample(n_draws, seed)
            if self.patterns.compact is not None:
                draw_matrix = self._sdwp_matrix(field, [item[0] for item in preliminary], draws)
                draw_values = {
                    item[0]: draw_matrix[:, index] for index, item in enumerate(preliminary)
                }
            else:
                draw_values = {
                    agent_id: np.array(
                        [self._sdwp_for_draw(field, agent_id, draw) for draw in draws],
                        dtype=np.float64,
                    )
                    for agent_id, _, _ in preliminary
                }
            for agent_id, value, failing in preliminary:
                values = draw_values[agent_id]
                lo, hi = np.quantile(values, [0.025, 0.975], method="linear")
                label = "prior_dependent_field" if failing else "field_identified"
                rows.append(
                    SdwpRow(
                        agent_id=agent_id,
                        sdwp=value,
                        contrast_status=statuses[agent_id],
                        field_label=label,
                        failing_contexts=failing,
                        labels=self._labels,
                        interval=Interval(
                            lo=float(lo), hi=float(hi), method="laplace", labels=self._labels
                        ),
                        sdwp_posterior_mean=float(np.mean(values)),
                    )
                )
        return SdwpReport(
            field_id=field.field_id,
            version=field.version,
            field_hash=field.field_hash,
            context_version_fingerprint=fingerprint,
            method=method,
            provenance=self._provenance,
            rows=tuple(rows),
        )

    def _predict_proba_compact(
        self, inputs: Sequence[PredictionInput]
    ) -> tuple[PredictionRow, ...]:
        compact = self.patterns.compact
        if compact is None:
            raise ValueError("compact prediction requires a compact pattern table")
        index = self.parameter_index
        agent_lookup = {agent: position for position, agent in enumerate(index.supported_agents)}
        deck_lookup = {deck: position for position, deck in enumerate(index.supported_decks)}
        count = len(inputs)
        agent_indices = np.array(
            [[agent_lookup[agent] for agent in context.agents] for context in inputs],
            dtype=np.int64,
        )
        deck_indices = np.array(
            [[deck_lookup[deck] for deck in context.decks] for context in inputs],
            dtype=np.int64,
        )
        theta_start = index.block_slices["theta"][0]
        utilities = (
            self.mode[theta_start + agent_indices]
            + self.mode[compact.deck_mu_positions[deck_indices]]
            + self.mode[compact.deck_delta_positions[deck_indices]]
            + self.mode[compact.seat_positions]
        )
        swaps = None
        eta_position = compact.eta_position
        if self.spec.swap_effect:
            swaps = np.array([context.swap_counts for context in inputs], dtype=np.int64)
            assert eta_position is not None
            utilities = utilities + self.mode[eta_position] * swaps
        probabilities = _probabilities_rows(utilities)

        failing = np.zeros((count, 4), dtype=bool)
        if self.null_space.shape[1] > 0:
            null = self.null_space
            parameter_count = len(index)
            chunk_size = 20_000
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                rows = np.arange(stop - start)
                for seat in range(1, 4):
                    difference = np.zeros((stop - start, parameter_count), dtype=np.float64)
                    difference[rows, theta_start + agent_indices[start:stop, seat]] += 1.0
                    difference[rows, theta_start + agent_indices[start:stop, 0]] -= 1.0
                    difference[
                        rows,
                        compact.deck_mu_positions[deck_indices[start:stop, seat]],
                    ] += 1.0
                    difference[
                        rows,
                        compact.deck_mu_positions[deck_indices[start:stop, 0]],
                    ] -= 1.0
                    difference[
                        rows,
                        compact.deck_delta_positions[deck_indices[start:stop, seat]],
                    ] += 1.0
                    difference[
                        rows,
                        compact.deck_delta_positions[deck_indices[start:stop, 0]],
                    ] -= 1.0
                    difference[rows, compact.seat_positions[seat]] += 1.0
                    difference[rows, compact.seat_positions[0]] -= 1.0
                    if swaps is not None and eta_position is not None:
                        difference[rows, eta_position] += (
                            swaps[start:stop, seat] - swaps[start:stop, 0]
                        )
                    projected = difference @ null
                    norms = np.linalg.norm(difference, axis=1)
                    residuals = np.zeros(stop - start, dtype=np.float64)
                    np.divide(
                        np.linalg.norm(projected, axis=1),
                        norms,
                        out=residuals,
                        where=norms > 0,
                    )
                    failing[start:stop, seat] = residuals > self.spec.tol_alias

        rows_out: list[PredictionRow] = []
        for row_index in range(count):
            failing_seats = tuple(int(seat) for seat in np.flatnonzero(failing[row_index]))
            rows_out.append(
                PredictionRow(
                    probabilities=(
                        float(probabilities[row_index, 0]),
                        float(probabilities[row_index, 1]),
                        float(probabilities[row_index, 2]),
                        float(probabilities[row_index, 3]),
                    ),
                    field_label=("prior_dependent_field" if failing_seats else "field_identified"),
                    failing_seats=failing_seats,
                    labels=self._labels,
                )
            )
        return tuple(rows_out)

    def predict_proba(
        self, contexts: Sequence[PredictionInput | Mapping[str, object]]
    ) -> tuple[PredictionRow, ...]:
        inputs = [coerce_prediction_input(context) for context in contexts]
        supported_agents = set(self.parameter_index.supported_agents)
        supported_decks = set(self.parameter_index.supported_decks)
        for context in inputs:
            for agent_id in context.agents:
                if agent_id not in self._agents.by_id:
                    raise UnknownEntityError(f"unknown agent {agent_id!r}")
            for deck_id in context.decks:
                if deck_id not in self._decks.by_id:
                    raise UnknownEntityError(f"unknown deck {deck_id!r}")
            for agent_id in context.agents:
                if agent_id not in supported_agents:
                    raise UnsupportedPredictionError(
                        f"agent {agent_id!r} has no likelihood-used games"
                    )
            for deck_id in context.decks:
                if deck_id not in supported_decks:
                    raise UnsupportedPredictionError(
                        f"deck {deck_id!r} has no likelihood-used games"
                    )
            if (
                not self.spec.swap_effect
                and any(context.swap_counts)
                and not self.spec.allow_pooled_variants
            ):
                raise SwapsNotModelledError(
                    "prediction has nonzero swap counts while the swap effect is off"
                )

        if self.patterns.compact is not None and inputs:
            return self._predict_proba_compact(inputs)

        rows: list[PredictionRow] = []
        for context in inputs:
            swaps = context.swap_counts if self.spec.swap_effect else (0, 0, 0, 0)
            design = design_row(
                context.agents, context.decks, swaps, self.parameter_index, self._decks
            )
            probabilities = _probabilities(design @ self.mode)
            failing: list[int] = []
            for seat in range(1, 4):
                residual = aliasing_residual(design[seat] - design[0], self.row_space)
                if residual is not None and residual > self.spec.tol_alias:
                    failing.append(seat)
            rows.append(
                PredictionRow(
                    probabilities=(
                        float(probabilities[0]),
                        float(probabilities[1]),
                        float(probabilities[2]),
                        float(probabilities[3]),
                    ),
                    field_label="prior_dependent_field" if failing else "field_identified",
                    failing_seats=tuple(failing),
                    labels=self._labels,
                )
            )
        return tuple(rows)

    def save(self, path: str | Path) -> None:
        payload = {
            "agents": [
                {
                    "agent_id": record.agent_id,
                    "kind": record.kind,
                    "policy_version": record.policy_version,
                    "is_anchor": record.is_anchor,
                    "tags": list(record.tags),
                    "metadata": dict(record.metadata),
                }
                for record in self._agents
            ],
            "decks": [
                {
                    "deck_id": record.deck_id,
                    "bracket": record.bracket,
                    "split": record.split,
                    "deck_version": record.deck_version,
                    "tags": dict(record.tags),
                    "metadata": dict(record.metadata),
                }
                for record in self._decks
            ],
            "spec": {
                "swap_effect": self.spec.swap_effect,
                "allow_pooled_variants": self.spec.allow_pooled_variants,
                "weak_ratio": self.spec.weak_ratio,
                "tol_eig": self.spec.tol_eig,
                "tol_alias": self.spec.tol_alias,
                "max_params": self.spec.max_params,
                "max_dense_bytes": self.spec.max_dense_bytes,
            },
            "inclusion": {
                "suites": None if self.inclusion.suites is None else sorted(self.inclusion.suites),
                "engine_versions": (
                    None
                    if self.inclusion.engine_versions is None
                    else sorted(self.inclusion.engine_versions)
                ),
                "splits": None if self.inclusion.splits is None else sorted(self.inclusion.splits),
                "agents": None if self.inclusion.agents is None else sorted(self.inclusion.agents),
                "played_at": [
                    None if value is None else value.isoformat()
                    for value in self.inclusion.played_at
                ],
                "clusters": self.inclusion.clusters,
                "manifest": (
                    None
                    if self.inclusion.manifest is None
                    else {
                        cluster_id: sorted(game_ids)
                        for cluster_id, game_ids in self.inclusion.manifest.items()
                    }
                ),
                "allow_mirror": self.inclusion.allow_mirror,
            },
            "index": {
                "names": list(self.parameter_index.names),
                "supported_agents": list(self.parameter_index.supported_agents),
                "supported_decks": list(self.parameter_index.supported_decks),
                "supported_brackets": list(self.parameter_index.supported_brackets),
                "has_eta": self.parameter_index.has_eta,
                "block_slices": {
                    name: list(bounds) for name, bounds in self.parameter_index.block_slices.items()
                },
            },
            "mode": self.mode.tolist(),
            "lower_factor": self.lower_factor.tolist(),
            "row_space": self.row_space.tolist(),
            "null_space": self.null_space.tolist(),
            "data_hessian": self.data_hessian.tolist(),
            "precision": self.precision.tolist(),
            "counts": {
                "received": self._counts.received,
                "exact_duplicates": self._counts.exact_duplicates,
                "eligible": self._counts.eligible,
                "excluded": dict(self._counts.excluded),
                "likelihood_used": self._counts.likelihood_used,
                "likelihood_bearing_clusters": self._counts.likelihood_bearing_clusters,
                "agents": {
                    agent_id: {
                        "games": values.games,
                        "seat_appearances": values.seat_appearances,
                    }
                    for agent_id, values in self._counts.agents.items()
                },
                "decks": {
                    deck_id: {
                        "games": values.games,
                        "seat_appearances": values.seat_appearances,
                    }
                    for deck_id, values in self._counts.decks.items()
                },
            },
            "components": _components_payload(self._components),
            "component_of": dict(self.component_of),
            "eligibility": _eligibility_payload(self._eligibility),
            "pooled_variants": self.pooled_variants,
            "provenance": {
                "package_version": self._provenance.package_version,
                "data_hash": self._provenance.data_hash,
                "agent_snapshot_id": self._provenance.agent_snapshot_id,
                "deck_snapshot_id": self._provenance.deck_snapshot_id,
                "field_hash": self._provenance.field_hash,
            },
            "data_hash": self.data_hash,
            "compact": None,
            "patterns": [],
            "cluster_counts": None,
            "n_games": self.patterns.n_games,
            "outcome_rates": [
                {
                    "agent_id": row.agent_id,
                    "suite": row.suite,
                    "win": row.counts.win,
                    "draw": row.counts.draw,
                    "cap": row.counts.cap,
                    "failed": row.counts.failed,
                }
                for row in self._outcome_rates.rows
            ],
            "cluster_strata_columns": sorted(self.cluster_strata),
            "stratum_crossings": [
                {"column": column, "indices": list(indices)}
                for column, indices in sorted(self.stratum_crossings.items())
            ],
        }
        arrays: dict[str, np.ndarray] = {
            "mode": np.asarray(self.mode, dtype=np.float64),
            "lower_factor": np.asarray(self.lower_factor, dtype=np.float64),
            "row_space": np.asarray(self.row_space, dtype=np.float64),
            "null_space": np.asarray(self.null_space, dtype=np.float64),
            "data_hessian": np.asarray(self.data_hessian, dtype=np.float64),
            "precision": np.asarray(self.precision, dtype=np.float64),
        }
        for name in (
            "mode",
            "lower_factor",
            "row_space",
            "null_space",
            "data_hessian",
            "precision",
        ):
            payload.pop(name)
        compact = self.patterns.compact
        if compact is not None:
            payload["compact"] = {
                "eta_position": compact.eta_position,
                "has_swap_counts": compact.swap_counts is not None,
            }
            arrays.update(
                {
                    "compact_agent_indices": compact.agent_indices,
                    "compact_deck_indices": compact.deck_indices,
                    "compact_targets": compact.targets,
                    "compact_deck_mu_positions": compact.deck_mu_positions,
                    "compact_deck_delta_positions": compact.deck_delta_positions,
                    "compact_seat_positions": compact.seat_positions,
                    "compact_reduction": compact.reduction,
                    "compact_cluster_patterns": compact.cluster_patterns,
                    "compact_cluster_winner_seats": compact.cluster_winner_seats,
                    "compact_cluster_indices": compact.cluster_indices,
                    "compact_cluster_counts": compact.cluster_counts,
                    "compact_cluster_key_tags": _string_bytes(
                        [key[0] for key in compact.cluster_keys]
                    ),
                    "compact_cluster_key_ids": _string_bytes(
                        [key[1] for key in compact.cluster_keys]
                    ),
                }
            )
            if compact.swap_counts is not None:
                arrays["compact_swap_counts"] = compact.swap_counts
        else:
            raise PersistenceError("posterior archive requires compact fitted pattern state")
        table_keys = sorted(self.cluster_table)
        arrays["cluster_table_suite"] = _string_bytes(
            [self.cluster_table[key][0] or "" for key in table_keys]
        )
        arrays["cluster_table_engine"] = _string_bytes(
            [self.cluster_table[key][1] or "" for key in table_keys]
        )
        arrays["cluster_table_suite_known"] = np.asarray(
            [self.cluster_table[key][0] is not None for key in table_keys],
            dtype=np.bool_,
        )
        arrays["cluster_table_engine_known"] = np.asarray(
            [self.cluster_table[key][1] is not None for key in table_keys],
            dtype=np.bool_,
        )
        for index, column in enumerate(sorted(self.cluster_strata)):
            values = self.cluster_strata[column]
            arrays[f"cluster_strata_{index}"] = _string_bytes([values[key] for key in table_keys])
        payload["package_version"] = __version__
        write_posterior_archive(path, payload, arrays)

    @classmethod
    def load(cls, path: str | Path) -> Posterior:
        try:
            return cls._load_unchecked(path)
        except PersistenceError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            raise PersistenceError(f"malformed posterior payload: {exc}") from exc

    @classmethod
    def _load_unchecked(cls, path: str | Path) -> Posterior:
        payload, arrays = read_posterior_archive(path)
        expected_arrays = _expected_array_names(payload)
        if set(arrays) != expected_arrays:
            extra = sorted(set(arrays) - expected_arrays)
            missing = sorted(expected_arrays - set(arrays))
            raise PersistenceError(
                f"posterior arrays do not match schema: extra={extra}, missing={missing}"
            )

        agents = AgentRegistry.from_records(_require_list(payload["agents"], "agents"))
        decks = DeckRegistry.from_records(_require_list(payload["decks"], "decks"))
        spec_payload = _require_mapping(payload["spec"], "spec")
        spec = ModelSpec(
            swap_effect=bool(spec_payload["swap_effect"]),
            allow_pooled_variants=bool(spec_payload["allow_pooled_variants"]),
            weak_ratio=float(spec_payload["weak_ratio"]),
            tol_eig=float(spec_payload["tol_eig"]),
            tol_alias=float(spec_payload["tol_alias"]),
            max_params=int(spec_payload["max_params"]),
            max_dense_bytes=int(spec_payload["max_dense_bytes"]),
        )
        inclusion_payload = _require_mapping(payload["inclusion"], "inclusion")
        played_raw = _require_list(inclusion_payload["played_at"], "played_at")
        if len(played_raw) != 2:
            raise PersistenceError("played_at must contain two entries")
        manifest_raw = inclusion_payload["manifest"]
        manifest = None
        if manifest_raw is not None:
            manifest_mapping = _require_mapping(manifest_raw, "manifest")
            manifest = {
                str(cluster_id): frozenset(
                    str(game_id) for game_id in _require_list(ids, "manifest ids")
                )
                for cluster_id, ids in manifest_mapping.items()
            }
        inclusion = Inclusion(
            suites=_optional_string_set(inclusion_payload["suites"]),
            engine_versions=_optional_string_set(inclusion_payload["engine_versions"]),
            splits=_optional_string_set(inclusion_payload["splits"]),
            agents=_optional_string_set(inclusion_payload["agents"]),
            played_at=cast(
                tuple[datetime | None, datetime | None],
                (
                    None if played_raw[0] is None else str(played_raw[0]),
                    None if played_raw[1] is None else str(played_raw[1]),
                ),
            ),
            clusters=inclusion_payload["clusters"],
            manifest=manifest,
            allow_mirror=bool(inclusion_payload["allow_mirror"]),
        )

        index_payload = _require_mapping(payload["index"], "index")
        names = tuple(str(name) for name in _require_list(index_payload["names"], "names"))
        positions = {name: index for index, name in enumerate(names)}
        block_slices = {
            str(name): (int(bounds[0]), int(bounds[1]))
            for name, bounds in _require_mapping(
                index_payload["block_slices"], "block_slices"
            ).items()
        }
        index = ParameterIndex(
            names=names,
            positions=MappingProxyType(positions),
            supported_agents=tuple(str(value) for value in index_payload["supported_agents"]),
            supported_decks=tuple(str(value) for value in index_payload["supported_decks"]),
            supported_brackets=tuple(int(value) for value in index_payload["supported_brackets"]),
            has_eta=bool(index_payload["has_eta"]),
            block_slices=MappingProxyType(block_slices),
        )

        compact = _compact_from_archive(payload["compact"], arrays)
        if compact is None:
            raise PersistenceError("posterior archive has no compact pattern state")
        patterns: tuple[Pattern, ...] = ()
        cluster_counts: dict[tuple[int, int, ClusterKey], int] = {}
        pattern_table = PatternTable(
            patterns=patterns,
            compact=compact,
            index=index,
            cluster_counts=MappingProxyType(cluster_counts),
            n_games=int(payload["n_games"]),
        )
        counts_payload = _require_mapping(payload["counts"], "counts")
        counts = CountsReport(
            received=int(counts_payload["received"]),
            exact_duplicates=int(counts_payload["exact_duplicates"]),
            eligible=int(counts_payload["eligible"]),
            excluded=MappingProxyType(
                {
                    str(key): int(value)
                    for key, value in _require_mapping(
                        counts_payload["excluded"], "excluded"
                    ).items()
                }
            ),
            likelihood_used=int(counts_payload["likelihood_used"]),
            likelihood_bearing_clusters=int(counts_payload["likelihood_bearing_clusters"]),
            agents=MappingProxyType(
                {
                    str(key): EntityCounts(
                        games=int(value["games"]),
                        seat_appearances=int(value["seat_appearances"]),
                    )
                    for key, value in _require_mapping(counts_payload["agents"], "agents").items()
                }
            ),
            decks=MappingProxyType(
                {
                    str(key): EntityCounts(
                        games=int(value["games"]),
                        seat_appearances=int(value["seat_appearances"]),
                    )
                    for key, value in _require_mapping(counts_payload["decks"], "decks").items()
                }
            ),
        )
        components = _components_from_payload(payload["components"])
        eligibility = _eligibility_from_payload(payload["eligibility"])
        provenance_payload = _require_mapping(payload["provenance"], "provenance")
        provenance = Provenance(
            package_version=str(provenance_payload["package_version"]),
            data_hash=str(provenance_payload["data_hash"]),
            agent_snapshot_id=str(provenance_payload["agent_snapshot_id"]),
            deck_snapshot_id=str(provenance_payload["deck_snapshot_id"]),
            field_hash=(
                None
                if provenance_payload["field_hash"] is None
                else str(provenance_payload["field_hash"])
            ),
        )
        outcome_rows: list[OutcomeRateRow] = []
        for raw in (
            _require_mapping(item, "outcome rate")
            for item in _require_list(payload["outcome_rates"], "outcome_rates")
        ):
            outcome_counts = OutcomeCounts(
                win=int(raw["win"]),
                draw=int(raw["draw"]),
                cap=int(raw["cap"]),
                failed=int(raw["failed"]),
            )
            total = outcome_counts.total
            outcome_rows.append(
                OutcomeRateRow(
                    agent_id=str(raw["agent_id"]),
                    suite=str(raw["suite"]),
                    counts=outcome_counts,
                    draw_rate=_rate(outcome_counts.draw, total),
                    cap_rate=_rate(outcome_counts.cap, total),
                    failed_rate=_rate(outcome_counts.failed, total),
                )
            )
        outcome_rates = OutcomeRatesReport(rows=tuple(outcome_rows))
        table_keys = compact.cluster_keys
        table_suites = _require_array(arrays, "cluster_table_suite")
        table_engines = _require_array(arrays, "cluster_table_engine")
        table_suite_known = _require_array(arrays, "cluster_table_suite_known")
        table_engine_known = _require_array(arrays, "cluster_table_engine_known")
        table_length = len(table_keys)
        for name, array in (
            ("cluster_table_suite", table_suites),
            ("cluster_table_engine", table_engines),
            ("cluster_table_suite_known", table_suite_known),
            ("cluster_table_engine_known", table_engine_known),
        ):
            if array.shape != (table_length,):
                raise PersistenceError(f"{name} has the wrong shape")
        suite_values = _decode_strings(table_suites)
        engine_values = _decode_strings(table_engines)
        cluster_table = {
            key: (
                suite_values[position] if bool(table_suite_known[position]) else None,
                engine_values[position] if bool(table_engine_known[position]) else None,
            )
            for position, key in enumerate(table_keys)
        }
        columns = _require_list(payload["cluster_strata_columns"], "cluster_strata_columns")
        retained_strata: dict[str, dict[ClusterKey, str]] = {}
        for position, column in enumerate(columns):
            if not isinstance(column, str) or not column:
                raise PersistenceError("cluster_strata column names must be non-empty strings")
            labels = _require_array(arrays, f"cluster_strata_{position}")
            if labels.shape != (table_length,):
                raise PersistenceError(f"cluster_strata_{position} has the wrong shape")
            retained_strata[column] = {
                key: label for key, label in zip(table_keys, _decode_strings(labels), strict=True)
            }
        stratum_crossings: dict[str, tuple[int, ...]] = {}
        for raw in (
            _require_mapping(item, "stratum crossing")
            for item in _require_list(payload["stratum_crossings"], "stratum_crossings")
        ):
            column = str(raw["column"])
            if column not in {"suite", "engine_version"} and column not in retained_strata:
                raise PersistenceError(f"unknown crossing stratum column {column!r}")
            indices = tuple(int(value) for value in _require_list(raw["indices"], "indices"))
            if any(index < 0 or index >= len(table_keys) for index in indices):
                raise PersistenceError(f"crossing indices for {column!r} are out of range")
            if tuple(sorted(set(indices))) != indices:
                raise PersistenceError(f"crossing indices for {column!r} are not unique")
            stratum_crossings[column] = indices
        _validate_compact_state(compact, index, cluster_table, int(payload["n_games"]))
        for column, values in retained_strata.items():
            if not {key for key in values} <= set(cluster_table):
                raise PersistenceError(
                    f"cluster_strata column {column!r} references non-frame keys"
                )
        return cls(
            mode=arrays["mode"],
            lower_factor=arrays["lower_factor"],
            parameter_index=index,
            spec=spec,
            inclusion=inclusion,
            _counts=counts,
            _agents=agents,
            _decks=decks,
            data_hash=str(payload["data_hash"]),
            patterns=pattern_table,
            precision=arrays["precision"],
            row_space=arrays["row_space"],
            null_space=arrays["null_space"],
            data_hessian=arrays["data_hessian"],
            _components=components,
            component_of=MappingProxyType(
                {
                    str(key): str(value)
                    for key, value in _require_mapping(
                        payload["component_of"], "component_of"
                    ).items()
                }
            ),
            _eligibility=eligibility,
            pooled_variants=bool(payload["pooled_variants"]),
            _provenance=provenance,
            _outcome_rates=outcome_rates,
            cluster_table=cluster_table,
            cluster_strata=retained_strata,
            stratum_crossings=stratum_crossings,
        )


def _validate_compact_state(
    compact: CompactPatterns | None,
    index: ParameterIndex,
    cluster_table: Mapping[ClusterKey, tuple[str | None, str | None]],
    n_games: int,
) -> None:
    if compact is None:
        return
    patterns = compact.n_patterns
    for name, array in (
        ("agent_indices", compact.agent_indices),
        ("deck_indices", compact.deck_indices),
        ("targets", compact.targets),
    ):
        if array.shape != (patterns, 4):
            raise PersistenceError(f"{name} has the wrong shape")
    if compact.agent_indices.size and (
        int(compact.agent_indices.min()) < 0
        or int(compact.agent_indices.max()) >= len(index.supported_agents)
    ):
        raise PersistenceError("agent_indices contains an out-of-range index")
    if compact.deck_indices.size and (
        int(compact.deck_indices.min()) < 0
        or int(compact.deck_indices.max()) >= len(index.supported_decks)
    ):
        raise PersistenceError("deck_indices contains an out-of-range index")
    if np.any(compact.targets < 0):
        raise PersistenceError("targets contains a negative count")
    if int(compact.targets.sum()) != n_games:
        raise PersistenceError("targets do not sum to n_games")
    if compact.swap_counts is not None:
        if compact.swap_counts.shape != (patterns, 4) or np.any(compact.swap_counts < 0):
            raise PersistenceError("swap_counts has the wrong shape or values")
    if compact.eta_position is None and compact.swap_counts is not None:
        raise PersistenceError("swap_counts are present while eta is off")
    if compact.reduction.shape != (len(index), index.reduced_size):
        raise PersistenceError("reduction matrix has the wrong shape")
    if len(compact.deck_mu_positions) != len(index.supported_decks):
        raise PersistenceError("deck_mu_positions has the wrong length")
    if len(compact.deck_delta_positions) != len(index.supported_decks):
        raise PersistenceError("deck_delta_positions has the wrong length")
    if compact.seat_positions.shape != (4,):
        raise PersistenceError("seat_positions has the wrong shape")
    cluster_count = len(compact.cluster_keys)
    if len(set(compact.cluster_keys)) != cluster_count:
        raise PersistenceError("cluster_keys contains duplicates")
    if len(compact.cluster_patterns) != len(compact.cluster_winner_seats):
        raise PersistenceError("cluster arrays have inconsistent lengths")
    if len(compact.cluster_patterns) != len(compact.cluster_indices):
        raise PersistenceError("cluster arrays have inconsistent lengths")
    if len(compact.cluster_patterns) != len(compact.cluster_counts):
        raise PersistenceError("cluster arrays have inconsistent lengths")
    if compact.cluster_patterns.size and (
        int(compact.cluster_patterns.min()) < 0 or int(compact.cluster_patterns.max()) >= patterns
    ):
        raise PersistenceError("cluster_patterns contains an out-of-range index")
    if compact.cluster_indices.size and (
        int(compact.cluster_indices.min()) < 0
        or int(compact.cluster_indices.max()) >= cluster_count
    ):
        raise PersistenceError("cluster_indices contains an out-of-range index")
    if compact.cluster_winner_seats.size and (
        int(compact.cluster_winner_seats.min()) < 0 or int(compact.cluster_winner_seats.max()) > 3
    ):
        raise PersistenceError("cluster_winner_seats contains an invalid seat")
    if np.any(compact.cluster_counts <= 0):
        raise PersistenceError("cluster_counts must be positive")
    if int(compact.cluster_counts.sum()) != n_games:
        raise PersistenceError("cluster counts do not sum to n_games")
    if tuple(sorted(cluster_table)) != tuple(compact.cluster_keys):
        raise PersistenceError("cluster_table does not match compact cluster keys")


def _string_bytes(values: list[str]) -> np.ndarray:
    return np.asarray([value.encode("utf-8") for value in values], dtype="S")


def _decode_strings(array: np.ndarray) -> list[str]:
    result: list[str] = []
    for raw in array.tolist():
        if isinstance(raw, bytes):
            try:
                result.append(raw.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise PersistenceError("archive string array is not valid UTF-8") from exc
        elif isinstance(raw, str):
            result.append(raw)
        else:
            raise PersistenceError("archive string array has a non-string value")
    return result


def _cluster_keys_from_arrays(
    arrays: Mapping[str, np.ndarray], tags_name: str, ids_name: str
) -> tuple[ClusterKey, ...]:
    tags = _require_array(arrays, tags_name)
    ids = _require_array(arrays, ids_name)
    if tags.shape != ids.shape or tags.ndim != 1:
        raise PersistenceError("cluster key arrays have inconsistent shapes")
    return tuple(
        cast(ClusterKey, (tag, identifier))
        for tag, identifier in zip(_decode_strings(tags), _decode_strings(ids), strict=True)
    )


def _expected_array_names(payload: Mapping[str, object]) -> set[str]:
    base = {
        "mode",
        "lower_factor",
        "row_space",
        "null_space",
        "data_hessian",
        "precision",
        "cluster_table_suite",
        "cluster_table_engine",
        "cluster_table_suite_known",
        "cluster_table_engine_known",
    }
    for index in range(
        len(_require_list(payload["cluster_strata_columns"], "cluster_strata_columns"))
    ):
        base.add(f"cluster_strata_{index}")
    compact_raw = payload["compact"]
    if compact_raw is None:
        raise PersistenceError("posterior archive has no compact pattern state")
    compact = _require_mapping(compact_raw, "compact")
    names = base | {
        "compact_agent_indices",
        "compact_deck_indices",
        "compact_targets",
        "compact_deck_mu_positions",
        "compact_deck_delta_positions",
        "compact_seat_positions",
        "compact_reduction",
        "compact_cluster_patterns",
        "compact_cluster_winner_seats",
        "compact_cluster_indices",
        "compact_cluster_counts",
        "compact_cluster_key_tags",
        "compact_cluster_key_ids",
    }
    if bool(compact["has_swap_counts"]):
        names.add("compact_swap_counts")
    return names


def _require_array(arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return arrays[name]
    except KeyError as exc:
        raise PersistenceError(f"missing archive array {name!r}") from exc


def _compact_from_archive(
    value: object, arrays: Mapping[str, np.ndarray]
) -> CompactPatterns | None:
    if value is None:
        return None
    raw = _require_mapping(value, "compact")
    swap_counts = arrays.get("compact_swap_counts")
    return CompactPatterns(
        agent_indices=_require_array(arrays, "compact_agent_indices"),
        deck_indices=_require_array(arrays, "compact_deck_indices"),
        targets=_require_array(arrays, "compact_targets"),
        swap_counts=swap_counts,
        deck_mu_positions=_require_array(arrays, "compact_deck_mu_positions"),
        deck_delta_positions=_require_array(arrays, "compact_deck_delta_positions"),
        seat_positions=_require_array(arrays, "compact_seat_positions"),
        eta_position=None if raw["eta_position"] is None else int(raw["eta_position"]),
        reduction=_require_array(arrays, "compact_reduction"),
        cluster_keys=_cluster_keys_from_arrays(
            arrays, "compact_cluster_key_tags", "compact_cluster_key_ids"
        ),
        cluster_patterns=_require_array(arrays, "compact_cluster_patterns"),
        cluster_winner_seats=_require_array(arrays, "compact_cluster_winner_seats"),
        cluster_indices=_require_array(arrays, "compact_cluster_indices"),
        cluster_counts=_require_array(arrays, "compact_cluster_counts"),
    )


def _require_mapping(value: object, name: str) -> Any:
    if not isinstance(value, Mapping):
        raise PersistenceError(f"{name} must be an object")
    return value


def _require_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise PersistenceError(f"{name} must be a list")
    return value


def _optional_string_set(value: object) -> frozenset[str] | None:
    if value is None:
        return None
    return frozenset(str(item) for item in _require_list(value, "string set"))


def _components_payload(components: ComponentsReport | AnchorUnsupported) -> dict[str, object]:
    if isinstance(components, AnchorUnsupported):
        return {"anchor_unsupported": components.anchor_id}
    return {
        "anchor_id": components.anchor_id,
        "anchor_component": list(components.anchor_component),
        "other_components": [list(group) for group in components.other_components],
    }


def _components_from_payload(value: object) -> ComponentsReport | AnchorUnsupported:
    raw = _require_mapping(value, "components")
    if "anchor_unsupported" in raw:
        return AnchorUnsupported(anchor_id=str(raw["anchor_unsupported"]))
    return ComponentsReport(
        anchor_id=str(raw["anchor_id"]),
        anchor_component=tuple(
            str(item) for item in _require_list(raw["anchor_component"], "anchor_component")
        ),
        other_components=tuple(
            tuple(str(item) for item in _require_list(group, "component"))
            for group in _require_list(raw["other_components"], "other_components")
        ),
    )


def _eligibility_payload(
    eligibility: tuple[EligibilityRow, ...] | AnchorUnsupported,
) -> object:
    if isinstance(eligibility, AnchorUnsupported):
        return {"anchor_unsupported": eligibility.anchor_id}
    return [
        {
            "agent_id": row.agent_id,
            "contrast_status": row.contrast_status,
            "eligible": row.eligible,
            "reasons": list(row.reasons),
        }
        for row in eligibility
    ]


def _eligibility_from_payload(
    value: object,
) -> tuple[EligibilityRow, ...] | AnchorUnsupported:
    if isinstance(value, Mapping):
        raw = _require_mapping(value, "eligibility")
        if "anchor_unsupported" in raw:
            return AnchorUnsupported(anchor_id=str(raw["anchor_unsupported"]))
        raise PersistenceError("invalid eligibility payload")
    return tuple(
        EligibilityRow(
            agent_id=str(raw["agent_id"]),
            contrast_status=raw["contrast_status"],
            eligible=bool(raw["eligible"]),
            reasons=tuple(str(item) for item in _require_list(raw["reasons"], "reasons")),
        )
        for raw in (
            _require_mapping(item, "eligibility row")
            for item in _require_list(value, "eligibility")
        )
    )


__all__ = [
    "OutcomeRateRow",
    "OutcomeRatesReport",
    "Posterior",
    "Provenance",
    "SdwpReport",
    "SdwpRow",
]
