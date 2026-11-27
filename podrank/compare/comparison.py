"""R2 comparison state machine (E0–E8, C2)."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from podrank._version import __version__
from podrank.compare.betting import Hypothesis, NonInferiority, Superiority, Wealth
from podrank.compare.blocks import focal_points, score_block
from podrank.compare.cs import ConfidenceSequence, confidence_sequence
from podrank.compare.descriptors import BlockDescriptor, expected_positions
from podrank.compare.pods import Design, PodDistribution
from podrank.data._canonical import canonical_json
from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.data.table import (
    GameTable,
    canonical_row_json,
    canonical_row_payload,
    normalize_game_row,
)
from podrank.errors import ValidationError
from podrank.types import AgentRecord, DeckRecord, GameRow, Role

type _PositionKey = tuple[str, int, Role | None]

Status = Literal[
    "CONTINUE",
    "ABOVE",
    "BELOW",
    "INCONCLUSIVE",
    "BUDGET_EXHAUSTED",
    "INCONCLUSIVE_INVALID",
]
_TERMINAL = frozenset({"ABOVE", "BELOW", "INCONCLUSIVE"})
_SUCCESS = frozenset({"win", "draw", "cap"})


@dataclass(frozen=True, slots=True)
class IngestIssue:
    index: int
    game_id: str | None
    kind: str
    reason: str


@dataclass(frozen=True, slots=True)
class IngestReport:
    accepted: int
    rejected: int
    transport_duplicates: int
    id_conflicts: int
    issues: tuple[IngestIssue, ...]


@dataclass(frozen=True, slots=True)
class EventRecord:
    number: int
    kind: str
    block_id: str | None
    rotation: int | None
    role: Role | None
    game_id: str | None


@dataclass(frozen=True, slots=True)
class FailureRecord:
    block_id: str
    rotation: int
    role: Role | None
    descriptor: BlockDescriptor
    event_number: int


@dataclass(frozen=True, slots=True)
class ExportReport:
    comparison_id: str
    export_revision: int
    evidence_valid: bool
    omissions: tuple[tuple[str, int, Role | None, str], ...]


@dataclass(frozen=True, slots=True)
class PairingReport:
    design: Design
    n_blocks: int
    var_x: float | None
    v_independent: float | None
    variance_ratio: float | None
    matched_positions: int
    draw_cap_difference: float | None


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    comparison_id: str
    status: Status
    decision_at_stop: str | None
    evidence_valid: bool
    invalidation_reasons: tuple[str, ...]
    design: Design
    replay: str
    hypothesis: str
    m_low: float
    m_up: float
    alpha_up: float
    alpha_low: float
    cs_alpha: float
    pod_distribution_id: str
    pod_distribution_version: int
    pod_distribution_hash: str
    blocks_consumed: int
    blocks_waiting: int
    blocks_failed: int
    mean_lift_bounds: tuple[float, float] | None
    confidence_sequence: ConfidenceSequence | None
    cs_reason: str | None
    draw_rate_a: float | None
    cap_rate_a: float | None
    draw_rate_b: float | None
    cap_rate_b: float | None
    draw_cap_difference: float | None
    pairing: PairingReport
    exhausted: bool


@dataclass(frozen=True, slots=True)
class _Resolved:
    row: GameRow | None
    points: float | None
    event_number: int


@dataclass(frozen=True, slots=True)
class _Consumed:
    block_id: str
    descriptor: BlockDescriptor
    x_up: float
    x_low: float
    complete: float | None
    failed: bool


@dataclass
class Comparison:
    """One declared, anytime-valid comparison between A and B."""

    a: str
    b: str
    design: Design
    agents: AgentRegistry
    decks: DeckRegistry
    engine_version: str
    pod_distribution: PodDistribution
    hypothesis: Hypothesis
    max_blocks: int
    alpha: float = 0.05
    alpha_up: float | None = None
    alpha_low: float | None = None
    cs_alpha: float | None = None
    replay: Literal["stochastic", "deterministic"] = "stochastic"
    min_blocks_for_failure_rule: int = 50
    max_failed_fraction: float = 0.02
    comparison_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.comparison_id = uuid.uuid4().hex
        for value, name in ((self.a, "a"), (self.b, "b")):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
            if value not in self.agents.by_id:
                raise ValueError(f"{name} is not in the agent snapshot")
        if self.a == self.b:
            raise ValueError("a and b must differ")
        if self.design not in {"duplicate", "paired"}:
            raise ValueError("design must be 'duplicate' or 'paired'")
        if self.design != self.pod_distribution.design:
            raise ValueError("design does not match the pod distribution")
        if not isinstance(self.engine_version, str) or not self.engine_version:
            raise ValueError("engine_version must be a non-empty string")
        if (
            isinstance(self.max_blocks, bool)
            or not isinstance(self.max_blocks, int)
            or self.max_blocks <= 0
        ):
            raise ValueError("max_blocks must be a positive integer")
        if self.replay not in {"stochastic", "deterministic"}:
            raise ValueError("replay must be 'stochastic' or 'deterministic'")
        if (
            isinstance(self.min_blocks_for_failure_rule, bool)
            or not isinstance(self.min_blocks_for_failure_rule, int)
            or self.min_blocks_for_failure_rule < 0
        ):
            raise ValueError("min_blocks_for_failure_rule must be a non-negative integer")
        if not 0 <= self.max_failed_fraction <= 1:
            raise ValueError("max_failed_fraction must lie in [0, 1]")
        up = self.alpha if self.alpha_up is None else self.alpha_up
        low = self.alpha if self.alpha_low is None else self.alpha_low
        for alpha_value, name in ((up, "alpha_up"), (low, "alpha_low")):
            if isinstance(alpha_value, bool) or not isinstance(alpha_value, (int, float)):
                raise ValueError(f"{name} must lie strictly in (0, 1)")
            if not 0 < float(alpha_value) < 1:
                raise ValueError(f"{name} must lie strictly in (0, 1)")
        cs = min(float(up), float(low)) if self.cs_alpha is None else self.cs_alpha
        if isinstance(cs, bool) or not isinstance(cs, (int, float)) or not 0 < float(cs) < 1:
            raise ValueError("cs_alpha must lie strictly in (0, 1)")
        self.alpha_up = float(up)
        self.alpha_low = float(low)
        self.cs_alpha = float(cs)
        self._wealth = Wealth(
            self.design,
            self.hypothesis,
            alpha_up=self.alpha_up,
            alpha_low=self.alpha_low,
        )
        self._descriptors: dict[str, BlockDescriptor] = {}
        self._declared: bool = False
        self._events: list[EventRecord] = []
        self._audit: list[IngestIssue] = []
        self._seen: dict[str, GameRow] = {}
        self._resolved: dict[_PositionKey, _Resolved] = {}
        self._provisional: dict[_PositionKey, GameRow] = {}
        self._quarantined: dict[_PositionKey, str] = {}
        self._consumed: list[_Consumed] = []
        self._consumed_positions: set[_PositionKey] = set()
        self._failure_records: list[FailureRecord] = []
        self._decision: str | None = None
        self._evidence_valid: bool = True
        self._invalidation_reasons: list[str] = []
        self._export_revision: int = 0
        self._seen_event_changed_export: bool = False

    # ------------------------------------------------------------------ declarations
    def declare_blocks(self, descriptors: object) -> None:
        if self._declared:
            raise ValueError("blocks have already been declared")
        if isinstance(descriptors, (str, bytes)) or not isinstance(descriptors, Sequence):
            raise ValueError("descriptors must be a sequence")
        items = [cast(BlockDescriptor, item) for item in descriptors]
        if not items:
            raise ValueError("at least one descriptor is required")
        ordered = sorted(items, key=lambda item: item.block_index)
        if [item.block_index for item in ordered] != list(range(len(ordered))):
            raise ValueError("block_index must be contiguous from 0")
        ids = {item.block_id for item in ordered}
        if len(ids) != len(ordered):
            raise ValueError("block_id must be unique")
        for descriptor in ordered:
            self._validate_descriptor(descriptor)
        self._descriptors = {item.block_id: item for item in ordered}
        self._declared = True

    def _validate_descriptor(self, descriptor: BlockDescriptor) -> None:
        if descriptor.design != self.design:
            raise ValueError("descriptor design does not match")
        if descriptor.engine_version != self.engine_version:
            raise ValueError("descriptor engine_version does not match")
        if (
            descriptor.pod_distribution_id != self.pod_distribution.pod_distribution_id
            or descriptor.pod_distribution_version != self.pod_distribution.version
            or descriptor.pod_distribution_hash != self.pod_distribution.pod_distribution_hash
        ):
            raise ValueError("descriptor pod distribution does not match")
        if any(descriptor.swap_counts):
            raise ValueError("R2 v1 accepts only zero swap counts")
        for deck, version in zip(descriptor.decks, descriptor.deck_versions, strict=True):
            deck_record = self.decks.by_id.get(deck)
            if deck_record is None:
                raise ValueError(f"unknown deck {deck!r}")
            if deck_record.deck_version != version:
                raise ValueError(f"deck version mismatch for {deck!r}")
        support = {item.key for item in self.pod_distribution.support}
        if (descriptor.decks, descriptor.opponents or ()) not in support:
            raise ValueError("descriptor is outside the pod distribution support")
        expected_agents: set[str] = {self.a, self.b}
        if descriptor.opponents is not None:
            expected_agents.update(descriptor.opponents)
        for agent_id in expected_agents:
            agent_record = self.agents.by_id.get(agent_id)
            if agent_record is None:
                raise ValueError(f"unknown agent {agent_id!r}")
            if descriptor.policy_versions.get(agent_id) != agent_record.policy_version:
                raise ValueError(f"policy version mismatch for {agent_id!r}")

    # ------------------------------------------------------------------ ingestion
    def ingest(self, games: GameTable | Iterable[Mapping[str, object]]) -> IngestReport:
        if not self._declared:
            raise ValueError("declare_blocks must be called before ingest")
        records = list(games)
        accepted = rejected = duplicates = conflicts = 0
        issues: list[IngestIssue] = []
        for index, record in enumerate(records):
            try:
                row = normalize_game_row(record)
            except ValidationError as exc:
                rejected += 1
                issues.append(IngestIssue(index, _game_id(record), "rejected", str(exc)))
                continue
            try:
                position = self._match_position(row)
            except ValueError as exc:
                rejected += 1
                issues.append(IngestIssue(index, row.game_id, "rejected", str(exc)))
                continue
            previous = self._seen.get(row.game_id)
            if previous is not None:
                if canonical_row_json(previous) == canonical_row_json(row):
                    duplicates += 1
                    issues.append(
                        IngestIssue(index, row.game_id, "transport_duplicate", "identical")
                    )
                    continue
                conflicts += 1
                self._audit.append(
                    IngestIssue(index, row.game_id, "id_conflict", "conflicting row for game_id")
                )
                issues.append(IngestIssue(index, row.game_id, "id_conflict", "conflicting row"))
                if self._position_of(previous) == position and (
                    previous.outcome,
                    previous.winner_seat,
                ) != (row.outcome, row.winner_seat):
                    self._quarantine(position, "game_id_integrity_violation")
                continue
            self._seen[row.game_id] = row
            accepted += 1
            self._record_event("game", row.cluster_id, row.rotation, row.role, row.game_id)
            self._seen_event_changed_export = False
            self._resolve(row, position)
            self._finish_event()
        return IngestReport(accepted, rejected, duplicates, conflicts, tuple(issues))

    def _position_of(self, row: GameRow) -> _PositionKey | None:
        try:
            return self._match_position(row)
        except ValueError:
            return None

    def _match_position(self, row: GameRow) -> _PositionKey:
        if row.cluster_id is None or row.cluster_id not in self._descriptors:
            raise ValueError("row does not reference a declared block")
        descriptor = self._descriptors[row.cluster_id]
        if row.block_id not in (None, descriptor.block_id):
            raise ValueError("row block_id does not match")
        if row.deal_id != descriptor.deal_id:
            raise ValueError("row deal_id does not match")
        if row.engine_version != self.engine_version:
            raise ValueError("row engine_version does not match")
        if tuple(row.decks) != descriptor.decks:
            raise ValueError("row decks do not match")
        if any(version is None for version in row.deck_versions):
            raise ValueError("R2 requires non-null deck versions")
        if tuple(row.deck_versions) != descriptor.deck_versions:
            raise ValueError("row deck versions do not match")
        if tuple(row.swap_counts) != descriptor.swap_counts:
            raise ValueError("row swap counts do not match")
        for seat, agent_id in enumerate(row.agents):
            if descriptor.policy_versions.get(agent_id) != row.policy_versions[seat]:
                raise ValueError(f"row policy version for {agent_id!r} does not match")
        if row.rotation is None or row.rotation not in range(4):
            raise ValueError("row rotation must be 0..3")
        rotation = row.rotation
        if self.design == "duplicate":
            if row.role is not None:
                raise ValueError("duplicate rows must have null role")
            expected = [self.b, self.b, self.b, self.b]
            expected[rotation] = self.a
            role: Role | None = None
        else:
            if row.role not in {"a", "b"}:
                raise ValueError("paired rows must have role 'a' or 'b'")
            role = row.role
            expected = [self.b, self.b, self.b, self.b]
            expected[rotation] = self.a if role == "a" else self.b
            assert descriptor.opponents is not None
            for offset, opponent in enumerate(descriptor.opponents, start=1):
                expected[(rotation + offset) % 4] = opponent
        if tuple(row.agents) != tuple(expected):
            raise ValueError("row agents do not match the declared position")
        return (descriptor.block_id, rotation, role)

    # ------------------------------------------------------------------ marks and budget
    def mark_failed(self, block_id: str, rotation: int, role: Role | None = None) -> None:
        descriptor, position = self._declared_position(block_id, rotation, role)
        self._record_event("mark_failed", block_id, rotation, role, None)
        self._seen_event_changed_export = False
        self._resolve_failed(position, descriptor, event_number=len(self._events))
        self._finish_event()

    def mark_block_failed(self, block_id: str) -> None:
        if block_id not in self._descriptors:
            raise ValueError(f"unknown block {block_id!r}")
        descriptor = self._descriptors[block_id]
        self._record_event("mark_block_failed", block_id, None, None, None)
        self._seen_event_changed_export = False
        event_number = len(self._events)
        for rotation, role in expected_positions(self.design):
            position = (block_id, rotation, role)
            if position not in self._resolved:
                self._resolve_failed(position, descriptor, event_number=event_number)
        self._finish_event()

    def extend_budget(self, new_max: int) -> None:
        if isinstance(new_max, bool) or not isinstance(new_max, int):
            raise ValueError("new_max must be an integer")
        if new_max <= self.max_blocks:
            raise ValueError("new_max must exceed max_blocks")
        if self._decision is not None:
            raise ValueError("cannot extend a terminal comparison")
        self.max_blocks = new_max
        self._seen_event_changed_export = False
        self._drain()
        self._finish_event()

    def _declared_position(
        self, block_id: str, rotation: int, role: Role | None
    ) -> tuple[BlockDescriptor, _PositionKey]:
        if block_id not in self._descriptors:
            raise ValueError(f"unknown block {block_id!r}")
        descriptor = self._descriptors[block_id]
        if rotation not in range(4):
            raise ValueError("rotation must be 0..3")
        if self.design == "duplicate":
            if role is not None:
                raise ValueError("duplicate positions have null role")
        elif role not in {"a", "b"}:
            raise ValueError("paired positions require role 'a' or 'b'")
        return descriptor, (block_id, rotation, role)

    def _resolve_failed(
        self, position: _PositionKey, descriptor: BlockDescriptor, *, event_number: int
    ) -> None:
        if position in self._resolved:
            return
        provisional = self._provisional.pop(position, None)
        self._resolved[position] = _Resolved(
            row=provisional, points=None, event_number=event_number
        )
        self._failure_records.append(
            FailureRecord(
                block_id=descriptor.block_id,
                rotation=position[1],
                role=position[2],
                descriptor=descriptor,
                event_number=event_number,
            )
        )

    # ------------------------------------------------------------------ resolution and drain
    def _record_event(
        self,
        kind: str,
        block_id: str | None,
        rotation: int | None,
        role: Role | None,
        game_id: str | None,
    ) -> None:
        self._events.append(
            EventRecord(len(self._events) + 1, kind, block_id, rotation, role, game_id)
        )

    def _resolve(self, row: GameRow, position: _PositionKey) -> None:
        if position in self._quarantined:
            return
        points = focal_points(row.outcome, row.winner_seat, position[1])
        existing = self._resolved.get(position)
        if self.replay == "stochastic":
            if existing is None:
                self._resolved[position] = _Resolved(row, points, len(self._events))
            return
        if row.outcome == "failed":
            if existing is None:
                self._provisional[position] = row
            return
        if existing is not None:
            if existing.row is not None and (existing.row.outcome, existing.row.winner_seat) != (
                row.outcome,
                row.winner_seat,
            ):
                self._quarantine(position, "replay_violation")
            return
        self._provisional.pop(position, None)
        self._resolved[position] = _Resolved(row, points, len(self._events))

    def _quarantine(self, position: _PositionKey, reason: str) -> None:
        self._quarantined[position] = reason
        if position in self._consumed_positions:
            self._export_revision += 1
            self._seen_event_changed_export = True
        self._invalidate(f"{reason}: block {position[0]}, rotation {position[1]}")

    def _invalidate(self, reason: str) -> None:
        if self._evidence_valid:
            self._evidence_valid = False
            self._invalidation_reasons.append(reason)
            if self._decision is None:
                self._decision = "INCONCLUSIVE"
            if not self._seen_event_changed_export:
                self._export_revision += 1
                self._seen_event_changed_export = True

    def _finish_event(self) -> None:
        if self._evidence_valid:
            self._drain()

    def _drain(self) -> None:
        while self._decision is None and len(self._consumed) < self.max_blocks:
            index = len(self._consumed)
            descriptor = next(
                item for item in self._descriptors.values() if item.block_index == index
            )
            positions = [
                (descriptor.block_id, rotation, role)
                for rotation, role in expected_positions(self.design)
            ]
            if any(position not in self._resolved for position in positions):
                break
            points = {
                (rotation, role): self._resolved[position].points
                for position, (rotation, role) in zip(
                    positions, expected_positions(self.design), strict=True
                )
            }
            scores = score_block(self.design, points)
            self._consumed.append(
                _Consumed(
                    block_id=descriptor.block_id,
                    descriptor=descriptor,
                    x_up=scores.x_up,
                    x_low=scores.x_low,
                    complete=scores.complete,
                    failed=scores.failed,
                )
            )
            self._consumed_positions.update(positions)
            self._wealth.update(scores.x_up, scores.x_low)
            self._export_revision += 1
            self._seen_event_changed_export = True
            self._evaluate()

    def _evaluate(self) -> None:
        if self._decision is not None:
            return
        up = self._wealth.crossed_up()
        low = self._wealth.crossed_low()
        if up and low:
            self._decision = "INCONCLUSIVE"
        elif up:
            self._decision = "ABOVE"
        elif low:
            self._decision = "BELOW"
        elif (
            len(self._consumed) >= self.min_blocks_for_failure_rule
            and self._failed_fraction() > self.max_failed_fraction
        ):
            self._decision = "INCONCLUSIVE"

    def _failed_fraction(self) -> float:
        return sum(1 for block in self._consumed if block.failed) / len(self._consumed)

    # ------------------------------------------------------------------ output
    @property
    def evidence_valid(self) -> bool:
        return self._evidence_valid

    def status(self) -> Status:
        if not self._evidence_valid:
            return "INCONCLUSIVE_INVALID"
        if self._decision in _TERMINAL:
            return self._decision  # type: ignore[return-value]
        if len(self._consumed) >= self.max_blocks:
            return "BUDGET_EXHAUSTED"
        return "CONTINUE"

    def report(self) -> ComparisonReport:
        bounds = None
        if self._consumed:
            bounds = (
                sum(block.x_up for block in self._consumed) / len(self._consumed),
                sum(block.x_low for block in self._consumed) / len(self._consumed),
            )
        cs: ConfidenceSequence | None = None
        cs_reason: str | None = None
        if not self._consumed:
            cs_reason = "no blocks consumed"
        elif not self._evidence_valid:
            cs_reason = "evidence invalid"
        else:
            cs = confidence_sequence(
                [block.x_up for block in self._consumed],
                [block.x_low for block in self._consumed],
                design=self.design,
                cs_alpha=cast(float, self.cs_alpha),
            )
        draw_a, cap_a = self._agent_rates(self.a)
        draw_b, cap_b = self._agent_rates(self.b)
        difference = self._paired_difference() if self.design == "paired" else None
        return ComparisonReport(
            comparison_id=self.comparison_id,
            status=self.status(),
            decision_at_stop=self._decision,
            evidence_valid=self._evidence_valid,
            invalidation_reasons=tuple(self._invalidation_reasons),
            design=self.design,
            replay=self.replay,
            hypothesis=type(self.hypothesis).__name__,
            m_low=self._wealth.thresholds.m_low,
            m_up=self._wealth.thresholds.m_up,
            alpha_up=cast(float, self.alpha_up),
            alpha_low=cast(float, self.alpha_low),
            cs_alpha=cast(float, self.cs_alpha),
            pod_distribution_id=self.pod_distribution.pod_distribution_id,
            pod_distribution_version=self.pod_distribution.version,
            pod_distribution_hash=self.pod_distribution.pod_distribution_hash,
            blocks_consumed=len(self._consumed),
            blocks_waiting=max(0, len(self._descriptors) - len(self._consumed)),
            blocks_failed=sum(1 for block in self._consumed if block.failed),
            mean_lift_bounds=bounds,
            confidence_sequence=cs,
            cs_reason=cs_reason,
            draw_rate_a=draw_a,
            cap_rate_a=cap_a,
            draw_rate_b=draw_b,
            cap_rate_b=cap_b,
            draw_cap_difference=difference,
            pairing=self.pairing_report(),
            exhausted=len(self._consumed) >= self.max_blocks and self._decision is None,
        )

    def pairing_report(self) -> PairingReport:
        if self.design == "paired":
            matched = self._paired_matched()
            return PairingReport(
                design="paired",
                n_blocks=len(self._consumed),
                var_x=None,
                v_independent=None,
                variance_ratio=None,
                matched_positions=len(matched),
                draw_cap_difference=(sum(matched) / len(matched) if matched else None),
            )
        complete = [block.complete for block in self._consumed if block.complete is not None]
        var_x = float(np.var(complete, ddof=1)) if len(complete) > 1 else None
        per_rotation: list[tuple[float, float]] = []
        for rotation in range(4):
            values = []
            for block in self._consumed:
                resolved = self._resolved.get((block.block_id, rotation, None))
                if resolved is None or resolved.row is None or resolved.points is None:
                    continue
                values.append(resolved.points)
            if len(values) < 2:
                per_rotation = []
                break
            array = np.asarray(values, dtype=np.float64)
            per_rotation.append((float(array.mean()), float((array**2).mean())))
        v_ind = None
        if per_rotation:
            variance = [second - first**2 for first, second in per_rotation]
            if all(value >= 0 for value in variance):
                v_ind = sum(variance) / 16.0
        ratio = var_x / v_ind if var_x is not None and v_ind is not None and v_ind > 0 else None
        return PairingReport(
            design="duplicate",
            n_blocks=len(self._consumed),
            var_x=var_x,
            v_independent=v_ind,
            variance_ratio=ratio,
            matched_positions=0,
            draw_cap_difference=None,
        )

    def _agent_rates(self, agent_id: str) -> tuple[float | None, float | None]:
        rows = self._consumed_rows()
        total = draw = cap = 0
        for row in rows:
            for agent in row.agents:
                if agent != agent_id:
                    continue
                total += 1
                draw += int(row.outcome == "draw")
                cap += int(row.outcome == "cap")
        if not total:
            return None, None
        return draw / total, cap / total

    def _consumed_rows(self) -> list[GameRow]:
        rows: list[GameRow] = []
        for block in self._consumed:
            for rotation, role in expected_positions(self.design):
                resolved = self._resolved.get((block.block_id, rotation, role))
                if resolved is not None and resolved.row is not None:
                    rows.append(resolved.row)
        return rows

    def _paired_difference(self) -> float | None:
        values = self._paired_matched()
        return sum(values) / len(values) if values else None

    def _paired_matched(self) -> list[float]:
        values: list[float] = []
        for block in self._consumed:
            for rotation in range(4):
                a_row = self._resolved.get((block.block_id, rotation, "a"))
                b_row = self._resolved.get((block.block_id, rotation, "b"))
                if a_row is None or b_row is None or a_row.row is None or b_row.row is None:
                    continue
                if a_row.row.outcome == "failed" or b_row.row.outcome == "failed":
                    continue
                a_score = 1.0 if a_row.row.outcome in {"draw", "cap"} else 0.0
                b_score = 1.0 if b_row.row.outcome in {"draw", "cap"} else 0.0
                values.append(a_score - b_score)
        return values

    def failure_records(self) -> tuple[FailureRecord, ...]:
        return tuple(self._failure_records)

    def rating_rows(self) -> tuple[GameTable, ExportReport]:
        rows: list[dict[str, object]] = []
        omissions: list[tuple[str, int, Role | None, str]] = []
        for block in self._consumed:
            for rotation, role in expected_positions(self.design):
                position = (block.block_id, rotation, role)
                resolved = self._resolved.get(position)
                if position in self._quarantined:
                    omissions.append((block.block_id, rotation, role, self._quarantined[position]))
                    continue
                if resolved is None:
                    continue
                if resolved.row is None:
                    continue
                rows.append(canonical_row_payload(resolved.row))
        return (
            GameTable.from_records(rows),
            ExportReport(
                comparison_id=self.comparison_id,
                export_revision=self._export_revision,
                evidence_valid=self._evidence_valid,
                omissions=tuple(omissions),
            ),
        )

    def save(self, path: str | Path) -> None:
        payload = self._state_payload()
        envelope = {
            "kind": "podrank.comparison",
            "schema_version": 1,
            "package_version": __version__,
            "payload": payload,
        }
        target = Path(path)
        text = canonical_json(envelope)
        descriptor = None
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = None
                handle.write(text)
            os.replace(temporary, target)
            temporary = None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass

    @classmethod
    def load(cls, path: str | Path) -> Comparison:
        try:
            return cls._load_unchecked(path)
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            raise ValueError(f"malformed comparison payload: {exc}") from exc

    @classmethod
    def _load_unchecked(cls, path: str | Path) -> Comparison:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read comparison: {exc}") from exc
        envelope = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(envelope, dict) or set(envelope) != {
            "kind",
            "schema_version",
            "package_version",
            "payload",
        }:
            raise ValueError("comparison envelope schema mismatch")
        if envelope["kind"] != "podrank.comparison" or envelope["schema_version"] != 1:
            raise ValueError("unsupported comparison schema")
        payload = envelope["payload"]
        if not isinstance(payload, dict):
            raise ValueError("comparison payload must be an object")
        config = payload["config"]
        agents = AgentRegistry.from_records(payload["agents"])
        decks = DeckRegistry.from_records(payload["decks"])
        pod_payload = payload["pod_distribution"]["payload"]
        pod = PodDistribution.from_support(
            pod_payload["pod_distribution_id"],
            pod_payload["version"],
            pod_payload["design"],
            pod_payload["support"],
            provenance=payload["pod_distribution"]["provenance"],
        )
        if pod.pod_distribution_hash != payload["pod_distribution"]["pod_distribution_hash"]:
            raise ValueError("pod distribution hash mismatch")
        hypothesis = _hypothesis_from_payload(config["hypothesis"])
        comparison = cls(
            a=config["a"],
            b=config["b"],
            design=config["design"],
            agents=agents,
            decks=decks,
            engine_version=config["engine_version"],
            pod_distribution=pod,
            hypothesis=hypothesis,
            max_blocks=config["max_blocks"],
            alpha=config["alpha"],
            alpha_up=config["alpha_up"],
            alpha_low=config["alpha_low"],
            cs_alpha=config["cs_alpha"],
            replay=config["replay"],
            min_blocks_for_failure_rule=config["min_blocks_for_failure_rule"],
            max_failed_fraction=config["max_failed_fraction"],
        )
        comparison.comparison_id = payload["comparison_id"]
        comparison._descriptors = {
            item["block_id"]: _descriptor_from_payload(item) for item in payload["descriptors"]
        }
        comparison._declared = bool(payload["declared"])
        comparison._events = [EventRecord(**item) for item in payload["events"]]
        comparison._audit = [IngestIssue(**item) for item in payload["audit"]]
        comparison._seen = {
            item["game_id"]: normalize_game_row(item["row"]) for item in payload["seen"]
        }
        comparison._resolved = {
            (item["block_id"], item["rotation"], item["role"]): _resolved_from_payload(item)
            for item in payload["resolved"]
        }
        comparison._provisional = {
            (item["block_id"], item["rotation"], item["role"]): normalize_game_row(item["row"])
            for item in payload["provisional"]
        }
        comparison._quarantined = {
            (item["block_id"], item["rotation"], item["role"]): item["reason"]
            for item in payload["quarantined"]
        }
        comparison._consumed = [
            _Consumed(
                block_id=item["block_id"],
                descriptor=comparison._descriptors[item["block_id"]],
                x_up=item["x_up"],
                x_low=item["x_low"],
                complete=item.get("complete"),
                failed=item["failed"],
            )
            for item in payload["consumed"]
        ]
        comparison._consumed_positions = {
            (item["block_id"], item["rotation"], item["role"])
            for item in payload["consumed_positions"]
        }
        comparison._failure_records = [
            FailureRecord(
                block_id=item["block_id"],
                rotation=item["rotation"],
                role=item["role"],
                descriptor=comparison._descriptors[item["block_id"]],
                event_number=item["event_number"],
            )
            for item in payload["failure_records"]
        ]
        comparison._decision = payload["decision"]
        comparison._evidence_valid = payload["evidence_valid"]
        comparison._invalidation_reasons = list(payload["invalidation_reasons"])
        comparison._export_revision = payload["export_revision"]
        comparison._seen_event_changed_export = payload["seen_event_changed_export"]
        comparison._wealth.log_wealth_up[:] = payload["log_wealth_up"]
        comparison._wealth.log_wealth_low[:] = payload["log_wealth_low"]
        return comparison

    def _state_payload(self) -> dict[str, object]:
        return {
            "comparison_id": self.comparison_id,
            "config": {
                "a": self.a,
                "b": self.b,
                "design": self.design,
                "engine_version": self.engine_version,
                "hypothesis": {
                    "kind": type(self.hypothesis).__name__,
                    "margin": self.hypothesis.margin,
                },
                "max_blocks": self.max_blocks,
                "alpha": self.alpha,
                "alpha_up": self.alpha_up,
                "alpha_low": self.alpha_low,
                "cs_alpha": self.cs_alpha,
                "replay": self.replay,
                "min_blocks_for_failure_rule": self.min_blocks_for_failure_rule,
                "max_failed_fraction": self.max_failed_fraction,
            },
            "agents": [_agent_payload(item) for item in self.agents],
            "decks": [_deck_payload(item) for item in self.decks],
            "pod_distribution": {
                "payload": self.pod_distribution.payload(),
                "pod_distribution_hash": self.pod_distribution.pod_distribution_hash,
                "provenance": dict(self.pod_distribution.provenance),
            },
            "descriptors": [_descriptor_payload(item) for item in self._descriptors.values()],
            "declared": self._declared,
            "events": [
                {
                    "number": item.number,
                    "kind": item.kind,
                    "block_id": item.block_id,
                    "rotation": item.rotation,
                    "role": item.role,
                    "game_id": item.game_id,
                }
                for item in self._events
            ],
            "audit": [
                {
                    "index": item.index,
                    "game_id": item.game_id,
                    "kind": item.kind,
                    "reason": item.reason,
                }
                for item in self._audit
            ],
            "seen": [
                {"game_id": key, "row": canonical_row_payload(value)}
                for key, value in self._seen.items()
            ],
            "resolved": [
                {
                    "block_id": key[0],
                    "rotation": key[1],
                    "role": key[2],
                    "row": None if value.row is None else canonical_row_payload(value.row),
                    "event_number": value.event_number,
                }
                for key, value in self._resolved.items()
            ],
            "provisional": [
                {
                    "block_id": key[0],
                    "rotation": key[1],
                    "role": key[2],
                    "row": canonical_row_payload(value),
                }
                for key, value in self._provisional.items()
            ],
            "quarantined": [
                {"block_id": key[0], "rotation": key[1], "role": key[2], "reason": reason}
                for key, reason in self._quarantined.items()
            ],
            "consumed": [
                {
                    "block_id": item.block_id,
                    "x_up": item.x_up,
                    "x_low": item.x_low,
                    "complete": item.complete,
                    "failed": item.failed,
                }
                for item in self._consumed
            ],
            "consumed_positions": [
                {"block_id": key[0], "rotation": key[1], "role": key[2]}
                for key in self._consumed_positions
            ],
            "failure_records": [
                {
                    "block_id": item.block_id,
                    "rotation": item.rotation,
                    "role": item.role,
                    "event_number": item.event_number,
                }
                for item in self._failure_records
            ],
            "decision": self._decision,
            "evidence_valid": self._evidence_valid,
            "invalidation_reasons": list(self._invalidation_reasons),
            "export_revision": self._export_revision,
            "seen_event_changed_export": self._seen_event_changed_export,
            "log_wealth_up": self._wealth.log_wealth_up.tolist(),
            "log_wealth_low": self._wealth.log_wealth_low.tolist(),
        }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _hypothesis_from_payload(payload: Any) -> Hypothesis:
    kind = payload["kind"]
    margin = payload["margin"]
    if kind == "Superiority":
        return Superiority(float(margin))
    if kind == "NonInferiority":
        return NonInferiority(float(margin))
    raise ValueError(f"unknown hypothesis {kind!r}")


def _agent_payload(record: AgentRecord) -> dict[str, object]:
    return {
        "agent_id": record.agent_id,
        "kind": record.kind,
        "policy_version": record.policy_version,
        "is_anchor": record.is_anchor,
        "tags": list(record.tags),
        "metadata": dict(record.metadata),
    }


def _deck_payload(record: DeckRecord) -> dict[str, object]:
    return {
        "deck_id": record.deck_id,
        "bracket": record.bracket,
        "split": record.split,
        "deck_version": record.deck_version,
        "tags": dict(record.tags),
        "metadata": dict(record.metadata),
    }


def _descriptor_payload(descriptor: BlockDescriptor) -> dict[str, object]:
    return {
        "block_id": descriptor.block_id,
        "block_index": descriptor.block_index,
        "design": descriptor.design,
        "deal_id": descriptor.deal_id,
        "decks": list(descriptor.decks),
        "deck_versions": list(descriptor.deck_versions),
        "swap_counts": list(descriptor.swap_counts),
        "opponents": None if descriptor.opponents is None else list(descriptor.opponents),
        "policy_versions": dict(descriptor.policy_versions),
        "engine_version": descriptor.engine_version,
        "pod_distribution_id": descriptor.pod_distribution_id,
        "pod_distribution_version": descriptor.pod_distribution_version,
        "pod_distribution_hash": descriptor.pod_distribution_hash,
    }


def _descriptor_from_payload(payload: Any) -> BlockDescriptor:
    return BlockDescriptor(
        block_id=payload["block_id"],
        block_index=payload["block_index"],
        design=payload["design"],
        deal_id=payload["deal_id"],
        decks=tuple(payload["decks"]),
        deck_versions=tuple(payload["deck_versions"]),
        swap_counts=tuple(payload["swap_counts"]),
        opponents=None if payload["opponents"] is None else tuple(payload["opponents"]),
        policy_versions=payload["policy_versions"],
        engine_version=payload["engine_version"],
        pod_distribution_id=payload["pod_distribution_id"],
        pod_distribution_version=payload["pod_distribution_version"],
        pod_distribution_hash=payload["pod_distribution_hash"],
    )


def _resolved_from_payload(payload: Any) -> _Resolved:
    raw_row = payload["row"]
    row = None if raw_row is None else normalize_game_row(raw_row)
    points = (
        None if row is None else focal_points(row.outcome, row.winner_seat, payload["rotation"])
    )
    return _Resolved(row=row, points=points, event_number=payload["event_number"])


def _game_id(record: Mapping[str, object]) -> str | None:
    value = record.get("game_id")
    return value if isinstance(value, str) else None


__all__ = [
    "Comparison",
    "ComparisonReport",
    "EventRecord",
    "ExportReport",
    "FailureRecord",
    "IngestIssue",
    "IngestReport",
    "PairingReport",
    "Status",
]

# ------------------------------------------------------------------ persistence
