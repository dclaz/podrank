"""Typed boundary records shared by the data and rating layers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Outcome = Literal["win", "draw", "cap", "failed"]
Role = Literal["a", "b"]
AgentKind = Literal["random", "heuristic", "checkpoint", "exploiter", "other"]
DeckSplit = Literal["train", "validation", "holdout"]
ClusterKey = tuple[Literal["cluster", "game"], str]
ExclusionReason = Literal[
    "split",
    "agent",
    "suite",
    "engine_version",
    "played_at",
    "cluster_missing",
    "cluster_filtered",
    "cluster_failed",
]


@dataclass(frozen=True, slots=True)
class AgentRecord:
    """One row of an immutable agent snapshot."""

    agent_id: str
    kind: AgentKind
    policy_version: str
    is_anchor: bool
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class DeckRecord:
    """One row of an immutable deck snapshot."""

    deck_id: str
    bracket: int
    split: DeckSplit
    deck_version: str
    tags: Mapping[str, str] = field(default_factory=dict, hash=False)
    metadata: Mapping[str, object] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class GameRow:
    """A fully normalised §3.2 game row.

    Raw input reaches :class:`podrank.GameTable` unchanged; I0 step 1 produces
    this record. Arrays are always length four in seat order and timestamps are
    timezone-aware UTC datetimes.
    """

    game_id: str
    played_at: datetime
    agents: tuple[str, str, str, str]
    decks: tuple[str, str, str, str]
    policy_versions: tuple[str, str, str, str]
    deck_versions: tuple[str | None, str | None, str | None, str | None]
    swap_counts: tuple[int, int, int, int]
    outcome: Outcome
    winner_seat: int | None
    suite: str
    cluster_id: str | None
    block_id: str | None
    rotation: int | None
    role: Role | None
    deal_id: str | None
    spec_hash: str | None
    engine_version: str
    audit: Mapping[str, object] | None = field(default=None, hash=False)


@dataclass(frozen=True, slots=True)
class EntityCounts:
    """Games containing an entity and its seat appearances."""

    games: int
    seat_appearances: int


@dataclass(frozen=True, slots=True)
class OutcomeCounts:
    """Outcome totals for one population."""

    win: int = 0
    draw: int = 0
    cap: int = 0
    failed: int = 0

    @property
    def total(self) -> int:
        return self.win + self.draw + self.cap + self.failed


@dataclass(frozen=True, slots=True)
class OutcomeRates:
    """Counts and rates for one outcome population."""

    counts: OutcomeCounts

    def _rate(self, count: int) -> float:
        total = self.counts.total
        return count / total if total else 0.0

    @property
    def win(self) -> float:
        return self._rate(self.counts.win)

    @property
    def draw(self) -> float:
        return self._rate(self.counts.draw)

    @property
    def cap(self) -> float:
        return self._rate(self.counts.cap)

    @property
    def failed(self) -> float:
        return self._rate(self.counts.failed)


@dataclass(frozen=True, slots=True)
class CountsReport:
    """I3 counts. Per-entity counts describe the eligible population."""

    received: int
    exact_duplicates: int
    eligible: int
    excluded: Mapping[str, int]
    likelihood_used: int
    likelihood_bearing_clusters: int
    agents: Mapping[str, EntityCounts]
    decks: Mapping[str, EntityCounts]

    @property
    def excluded_total(self) -> int:
        return sum(self.excluded.values())

    @property
    def reconciles(self) -> bool:
        return self.received - self.exact_duplicates == self.eligible + self.excluded_total


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One non-fatal error collected by ``validate(strict=False)``."""

    row_index: int | None
    game_id: str | None
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Preview result from I0 steps 1–3."""

    received: int
    exact_duplicates: int
    valid_rows: tuple[GameRow, ...]
    data_hash: str
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[str, ...]
    agent_snapshot_id: str
    deck_snapshot_id: str

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def deduplicated(self) -> int:
        return self.received - self.exact_duplicates


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Result of the full I0 pipeline (steps 1–6)."""

    received: int
    exact_duplicates: int
    data_hash: str
    deduplicated_rows: tuple[GameRow, ...]
    eligible_rows: tuple[GameRow, ...]
    likelihood_rows: tuple[GameRow, ...]
    counts: CountsReport
    partial_clusters: Mapping[str, int]
    cluster_reasons: Mapping[str, str]
    pre_inclusion_rates: OutcomeRates
    eligible_rates: OutcomeRates
    supported_agents: frozenset[str]
    unrankable_agents: frozenset[str]
    saturated_agents: frozenset[str]
    supported_decks: frozenset[str]
    unrankable_decks: frozenset[str]
    agent_snapshot_id: str
    deck_snapshot_id: str
    warnings: tuple[str, ...]


__all__ = [
    "AgentKind",
    "AgentRecord",
    "ClusterKey",
    "CountsReport",
    "DeckRecord",
    "DeckSplit",
    "EntityCounts",
    "ExclusionReason",
    "GameRow",
    "Outcome",
    "OutcomeCounts",
    "OutcomeRates",
    "PipelineResult",
    "Role",
    "ValidationIssue",
    "ValidationReport",
]
