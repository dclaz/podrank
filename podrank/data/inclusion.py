"""Inclusion configuration and the single I0 validation/inclusion pipeline."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Literal, overload

from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.data.table import (
    GameTable,
    canonical_row_json,
    compute_data_hash,
    normalize_game_row,
    normalize_timestamp,
)
from podrank.errors import (
    DuplicateGameError,
    DuplicateRowWarning,
    IdentityMismatchError,
    InclusionError,
    ManifestMismatchError,
    NoRankableEvidence,
    UnknownEntityError,
    ValidationError,
)
from podrank.types import (
    ClusterKey,
    CountsReport,
    EntityCounts,
    ExclusionReason,
    GameRow,
    OutcomeCounts,
    OutcomeRates,
    PipelineResult,
    ValidationIssue,
    ValidationReport,
)

_EXCLUSION_REASONS: tuple[ExclusionReason, ...] = (
    "split",
    "agent",
    "suite",
    "engine_version",
    "played_at",
    "cluster_missing",
    "cluster_filtered",
    "cluster_failed",
)
_DECK_SPLITS = frozenset({"train", "validation", "holdout"})
_WUBRG = frozenset("wubrg")


@dataclass(frozen=True, slots=True)
class _TagSchema:
    allowed: Mapping[str, frozenset[str] | None]
    required: frozenset[str]


@dataclass(frozen=True, slots=True)
class Inclusion:
    """Immutable inclusion settings from Appendix C.1."""

    suites: frozenset[str] | None = None
    engine_versions: frozenset[str] | None = None
    splits: frozenset[str] | None = None
    agents: frozenset[str] | None = None
    played_at: tuple[datetime | None, datetime | None] = (None, None)
    clusters: Literal["use_available", "complete_only"] = "use_available"
    manifest: Mapping[str, frozenset[str]] | None = None
    allow_mirror: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "suites", _normalise_filter(self.suites, "suites"))
        object.__setattr__(
            self, "engine_versions", _normalise_filter(self.engine_versions, "engine_versions")
        )
        object.__setattr__(
            self, "splits", _normalise_filter(self.splits, "splits", allowed=_DECK_SPLITS)
        )
        object.__setattr__(self, "agents", _normalise_filter(self.agents, "agents"))
        if not isinstance(self.allow_mirror, bool):
            raise InclusionError("allow_mirror must be a boolean")
        if self.clusters not in {"use_available", "complete_only"}:
            raise InclusionError("clusters must be 'use_available' or 'complete_only'")

        if not isinstance(self.played_at, tuple) or len(self.played_at) != 2:
            raise InclusionError("played_at must be a (lower, upper) tuple")
        lower = (
            None
            if self.played_at[0] is None
            else normalize_timestamp(self.played_at[0], "played_at lower bound")
        )
        upper = (
            None
            if self.played_at[1] is None
            else normalize_timestamp(self.played_at[1], "played_at upper bound")
        )
        if lower is not None and upper is not None and lower > upper:
            raise InclusionError("played_at lower bound must not exceed the upper bound")
        object.__setattr__(self, "played_at", (lower, upper))

        if self.clusters == "use_available":
            if self.manifest is not None:
                raise InclusionError("manifest must be absent when clusters='use_available'")
        else:
            if self.manifest is None:
                raise InclusionError("manifest is required when clusters='complete_only'")
            object.__setattr__(self, "manifest", _normalise_manifest(self.manifest))

    @classmethod
    def default(cls) -> Inclusion:
        return cls()


def _normalise_filter(
    value: object, field_name: str, *, allowed: frozenset[str] | None = None
) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise InclusionError(f"{field_name} must be an iterable of strings or None")
    items = frozenset(value)
    for item in items:
        if not isinstance(item, str) or not item:
            raise InclusionError(f"{field_name} must contain non-empty strings")
    if allowed is not None and not items <= allowed:
        raise InclusionError(f"{field_name} must be a subset of {sorted(allowed)}")
    return items


def _normalise_manifest(value: object) -> Mapping[str, frozenset[str]]:
    if not isinstance(value, Mapping):
        raise InclusionError("manifest must be a mapping of cluster_id to game IDs")
    result: dict[str, frozenset[str]] = {}
    for cluster_id, game_ids in value.items():
        if not isinstance(cluster_id, str) or not cluster_id:
            raise InclusionError("manifest cluster IDs must be non-empty strings")
        if isinstance(game_ids, (str, bytes, bytearray)) or not isinstance(game_ids, Iterable):
            raise InclusionError("manifest values must be iterables of game IDs")
        ids = frozenset(game_ids)
        if not ids:
            raise InclusionError(f"manifest cluster {cluster_id!r} must list at least one game")
        for game_id in ids:
            if not isinstance(game_id, str) or not game_id:
                raise InclusionError("manifest game IDs must be non-empty strings")
        result[cluster_id] = ids
    if not result:
        raise InclusionError("manifest must not be empty")
    return MappingProxyType(dict(sorted(result.items())))


def _normalise_split_policy(
    value: Mapping[str, Collection[str]] | None,
) -> Mapping[str, frozenset[str]] | None:
    if value is None:
        return None
    result: dict[str, frozenset[str]] = {}
    for split, suites in value.items():
        if split not in _DECK_SPLITS:
            raise InclusionError(f"split_policy has unknown split {split!r}")
        if isinstance(suites, (str, bytes, bytearray)) or not isinstance(suites, Iterable):
            raise InclusionError(f"split_policy[{split!r}] must be an iterable of suites")
        items = frozenset(suites)
        for suite in items:
            if not isinstance(suite, str) or not suite:
                raise InclusionError("split_policy suite names must be non-empty strings")
        result[split] = items
    return MappingProxyType(dict(sorted(result.items())))


def _string_collection(value: object, field_name: str) -> frozenset[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise ValidationError(f"{field_name} must be an iterable of strings")
    items = frozenset(value)
    for item in items:
        if not isinstance(item, str) or not item:
            raise ValidationError(f"{field_name} must contain non-empty strings")
    return items


def _normalise_tag_schema(value: Mapping[str, object] | None) -> _TagSchema | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValidationError("tag_schema must be a mapping or None")
    required: frozenset[str] = frozenset()
    allowed: dict[str, frozenset[str] | None] = {}
    for key, specification in value.items():
        if not isinstance(key, str) or not key:
            raise ValidationError("tag_schema keys must be non-empty strings")
        if key == "required":
            required = _string_collection(specification, "tag_schema['required']")
            continue
        if specification == "wubrg":
            allowed[key] = None
        elif isinstance(specification, str):
            if not specification:
                raise ValidationError(f"tag_schema[{key!r}] must not be empty")
            allowed[key] = frozenset({specification})
        else:
            allowed[key] = _string_collection(specification, f"tag_schema[{key!r}]")
    return _TagSchema(
        allowed=MappingProxyType(dict(sorted(allowed.items()))),
        required=required,
    )


def _check_tags(deck_id: str, tags: Mapping[str, str], schema: _TagSchema) -> None:
    for key in schema.required:
        if key not in tags:
            raise ValidationError(f"deck {deck_id!r} is missing required tag {key!r}")
    for key, allowed in schema.allowed.items():
        if key not in tags:
            continue
        value = tags[key]
        if allowed is None:
            if not value or any(character.lower() not in _WUBRG for character in value):
                raise ValidationError(
                    f"deck {deck_id!r} tag {key!r} must use only wubrg characters"
                )
        elif value not in allowed:
            raise ValidationError(f"deck {deck_id!r} tag {key!r} must be one of {sorted(allowed)}")


def _check_identity(
    row: GameRow,
    agents: AgentRegistry,
    decks: DeckRegistry,
    inclusion: Inclusion,
    split_policy: Mapping[str, frozenset[str]] | None,
    tag_schema: _TagSchema | None,
) -> None:
    for seat in range(4):
        agent_id = row.agents[seat]
        deck_id = row.decks[seat]
        if agent_id not in agents.by_id:
            raise UnknownEntityError(f"game {row.game_id!r} references unknown agent {agent_id!r}")
        if deck_id not in decks.by_id:
            raise UnknownEntityError(f"game {row.game_id!r} references unknown deck {deck_id!r}")
        if row.policy_versions[seat] != agents.by_id[agent_id].policy_version:
            raise IdentityMismatchError(
                f"game {row.game_id!r} policy version for agent {agent_id!r} "
                "does not match the snapshot"
            )
        deck_version = row.deck_versions[seat]
        if deck_version is not None and deck_version != decks.by_id[deck_id].deck_version:
            raise IdentityMismatchError(
                f"game {row.game_id!r} deck version for deck {deck_id!r} "
                "does not match the snapshot"
            )

    if not inclusion.allow_mirror and len(set(row.agents)) != 4:
        raise ValidationError(
            f"game {row.game_id!r} repeats an agent but mirror seats are disabled"
        )

    if split_policy is not None:
        for deck_id in row.decks:
            split = decks.by_id[deck_id].split
            allowed_suites = split_policy.get(split)
            if allowed_suites is not None and row.suite not in allowed_suites:
                raise ValidationError(
                    f"game {row.game_id!r} suite {row.suite!r} is not allowed "
                    f"for deck split {split!r}"
                )

    if tag_schema is not None:
        for deck_id in row.decks:
            _check_tags(deck_id, decks.by_id[deck_id].tags, tag_schema)


def _first_filter_reason(
    row: GameRow, inclusion: Inclusion, decks: DeckRegistry
) -> ExclusionReason | None:
    if inclusion.splits is not None and not all(
        decks.by_id[deck_id].split in inclusion.splits for deck_id in row.decks
    ):
        return "split"
    if inclusion.agents is not None and not any(
        agent_id in inclusion.agents for agent_id in row.agents
    ):
        return "agent"
    if inclusion.suites is not None and row.suite not in inclusion.suites:
        return "suite"
    if (
        inclusion.engine_versions is not None
        and row.engine_version not in inclusion.engine_versions
    ):
        return "engine_version"
    lower, upper = inclusion.played_at
    if (lower is not None and row.played_at < lower) or (
        upper is not None and row.played_at > upper
    ):
        return "played_at"
    return None


def _cluster_key(row: GameRow) -> ClusterKey:
    if row.cluster_id is None:
        return ("game", row.game_id)
    return ("cluster", row.cluster_id)


def _outcome_counts(rows: Iterable[GameRow]) -> OutcomeCounts:
    counts = {"win": 0, "draw": 0, "cap": 0, "failed": 0}
    for row in rows:
        counts[row.outcome] += 1
    return OutcomeCounts(**counts)


def _outcome_rates(rows: Iterable[GameRow]) -> OutcomeRates:
    return OutcomeRates(counts=_outcome_counts(rows))


def _entity_counts_pair(
    rows: Iterable[GameRow],
    agent_ids: Iterable[str],
    deck_ids: Iterable[str],
) -> tuple[Mapping[str, EntityCounts], Mapping[str, EntityCounts]]:
    def counters(ids: Iterable[str]) -> tuple[dict[str, int], dict[str, int]]:
        return {entity_id: 0 for entity_id in ids}, {entity_id: 0 for entity_id in ids}

    agent_games, agent_appearances = counters(agent_ids)
    deck_games, deck_appearances = counters(deck_ids)
    for row in rows:
        for entity_id in set(row.agents):
            agent_games[entity_id] += 1
        for entity_id in row.agents:
            agent_appearances[entity_id] += 1
        for entity_id in set(row.decks):
            deck_games[entity_id] += 1
        for entity_id in row.decks:
            deck_appearances[entity_id] += 1

    def result(games: dict[str, int], appearances: dict[str, int]) -> Mapping[str, EntityCounts]:
        return MappingProxyType(
            {
                entity_id: EntityCounts(
                    games=games[entity_id], seat_appearances=appearances[entity_id]
                )
                for entity_id in sorted(games)
            }
        )

    return result(agent_games, agent_appearances), result(deck_games, deck_appearances)


def _check_manifest_membership(rows: Iterable[GameRow], inclusion: Inclusion) -> None:
    manifest = inclusion.manifest
    if manifest is None:
        return
    reverse: dict[str, str] = {}
    for cluster_id, game_ids in manifest.items():
        for game_id in game_ids:
            if game_id in reverse:
                raise ManifestMismatchError(
                    f"game {game_id!r} is listed under both {reverse[game_id]!r} and {cluster_id!r}"
                )
            reverse[game_id] = cluster_id

    for row in rows:
        expected = reverse.get(row.game_id)
        if row.cluster_id is None:
            if expected is not None:
                raise ManifestMismatchError(
                    f"game {row.game_id!r} is listed in cluster {expected!r} "
                    "but has null cluster_id"
                )
        elif expected != row.cluster_id:
            if expected is None:
                raise ManifestMismatchError(
                    f"game {row.game_id!r} has cluster_id {row.cluster_id!r} "
                    "but is absent from the manifest"
                )
            raise ManifestMismatchError(
                f"game {row.game_id!r} has cluster_id {row.cluster_id!r} "
                f"but the manifest lists it under {expected!r}"
            )


def _apply_completeness(
    identity_rows: list[tuple[int, GameRow]],
    eligible: list[tuple[int, GameRow]],
    inclusion: Inclusion,
    excluded: dict[str, int],
) -> tuple[list[tuple[int, GameRow]], dict[str, str]]:
    manifest = inclusion.manifest
    if manifest is None:  # guarded by Inclusion, kept for the type checker
        raise InclusionError("complete_only requires a manifest")

    present: dict[str, dict[str, GameRow]] = {}
    for _, row in identity_rows:
        if row.cluster_id is not None:
            present.setdefault(row.cluster_id, {})[row.game_id] = row
    eligible_ids = {row.game_id for _, row in eligible}
    cluster_reasons: dict[str, str] = {}

    for cluster_id in sorted(manifest):
        expected = manifest[cluster_id]
        rows = present.get(cluster_id, {})
        missing = any(game_id not in rows for game_id in expected)
        filtered = any(game_id in rows and game_id not in eligible_ids for game_id in expected)
        failed = any(
            game_id in rows and game_id in eligible_ids and rows[game_id].outcome == "failed"
            for game_id in expected
        )
        if missing:
            reason = "missing"
            exclusion_reason = "cluster_missing"
        elif filtered:
            reason = "filtered"
            exclusion_reason = "cluster_filtered"
        elif failed:
            reason = "failed"
            exclusion_reason = "cluster_failed"
        else:
            continue
        cluster_reasons[cluster_id] = reason
        kept: list[tuple[int, GameRow]] = []
        for index, row in eligible:
            if row.cluster_id == cluster_id:
                excluded[exclusion_reason] += 1
            else:
                kept.append((index, row))
        eligible = kept

    kept = []
    for index, row in eligible:
        if row.cluster_id is None and row.outcome == "failed":
            excluded["cluster_failed"] += 1
        else:
            kept.append((index, row))
    eligible = kept
    return eligible, cluster_reasons


@overload
def run_pipeline(
    games: GameTable,
    agents: AgentRegistry,
    decks: DeckRegistry,
    *,
    inclusion: Inclusion,
    split_policy: Mapping[str, Collection[str]] | None = None,
    tag_schema: Mapping[str, object] | None = None,
    validate_only: Literal[True],
    strict: bool = True,
) -> ValidationReport: ...


@overload
def run_pipeline(
    games: GameTable,
    agents: AgentRegistry,
    decks: DeckRegistry,
    *,
    inclusion: Inclusion,
    split_policy: Mapping[str, Collection[str]] | None = None,
    tag_schema: Mapping[str, object] | None = None,
    validate_only: Literal[False] = False,
    strict: bool = True,
) -> PipelineResult: ...


def run_pipeline(
    games: GameTable,
    agents: AgentRegistry,
    decks: DeckRegistry,
    *,
    inclusion: Inclusion,
    split_policy: Mapping[str, Collection[str]] | None = None,
    tag_schema: Mapping[str, object] | None = None,
    validate_only: bool = False,
    strict: bool = True,
) -> PipelineResult | ValidationReport:
    """Run I0 steps 1–3 (preview) or 1–6 (full pipeline)."""

    split_policy_normalised = _normalise_split_policy(split_policy)
    tag_schema_normalised = _normalise_tag_schema(tag_schema)

    errors: list[ValidationIssue] = []
    warning_messages: list[str] = []
    received = len(games)

    def issue(error: Exception, row_index: int | None, game_id: str | None) -> None:
        errors.append(
            ValidationIssue(
                row_index=row_index,
                game_id=game_id,
                error_type=type(error).__name__,
                message=str(error),
            )
        )

    # I0 step 1 + 2: normalise once, deduplicating exact duplicates as we go.
    deduplicated: list[tuple[int, GameRow]] = []
    seen: dict[str, GameRow] = {}
    encoded_first: dict[str, str] = {}
    exact_duplicates = 0
    for index, record in enumerate(games):
        try:
            row = normalize_game_row(record)
        except ValidationError as exc:
            if strict:
                raise
            issue(exc, index, _record_game_id(record))
            continue
        previous = seen.get(row.game_id)
        if previous is None:
            seen[row.game_id] = row
            deduplicated.append((index, row))
            continue
        first_encoded = encoded_first.get(row.game_id)
        if first_encoded is None:
            first_encoded = canonical_row_json(previous)
            encoded_first[row.game_id] = first_encoded
        encoded = canonical_row_json(row)
        if first_encoded == encoded:
            exact_duplicates += 1
            message = f"game {row.game_id!r} was supplied more than once; duplicate removed"
            warning_messages.append(message)
            warnings.warn(message, DuplicateRowWarning, stacklevel=2)
        else:
            duplicate_error = DuplicateGameError(f"game_id {row.game_id!r} has conflicting rows")
            if strict:
                raise duplicate_error
            issue(duplicate_error, index, row.game_id)

    # I0 step 3: identity, mirror, split and tag policies.
    if len(agents.anchors) != 1:
        anchor_error = IdentityMismatchError(
            f"exactly one anchor is required; found {len(agents.anchors)}"
        )
        if strict:
            raise anchor_error
        issue(anchor_error, None, None)

    identity_rows: list[tuple[int, GameRow]] = []
    for index, row in deduplicated:
        try:
            _check_identity(
                row,
                agents,
                decks,
                inclusion,
                split_policy_normalised,
                tag_schema_normalised,
            )
        except (UnknownEntityError, ValidationError) as error:
            if strict:
                raise
            issue(error, index, row.game_id)
            continue
        identity_rows.append((index, row))

    if validate_only:
        rows = tuple(row for _, row in identity_rows)
        return ValidationReport(
            received=received,
            exact_duplicates=exact_duplicates,
            valid_rows=rows,
            data_hash=compute_data_hash(rows),
            errors=tuple(errors),
            warnings=tuple(warning_messages),
            agent_snapshot_id=agents.snapshot_id,
            deck_snapshot_id=decks.snapshot_id,
        )

    # Full path: complete-only manifest ownership is checked before I1.
    _check_manifest_membership((row for _, row in identity_rows), inclusion)

    # I0 step 4: row filters in I3 order, counting the first removal reason.
    excluded: dict[str, int] = {reason: 0 for reason in _EXCLUSION_REASONS}
    before_clusters: dict[str, int] = {}
    for _, row in identity_rows:
        if row.cluster_id is not None:
            before_clusters[row.cluster_id] = before_clusters.get(row.cluster_id, 0) + 1

    eligible: list[tuple[int, GameRow]] = []
    for index, row in identity_rows:
        reason = _first_filter_reason(row, inclusion, decks)
        if reason is None:
            eligible.append((index, row))
        else:
            excluded[reason] += 1

    after_clusters: dict[str, int] = {}
    for _, row in eligible:
        if row.cluster_id is not None:
            after_clusters[row.cluster_id] = after_clusters.get(row.cluster_id, 0) + 1
    partial_clusters = {
        cluster_id: before_clusters[cluster_id] - count
        for cluster_id, count in after_clusters.items()
        if 0 < count < before_clusters.get(cluster_id, 0)
    }

    # I0 step 5: completeness filter.
    cluster_reasons: dict[str, str] = {}
    if inclusion.clusters == "complete_only":
        eligible, cluster_reasons = _apply_completeness(
            identity_rows, eligible, inclusion, excluded
        )

    # I0 step 6: outcome selection and evidence support.
    eligible_rows = tuple(row for _, row in eligible)
    likelihood_rows = tuple(row for row in eligible_rows if row.outcome == "win")
    if not likelihood_rows:
        raise NoRankableEvidence("no likelihood-used games remain after inclusion")

    likelihood_bearing = {_cluster_key(row) for row in likelihood_rows}
    supported_agents = frozenset(agent for row in likelihood_rows for agent in row.agents)
    supported_decks = frozenset(deck for row in likelihood_rows for deck in row.decks)
    all_agents = frozenset(agents.by_id)
    all_decks = frozenset(decks.by_id)

    decisive: dict[str, list[bool]] = {agent: [] for agent in supported_agents}
    for row in likelihood_rows:
        winners = {row.agents[seat] for seat in range(4) if row.winner_seat == seat}
        for agent in set(row.agents):
            decisive[agent].append(agent in winners)
    saturated_agents = frozenset(
        agent
        for agent, outcomes in decisive.items()
        if outcomes and (all(outcomes) or not any(outcomes))
    )

    agent_counts, deck_counts = _entity_counts_pair(eligible_rows, agents.by_id, decks.by_id)
    counts = CountsReport(
        received=received,
        exact_duplicates=exact_duplicates,
        eligible=len(eligible_rows),
        excluded=MappingProxyType({reason: excluded[reason] for reason in _EXCLUSION_REASONS}),
        likelihood_used=len(likelihood_rows),
        likelihood_bearing_clusters=len(likelihood_bearing),
        agents=agent_counts,
        decks=deck_counts,
    )

    return PipelineResult(
        received=received,
        exact_duplicates=exact_duplicates,
        data_hash=compute_data_hash(tuple(row for _, row in identity_rows)),
        deduplicated_rows=tuple(row for _, row in identity_rows),
        eligible_rows=eligible_rows,
        likelihood_rows=likelihood_rows,
        counts=counts,
        partial_clusters=MappingProxyType(dict(sorted(partial_clusters.items()))),
        cluster_reasons=MappingProxyType(dict(sorted(cluster_reasons.items()))),
        pre_inclusion_rates=_outcome_rates(row for _, row in identity_rows),
        eligible_rates=_outcome_rates(eligible_rows),
        supported_agents=supported_agents,
        unrankable_agents=all_agents - supported_agents,
        saturated_agents=saturated_agents,
        supported_decks=supported_decks,
        unrankable_decks=all_decks - supported_decks,
        agent_snapshot_id=agents.snapshot_id,
        deck_snapshot_id=decks.snapshot_id,
        warnings=tuple(warning_messages),
    )


def _record_game_id(record: Mapping[str, object]) -> str | None:
    value = record.get("game_id")
    return value if isinstance(value, str) else None


__all__ = ["Inclusion", "run_pipeline"]
