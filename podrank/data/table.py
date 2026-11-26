"""Raw game table, row normalisation and canonical data hashing.

``GameTable`` deliberately preserves the supplied row order and does not
deduplicate or validate rows at construction (Appendix C and §3.5). I0 step 1
normalises rows into :class:`GameRow`; I0 step 2 removes exact duplicates.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import cast

from podrank.data._canonical import canonical_json, json_value, sha256_hex
from podrank.errors import ValidationError
from podrank.types import GameRow, Outcome, Role

_ALLOWED_FIELDS = frozenset(
    {
        "game_id",
        "played_at",
        "agents",
        "decks",
        "policy_versions",
        "deck_versions",
        "swap_counts",
        "outcome",
        "winner_seat",
        "suite",
        "cluster_id",
        "block_id",
        "rotation",
        "role",
        "deal_id",
        "spec_hash",
        "engine_version",
        "audit",
    }
)
_REQUIRED_FIELDS = frozenset(
    {"game_id", "played_at", "agents", "decks", "policy_versions", "outcome"}
)
_OUTCOMES = frozenset({"win", "draw", "cap", "failed"})
_ROLES = frozenset({"a", "b"})
_RFC3339 = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})[Tt]"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d+))?"
    r"(?P<offset>[Zz]|[+-]\d{2}:\d{2})$"
)


def _string(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string")
    if not allow_empty and not value:
        raise ValidationError(f"{field_name} must not be empty")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name)


def _sequence4(value: object, field_name: str) -> list[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValidationError(f"{field_name} must be a sequence of length 4")
    items = list(value)
    if len(items) != 4:
        raise ValidationError(f"{field_name} must have length 4, got {len(items)}")
    return items


def _integer(value: object, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValidationError(f"{field_name} must be >= {minimum}")
    return value


def _outcome(value: object) -> Outcome:
    if not isinstance(value, str) or value not in _OUTCOMES:
        raise ValidationError(f"outcome must be one of {sorted(_OUTCOMES)}")
    return cast(Outcome, value)


def _role(value: object) -> Role:
    if not isinstance(value, str) or value not in _ROLES:
        raise ValidationError(f"role must be one of {sorted(_ROLES)}")
    return cast(Role, value)


def _audit(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValidationError("audit must be a JSON object or null")
    cloned = json_value(value, field_name="audit")
    if not isinstance(cloned, dict):
        raise ValidationError("audit must be a JSON object or null")
    return MappingProxyType(cloned)


def _freeze_raw(record: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(record, Mapping):
        raise ValidationError("each game record must be a mapping")
    for key in record:
        if not isinstance(key, str):
            raise ValidationError("game record keys must be strings")
    try:
        copied = copy.deepcopy(dict(record))
    except Exception as exc:  # pragma: no cover - defensive for exotic inputs
        raise ValidationError(f"game record could not be copied: {exc}") from exc
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class GameTable:
    """Immutable, order-preserving container of raw game records."""

    rows: tuple[Mapping[str, object], ...]

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> GameTable:
        return cls(tuple(_freeze_raw(record) for record in records))

    def to_records(self) -> list[dict[str, object]]:
        return [copy.deepcopy(dict(row)) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        return iter(self.rows)


def normalize_timestamp(value: object, field_name: str = "played_at") -> datetime:
    """Parse a timezone-aware datetime or RFC 3339 string and return UTC."""

    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValidationError(f"{field_name} must be timezone-aware")
        return value.astimezone(UTC)
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be an aware datetime or RFC 3339 string")
    match = _RFC3339.fullmatch(value)
    if match is None:
        raise ValidationError(f"{field_name} is not an RFC 3339 timestamp with an explicit offset")
    fraction = match.group("fraction") or ""
    if len(fraction) > 6 and any(digit != "0" for digit in fraction[6:]):
        raise ValidationError(f"{field_name} has unsupported sub-microsecond precision")
    micros = fraction[:6].ljust(6, "0")
    offset = match.group("offset")
    suffix = "+00:00" if offset in {"Z", "z"} else offset
    normalised = f"{match.group('date')}T{match.group('time')}.{micros}{suffix}"
    try:
        parsed = datetime.fromisoformat(normalised)
    except ValueError as exc:
        raise ValidationError(f"{field_name} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValidationError(f"{field_name} must include an explicit offset")
    return parsed.astimezone(UTC)


def format_timestamp(value: datetime) -> str:
    """Format a UTC timestamp exactly as required by C.2."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError("played_at must be timezone-aware")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def normalize_game_row(record: Mapping[str, object]) -> GameRow:
    """Apply I0 step 1 to one raw record."""

    keys = set(record)
    missing = _REQUIRED_FIELDS - keys
    if missing:
        raise ValidationError(f"missing required game fields: {sorted(missing)}")
    unknown = keys - _ALLOWED_FIELDS
    if unknown:
        raise ValidationError(f"unknown game fields: {sorted(unknown)}")

    game_id = _string(record["game_id"], "game_id")
    played_at = normalize_timestamp(record["played_at"])

    agent_items = _sequence4(record["agents"], "agents")
    deck_items = _sequence4(record["decks"], "decks")
    policy_items = _sequence4(record["policy_versions"], "policy_versions")
    agents = tuple(_string(item, "agents") for item in agent_items)
    decks = tuple(_string(item, "decks") for item in deck_items)
    policy_versions = tuple(_string(item, "policy_versions") for item in policy_items)

    if "deck_versions" in record and record["deck_versions"] is not None:
        deck_version_items = _sequence4(record["deck_versions"], "deck_versions")
        deck_versions = tuple(
            None if item is None else _string(item, "deck_versions") for item in deck_version_items
        )
    else:
        deck_versions = (None, None, None, None)

    if "swap_counts" in record:
        swap_items = _sequence4(record["swap_counts"], "swap_counts")
        swap_counts = tuple(_integer(item, "swap_counts", minimum=0) for item in swap_items)
    else:
        swap_counts = (0, 0, 0, 0)

    outcome = _outcome(record["outcome"])
    raw_winner = record.get("winner_seat")
    if outcome == "win":
        if raw_winner is None:
            raise ValidationError("winner_seat is required when outcome is 'win'")
        parsed_winner = _integer(raw_winner, "winner_seat", minimum=0)
        if parsed_winner > 3:
            raise ValidationError("winner_seat must be 0..3 when outcome is 'win'")
        winner_seat: int | None = parsed_winner
    else:
        if raw_winner is not None:
            raise ValidationError("winner_seat must be null unless outcome is 'win'")
        winner_seat = None

    suite = _string(record.get("suite", ""), "suite", allow_empty=True)
    cluster_id = _optional_string(record.get("cluster_id"), "cluster_id")
    block_id = _optional_string(record.get("block_id"), "block_id")
    deal_id = _optional_string(record.get("deal_id"), "deal_id")
    spec_hash = _optional_string(record.get("spec_hash"), "spec_hash")

    raw_rotation = record.get("rotation")
    rotation = None if raw_rotation is None else _integer(raw_rotation, "rotation")

    raw_role = record.get("role")
    role = None if raw_role is None else _role(raw_role)

    engine_version = _string(record.get("engine_version", ""), "engine_version", allow_empty=True)

    return GameRow(
        game_id=game_id,
        played_at=played_at,
        agents=cast(tuple[str, str, str, str], agents),
        decks=cast(tuple[str, str, str, str], decks),
        policy_versions=cast(tuple[str, str, str, str], policy_versions),
        deck_versions=cast(tuple[str | None, str | None, str | None, str | None], deck_versions),
        swap_counts=cast(tuple[int, int, int, int], swap_counts),
        outcome=outcome,
        winner_seat=winner_seat,
        suite=suite,
        cluster_id=cluster_id,
        block_id=block_id,
        rotation=rotation,
        role=role,
        deal_id=deal_id,
        spec_hash=spec_hash,
        engine_version=engine_version,
        audit=_audit(record.get("audit")),
    )


def canonical_row_payload(row: GameRow) -> dict[str, object]:
    """Return the complete normalised §3.2 row object used for hashing."""

    return {
        "game_id": row.game_id,
        "played_at": format_timestamp(row.played_at),
        "agents": list(row.agents),
        "decks": list(row.decks),
        "policy_versions": list(row.policy_versions),
        "deck_versions": list(row.deck_versions),
        "swap_counts": list(row.swap_counts),
        "outcome": row.outcome,
        "winner_seat": row.winner_seat,
        "suite": row.suite,
        "cluster_id": row.cluster_id,
        "block_id": row.block_id,
        "rotation": row.rotation,
        "role": row.role,
        "deal_id": row.deal_id,
        "spec_hash": row.spec_hash,
        "engine_version": row.engine_version,
        "audit": None if row.audit is None else dict(row.audit),
    }


def canonical_row_json(row: GameRow) -> str:
    """Canonical JSON for one row, used for exact-duplicate equality."""

    return canonical_json(canonical_row_payload(row))


def compute_data_hash(rows: Sequence[GameRow]) -> str:
    """Hash the canonical rows after deduplication, sorted by ``game_id``."""

    ordered = sorted(rows, key=lambda row: row.game_id)
    return sha256_hex(canonical_json([canonical_row_payload(row) for row in ordered]))


__all__ = [
    "GameTable",
    "canonical_row_json",
    "canonical_row_payload",
    "compute_data_hash",
    "format_timestamp",
    "normalize_game_row",
    "normalize_timestamp",
]
