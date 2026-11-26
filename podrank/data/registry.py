"""Immutable, content-hashed agent and deck registry snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from podrank.data._canonical import canonical_json, json_value, sha256_hex
from podrank.errors import ValidationError
from podrank.types import AgentKind, AgentRecord, DeckRecord, DeckSplit

_AGENT_FIELDS = frozenset({"agent_id", "kind", "policy_version", "is_anchor", "tags", "metadata"})
_AGENT_REQUIRED = frozenset({"agent_id", "kind", "policy_version", "is_anchor"})
_DECK_FIELDS = frozenset({"deck_id", "bracket", "split", "deck_version", "tags", "metadata"})
_DECK_REQUIRED = frozenset({"deck_id", "bracket", "split", "deck_version"})
_AGENT_KINDS = frozenset({"random", "heuristic", "checkpoint", "exploiter", "other"})
_DECK_SPLITS = frozenset({"train", "validation", "holdout"})


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise ValidationError(f"{field_name} keys must be strings")
    return cast(Mapping[str, object], value)


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field_name} must be a non-empty string")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{field_name} must be a boolean")
    return value


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{field_name} must be an integer")
    return value


def _json_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    cloned = json_value(_mapping(value, field_name), field_name=field_name)
    if not isinstance(cloned, dict):
        raise ValidationError(f"{field_name} must be a JSON object")
    return MappingProxyType(cloned)


def _tags(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValidationError(f"{field_name} must be a sequence of strings")
    tags = {_string(item, field_name) for item in value}
    return tuple(sorted(tags))


def _deck_tags(value: object, field_name: str) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    mapping = _mapping(value, field_name)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        result[key] = _string(item, f"{field_name}[{key!r}]")
        if not key:
            raise ValidationError(f"{field_name} keys must be non-empty")
    return MappingProxyType(dict(sorted(result.items())))


@dataclass(frozen=True, slots=True)
class AgentRegistry:
    """Immutable snapshot of agent rows with a content hash."""

    rows: tuple[AgentRecord, ...]
    by_id: Mapping[str, AgentRecord] = field(repr=False, compare=False, hash=False)
    snapshot_id: str

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> AgentRegistry:
        parsed: list[AgentRecord] = []
        seen: set[str] = set()
        for record in records:
            mapping = _mapping(record, "agent record")
            missing = _AGENT_REQUIRED - set(mapping)
            if missing:
                raise ValidationError(f"missing required agent fields: {sorted(missing)}")
            unknown = set(mapping) - _AGENT_FIELDS
            if unknown:
                raise ValidationError(f"unknown agent fields: {sorted(unknown)}")
            agent_id = _string(mapping["agent_id"], "agent_id")
            if agent_id in seen:
                raise ValidationError(f"duplicate agent_id: {agent_id!r}")
            seen.add(agent_id)
            kind = _string(mapping["kind"], "kind")
            if kind not in _AGENT_KINDS:
                raise ValidationError(f"kind must be one of {sorted(_AGENT_KINDS)}")
            parsed.append(
                AgentRecord(
                    agent_id=agent_id,
                    kind=cast(AgentKind, kind),
                    policy_version=_string(mapping["policy_version"], "policy_version"),
                    is_anchor=_boolean(mapping["is_anchor"], "is_anchor"),
                    tags=_tags(mapping.get("tags"), "tags"),
                    metadata=_json_mapping(mapping.get("metadata"), "metadata"),
                )
            )
        rows = tuple(sorted(parsed, key=lambda row: row.agent_id))
        payload = [
            {
                "agent_id": row.agent_id,
                "kind": row.kind,
                "policy_version": row.policy_version,
                "is_anchor": row.is_anchor,
                "tags": list(row.tags),
                "metadata": dict(row.metadata),
            }
            for row in rows
        ]
        return cls(
            rows=rows,
            by_id=MappingProxyType({row.agent_id: row for row in rows}),
            snapshot_id=sha256_hex(canonical_json(payload)),
        )

    @property
    def anchors(self) -> tuple[AgentRecord, ...]:
        return tuple(row for row in self.rows if row.is_anchor)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[AgentRecord]:
        return iter(self.rows)

    def __getitem__(self, agent_id: str) -> AgentRecord:
        return self.by_id[agent_id]


@dataclass(frozen=True, slots=True)
class DeckRegistry:
    """Immutable snapshot of deck rows with a content hash."""

    rows: tuple[DeckRecord, ...]
    by_id: Mapping[str, DeckRecord] = field(repr=False, compare=False, hash=False)
    snapshot_id: str

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, object]]) -> DeckRegistry:
        parsed: list[DeckRecord] = []
        seen: set[str] = set()
        for record in records:
            mapping = _mapping(record, "deck record")
            missing = _DECK_REQUIRED - set(mapping)
            if missing:
                raise ValidationError(f"missing required deck fields: {sorted(missing)}")
            unknown = set(mapping) - _DECK_FIELDS
            if unknown:
                raise ValidationError(f"unknown deck fields: {sorted(unknown)}")
            deck_id = _string(mapping["deck_id"], "deck_id")
            if deck_id in seen:
                raise ValidationError(f"duplicate deck_id: {deck_id!r}")
            seen.add(deck_id)
            split = _string(mapping["split"], "split")
            if split not in _DECK_SPLITS:
                raise ValidationError(f"split must be one of {sorted(_DECK_SPLITS)}")
            parsed.append(
                DeckRecord(
                    deck_id=deck_id,
                    bracket=_integer(mapping["bracket"], "bracket"),
                    split=cast(DeckSplit, split),
                    deck_version=_string(mapping["deck_version"], "deck_version"),
                    tags=_deck_tags(mapping.get("tags"), "tags"),
                    metadata=_json_mapping(mapping.get("metadata"), "metadata"),
                )
            )
        rows = tuple(sorted(parsed, key=lambda row: row.deck_id))
        payload = [
            {
                "deck_id": row.deck_id,
                "bracket": row.bracket,
                "split": row.split,
                "deck_version": row.deck_version,
                "tags": dict(row.tags),
                "metadata": dict(row.metadata),
            }
            for row in rows
        ]
        return cls(
            rows=rows,
            by_id=MappingProxyType({row.deck_id: row for row in rows}),
            snapshot_id=sha256_hex(canonical_json(payload)),
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[DeckRecord]:
        return iter(self.rows)

    def __getitem__(self, deck_id: str) -> DeckRecord:
        return self.by_id[deck_id]


__all__ = ["AgentRegistry", "DeckRegistry"]
