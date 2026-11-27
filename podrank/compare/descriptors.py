"""Block descriptors declared before play (C1, C2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

from podrank.compare.pods import Design, PodDraw
from podrank.types import Role

Position = tuple[int, Role | None]


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _strings(value: object, name: str, length: int, *, distinct: bool = False) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    items = tuple(_string(item, name) for item in value)
    if len(items) != length:
        raise ValueError(f"{name} must have length {length}")
    if distinct and len(set(items)) != length:
        raise ValueError(f"{name} must contain {length} distinct IDs")
    return items


def _swaps(value: object) -> tuple[int, int, int, int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("swap_counts must be a sequence of four integers")
    items = tuple(_integer(item, "swap_counts") for item in value)
    if len(items) != 4:
        raise ValueError("swap_counts must have length 4")
    return items


def expected_positions(design: Design) -> tuple[Position, ...]:
    if design == "duplicate":
        return tuple((rotation, None) for rotation in range(4))
    return tuple((rotation, role) for rotation in range(4) for role in ("a", "b"))


@dataclass(frozen=True, slots=True)
class BlockDescriptor:
    """One declared block, bound to snapshots and a pod distribution."""

    block_id: str
    block_index: int
    design: Design
    deal_id: str
    decks: tuple[str, str, str, str]
    deck_versions: tuple[str, str, str, str]
    swap_counts: tuple[int, int, int, int]
    opponents: tuple[str, str, str] | None
    policy_versions: Mapping[str, str] = field(hash=False, compare=False)
    engine_version: str = ""
    pod_distribution_id: str = ""
    pod_distribution_version: int = 0
    pod_distribution_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_id", _string(self.block_id, "block_id"))
        object.__setattr__(self, "block_index", _integer(self.block_index, "block_index"))
        if self.design not in {"duplicate", "paired"}:
            raise ValueError("design must be 'duplicate' or 'paired'")
        object.__setattr__(self, "deal_id", _string(self.deal_id, "deal_id"))
        object.__setattr__(
            self,
            "decks",
            cast(tuple[str, str, str, str], _strings(self.decks, "decks", 4, distinct=True)),
        )
        object.__setattr__(
            self,
            "deck_versions",
            cast(tuple[str, str, str, str], _strings(self.deck_versions, "deck_versions", 4)),
        )
        object.__setattr__(self, "swap_counts", _swaps(self.swap_counts))
        if self.opponents is not None:
            object.__setattr__(
                self,
                "opponents",
                cast(tuple[str, str, str], _strings(self.opponents, "opponents", 3)),
            )
        if self.design == "duplicate" and self.opponents is not None:
            raise ValueError("duplicate blocks must have null opponents")
        if self.design == "paired" and self.opponents is None:
            raise ValueError("paired blocks must have three opponents")
        if not isinstance(self.policy_versions, Mapping):
            raise ValueError("policy_versions must be a mapping")
        versions: dict[str, str] = {}
        for agent_id, version in self.policy_versions.items():
            versions[_string(agent_id, "policy_versions key")] = _string(
                version, f"policy_versions[{agent_id!r}]"
            )
        if not versions:
            raise ValueError("policy_versions must not be empty")
        object.__setattr__(
            self, "policy_versions", MappingProxyType(dict(sorted(versions.items())))
        )
        object.__setattr__(self, "engine_version", _string(self.engine_version, "engine_version"))
        object.__setattr__(
            self, "pod_distribution_id", _string(self.pod_distribution_id, "pod_distribution_id")
        )
        object.__setattr__(
            self,
            "pod_distribution_version",
            _integer(self.pod_distribution_version, "pod_distribution_version"),
        )
        object.__setattr__(
            self,
            "pod_distribution_hash",
            _string(self.pod_distribution_hash, "pod_distribution_hash"),
        )

    @property
    def positions(self) -> tuple[Position, ...]:
        return expected_positions(self.design)

    @property
    def key(self) -> tuple[str, int]:
        return (self.block_id, self.block_index)

    @classmethod
    def from_draw(
        cls,
        draw: PodDraw,
        *,
        block_id: str,
        design: Design,
        engine_version: str,
        policy_versions: Mapping[str, str],
        deck_versions: Mapping[str, str],
    ) -> BlockDescriptor:
        if draw.opponents is not None and design == "duplicate":
            raise ValueError("duplicate blocks must have null opponents")
        if draw.opponents is None and design == "paired":
            raise ValueError("paired blocks must have three opponents")
        try:
            versions = tuple(deck_versions[deck] for deck in draw.decks)
        except KeyError as exc:
            raise ValueError(f"missing deck version for {exc.args[0]!r}") from exc
        return cls(
            block_id=block_id,
            block_index=draw.block_index,
            design=design,
            deal_id=draw.deal_id,
            decks=draw.decks,
            deck_versions=cast(tuple[str, str, str, str], versions),
            swap_counts=(0, 0, 0, 0),
            opponents=draw.opponents,
            policy_versions=policy_versions,
            engine_version=engine_version,
            pod_distribution_id=draw.pod_distribution_id,
            pod_distribution_version=draw.pod_distribution_version,
            pod_distribution_hash=draw.pod_distribution_hash,
        )


__all__ = ["BlockDescriptor", "Position", "expected_positions"]
