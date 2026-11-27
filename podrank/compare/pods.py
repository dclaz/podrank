"""Frozen pod distributions and their sampling stream (C3, F23)."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import numpy as np

from podrank._version import __version__
from podrank.data._canonical import canonical_json, sha256_hex
from podrank.errors import PersistenceError

Design = Literal["duplicate", "paired"]
_DESIGNS = frozenset({"duplicate", "paired"})
_SUPPORT_FIELDS = frozenset({"decks", "opponents", "weight"})
_PAYLOAD_FIELDS = frozenset({"design", "pod_distribution_id", "version", "support"})
_ENVELOPE_FIELDS = frozenset({"payload", "pod_distribution_hash", "provenance"})
_PROVENANCE_FIELDS = frozenset({"recipe", "deck_snapshot_id", "materialised_with"})


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _weight(value: object, name: str = "weight") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return numeric


def _strings4(value: object, name: str, *, distinct: bool) -> tuple[str, str, str, str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of four strings")
    items = tuple(_string(item, name) for item in value)
    if len(items) != 4:
        raise ValueError(f"{name} must have length 4")
    if distinct and len(set(items)) != 4:
        raise ValueError(f"{name} must contain four distinct IDs")
    return items


def _opponents(value: object) -> tuple[str, str, str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError("opponents must be null or a sequence of three strings")
    items = tuple(_string(item, "opponents") for item in value)
    if len(items) != 3:
        raise ValueError("opponents must have length 3")
    return items


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PersistenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class PodSupport:
    """One canonical pod support point."""

    decks: tuple[str, str, str, str]
    opponents: tuple[str, str, str] | None
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "decks", _strings4(self.decks, "decks", distinct=True))
        object.__setattr__(self, "opponents", _opponents(self.opponents))
        object.__setattr__(self, "weight", _weight(self.weight))

    @property
    def key(self) -> tuple[tuple[str, str, str, str], tuple[str, ...]]:
        return (self.decks, self.opponents or ())

    def payload(self) -> dict[str, object]:
        return {
            "decks": list(self.decks),
            "opponents": None if self.opponents is None else list(self.opponents),
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class PodDraw:
    """One sampled pod, before the caller binds game identity."""

    block_index: int
    deal_id: str
    decks: tuple[str, str, str, str]
    opponents: tuple[str, str, str] | None
    pod_distribution_id: str
    pod_distribution_version: int
    pod_distribution_hash: str


@dataclass(frozen=True, slots=True)
class PodDistribution:
    """A canonical finite support with a payload-only hash."""

    pod_distribution_id: str
    version: int
    design: Design
    support: tuple[PodSupport, ...]
    pod_distribution_hash: str
    provenance: Mapping[str, object] = field(hash=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "pod_distribution_id", _string(self.pod_distribution_id, "pod_distribution_id")
        )
        object.__setattr__(self, "version", _integer(self.version, "version"))
        if self.design not in _DESIGNS:
            raise ValueError(f"design must be one of {sorted(_DESIGNS)}")
        if not self.support:
            raise ValueError("support must not be empty")
        keys = [item.key for item in self.support]
        if len(set(keys)) != len(keys):
            raise ValueError("support keys must be unique")
        if keys != sorted(keys):
            raise ValueError("support must be in canonical order")
        if self.design == "duplicate":
            if any(item.opponents is not None for item in self.support):
                raise ValueError("duplicate distributions must have null opponents")
        elif any(item.opponents is None for item in self.support):
            raise ValueError("paired distributions must have three opponents")
        weights = [item.weight for item in self.support]
        if abs(math.fsum(weights) - 1.0) > 1e-12:
            raise ValueError("support weights must sum to one within 1e-12")
        object.__setattr__(
            self,
            "pod_distribution_hash",
            _string(self.pod_distribution_hash, "pod_distribution_hash"),
        )
        if set(self.provenance) != _PROVENANCE_FIELDS:
            raise ValueError(
                "provenance must have exactly recipe, deck_snapshot_id, materialised_with"
            )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @classmethod
    def from_support(
        cls,
        pod_distribution_id: str,
        version: int,
        design: Design,
        support: Iterable[PodSupport | Mapping[str, object]],
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> PodDistribution:
        if design not in _DESIGNS:
            raise ValueError(f"design must be one of {sorted(_DESIGNS)}")
        parsed: list[PodSupport] = []
        for item in support:
            if isinstance(item, PodSupport):
                parsed.append(item)
            else:
                if not isinstance(item, Mapping):
                    raise ValueError("each support item must be a PodSupport or mapping")
                if set(item) != _SUPPORT_FIELDS:
                    raise ValueError("support items must have exactly decks, opponents, weight")
                parsed.append(
                    PodSupport(
                        decks=cast(tuple[str, str, str, str], item["decks"]),
                        opponents=cast(tuple[str, str, str] | None, item["opponents"]),
                        weight=cast(float, item["weight"]),
                    )
                )
        if not parsed:
            raise ValueError("support must not be empty")
        for item in parsed:
            if design == "duplicate" and item.opponents is not None:
                raise ValueError("duplicate distributions must have null opponents")
            if design == "paired" and item.opponents is None:
                raise ValueError("paired distributions must have three opponents")

        grouped: dict[tuple[tuple[str, str, str, str], tuple[str, ...]], list[float]] = {}
        for item in parsed:
            grouped.setdefault(item.key, []).append(item.weight)
        keys = sorted(grouped)
        raw_weights = [math.fsum(grouped[key]) for key in keys]
        total = math.fsum(raw_weights)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("support weights must have a positive finite sum")
        normalised = [weight / total for weight in raw_weights]
        if abs(math.fsum(normalised) - 1.0) > 1e-12:
            raise ValueError("canonical support weights do not sum to one within 1e-12")

        canonical: list[PodSupport] = []
        for key, weight in zip(keys, normalised, strict=True):
            decks, opponents = key
            canonical.append(
                PodSupport(
                    decks=decks,
                    opponents=cast(tuple[str, str, str] | None, opponents or None),
                    weight=weight,
                )
            )
        resolved_provenance: Mapping[str, object] = (
            {
                "recipe": None,
                "deck_snapshot_id": None,
                "materialised_with": __version__,
            }
            if provenance is None
            else provenance
        )
        if set(resolved_provenance) != _PROVENANCE_FIELDS:
            raise ValueError(
                "provenance must have exactly recipe, deck_snapshot_id, materialised_with"
            )
        payload = {
            "design": design,
            "pod_distribution_id": pod_distribution_id,
            "version": version,
            "support": [item.payload() for item in canonical],
        }
        return cls(
            pod_distribution_id=pod_distribution_id,
            version=version,
            design=design,
            support=tuple(canonical),
            pod_distribution_hash=sha256_hex(canonical_json(payload)),
            provenance=resolved_provenance,
        )

    def payload(self) -> dict[str, object]:
        return {
            "design": self.design,
            "pod_distribution_id": self.pod_distribution_id,
            "version": self.version,
            "support": [item.payload() for item in self.support],
        }

    def weights(self) -> np.ndarray:
        return np.asarray([item.weight for item in self.support], dtype=np.float64)

    def sample(self, n_blocks: int, seed: int, start_index: int = 0) -> tuple[PodDraw, ...]:
        if isinstance(n_blocks, bool) or not isinstance(n_blocks, int) or n_blocks <= 0:
            raise ValueError("n_blocks must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
            raise ValueError("start_index must be a non-negative integer")
        weights = self.weights()
        normalised = weights / float(weights.sum())
        draws: list[PodDraw] = []
        for offset in range(n_blocks):
            index = start_index + offset
            rng = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(index,)))
            )
            chosen = int(rng.choice(len(self.support), size=None, p=normalised))
            item = self.support[chosen]
            draws.append(
                PodDraw(
                    block_index=index,
                    deal_id=f"{self.pod_distribution_hash[:12]}:{seed}:{index}",
                    decks=item.decks,
                    opponents=item.opponents,
                    pod_distribution_id=self.pod_distribution_id,
                    pod_distribution_version=self.version,
                    pod_distribution_hash=self.pod_distribution_hash,
                )
            )
        return tuple(draws)

    def save(self, path: str | Path) -> None:
        envelope = {
            "payload": self.payload(),
            "pod_distribution_hash": self.pod_distribution_hash,
            "provenance": dict(self.provenance),
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
    def load(cls, path: str | Path) -> PodDistribution:
        try:
            return cls._load_unchecked(path)
        except PersistenceError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            raise PersistenceError(f"malformed pod distribution: {exc}") from exc

    @classmethod
    def _load_unchecked(cls, path: str | Path) -> PodDistribution:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise PersistenceError(f"could not read pod distribution: {exc}") from exc
        try:
            envelope = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, PersistenceError) as exc:
            raise PersistenceError(f"invalid pod distribution JSON: {exc}") from exc
        if not isinstance(envelope, dict) or set(envelope) != _ENVELOPE_FIELDS:
            raise PersistenceError(
                "pod distribution envelope must have exactly payload, "
                "pod_distribution_hash, provenance"
            )
        payload = envelope["payload"]
        if not isinstance(payload, dict) or set(payload) != _PAYLOAD_FIELDS:
            raise PersistenceError("invalid pod distribution payload schema")
        design = payload["design"]
        if design not in _DESIGNS:
            raise PersistenceError("invalid pod distribution design")
        raw_support = payload["support"]
        if isinstance(raw_support, (str, bytes)) or not isinstance(raw_support, list):
            raise PersistenceError("pod distribution support must be a list")
        if not raw_support:
            raise PersistenceError("pod distribution support must not be empty")
        support: list[PodSupport] = []
        for raw in raw_support:
            if not isinstance(raw, dict) or set(raw) != _SUPPORT_FIELDS:
                raise PersistenceError("invalid pod support schema")
            try:
                support.append(
                    PodSupport(
                        decks=cast(tuple[str, str, str, str], raw["decks"]),
                        opponents=cast(tuple[str, str, str] | None, raw["opponents"]),
                        weight=cast(float, raw["weight"]),
                    )
                )
            except ValueError as exc:
                raise PersistenceError(f"invalid pod support: {exc}") from exc
        keys = [item.key for item in support]
        if keys != sorted(keys):
            raise PersistenceError("pod support is not in canonical order")
        if len(set(keys)) != len(keys):
            raise PersistenceError("pod support keys must be unique")
        weights = [item.weight for item in support]
        if abs(math.fsum(weights) - 1.0) > 1e-12:
            raise PersistenceError("pod support weights do not sum to one within 1e-12")
        canonical_payload = {
            "design": design,
            "pod_distribution_id": payload["pod_distribution_id"],
            "version": payload["version"],
            "support": [item.payload() for item in support],
        }
        expected_hash = sha256_hex(canonical_json(canonical_payload))
        stored_hash = envelope["pod_distribution_hash"]
        if not isinstance(stored_hash, str) or stored_hash != expected_hash:
            raise PersistenceError("pod distribution hash does not match the payload")
        provenance = envelope["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
            raise PersistenceError("invalid pod distribution provenance schema")
        return cls(
            pod_distribution_id=cast(str, payload["pod_distribution_id"]),
            version=cast(int, payload["version"]),
            design=cast(Design, design),
            support=tuple(support),
            pod_distribution_hash=stored_hash,
            provenance=provenance,
        )


__all__ = ["Design", "PodDistribution", "PodDraw", "PodSupport"]
