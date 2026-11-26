"""Reference fields: canonical contexts, materialisation and persistence."""

from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np

from podrank._version import __version__
from podrank.data._canonical import canonical_json, sha256_hex
from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.errors import FieldIntegrityError, NoEligiblePodsError

_CTX_FIELDS = frozenset({"decks", "swap_counts", "opponents", "focal_seat", "weight"})
_SPLITS = frozenset({"train", "validation", "holdout"})


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


@dataclass(frozen=True, slots=True)
class FieldContext:
    decks: tuple[str, str, str, str]
    swap_counts: tuple[int, int, int, int]
    opponents: tuple[str, str, str]
    focal_seat: int
    weight: float

    @property
    def key(
        self,
    ) -> tuple[
        int,
        tuple[str, str, str, str],
        tuple[int, int, int, int],
        tuple[str, str, str],
    ]:
        return (self.focal_seat, self.decks, self.swap_counts, self.opponents)


def _coerce_context(value: object) -> FieldContext:
    if isinstance(value, FieldContext):
        return value
    if isinstance(value, tuple) and len(value) == 2:
        inner, weight = value
        base = _coerce_context(inner)
        return FieldContext(
            decks=base.decks,
            swap_counts=base.swap_counts,
            opponents=base.opponents,
            focal_seat=base.focal_seat,
            weight=_positive_float(weight, "weight"),
        )
    if not isinstance(value, Mapping):
        raise ValueError("each field context must be a FieldContext, mapping or (context, weight)")
    unknown = set(value) - _CTX_FIELDS
    if unknown:
        raise ValueError(f"unknown field context keys: {sorted(unknown)}")
    missing = _CTX_FIELDS - set(value)
    if missing:
        raise ValueError(f"missing field context keys: {sorted(missing)}")

    decks_raw = value["decks"]
    opponents_raw = value["opponents"]
    swaps_raw = value["swap_counts"]
    if isinstance(decks_raw, (str, bytes)) or not isinstance(decks_raw, Sequence):
        raise ValueError("decks must be a sequence of four strings")
    if isinstance(opponents_raw, (str, bytes)) or not isinstance(opponents_raw, Sequence):
        raise ValueError("opponents must be a sequence of three strings")
    if isinstance(swaps_raw, (str, bytes)) or not isinstance(swaps_raw, Sequence):
        raise ValueError("swap_counts must be a sequence of four integers")
    decks = tuple(_string(item, "decks") for item in decks_raw)
    opponents = tuple(_string(item, "opponents") for item in opponents_raw)
    swaps = tuple(_integer(item, "swap_counts", minimum=0) for item in swaps_raw)
    if len(decks) != 4:
        raise ValueError("decks must have length 4")
    if len(opponents) != 3:
        raise ValueError("opponents must have length 3")
    if len(swaps) != 4:
        raise ValueError("swap_counts must have length 4")
    focal_seat = _integer(value["focal_seat"], "focal_seat", minimum=0)
    if focal_seat > 3:
        raise ValueError("focal_seat must be 0..3")
    return FieldContext(
        decks=decks,
        swap_counts=swaps,
        opponents=opponents,
        focal_seat=focal_seat,
        weight=_positive_float(value["weight"], "weight"),
    )


@dataclass(frozen=True, slots=True)
class Field:
    """Canonical finite weighted list of evaluation contexts."""

    field_id: str
    version: int
    contexts: tuple[FieldContext, ...]
    field_hash: str
    provenance: Mapping[str, object] = field(hash=False, compare=False)

    @classmethod
    def from_contexts(
        cls,
        field_id: str,
        version: int,
        contexts: Iterable[FieldContext | Mapping[str, object]],
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> Field:
        field_id = _string(field_id, "field_id")
        version = _integer(version, "version")
        coerced = [_coerce_context(context) for context in contexts]
        if not coerced:
            raise ValueError("a field must contain at least one context")

        grouped: dict[
            tuple[int, tuple[str, str, str, str], tuple[int, int, int, int], tuple[str, str, str]],
            list[float],
        ] = {}
        for context in coerced:
            grouped.setdefault(context.key, []).append(context.weight)
        keys = sorted(grouped)
        raw_weights = [math.fsum(grouped[key]) for key in keys]
        total = math.fsum(raw_weights)
        if not math.isfinite(total) or total <= 0:
            raise ValueError("field weights must have a positive finite sum")
        weights = [weight / total for weight in raw_weights]
        if abs(math.fsum(weights) - 1.0) > 1e-12:
            raise ValueError("canonical field weights do not sum to one within 1e-12")

        canonical = tuple(
            FieldContext(
                decks=key[1],
                swap_counts=key[2],
                opponents=key[3],
                focal_seat=key[0],
                weight=weight,
            )
            for key, weight in zip(keys, weights, strict=True)
        )
        payload = _payload(field_id, version, canonical)
        field_hash = sha256_hex(canonical_json(payload))
        if provenance is None:
            provenance = {
                "recipe": None,
                "deck_snapshot_id": None,
                "materialised_with": __version__,
            }
        return cls(
            field_id=field_id,
            version=version,
            contexts=canonical,
            field_hash=field_hash,
            provenance=MappingProxyType(dict(provenance)),
        )

    @classmethod
    def materialise(cls, recipe: Mapping[str, object], *, decks: DeckRegistry) -> Field:
        top = _mapping(recipe, "recipe")
        unknown = set(top) - {"field_id", "version", "recipe"}
        if unknown:
            raise ValueError(f"unknown recipe keys: {sorted(unknown)}")
        missing = {"field_id", "version", "recipe"} - set(top)
        if missing:
            raise ValueError(f"missing recipe keys: {sorted(missing)}")
        field_id = _string(top["field_id"], "field_id")
        version = _integer(top["version"], "version")
        inner = _mapping(top["recipe"], "recipe.recipe")
        unknown_inner = set(inner) - {
            "deck_split",
            "pod",
            "opponents",
            "focal_seat",
            "n_contexts",
            "seed",
        }
        if unknown_inner:
            raise ValueError(f"unknown recipe keys: {sorted(unknown_inner)}")
        missing_inner = {
            "deck_split",
            "pod",
            "opponents",
            "focal_seat",
            "n_contexts",
            "seed",
        } - set(inner)
        if missing_inner:
            raise ValueError(f"missing recipe keys: {sorted(missing_inner)}")

        split = _string(inner["deck_split"], "deck_split")
        if split not in _SPLITS:
            raise ValueError(f"deck_split must be one of {sorted(_SPLITS)}")
        pod = _mapping(inner["pod"], "pod")
        unknown_pod = set(pod) - {
            "bracket_policy",
            "bracket_weights",
            "min_decks_per_bracket",
            "distinct_decks",
            "swap_counts",
        }
        if unknown_pod:
            raise ValueError(f"unknown pod keys: {sorted(unknown_pod)}")
        bracket_policy = _string(pod.get("bracket_policy"), "bracket_policy")
        if bracket_policy not in {"same_bracket", "any"}:
            raise ValueError("bracket_policy must be 'same_bracket' or 'any'")
        if _string(pod.get("bracket_weights"), "bracket_weights") != "equal":
            raise ValueError("bracket_weights must be 'equal'")
        minimum = _integer(pod.get("min_decks_per_bracket", 1), "min_decks_per_bracket", minimum=1)
        if pod.get("distinct_decks") is not True:
            raise ValueError("distinct_decks must be true")
        swaps = _integer(pod.get("swap_counts", 0), "swap_counts", minimum=0)

        opponents_spec = _mapping(inner["opponents"], "opponents")
        unknown_opponents = set(opponents_spec) - {"pool", "draw"}
        if unknown_opponents:
            raise ValueError(f"unknown opponents keys: {sorted(unknown_opponents)}")
        if _string(opponents_spec.get("draw"), "opponents.draw") != "independent_uniform_per_seat":
            raise ValueError("opponents.draw must be 'independent_uniform_per_seat'")
        pool_raw = opponents_spec.get("pool")
        if isinstance(pool_raw, (str, bytes)) or not isinstance(pool_raw, Sequence):
            raise ValueError("opponents.pool must be a non-empty sequence of strings")
        pool = sorted(_string(item, "opponents.pool") for item in pool_raw)
        if not pool:
            raise ValueError("opponents.pool must be non-empty")
        if len(set(pool)) != len(pool):
            raise ValueError("opponents.pool must not contain duplicates")
        if _string(inner["focal_seat"], "focal_seat") != "uniform":
            raise ValueError("focal_seat must be 'uniform'")
        n_contexts = _integer(inner["n_contexts"], "n_contexts", minimum=1)
        seed = _integer(inner["seed"], "seed", minimum=0)

        by_bracket: dict[int, list[str]] = {}
        all_decks: list[str] = []
        for deck in decks:
            if deck.split == split:
                by_bracket.setdefault(deck.bracket, []).append(deck.deck_id)
                all_decks.append(deck.deck_id)
        for bracket_decks in by_bracket.values():
            bracket_decks.sort()
        all_decks.sort()

        effective_minimum = max(4, minimum)
        if bracket_policy == "same_bracket":
            eligible_brackets = sorted(
                bracket
                for bracket, values in by_bracket.items()
                if len(values) >= effective_minimum
            )
            if not eligible_brackets:
                counts = {bracket: len(values) for bracket, values in sorted(by_bracket.items())}
                raise NoEligiblePodsError(
                    f"no bracket has {effective_minimum} decks in split {split!r}: {counts}"
                )
            for bracket in sorted(by_bracket):
                if bracket not in eligible_brackets:
                    warnings.warn(
                        f"bracket {bracket!r} has fewer than {effective_minimum} decks "
                        f"in split {split!r} and is excluded",
                        UserWarning,
                        stacklevel=2,
                    )
        else:
            if len(all_decks) < effective_minimum:
                raise NoEligiblePodsError(
                    f"split {split!r} has {len(all_decks)} decks; {effective_minimum} required"
                )

        contexts: list[FieldContext] = []
        for index in range(n_contexts):
            rng = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(index,)))
            )
            if bracket_policy == "same_bracket":
                bracket = eligible_brackets[int(rng.integers(0, len(eligible_brackets)))]
                deck_pool = by_bracket[bracket]
            else:
                deck_pool = all_decks
            deck_indices = rng.choice(len(deck_pool), size=4, replace=False)
            chosen_decks = tuple(deck_pool[int(position)] for position in deck_indices)
            focal_seat = int(rng.integers(0, 4))
            opponent_indices = rng.integers(0, len(pool), size=3)
            chosen_opponents = tuple(pool[int(position)] for position in opponent_indices)
            contexts.append(
                FieldContext(
                    decks=cast(tuple[str, str, str, str], chosen_decks),
                    swap_counts=(swaps, swaps, swaps, swaps),
                    opponents=cast(tuple[str, str, str], chosen_opponents),
                    focal_seat=focal_seat,
                    weight=1.0,
                )
            )

        provenance = {
            "recipe": dict(recipe),
            "deck_snapshot_id": decks.snapshot_id,
            "materialised_with": __version__,
        }
        return cls.from_contexts(field_id, version, contexts, provenance=provenance)

    def payload(self) -> dict[str, object]:
        return _payload(self.field_id, self.version, self.contexts)

    def save(self, path: str | Path) -> None:
        envelope = {
            "payload": self.payload(),
            "field_hash": self.field_hash,
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
    def load(cls, path: str | Path) -> Field:
        try:
            return cls._load_unchecked(path)
        except FieldIntegrityError:
            raise
        except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
            raise FieldIntegrityError(f"malformed field payload: {exc}") from exc

    @classmethod
    def _load_unchecked(cls, path: str | Path) -> Field:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise FieldIntegrityError(f"could not read field file: {exc}") from exc
        try:
            envelope = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except (json.JSONDecodeError, FieldIntegrityError) as exc:
            raise FieldIntegrityError(f"invalid field JSON: {exc}") from exc
        if not isinstance(envelope, dict) or set(envelope) != {
            "payload",
            "field_hash",
            "provenance",
        }:
            raise FieldIntegrityError(
                "field envelope must have exactly payload, field_hash, provenance"
            )
        payload = envelope["payload"]
        if not isinstance(payload, dict) or set(payload) != {"contexts", "field_id", "version"}:
            raise FieldIntegrityError("invalid field payload schema")
        field_id = _string(payload["field_id"], "field_id")
        version = _integer(payload["version"], "version")
        raw_contexts = payload["contexts"]
        if isinstance(raw_contexts, (str, bytes)) or not isinstance(raw_contexts, list):
            raise FieldIntegrityError("field contexts must be a list")
        if not raw_contexts:
            raise FieldIntegrityError("field contexts must not be empty")
        contexts: list[FieldContext] = []
        for raw in raw_contexts:
            if not isinstance(raw, dict) or set(raw) != _CTX_FIELDS:
                raise FieldIntegrityError("invalid field context schema")
            try:
                context = _coerce_context(raw)
            except ValueError as exc:
                raise FieldIntegrityError(f"invalid field context: {exc}") from exc
            contexts.append(context)
        keys = [context.key for context in contexts]
        if keys != sorted(keys):
            raise FieldIntegrityError("field contexts are not in canonical order")
        if len(set(keys)) != len(keys):
            raise FieldIntegrityError("field contexts must be unique")
        weights = [context.weight for context in contexts]
        if abs(math.fsum(weights) - 1.0) > 1e-12:
            raise FieldIntegrityError("field weights do not sum to one within 1e-12")
        canonical_payload = _payload(field_id, version, tuple(contexts))
        expected_hash = sha256_hex(canonical_json(canonical_payload))
        stored_hash = envelope["field_hash"]
        if not isinstance(stored_hash, str) or stored_hash != expected_hash:
            raise FieldIntegrityError("field payload hash does not match field_hash")
        provenance = envelope["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != {
            "recipe",
            "deck_snapshot_id",
            "materialised_with",
        }:
            raise FieldIntegrityError("invalid field provenance schema")
        return cls(
            field_id=field_id,
            version=version,
            contexts=tuple(contexts),
            field_hash=stored_hash,
            provenance=MappingProxyType(provenance),
        )

    def context_version_fingerprint(self, agents: AgentRegistry, decks: DeckRegistry) -> str:
        agent_ids = sorted(
            {opponent for context in self.contexts for opponent in context.opponents}
        )
        deck_ids = sorted({deck for context in self.contexts for deck in context.decks})
        payload = {
            "agents": [[agent_id, agents.by_id[agent_id].policy_version] for agent_id in agent_ids],
            "decks": [[deck_id, decks.by_id[deck_id].deck_version] for deck_id in deck_ids],
        }
        return sha256_hex(canonical_json(payload))

    def __len__(self) -> int:
        return len(self.contexts)

    def __iter__(self) -> Iterator[FieldContext]:
        return iter(self.contexts)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _payload(field_id: str, version: int, contexts: Sequence[FieldContext]) -> dict[str, object]:
    return {
        "contexts": [
            {
                "decks": list(context.decks),
                "focal_seat": context.focal_seat,
                "opponents": list(context.opponents),
                "swap_counts": list(context.swap_counts),
                "weight": context.weight,
            }
            for context in contexts
        ],
        "field_id": field_id,
        "version": version,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FieldIntegrityError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


__all__ = ["Field", "FieldContext", "FieldIntegrityError", "NoEligiblePodsError"]
