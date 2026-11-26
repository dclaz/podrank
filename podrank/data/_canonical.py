"""Small canonical-JSON helpers shared by data snapshots and the game table.

The encoding is the one pinned by IMPLEMENTATION_PLAN.md FD3: UTF-8 JSON with
sorted keys, compact separators, ASCII escapes and no non-finite numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

from podrank.errors import ValidationError


def canonical_json(value: object) -> str:
    """Serialise ``value`` using the FD3 canonical JSON rules."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_hex(text: str) -> str:
    """Return the lowercase SHA-256 hex digest of the UTF-8 encoding of ``text``."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_value(value: object, *, field_name: str) -> object:
    """Validate and clone a JSON-compatible value, rejecting non-finite floats."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"{field_name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"{field_name} has a non-string key")
            result[key] = json_value(item, field_name=field_name)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_value(item, field_name=field_name) for item in value]
    raise ValidationError(f"{field_name} contains a non-JSON value")


__all__ = ["canonical_json", "json_value", "sha256_hex"]
