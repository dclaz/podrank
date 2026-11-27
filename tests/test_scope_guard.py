"""R2 public-name scope guard."""

from __future__ import annotations

import inspect

import podrank.compare as compare

FORBIDDEN = {"promote", "promotion", "incumbent", "candidate", "gate", "attempt", "checkpoint"}


def _tokens(name: str) -> set[str]:
    return {token for token in name.lower().split("_") if token}


def _public_names() -> set[str]:
    names = set(compare.__all__)
    for exported in compare.__all__:
        value = getattr(compare, exported)
        if inspect.isclass(value) and value.__module__.startswith("podrank.compare"):
            names.update(name for name, _ in inspect.getmembers(value) if not name.startswith("_"))
    return names


def test_public_names_do_not_claim_promotion_semantics() -> None:
    offenders = {
        name: FORBIDDEN & _tokens(name) for name in _public_names() if FORBIDDEN & _tokens(name)
    }
    assert offenders == {}
