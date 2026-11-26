"""Outcome-free prediction inputs and rows (Appendix C.1)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast


def _tuple4_strings(value: object, name: str) -> tuple[str, str, str, str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of four strings")
    items = list(value)
    if len(items) != 4 or any(not isinstance(item, str) or not item for item in items):
        raise ValueError(f"{name} must be a sequence of four non-empty strings")
    return cast(tuple[str, str, str, str], tuple(items))


def _tuple4_swaps(value: object) -> tuple[int, int, int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("swap_counts must be a sequence of four non-negative integers")
    items = list(value)
    if len(items) != 4:
        raise ValueError("swap_counts must have length 4")
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError("swap_counts must be four non-negative integers")
    return cast(tuple[int, int, int, int], tuple(items))


@dataclass(frozen=True, slots=True)
class PredictionInput:
    agents: tuple[str, str, str, str]
    decks: tuple[str, str, str, str]
    swap_counts: tuple[int, int, int, int] = (0, 0, 0, 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agents", _tuple4_strings(self.agents, "agents"))
        object.__setattr__(self, "decks", _tuple4_strings(self.decks, "decks"))
        object.__setattr__(self, "swap_counts", _tuple4_swaps(self.swap_counts))


@dataclass(frozen=True, slots=True)
class PredictionRow:
    probabilities: tuple[float, float, float, float]
    field_label: str
    failing_seats: tuple[int, ...]
    labels: tuple[str, ...]


def coerce_prediction_input(value: object) -> PredictionInput:
    if isinstance(value, PredictionInput):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("each prediction input must be a PredictionInput or mapping")
    unknown = set(value) - {"agents", "decks", "swap_counts"}
    if unknown:
        raise ValueError(f"unknown prediction input keys: {sorted(unknown)}")
    prediction = PredictionInput(
        agents=value.get("agents"),  # type: ignore[arg-type]
        decks=value.get("decks"),  # type: ignore[arg-type]
        swap_counts=value.get("swap_counts", (0, 0, 0, 0)),
    )
    return prediction


__all__ = ["PredictionInput", "PredictionRow", "coerce_prediction_input"]
