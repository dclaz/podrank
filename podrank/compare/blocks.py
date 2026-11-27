"""Block scoring and directional bounds (F2, F4, E4)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from podrank.compare.descriptors import Position, expected_positions
from podrank.compare.pods import Design
from podrank.types import Outcome, Role

_FOCAL_VALUES = frozenset({0.0, 0.25, 1.0})


def focal_points(outcome: Outcome, winner_seat: int | None, seat: int) -> float | None:
    """Return the R2 focal score for one report."""

    if outcome == "failed":
        return None
    if outcome in {"draw", "cap"}:
        return 0.25
    if winner_seat is None:
        raise ValueError("a win requires winner_seat")
    return 1.0 if winner_seat == seat else 0.0


def _coefficient(design: Design, rotation: int, role: Role | None) -> float:
    if design == "duplicate":
        return 1.0 / 4.0
    sign = 1.0 if role == "a" else -1.0
    return sign / 4.0


def _substitute(design: Design, rotation: int, role: Role | None, *, upper: bool) -> float:
    if design == "duplicate":
        return 0.0 if upper else 1.0
    if role == "a":
        return 0.0 if upper else 1.0
    return 1.0 if upper else 0.0


def _score(
    design: Design,
    points: Mapping[Position, float | None],
    *,
    upper: bool,
) -> float:
    total = 0.0
    for rotation, role in expected_positions(design):
        value = points[(rotation, role)]
        if value is None:
            value = _substitute(design, rotation, role, upper=upper)
        total += _coefficient(design, rotation, role) * value
    return total - (0.25 if design == "duplicate" else 0.0)


@dataclass(frozen=True, slots=True)
class BlockScores:
    """Directional and complete-data scores for one block."""

    x_up: float
    x_low: float
    complete: float | None
    failed: bool


def score_block(design: Design, points: Mapping[Position, float | None]) -> BlockScores:
    expected = expected_positions(design)
    if set(points) != set(expected):
        missing = sorted(set(expected) - set(points))
        extra = sorted(set(points) - set(expected), key=repr)
        raise ValueError(f"block positions mismatch: missing={missing}, extra={extra}")
    resolved: dict[Position, float | None] = {}
    for key in expected:
        value = points[key]
        if value is not None:
            numeric = float(value)
            if numeric not in _FOCAL_VALUES:
                raise ValueError("focal points must be 0, 0.25, 1 or None")
            resolved[key] = numeric
        else:
            resolved[key] = None
    failed = any(value is None for value in resolved.values())
    complete = None if failed else _score(design, resolved, upper=True)
    return BlockScores(
        x_up=_score(design, resolved, upper=True),
        x_low=_score(design, resolved, upper=False),
        complete=complete,
        failed=failed,
    )


@dataclass(frozen=True, slots=True)
class ScoredBlockInput:
    """One already-resolved block for the descriptive summary (E7.8)."""

    block_id: str
    points: Mapping[Position, float | None]

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, str) or not self.block_id:
            raise ValueError("block_id must be a non-empty string")
        if not isinstance(self.points, Mapping):
            raise ValueError("points must be a mapping")
        object.__setattr__(self, "points", MappingProxyType(dict(self.points)))


@dataclass(frozen=True, slots=True)
class DescriptiveSummary:
    """Descriptive block summary; no wealth, status or confidence sequence."""

    n_blocks: int
    n_failed_blocks: int
    failed_block_fraction: float | None
    mean_lift_bounds: tuple[float, float] | None


def summarise_blocks(
    blocks: Iterable[ScoredBlockInput],
    *,
    design: Design,
) -> DescriptiveSummary:
    """Summarise already-resolved blocks without creating a comparison."""

    if design not in {"duplicate", "paired"}:
        raise ValueError("design must be 'duplicate' or 'paired'")
    items = list(blocks)
    if any(not isinstance(item, ScoredBlockInput) for item in items):
        raise ValueError("blocks must contain ScoredBlockInput records")
    if len({item.block_id for item in items}) != len(items):
        raise ValueError("block_id values must be unique")
    if not items:
        return DescriptiveSummary(0, 0, None, None)
    scores = [score_block(design, item.points) for item in items]
    failed = sum(1 for score in scores if score.failed)
    return DescriptiveSummary(
        n_blocks=len(scores),
        n_failed_blocks=failed,
        failed_block_fraction=failed / len(scores),
        mean_lift_bounds=(
            sum(score.x_up for score in scores) / len(scores),
            sum(score.x_low for score in scores) / len(scores),
        ),
    )


__all__ = [
    "BlockScores",
    "DescriptiveSummary",
    "ScoredBlockInput",
    "focal_points",
    "score_block",
    "summarise_blocks",
]
