"""Posterior predictive checks for decisive games (R1 diagnostics, §10.2)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from podrank.data.table import GameTable, normalize_game_row
from podrank.diagnostics.calibration import _probabilities, _utilities
from podrank.rating.posterior import Posterior

PpcGroup = Literal["agent", "deck", "seat", "tag"]


@dataclass(frozen=True, slots=True)
class PpcRow:
    group: str
    key: str
    observed: int
    expected: float
    z: float
    flagged: bool


@dataclass(frozen=True, slots=True)
class PpcReport:
    n_games: int
    n_excluded: int
    rows: tuple[PpcRow, ...]


def _tags(post: Posterior, deck: str) -> tuple[str, ...]:
    record = post._decks.by_id.get(deck)  # noqa: SLF001
    return () if record is None else tuple(record.tags)


def posterior_predictive(
    post: Posterior,
    games: GameTable | Iterable[Mapping[str, object]],
    *,
    by: tuple[PpcGroup, ...] = ("agent", "deck", "seat", "tag"),
) -> PpcReport:
    """Compare observed decisive win rates with MAP predictions."""

    observed: dict[tuple[str, str], int] = {}
    expected: dict[tuple[str, str], float] = {}
    variance: dict[tuple[str, str], float] = {}
    n_games = 0
    excluded = 0
    for row in (normalize_game_row(record) for record in games):
        if row.outcome != "win" or row.winner_seat is None:
            continue
        utilities = _utilities(post, row, "full")
        if utilities is None:
            excluded += 1
            continue
        probabilities = _probabilities(utilities)
        n_games += 1
        winner_seat = row.winner_seat
        winner_agent = row.agents[winner_seat]
        winner_deck = row.decks[winner_seat]
        winner_tags = set(_tags(post, winner_deck))
        for group in by:
            if group == "agent":
                for seat, agent in enumerate(row.agents):
                    key = ("agent", agent)
                    expected[key] = expected.get(key, 0.0) + float(probabilities[seat])
                    variance[key] = variance.get(key, 0.0) + float(
                        probabilities[seat] * (1.0 - probabilities[seat])
                    )
                    if agent == winner_agent:
                        observed[key] = observed.get(key, 0) + 1
            elif group == "deck":
                for seat, deck in enumerate(row.decks):
                    key = ("deck", deck)
                    expected[key] = expected.get(key, 0.0) + float(probabilities[seat])
                    variance[key] = variance.get(key, 0.0) + float(
                        probabilities[seat] * (1.0 - probabilities[seat])
                    )
                    if deck == winner_deck:
                        observed[key] = observed.get(key, 0) + 1
            elif group == "seat":
                key = ("seat", str(winner_seat))
                for seat in range(4):
                    seat_key = ("seat", str(seat))
                    expected[seat_key] = expected.get(seat_key, 0.0) + float(probabilities[seat])
                    variance[seat_key] = variance.get(seat_key, 0.0) + float(
                        probabilities[seat] * (1.0 - probabilities[seat])
                    )
                observed[key] = observed.get(key, 0) + 1
            else:
                deck_tags = {tag for deck in row.decks for tag in _tags(post, deck)}
                for tag in deck_tags:
                    key = ("tag", tag)
                    expected[key] = expected.get(key, 0.0) + sum(
                        float(probabilities[seat])
                        for seat, deck in enumerate(row.decks)
                        if tag in _tags(post, deck)
                    )
                    variance[key] = variance.get(key, 0.0) + sum(
                        float(probabilities[seat] * (1.0 - probabilities[seat]))
                        for seat, deck in enumerate(row.decks)
                        if tag in _tags(post, deck)
                    )
                    if tag in winner_tags:
                        observed[key] = observed.get(key, 0) + 1
    rows: list[PpcRow] = []
    for key in sorted(expected):
        group_name, name = key
        observed_count = observed.get(key, 0)
        expected_count = expected[key]
        sd = math.sqrt(variance[key]) if variance[key] > 0 else 0.0
        z = (observed_count - expected_count) / sd if sd > 0 else 0.0
        rows.append(
            PpcRow(
                group=group_name,
                key=name,
                observed=observed_count,
                expected=expected_count,
                z=z,
                flagged=abs(z) > 3.0,
            )
        )
    return PpcReport(n_games=n_games, n_excluded=excluded, rows=tuple(rows))


__all__ = ["PpcGroup", "PpcReport", "PpcRow", "posterior_predictive"]
