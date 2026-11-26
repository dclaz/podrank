"""Posterior reporting methods not covered by the main fixtures."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from podrank import (
    EffectNotModelled,
    Field,
    FieldContext,
    GameTable,
    ModelSpec,
    UnknownEntityError,
    UnsupportedFieldError,
    fit,
)
from podrank.errors import SwapsNotModelledError
from tests._fixtures import make_game, registries
from tests.test_rating_fit import _f11_rows, _registry


def _f17_posterior():
    agents, decks = _registry("HABCDE", 8, split="holdout")
    rows = _f11_rows()
    for index, permutation in enumerate(sorted(set(permutations(["C", "C", "D", "D"])))):
        rows.append(
            make_game(
                f"cd{index}",
                agents=permutation,
                decks=("d5", "d6", "d7", "d8"),
                outcome="win",
                winner_seat=index % 4,
                cluster_id=f"c{index}",
            )
        )
    rows.append(make_game("e1", agents=("E", "H", "A", "B"), outcome="draw"))
    return fit(GameTable.from_records(rows), agents, decks)


def _f14_posterior():
    agents, decks = registries()
    row = make_game(
        "g1",
        agents=("H", "A", "B", "C"),
        decks=("d1", "d2", "d3", "d4"),
        winner_seat=0,
    )
    return fit(GameTable.from_records([row]), agents, decks)


def test_log_worth_unrankable_and_outcome_reports() -> None:
    post = _f17_posterior()
    included = post.log_worth(include_withheld=True)
    assert isinstance(included, tuple)
    by_agent = {row.agent_id: row for row in included}
    assert by_agent["H"].status == "zero"
    assert by_agent["A"].status == "aliased"
    assert by_agent["A"].value is None
    assert by_agent["A"].prior_dependent_value is not None
    assert by_agent["C"].status == "disconnected"
    assert by_agent["C"].prior_dependent_value is not None
    assert by_agent["E"].status == "unrankable"
    assert by_agent["E"].value is None

    withheld = {row.agent_id: row for row in post.log_worth()}
    assert withheld["A"].prior_dependent_value is None

    unrankable = post.unrankable()
    assert "E" in unrankable.agents
    assert unrankable.decks == ()

    rates = post.outcome_rates()
    e_rows = [row for row in rates.rows if row.agent_id == "E"]
    assert e_rows
    assert e_rows[0].counts.draw == 1
    assert e_rows[0].draw_rate == pytest.approx(1.0)

    seats = post.seats()
    assert len(seats) == 4
    assert [row.seat for row in seats] == [0, 1, 2, 3]
    assert all(row.status in {"zero", "weak", "identified", "aliased"} for row in seats)

    zero = post.contrast_vector(np.zeros(len(post.parameter_index)))
    assert zero.status == "zero"
    nonzero = post.contrast_vector(np.ones(len(post.parameter_index)))
    assert nonzero.status in {"aliased", "weak", "identified"}

    report = post.identification_report()
    assert report.components == post.components()
    assert isinstance(report.weak_directions, tuple)
    assert report.n_gauge > 0

    assert isinstance(post.swap_effect(), EffectNotModelled)


def test_deck_nuisance_and_swap_effect() -> None:
    post = _f14_posterior()
    decks = post.decks()
    assert decks.baseline_status == "available"
    assert decks.reference_deck_ids == ("d1", "d2", "d3", "d4")
    assert len(decks.rows) == 4
    assert all("prior_dependent_split" in row.labels for row in decks.rows)

    agents, deck_registry = registries()
    swapped = make_game(
        "g1",
        agents=("H", "A", "B", "C"),
        decks=("d1", "d2", "d3", "d4"),
        swap_counts=(1, 0, 0, 0),
        winner_seat=0,
    )
    eta_post = fit(
        GameTable.from_records([swapped]),
        agents,
        deck_registry,
        spec=ModelSpec(swap_effect=True),
    )
    effect = eta_post.swap_effect()
    assert not isinstance(effect, EffectNotModelled)


def test_sdwp_field_support_errors() -> None:
    post = _f14_posterior()

    unknown_deck = Field.from_contexts(
        "bad_deck",
        1,
        [
            FieldContext(
                decks=("dx", "d2", "d3", "d4"),
                swap_counts=(0, 0, 0, 0),
                opponents=("H", "H", "H"),
                focal_seat=0,
                weight=1.0,
            )
        ],
    )
    with pytest.raises(UnsupportedFieldError, match="deck"):
        post.sdwp(unknown_deck)

    unknown_opponent = Field.from_contexts(
        "bad_opponent",
        1,
        [
            FieldContext(
                decks=("d1", "d2", "d3", "d4"),
                swap_counts=(0, 0, 0, 0),
                opponents=("Z", "Z", "Z"),
                focal_seat=0,
                weight=1.0,
            )
        ],
    )
    with pytest.raises(UnsupportedFieldError, match="opponent"):
        post.sdwp(unknown_opponent)

    swapped = Field.from_contexts(
        "swapped",
        1,
        [
            FieldContext(
                decks=("d1", "d2", "d3", "d4"),
                swap_counts=(1, 0, 0, 0),
                opponents=("H", "H", "H"),
                focal_seat=0,
                weight=1.0,
            )
        ],
    )
    with pytest.raises(SwapsNotModelledError):
        post.sdwp(swapped)
    with pytest.raises(ValueError, match="n_draws"):
        post.sdwp(swapped, n_draws=0)


def test_prediction_input_errors_and_sample_validation() -> None:
    post = _f14_posterior()
    with pytest.raises(ValueError, match="mapping"):
        post.predict_proba([["H", "A", "B", "C"]])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unknown prediction input keys"):
        post.predict_proba([{"agents": ("H", "A", "B", "C"), "extra": 1}])
    with pytest.raises(ValueError, match="four"):
        post.predict_proba([{"agents": ("H", "A"), "decks": ("d1", "d2", "d3", "d4")}])
    with pytest.raises(ValueError, match="non-negative"):
        post.predict_proba(
            [
                {
                    "agents": ("H", "A", "B", "C"),
                    "decks": ("d1", "d2", "d3", "d4"),
                    "swap_counts": (-1, 0, 0, 0),
                }
            ]
        )
    with pytest.raises(UnknownEntityError):
        post.predict_proba([{"agents": ("Z", "A", "B", "C"), "decks": ("d1", "d2", "d3", "d4")}])
    with pytest.raises(ValueError, match="seed"):
        post.sample(2, seed=-1)
