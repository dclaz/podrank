"""Fit, identification statuses and prediction fixtures (F11, F11b, F14, F17, F20)."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from podrank import (
    AgentRegistry,
    DeckRegistry,
    Field,
    FieldContext,
    GameTable,
    ModelSpec,
    PredictionInput,
    UnknownEntityError,
    UnsupportedPredictionError,
    fit,
)
from podrank.errors import SwapsNotModelledError
from podrank.rating.design import design_row
from podrank.rating.identification import AnchorUnsupported, agent_vector, aliasing_residual
from podrank.rating.likelihood import likelihood_hess
from tests._fixtures import make_game, registries


def _registry(agent_ids: str, deck_ids: int, *, split: str = "train"):
    agent_rows = [
        {
            "agent_id": agent_id,
            "kind": "heuristic" if agent_id == "H" else "checkpoint",
            "policy_version": f"{agent_id}1",
            "is_anchor": agent_id == "H",
        }
        for agent_id in agent_ids
    ]
    deck_rows = [
        {
            "deck_id": f"d{number}",
            "bracket": 1,
            "split": split,
            "deck_version": f"d{number}v1",
        }
        for number in range(1, deck_ids + 1)
    ]
    return AgentRegistry.from_records(agent_rows), DeckRegistry.from_records(deck_rows)


def _f11_rows() -> list[dict[str, object]]:
    mapping = {"A": "d1", "B": "d2", "H1": "d3", "H2": "d4"}
    rows: list[dict[str, object]] = []
    for index, permutation in enumerate(sorted(set(permutations(["A", "B", "H1", "H2"])))):
        agents = tuple("H" if item.startswith("H") else item for item in permutation)
        decks = tuple(mapping[item] for item in permutation)
        rows.append(
            make_game(
                f"g{index}",
                agents=agents,
                decks=decks,
                outcome="win",
                winner_seat=index % 4,
                cluster_id=f"b{index}",
            )
        )
    return rows


def test_f14_wide_design_aliasing() -> None:
    agents, decks = registries()
    row = make_game(
        "g1",
        agents=("H", "A", "B", "C"),
        decks=("d1", "d2", "d3", "d4"),
        winner_seat=0,
    )
    post = fit(GameTable.from_records([row]), agents, decks)

    assert len(post.parameter_index) == 13
    assert post.row_space.shape == (13, 3)
    result = post.contrast("A", "H")
    assert result.status == "aliased"
    assert result.aliasing_residual == pytest.approx(0.816497, abs=1e-6)
    assert result.value is None

    exact = post.contrast("H", "H")
    assert exact.status == "zero"
    assert exact.value == 0.0
    assert exact.interval is not None
    assert (exact.interval.lo, exact.interval.hi) == (0.0, 0.0)


def test_f11_aliasing_before_and_after_crossover() -> None:
    agents, decks = registries()
    first = fit(GameTable.from_records(_f11_rows()), agents, decks)

    assert first.parameter_index.names == (
        "theta[A]",
        "theta[B]",
        "theta[H]",
        "seat[0]",
        "seat[1]",
        "seat[2]",
        "seat[3]",
        "mu[1]",
        "delta[d1]",
        "delta[d2]",
        "delta[d3]",
        "delta[d4]",
    )
    assert first.row_space.shape[1] == 6
    for left, right, expected in (
        ("A", "H", 0.65192),
        ("B", "H", 0.65192),
        ("A", "B", 0.70711),
    ):
        vector = agent_vector(first.parameter_index, left) - agent_vector(
            first.parameter_index, right
        )
        assert aliasing_residual(vector, first.row_space) == pytest.approx(expected, abs=1e-5)
        assert first.contrast(left, right).status == "aliased"

    mapping = {"A": "d3", "B": "d4", "H1": "d1", "H2": "d2"}
    rows = _f11_rows()
    for offset, permutation in enumerate(sorted(set(permutations(["A", "B", "H1", "H2"])))):
        index = len(rows)
        agents_row = tuple("H" if item.startswith("H") else item for item in permutation)
        decks_row = tuple(mapping[item] for item in permutation)
        rows.append(
            make_game(
                f"x{offset}",
                agents=agents_row,
                decks=decks_row,
                outcome="win",
                winner_seat=index % 4,
                cluster_id=f"x{offset}",
            )
        )
    after = fit(GameTable.from_records(rows), agents, decks)
    assert after.row_space.shape[1] == 8
    assert after.contrast("A", "H").status in {"weak", "identified"}
    assert after.contrast("A", "H").aliasing_residual == pytest.approx(0.0, abs=1e-10)


def test_f11b_unsupported_anchor_still_reports_explicit_contrasts() -> None:
    agents, decks = registries()
    rows: list[dict[str, object]] = []
    for index, permutation in enumerate(sorted(set(permutations(["A", "B", "C", "C"])))):
        rows.append(
            make_game(
                f"g{index}",
                agents=permutation,
                decks=("d1", "d2", "d3", "d4"),
                outcome="win",
                winner_seat=index % 4,
                cluster_id=f"b{index % 4}",
            )
        )
    rows.append(make_game("drawH", agents=("H", "A", "B", "C"), outcome="draw"))
    post = fit(GameTable.from_records(rows), agents, decks)

    assert isinstance(post.components(), AnchorUnsupported)
    assert isinstance(post.log_worth(), AnchorUnsupported)
    assert isinstance(post.headline_eligibility(), AnchorUnsupported)
    explicit = post.contrast("A", "B")
    assert explicit.status in {"weak", "identified"}
    assert explicit.value is not None
    assert post.contrast("A", "H").status == "unrankable"

    field = Field.from_contexts(
        "f",
        1,
        [
            FieldContext(
                decks=("d1", "d2", "d3", "d4"),
                swap_counts=(0, 0, 0, 0),
                opponents=("B", "B", "B"),
                focal_seat=0,
                weight=1.0,
            )
        ],
    )
    assert isinstance(post.sdwp(field), AnchorUnsupported)


def test_f17_output_statuses() -> None:
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
    post = fit(GameTable.from_records(rows), agents, decks)

    exact = post.contrast("H", "H")
    assert exact.status == "zero"
    assert exact.value == 0.0
    assert exact.interval is not None
    assert (exact.interval.lo, exact.interval.hi) == (0.0, 0.0)

    assert post.contrast("A", "H").status == "aliased"
    assert post.contrast("A", "H").value is None
    withheld = post.contrast("A", "H", include_withheld=True)
    assert withheld.value is None
    assert withheld.prior_dependent_value is not None
    assert post.contrast("C", "H").status == "disconnected"
    assert post.contrast("C", "D").status != "disconnected"
    assert post.contrast("E", "E").status == "unrankable"
    with pytest.raises(UnknownEntityError):
        post.contrast("Z", "H")

    baseline = post.decks()
    assert baseline.baseline_status == "unavailable"
    assert baseline.rows == ()

    eligibility = post.headline_eligibility()
    assert isinstance(eligibility, tuple)
    statuses = {row.agent_id: (row.contrast_status, row.eligible) for row in eligibility}
    assert statuses["H"] == ("zero", True)
    assert statuses["A"] == ("aliased", False)
    assert statuses["B"] == ("aliased", False)
    assert statuses["C"] == ("disconnected", False)
    assert statuses["D"] == ("disconnected", False)
    assert statuses["E"] == ("unrankable", False)

    field = Field.from_contexts(
        "f",
        1,
        [
            FieldContext(
                decks=("d1", "d2", "d3", "d4"),
                swap_counts=(0, 0, 0, 0),
                opponents=("H", "H", "H"),
                focal_seat=0,
                weight=1.0,
            )
        ],
    )
    report = post.sdwp(field)
    assert not isinstance(report, AnchorUnsupported)
    assert [row.agent_id for row in report.rows] == ["H"]
    assert report.rows[0].sdwp > 0


def test_fit_reuses_solver_likelihood_hessian() -> None:
    agents, decks = registries()
    rows = [
        make_game("g1", agents=("H", "A", "B", "C"), winner_seat=0),
        make_game("g2", agents=("A", "B", "C", "H"), winner_seat=1),
        make_game("g3", agents=("B", "C", "H", "A"), winner_seat=2),
        make_game("g4", agents=("C", "H", "A", "B"), winner_seat=3),
    ]
    post = fit(GameTable.from_records(rows), agents, decks)
    expected = likelihood_hess(post.mode, post.patterns)
    assert np.allclose(post.data_hessian, expected, rtol=1e-10, atol=1e-12)


def test_compact_predict_proba_matches_manual_designs() -> None:
    agents, decks = registries()
    rows = [
        make_game(
            f"g{index}",
            agents=("H", "A", "B", "C"),
            decks=("d1", "d2", "d3", "d4"),
            winner_seat=index % 4,
        )
        for index in range(4)
    ]
    post = fit(GameTable.from_records(rows), agents, decks)
    contexts = [
        PredictionInput(agents=("H", "A", "B", "C"), decks=("d1", "d2", "d3", "d4")),
        PredictionInput(agents=("C", "B", "A", "H"), decks=("d4", "d3", "d2", "d1")),
        PredictionInput(agents=("H", "A", "B", "C"), decks=("d2", "d1", "d3", "d4")),
    ]
    output = post.predict_proba(contexts)
    for context, row in zip(contexts, output, strict=True):
        design = design_row(
            context.agents, context.decks, context.swap_counts, post.parameter_index, decks
        )
        utilities = design @ post.mode
        shifted = np.exp(utilities - utilities.max())
        expected = shifted / shifted.sum()
        assert row.probabilities == pytest.approx(tuple(float(value) for value in expected))
        failing: list[int] = []
        for seat in range(1, 4):
            residual = aliasing_residual(design[seat] - design[0], post.row_space)
            if residual is not None and residual > post.spec.tol_alias:
                failing.append(seat)
        assert row.failing_seats == tuple(failing)
        assert row.field_label == ("prior_dependent_field" if failing else "field_identified")


def test_f20_prediction_and_pooled_swaps() -> None:
    agents, decks = registries()
    observed = make_game(
        "g1",
        agents=("H", "A", "B", "C"),
        decks=("d1", "d2", "d3", "d4"),
        winner_seat=0,
    )
    post = fit(GameTable.from_records([observed]), agents, decks)

    predictions = post.predict_proba(
        [{"agents": ("H", "A", "B", "C"), "decks": ("d1", "d2", "d3", "d4")}]
    )
    assert sum(predictions[0].probabilities) == pytest.approx(1.0)
    assert predictions[0].field_label == "field_identified"

    moved = post.predict_proba(
        [{"agents": ("H", "A", "B", "C"), "decks": ("d2", "d1", "d3", "d4")}]
    )
    assert sum(moved[0].probabilities) == pytest.approx(1.0)
    assert moved[0].field_label == "prior_dependent_field"

    with pytest.raises(UnknownEntityError):
        post.predict_proba([{"agents": ("H", "A", "B", "C"), "decks": ("dx", "d2", "d3", "d4")}])
    with pytest.raises(UnsupportedPredictionError):
        post.predict_proba([{"agents": ("H", "A", "B", "C"), "decks": ("d5", "d2", "d3", "d4")}])

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

    pooled_row = make_game(
        "g2",
        agents=("H", "A", "B", "C"),
        decks=("d1", "d2", "d3", "d4"),
        swap_counts=(1, 0, 0, 0),
        winner_seat=0,
    )
    pooled = fit(
        GameTable.from_records([pooled_row]),
        agents,
        decks,
        spec=ModelSpec(allow_pooled_variants=True),
    )
    pooled_prediction = pooled.predict_proba(
        [
            {
                "agents": ("H", "A", "B", "C"),
                "decks": ("d1", "d2", "d3", "d4"),
                "swap_counts": (1, 0, 0, 0),
            }
        ]
    )[0]
    assert "pooled_variants" in pooled_prediction.labels
    assert sum(pooled_prediction.probabilities) == pytest.approx(1.0)
