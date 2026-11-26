"""Fit boundary errors and configuration paths."""

from __future__ import annotations

import pytest

from podrank import GameTable, ModelSpec, fit
from podrank.errors import ModelTooLargeError, SwapsNotModelledError, ValidationError
from tests._fixtures import make_game, registries


def _f14_row(**kwargs: object) -> dict[str, object]:
    return make_game(
        "g1",
        agents=("H", "A", "B", "C"),
        decks=("d1", "d2", "d3", "d4"),
        winner_seat=0,
        **kwargs,
    )


def test_fit_raises_on_malformed_rows_without_prior_validate() -> None:
    agents, decks = registries()
    row = make_game("g1", agents=("H", "A", "B"))
    with pytest.raises(ValidationError):
        fit(GameTable.from_records([row]), agents, decks)


def test_fit_applies_tag_schema_and_split_policy() -> None:
    agents, decks = registries()
    row = _f14_row()
    with pytest.raises(ValidationError, match="missing required tag"):
        fit(
            GameTable.from_records([row]),
            agents,
            decks,
            tag_schema={"power": {"high"}, "required": ["power"]},
        )
    with pytest.raises(ValidationError, match="not allowed"):
        fit(
            GameTable.from_records([row]),
            agents,
            decks,
            split_policy={"train": ("other",)},
        )


def test_fit_rejects_nonzero_swaps_when_swap_effect_is_off() -> None:
    agents, decks = registries()
    row = _f14_row(swap_counts=(1, 0, 0, 0))
    with pytest.raises(SwapsNotModelledError):
        fit(GameTable.from_records([row]), agents, decks)


def test_fit_enforces_dense_size_limits() -> None:
    agents, decks = registries()
    table = GameTable.from_records([_f14_row()])
    with pytest.raises(ModelTooLargeError, match="max_params"):
        fit(table, agents, decks, spec=ModelSpec(max_params=5))
    with pytest.raises(ModelTooLargeError, match="max_dense_bytes"):
        fit(table, agents, decks, spec=ModelSpec(max_dense_bytes=1))


def test_fit_warm_start_names() -> None:
    agents, decks = registries()
    table = GameTable.from_records([_f14_row()])
    post = fit(
        table,
        agents,
        decks,
        warm_start={"theta[A]": 0.2, "not_a_parameter": 99.0},
    )
    assert post.mode.shape == (len(post.parameter_index),)
    with pytest.raises(ValueError, match="finite"):
        fit(table, agents, decks, warm_start={"theta[A]": float("nan")})


def test_fit_cluster_strata_validation() -> None:
    agents, decks = registries()
    table = GameTable.from_records([_f14_row()])
    frame = {("game", "g1"): "a"}

    assert fit(table, agents, decks, cluster_strata={"custom": frame}) is not None

    with pytest.raises(ValueError, match="mapping"):
        fit(table, agents, decks, cluster_strata=[])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reserved"):
        fit(table, agents, decks, cluster_strata={"suite": frame})
    with pytest.raises(ValueError, match="must be a mapping"):
        fit(table, agents, decks, cluster_strata={"custom": "x"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="missing frame keys"):
        fit(table, agents, decks, cluster_strata={"custom": {("game", "other"): "a"}})
    with pytest.raises(ValueError, match="labels"):
        fit(table, agents, decks, cluster_strata={"custom": {("game", "g1"): ""}})
