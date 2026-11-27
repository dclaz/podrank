"""R1 predictive-quality diagnostics."""

from __future__ import annotations

import math

import pytest

from podrank import GameTable, fit, sim
from podrank.diagnostics import predictive_quality


def test_predictive_quality_on_doc_tiny() -> None:
    world = sim.reference_world("doc_tiny")
    post = fit(
        GameTable.from_records(world.game_rows()),
        world.agent_registry(),
        world.deck_registry(),
    )
    report = predictive_quality(post, world.game_rows())
    assert report.n_games == 223
    assert report.n_clusters == 60
    assert report.n_excluded == 0
    assert report.log_loss["uniform"] == pytest.approx(math.log(4.0))
    assert report.brier["uniform"] == pytest.approx(0.75)
    for name in ("seat", "theta", "theta_deck", "full"):
        assert report.log_loss[name] is not None
        assert report.brier[name] is not None
        assert report.log_loss[name] < math.log(4.0)
    assert sum(bin.count for bin in report.reliability) == report.n_games
    for bin in report.reliability:
        assert 0 <= bin.predicted <= 1
        assert 0 <= bin.observed <= 1


def test_predictive_quality_rejects_zero_bins() -> None:
    world = sim.reference_world("doc_tiny")
    post = fit(
        GameTable.from_records(world.game_rows()),
        world.agent_registry(),
        world.deck_registry(),
    )
    with pytest.raises(ValueError, match="reliability_bins"):
        predictive_quality(post, world.game_rows(), reliability_bins=0)
