"""R1 predictive-quality diagnostics."""

from __future__ import annotations

import math

import pytest

from podrank import GameTable, ModelSpec, fit, sim
from podrank.diagnostics import (
    cluster_cv,
    forward_split,
    log_evidence,
    posterior_predictive,
    predictive_quality,
    tune_sigma_beta,
    tune_sigma_beta_cv,
)


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


def test_posterior_predictive_on_doc_tiny() -> None:
    world = sim.reference_world("doc_tiny")
    post = fit(
        GameTable.from_records(world.game_rows()),
        world.agent_registry(),
        world.deck_registry(),
    )
    report = posterior_predictive(post, world.game_rows())
    assert report.n_games == 223
    assert report.n_excluded == 0
    groups = {row.group for row in report.rows}
    assert groups == {"agent", "deck", "seat"}
    for group in ("agent", "deck", "seat"):
        rows = [row for row in report.rows if row.group == group]
        assert sum(row.observed for row in rows) == report.n_games
        assert sum(row.expected for row in rows) == pytest.approx(report.n_games)
        assert all(math.isfinite(row.z) for row in rows)

    seat_only = posterior_predictive(post, world.game_rows(), by=("seat",))
    assert len(seat_only.rows) == 4


def test_forward_split_on_doc_tiny() -> None:
    world = sim.reference_world("doc_tiny")
    report = forward_split(
        GameTable.from_records(world.game_rows()),
        world.agent_registry(),
        world.deck_registry(),
        n_windows=3,
        n_burn=2,
    )
    assert len(report.windows) == 2
    assert report.full_log_loss is not None
    assert report.best_baseline in {"uniform", "seat", "theta", "theta_deck"}
    assert report.not_worse is not None
    assert all(window.scored_games > 0 for window in report.windows)
    assert report.windows[0].train_clusters >= 20


def test_sigma_beta_evidence_and_selection() -> None:
    world = sim.reference_world("doc_tiny")
    table = GameTable.from_records(world.game_rows())
    agents = world.agent_registry()
    decks = world.deck_registry()
    post = fit(table, agents, decks)
    assert math.isfinite(log_evidence(post))
    selection = tune_sigma_beta(table, agents, decks, grid=(0.25, 0.5, 1.0))
    assert len(selection.results) == 3
    assert selection.best.sigma_beta in {0.25, 0.5, 1.0}
    assert all(math.isfinite(result.log_evidence) for result in selection.results)
    cv = tune_sigma_beta_cv(table, agents, decks, grid=(0.5,), folds=3)
    assert cv.best.sigma_beta == 0.5
    assert len(cv.results) == 1
    assert ModelSpec(sigma_beta=1.0).sigma_beta == 1.0
    with pytest.raises(ValueError, match="sigma_beta"):
        ModelSpec(sigma_beta=0.0)


def test_whole_cluster_cv_on_doc_tiny() -> None:
    world = sim.reference_world("doc_tiny")
    report = cluster_cv(
        GameTable.from_records(world.game_rows()),
        world.agent_registry(),
        world.deck_registry(),
        folds=5,
    )
    assert len(report.folds) == 5
    assert report.full_log_loss is not None
    assert report.best_baseline in {"uniform", "seat", "theta", "theta_deck"}
    assert report.not_worse is not None
    assert sum(fold.scored_games for fold in report.folds) == 223


def test_sigma_beta_evidence_selects_reference_truth() -> None:
    world = sim.reference_world("standard")
    selection = tune_sigma_beta(
        GameTable.from_records(world.game_rows()),
        world.agent_registry(),
        world.deck_registry(),
        grid=(0.25, 0.5, 1.0, 2.0),
    )
    assert selection.best.sigma_beta == 0.5
