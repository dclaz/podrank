"""P3 `doc_tiny_balanced` vertical slice."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from podrank import GameTable, Posterior, fit, sim, uncertainty


def _explicit_sdwp(post: Posterior, field, agent_id: str) -> float:
    total = 0.0
    for context in field.contexts:
        design = post._context_design(context, agent_id)
        utilities = design @ post.mode
        maximum = float(np.max(utilities))
        weights = np.exp(utilities - maximum)
        probabilities = weights / float(np.sum(weights))
        total += context.weight * float(probabilities[context.focal_seat])
    return total


def test_doc_tiny_balanced_vertical_slice(tmp_path) -> None:
    world = sim.reference_world("doc_tiny_balanced")
    agents = world.agent_registry()
    decks = world.deck_registry()
    rows = world.game_rows()
    field = world.field(24)
    post = fit(GameTable.from_records(rows), agents, decks)

    assert post.counts().likelihood_used == 240
    assert post.counts().likelihood_bearing_clusters == 60
    eligibility = post.headline_eligibility()
    assert isinstance(eligibility, tuple)
    assert len(eligibility) == 6
    assert all(row.eligible for row in eligibility)
    anchor = next(row for row in eligibility if row.agent_id == "agent_00")
    assert anchor.contrast_status == "zero"
    assert all(row.contrast_status in {"zero", "weak", "identified"} for row in eligibility)

    report = post.sdwp(field)
    assert len(report.rows) == 6
    values = [row.sdwp for row in report.rows]
    assert values == sorted(values, reverse=True)
    for row in report.rows:
        assert row.field_label == "field_identified"
        assert row.failing_contexts == ()
        assert row.sdwp == pytest.approx(_explicit_sdwp(post, field, row.agent_id), abs=1e-12)

    path = tmp_path / "posterior.podrank"
    post.save(path)
    loaded = Posterior.load(path)
    assert loaded.sdwp(field).rows == report.rows

    boot = uncertainty.cluster_bootstrap(post, n=100, seed=101)
    assert boot.n_converged == 100
    assert boot.failed_fraction == 0.0
    boot_report = boot.sdwp(field)
    assert len(boot_report.rows) == 6
    for row in boot_report.rows:
        assert row.interval is not None
        assert row.interval.method == "cluster_bootstrap"
        assert row.interval.lo < row.interval.hi

    assert "podrank.compare" not in sys.modules
