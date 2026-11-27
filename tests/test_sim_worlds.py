"""Reference-world size and registry smoke tests."""

from __future__ import annotations

import pytest

from podrank import sim


def test_reference_world_sizes() -> None:
    expected = {
        "doc_tiny": (6, 3, 4, 60),
        "doc_tiny_balanced": (6, 3, 4, 60),
        "ci_smoke": (10, 3, 8, 200),
        "standard": (20, 4, 12, 600),
        "perf_large": (50, 5, 28, 250_000),
    }
    for name, (agents, brackets, per_bracket, clusters) in expected.items():
        world = sim.reference_world(name)
        assert world.agents == agents
        assert world.brackets == brackets
        assert world.decks_per_bracket == per_bracket
        assert world.clusters == clusters
        assert len(world.agent_registry()) == agents
        assert len(world.deck_registry()) == brackets * per_bracket
        assert world.agent_registry().anchors[0].agent_id == "agent_00"


def test_standard_rows_matches_world_generator_and_run_seed() -> None:
    for name in ("doc_tiny", "ci_smoke"):
        world = sim.reference_world(name)
        assert sim.standard_rows(world) == world.game_rows()

    world = sim.reference_world("ci_smoke")
    first = sim.standard_rows(world, run_seed=1234)
    assert first == sim.standard_rows(world, run_seed=1234)
    assert first != sim.standard_rows(world, run_seed=5678)

    zero_delta = {deck: 0.0 for deck in world.deck_ids}
    overridden = sim.standard_rows(world, run_seed=1234, deck_effects=zero_delta, rho_div=0.0)
    assert [row["agents"] for row in overridden] == [row["agents"] for row in first]
    assert [row["decks"] for row in overridden] == [row["decks"] for row in first]
    # Outcome categories use their own stream; δ and shared luck change winners.
    assert [row["outcome"] for row in overridden] == [row["outcome"] for row in first]
    assert [row["winner_seat"] for row in overridden] != [row["winner_seat"] for row in first]
    assert overridden == sim.standard_rows(
        world, run_seed=1234, deck_effects=zero_delta, rho_div=0.0
    )


def test_ci_smoke_world_generates_fixed_size() -> None:
    world = sim.reference_world("ci_smoke")
    rows = world.game_rows()
    assert len(rows) == 800
    assert len({row["cluster_id"] for row in rows}) == 200
    assert all(row["outcome"] in {"win", "draw", "cap", "failed"} for row in rows)


def test_compare_worlds_are_calibrated() -> None:
    from podrank.sim import reference_compare_world

    null = reference_compare_world("cmp_null")
    assert [cell.label for cell in null.cells] == [
        "superiority_0",
        "superiority_002",
        "noninferiority_002",
    ]
    effect = reference_compare_world("cmp_effect")
    assert [cell.mean for cell in effect.cells] == [0.005, 0.01, 0.02, 0.05]
    for world in (null, effect):
        for cell in world.cells:
            for design in ("duplicate", "paired"):
                theta = cell.theta_a(design)
                assert world.mean_lift(theta, design) == pytest.approx(cell.mean, abs=1e-12)
