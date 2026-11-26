"""Reference-world size and registry smoke tests."""

from __future__ import annotations

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


def test_ci_smoke_world_generates_fixed_size() -> None:
    world = sim.reference_world("ci_smoke")
    rows = world.game_rows()
    assert len(rows) == 800
    assert len({row["cluster_id"] for row in rows}) == 200
    assert all(row["outcome"] in {"win", "draw", "cap", "failed"} for row in rows)
