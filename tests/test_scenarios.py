"""Deterministic scenario checks: SC3b, SC5–SC8.

Full SC1/SC2/SC3/SC4/SC7 statistical gates remain open; the checks here are the
deterministic hard gates from §11.3.
"""

from __future__ import annotations

import pytest

from podrank import GameTable, fit, sim
from podrank.errors import NoRankableEvidence
from podrank.rating.identification import AnchorUnsupported
from tests._fixtures import make_game


def _draw(row: dict[str, object]) -> dict[str, object]:
    updated = dict(row)
    updated["outcome"] = "draw"
    updated["winner_seat"] = None
    return updated


def test_sc6_all_draw_raises_no_rankable_evidence() -> None:
    world = sim.reference_world("doc_tiny")
    rows = [_draw(row) for row in world.game_rows()]
    with pytest.raises(NoRankableEvidence):
        fit(GameTable.from_records(rows), world.agent_registry(), world.deck_registry())


def test_sc6_draw_only_agent_is_unrankable() -> None:
    world = sim.reference_world("ci_smoke")
    rows = [
        _draw(row) if "agent_09" in row["agents"] else row  # type: ignore[operator]
        for row in world.game_rows()
    ]
    post = fit(GameTable.from_records(rows), world.agent_registry(), world.deck_registry())
    assert "agent_09" in post.unrankable().agents


def test_sc5_disconnected_components() -> None:
    world = sim.reference_world("standard")
    agents = world.agent_registry()
    decks = world.deck_registry()
    rows = []
    first_group = world.agent_ids[:10]
    second_group = world.agent_ids[10:]
    for cluster in range(60):
        group = first_group if cluster < 30 else second_group
        deck_start = (cluster * 4) % len(world.deck_ids)
        deck_order = [
            world.deck_ids[(deck_start + offset) % len(world.deck_ids)] for offset in range(4)
        ]
        for rotation in range(4):
            seated = [group[(cluster + rotation + offset) % len(group)] for offset in range(4)]
            rows.append(
                make_game(
                    f"b{cluster:03d}r{rotation}",
                    agents=tuple(seated),
                    decks=tuple(deck_order),
                    policy_versions=tuple(agents.by_id[agent].policy_version for agent in seated),
                    winner_seat=(cluster + rotation) % 4,
                    cluster_id=f"b{cluster:03d}",
                )
            )
    post = fit(GameTable.from_records(rows), agents, decks)
    components = post.components()
    assert not isinstance(components, AnchorUnsupported)
    assert components.other_components
    eligibility = post.headline_eligibility()
    assert isinstance(eligibility, tuple)
    disconnected = {row.agent_id for row in eligibility if row.contrast_status == "disconnected"}
    assert set(second_group) <= disconnected


def test_sc8_unsupported_anchor_returns_anchor_unsupported() -> None:
    world = sim.reference_world("doc_tiny")
    rows = [
        _draw(row) if "agent_00" in row["agents"] else row  # type: ignore[operator]
        for row in world.game_rows()
    ]
    post = fit(GameTable.from_records(rows), world.agent_registry(), world.deck_registry())
    assert isinstance(post.components(), AnchorUnsupported)
    assert isinstance(post.log_worth(), AnchorUnsupported)
    assert isinstance(post.headline_eligibility(), AnchorUnsupported)
    assert isinstance(post.sdwp(world.field(4)), AnchorUnsupported)


def test_pinned_roles_follow_world_size() -> None:
    assert sim.pinned_roles(sim.reference_world("standard")) == {
        "agent_18": "deck_b1_00",
        "agent_19": "deck_b2_00",
    }
    assert sim.pinned_roles(sim.reference_world("ci_smoke")) == {
        "agent_08": "deck_b1_00",
        "agent_09": "deck_b2_00",
    }
    with pytest.raises(ValueError, match="pinned-deck roles"):
        sim.pinned_roles(sim.reference_world("doc_tiny"))


def test_sc7_pinned_deck_aliasing() -> None:
    world = sim.reference_world("standard")
    pinned = sim.pinned_roles(world)
    rows = sim.sc7_rows(world)

    # Generator invariants (a)–(c): exclusivity, reserve size and connectivity.
    sim.assert_pinned_design(world, rows, pinned)
    assert len({row["cluster_id"] for row in rows}) == world.clusters
    for agent, deck in pinned.items():
        decisive = {
            str(row["cluster_id"])
            for row in rows
            if row["outcome"] == "win" and agent in row["agents"]  # type: ignore[operator]
        }
        assert len(decisive) == 30
        assert all(
            row["decks"][row["agents"].index(agent)] == deck  # type: ignore[union-attr]
            for row in rows
            if agent in row["agents"]  # type: ignore[operator]
        )
        assert all(
            row["agents"][row["decks"].index(deck)] == agent  # type: ignore[union-attr]
            for row in rows
            if deck in row["decks"]  # type: ignore[operator]
        )

    post = fit(GameTable.from_records(rows), world.agent_registry(), world.deck_registry())

    # (d) the alias directions lie in the D4 null space, then D4 is tested.
    for agent, deck in pinned.items():
        sim.assert_alias_in_null_space(post.patterns, agent, deck)

    log_worth = {row.agent_id: row for row in post.log_worth()}
    eligibility = {row.agent_id: row for row in post.headline_eligibility()}
    for agent in pinned:
        assert log_worth[agent].status == "aliased"
        assert log_worth[agent].value is None
        assert log_worth[agent].interval is None
        assert eligibility[agent].contrast_status == "aliased"
        assert not eligibility[agent].eligible

    assert log_worth["agent_00"].status == "zero"
    assert log_worth["agent_00"].interval is not None
    assert (log_worth["agent_00"].interval.lo, log_worth["agent_00"].interval.hi) == (0.0, 0.0)

    eligible = {
        row.agent_id
        for row in post.headline_eligibility()
        if row.contrast_status in {"weak", "identified"}
    }
    assert set(world.agent_ids) - set(pinned) - {"agent_00"} <= eligible

    sdwp_agents = {row.agent_id for row in post.sdwp(world.field(24)).rows}
    assert sdwp_agents.isdisjoint(pinned)


def test_sc7_generator_rejects_broken_designs() -> None:
    world = sim.reference_world("ci_smoke")
    pinned = sim.pinned_roles(world)
    rows = sim.sc7_rows(world)
    sim.assert_pinned_design(world, rows, pinned)

    agent, deck = next(iter(pinned.items()))
    broken = [dict(row) for row in rows]
    for row in broken:
        agents = list(row["agents"])  # type: ignore[arg-type]
        if agent in agents and deck in row["decks"]:  # type: ignore[operator]
            seat = agents.index(agent)
            agents[seat] = "agent_00" if agent != "agent_00" else "agent_01"
            row["agents"] = agents
            break
    with pytest.raises(AssertionError, match="generator bug"):
        sim.assert_pinned_design(world, broken, pinned)

    sparse = sim.pinned_deck_rows(world, pinned, reserved=5, ordinary=5)
    with pytest.raises(AssertionError, match="decisive clusters"):
        sim.assert_pinned_design(world, sparse, pinned)


def test_sc3b_crossover_breaks_first_alias_only() -> None:
    world = sim.reference_world("standard")
    pinned = sim.pinned_roles(world)
    first = ("agent_18", "deck_b1_00")
    second = ("agent_19", "deck_b2_00")
    assert first[0] in pinned and second[0] in pinned

    base = sim.sc7_rows(world)
    sim.assert_pinned_design(world, base, pinned)
    augmented = sim.append_crossover_clusters(world, base, first, second)

    assert len({row["cluster_id"] for row in augmented}) == world.clusters + 2
    assert len(sim.crossover_cluster_ids(augmented, first)) == 2

    post = fit(GameTable.from_records(augmented), world.agent_registry(), world.deck_registry())
    sim.assert_alias_in_null_space(post.patterns, second[0], second[1])
    log_worth = {row.agent_id: row for row in post.log_worth()}
    assert log_worth[first[0]].status in {"weak", "identified"}
    assert log_worth[first[0]].interval is not None
    assert log_worth[second[0]].status == "aliased"
    assert log_worth[second[0]].interval is None

    sdwp_agents = {row.agent_id for row in post.sdwp(world.field(24)).rows}
    assert first[0] in sdwp_agents
    assert second[0] not in sdwp_agents
