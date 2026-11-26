"""Deterministic scenario smoke checks (SC6, SC8); full SC1–SC8 gates remain open."""

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
