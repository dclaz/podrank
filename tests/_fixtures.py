"""Small hand-built records used by the P1 data tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from podrank import AgentRegistry, DeckRegistry

AGENT_VERSIONS = {"H": "H1", "A": "A1", "B": "B1", "C": "C1", "D": "D1"}
AGENT_ORDER = ("C", "A", "H", "B")


def agent_rows(
    *,
    anchors: Sequence[str] = ("H",),
    extra_ids: Sequence[str] = (),
) -> list[dict[str, object]]:
    ids = (*AGENT_ORDER, *extra_ids)
    rows: list[dict[str, object]] = []
    for agent_id in ids:
        rows.append(
            {
                "agent_id": agent_id,
                "kind": "heuristic" if agent_id == "H" else "checkpoint",
                "policy_version": AGENT_VERSIONS[agent_id],
                "is_anchor": agent_id in anchors,
                "tags": ["anchor"] if agent_id in anchors else [],
                "metadata": {"source": "fixture"},
            }
        )
    return rows


def deck_rows(*, colors: Mapping[str, str] | None = None) -> list[dict[str, object]]:
    values = {
        "d1": "wu",
        "d2": "ub",
        "d3": "br",
        "d4": "rg",
        "d5": "wg",
    }
    if colors is not None:
        values.update(colors)
    rows: list[dict[str, object]] = []
    for deck_id in ("d3", "d1", "d5", "d2", "d4"):
        rows.append(
            {
                "deck_id": deck_id,
                "bracket": 1 if deck_id != "d5" else 2,
                "split": "holdout" if deck_id == "d5" else "train",
                "deck_version": f"{deck_id}v1",
                "tags": {"colors": values[deck_id], "speed": "fast"},
                "metadata": {"source": "fixture"},
            }
        )
    return rows


def registries(
    *,
    anchors: Sequence[str] = ("H",),
    extra_ids: Sequence[str] = (),
    colors: Mapping[str, str] | None = None,
) -> tuple[AgentRegistry, DeckRegistry]:
    return (
        AgentRegistry.from_records(agent_rows(anchors=anchors, extra_ids=extra_ids)),
        DeckRegistry.from_records(deck_rows(colors=colors)),
    )


def make_game(
    game_id: str,
    *,
    played_at: str | datetime = "2026-01-01T00:00:00Z",
    agents: Sequence[str] = ("H", "A", "B", "C"),
    decks: Sequence[str] = ("d1", "d2", "d3", "d4"),
    policy_versions: Sequence[str | None] | None = None,
    deck_versions: Sequence[str | None] | None = None,
    swap_counts: Sequence[int] = (0, 0, 0, 0),
    outcome: str = "win",
    winner_seat: int | None = None,
    suite: str = "train",
    cluster_id: str | None = None,
    **extra: object,
) -> dict[str, object]:
    if policy_versions is None:
        versions = tuple(AGENT_VERSIONS[str(agent)] for agent in agents)
    else:
        versions = tuple(policy_versions)
    if deck_versions is None:
        deck_version_values: tuple[str | None, ...] = tuple(None for _ in decks)
    else:
        deck_version_values = tuple(deck_versions)
    if winner_seat is None and outcome == "win":
        winner_seat = 0
    record: dict[str, object] = {
        "game_id": game_id,
        "played_at": played_at,
        "agents": list(agents),
        "decks": list(decks),
        "policy_versions": list(versions),
        "deck_versions": list(deck_version_values),
        "swap_counts": list(swap_counts),
        "outcome": outcome,
        "winner_seat": winner_seat,
        "suite": suite,
        "cluster_id": cluster_id,
        "engine_version": "e1",
    }
    record.update(extra)
    return record


T0 = "2026-01-01T00:00:00Z"
T1 = "2026-01-01T00:00:01Z"
T2 = "2026-01-01T00:00:02Z"
T3 = "2026-01-01T00:00:03Z"
T3_DT = datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)


def f7_rows() -> list[dict[str, object]]:
    return [
        make_game("g1", played_at=T0, outcome="win", winner_seat=0, cluster_id="b1"),
        make_game("g2", played_at=T1, outcome="draw", cluster_id="b1"),
        make_game("g3", played_at=T2, outcome="win", winner_seat=1, cluster_id="b1"),
        make_game("g4", played_at=T3, outcome="win", winner_seat=2, cluster_id="b1"),
    ]


def f7_manifest() -> dict[str, set[str]]:
    return {"b1": {"g1", "g2", "g3", "g4"}}
