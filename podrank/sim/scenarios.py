"""Frozen scenario generators for the predefined R1 scenarios (SC1–SC8).

Only the deterministic generators needed by the hard-gate scenarios live here.
The ordinary world generator remains in :mod:`podrank.sim.world`; these helpers
override cluster composition where a scenario prescribes it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.rating.design import ParameterIndex, PatternTable
from podrank.rating.identification import difference_gram
from podrank.sim.world import SimWorld

# Role mapping from §11.2. Keys are world agent counts, values are
# ``pinned agent -> exclusive deck``.
_PINNED_BY_AGENT_COUNT: dict[int, dict[str, str]] = {
    20: {"agent_18": "deck_b1_00", "agent_19": "deck_b2_00"},
    10: {"agent_08": "deck_b1_00", "agent_09": "deck_b2_00"},
}

# Role mapping from §11.2 for SC3: sparse agent -> reserved cluster count and
# pinned agent -> exclusive deck.
_SC3_BY_AGENT_COUNT: dict[int, tuple[dict[str, int], dict[str, str]]] = {
    20: (
        {"agent_17": 3, "agent_18": 6, "agent_19": 12},
        {"agent_16": "deck_b1_00"},
    ),
    10: (
        {"agent_07": 3, "agent_08": 6, "agent_09": 12},
        {"agent_06": "deck_b1_00"},
    ),
}


@dataclass(frozen=True, slots=True)
class _Context:
    world: SimWorld
    agents: AgentRegistry
    decks: DeckRegistry
    theta: Mapping[str, float]
    mu: Mapping[int, float]
    delta: Mapping[str, float]
    seats: tuple[float, ...]


def _context(world: SimWorld) -> _Context:
    return _Context(
        world=world,
        agents=world.agent_registry(),
        decks=world.deck_registry(),
        theta=world.theta,
        mu=world.bracket_effects,
        delta=world.deck_effects,
        seats=world.seats,
    )


def pinned_roles(world: SimWorld) -> dict[str, str]:
    """Return the SC7 pinned agent/deck mapping for a reference world."""

    try:
        return dict(_PINNED_BY_AGENT_COUNT[world.agents])
    except KeyError as error:
        raise ValueError(f"no pinned-deck roles for a world with {world.agents} agents") from error


def sc3_roles(world: SimWorld) -> tuple[dict[str, int], dict[str, str]]:
    """Return the SC3 sparse and pinned role mappings for a reference world."""

    try:
        sparse, pinned = _SC3_BY_AGENT_COUNT[world.agents]
    except KeyError as error:
        raise ValueError(f"no SC3 roles for a world with {world.agents} agents") from error
    return dict(sparse), dict(pinned)


def _row(
    context: _Context,
    cluster_index: int,
    cluster_id: str,
    rotation: int,
    agents_seated: Sequence[str],
    decks_seated: Sequence[str],
    outcome: str,
    winner_seat: int | None,
) -> dict[str, object]:
    return {
        "game_id": f"{cluster_id}r{rotation}",
        "played_at": f"2026-01-01T00:00:{cluster_index % 60:02d}Z",
        "agents": list(agents_seated),
        "decks": list(decks_seated),
        "policy_versions": [context.agents.by_id[agent].policy_version for agent in agents_seated],
        "swap_counts": [0, 0, 0, 0],
        "outcome": outcome,
        "winner_seat": winner_seat,
        "suite": "train",
        "cluster_id": cluster_id,
        "engine_version": "world",
    }


def _utilities(
    context: _Context,
    agents_seated: Sequence[str],
    decks_seated: Sequence[str],
    bracket: int,
) -> np.ndarray:
    mu = context.mu[bracket]
    return np.array(
        [
            context.theta[agents_seated[seat]]
            + mu
            + context.delta[decks_seated[seat]]
            + context.seats[seat]
            for seat in range(4)
        ],
        dtype=np.float64,
    )


def _sampled_outcome(
    context: _Context,
    cluster_index: int,
    rotation: int,
    utilities: np.ndarray,
    base_gumbel: np.ndarray,
    gumbel_rng: np.random.Generator,
    *,
    force_decisive: bool,
) -> tuple[str, int | None]:
    world = context.world
    if force_decisive:
        return "win", int(np.argmax(utilities + base_gumbel))
    gumbel = base_gumbel
    if world.rho_div > 0:
        redraw = gumbel_rng.random(4) < world.rho_div
        gumbel = np.where(redraw, gumbel_rng.gumbel(size=4), base_gumbel)
    outcome_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(world.seed, spawn_key=(3, cluster_index, rotation)))
    )
    draw = float(outcome_rng.random())
    if draw < world.draw_rate:
        return "draw", None
    if draw < world.draw_rate + world.cap_rate:
        return "cap", None
    if draw < world.draw_rate + world.cap_rate + world.failure_rate:
        return "failed", None
    return "win", int(np.argmax(utilities + gumbel))


def _reserved_cluster(
    context: _Context,
    cluster_index: int,
    cluster_id: str,
    pinned_agent: str,
    pinned_deck: str,
    free_agents: Sequence[str],
    *,
    include_anchor: bool,
) -> list[dict[str, object]]:
    world = context.world
    bracket = context.decks.by_id[pinned_deck].bracket
    other_deck_pool = sorted(
        deck
        for deck in world.deck_ids
        if context.decks.by_id[deck].bracket == bracket and deck != pinned_deck
    )
    if len(other_deck_pool) < 3:
        raise ValueError("pinned deck bracket has fewer than four usable decks")

    composition = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(world.seed, spawn_key=(1, cluster_index)))
    )
    if include_anchor and "agent_00" in free_agents:
        first = "agent_00"
        remaining = [agent for agent in free_agents if agent != first]
        others = [
            first,
            *(str(agent) for agent in composition.choice(remaining, size=2, replace=False)),
        ]
    else:
        others = [
            str(agent) for agent in composition.choice(list(free_agents), size=3, replace=False)
        ]
    other_decks = [str(deck) for deck in composition.choice(other_deck_pool, size=3, replace=False)]
    pinned_seat = int(composition.integers(0, 4))
    remaining_seats = [seat for seat in range(4) if seat != pinned_seat]

    ordered_agents: list[str | None] = [None] * 4
    ordered_decks: list[str | None] = [None] * 4
    ordered_agents[pinned_seat] = pinned_agent
    ordered_decks[pinned_seat] = pinned_deck
    for position, seat in enumerate(remaining_seats):
        ordered_agents[seat] = others[position]
        ordered_decks[seat] = other_decks[position]

    gumbel_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(world.seed, spawn_key=(2, cluster_index)))
    )
    base_gumbel = gumbel_rng.gumbel(size=4)

    rows: list[dict[str, object]] = []
    for rotation in range(4):
        seated_agents: list[str] = [str(agent) for agent in ordered_agents]
        for position, seat in enumerate(remaining_seats):
            seated_agents[seat] = others[(position + rotation) % 3]
        seated_decks = [str(deck) for deck in ordered_decks]
        utilities = _utilities(context, seated_agents, seated_decks, bracket)
        outcome, winner = _sampled_outcome(
            context,
            cluster_index,
            rotation,
            utilities,
            base_gumbel,
            gumbel_rng,
            force_decisive=rotation == 0,
        )
        rows.append(
            _row(
                context,
                cluster_index,
                cluster_id,
                rotation,
                seated_agents,
                seated_decks,
                outcome,
                winner,
            )
        )
    return rows


def _ordinary_cluster(
    context: _Context,
    cluster_index: int,
    cluster_id: str,
    free_agents: Sequence[str],
    allowed_decks: Mapping[int, Sequence[str]],
) -> list[dict[str, object]]:
    world = context.world
    composition = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(world.seed, spawn_key=(1, cluster_index)))
    )
    bracket = int(composition.integers(0, world.brackets))
    bracket_decks = list(allowed_decks[bracket])
    if len(bracket_decks) < 4:
        raise ValueError("scenario bracket has fewer than four usable decks")
    ordered_decks = [
        str(bracket_decks[int(index)])
        for index in composition.choice(len(bracket_decks), size=4, replace=False)
    ]
    ordered_agents = [
        str(free_agents[int(index)])
        for index in composition.choice(len(free_agents), size=4, replace=False)
    ]

    gumbel_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(world.seed, spawn_key=(2, cluster_index)))
    )
    base_gumbel = gumbel_rng.gumbel(size=4)

    rows: list[dict[str, object]] = []
    for rotation in range(4):
        seated_agents = [ordered_agents[(position + rotation) % 4] for position in range(4)]
        utilities = _utilities(context, seated_agents, ordered_decks, bracket)
        outcome, winner = _sampled_outcome(
            context,
            cluster_index,
            rotation,
            utilities,
            base_gumbel,
            gumbel_rng,
            force_decisive=False,
        )
        rows.append(
            _row(
                context,
                cluster_index,
                cluster_id,
                rotation,
                seated_agents,
                ordered_decks,
                outcome,
                winner,
            )
        )
    return rows


def _sparse_cluster(
    context: _Context,
    cluster_index: int,
    cluster_id: str,
    sparse_agent: str,
    ordinary_agents: Sequence[str],
    allowed_decks: Mapping[int, Sequence[str]],
    *,
    include_anchor: bool,
) -> list[dict[str, object]]:
    world = context.world
    composition = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(world.seed, spawn_key=(1, cluster_index)))
    )
    bracket = int(composition.integers(0, world.brackets))
    deck_pool = list(allowed_decks[bracket])
    if len(deck_pool) < 4:
        raise ValueError("scenario bracket has fewer than four usable decks")
    ordered_decks = [
        str(deck_pool[int(index)])
        for index in composition.choice(len(deck_pool), size=4, replace=False)
    ]
    if include_anchor and "agent_00" in ordinary_agents:
        first = "agent_00"
        remaining = [agent for agent in ordinary_agents if agent != first]
        others = [
            first,
            *(str(agent) for agent in composition.choice(remaining, size=2, replace=False)),
        ]
    else:
        others = [
            str(agent) for agent in composition.choice(list(ordinary_agents), size=3, replace=False)
        ]
    ordered_agents = [sparse_agent, *others]

    gumbel_rng = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(world.seed, spawn_key=(2, cluster_index)))
    )
    base_gumbel = gumbel_rng.gumbel(size=4)

    rows: list[dict[str, object]] = []
    for rotation in range(4):
        seated_agents = [ordered_agents[(position + rotation) % 4] for position in range(4)]
        utilities = _utilities(context, seated_agents, ordered_decks, bracket)
        outcome, winner = _sampled_outcome(
            context,
            cluster_index,
            rotation,
            utilities,
            base_gumbel,
            gumbel_rng,
            force_decisive=rotation == 0,
        )
        rows.append(
            _row(
                context,
                cluster_index,
                cluster_id,
                rotation,
                seated_agents,
                ordered_decks,
                outcome,
                winner,
            )
        )
    return rows


def sc3_rows(
    world: SimWorld,
    *,
    pinned_reserved: int = 30,
    ordinary: int | None = None,
) -> list[dict[str, object]]:
    """Generate the SC3 sparse-support variant of a reference world.

    Twenty-one clusters are reserved across the three sparse agents, a fourth
    agent is pinned to an exclusive deck, and ordinary clusters exclude every
    sparse or pinned agent and the pinned deck.
    """

    sparse, pinned = sc3_roles(world)
    pinned_decks = set(pinned.values())
    sparse_agents = set(sparse)
    pinned_agents = set(pinned)
    ordinary_agents = [
        agent
        for agent in world.agent_ids
        if agent not in sparse_agents and agent not in pinned_agents
    ]
    if len(ordinary_agents) < 4:
        raise ValueError("SC3 needs at least four ordinary agents")
    for agent, deck in pinned.items():
        if agent not in world.agent_ids:
            raise ValueError(f"unknown pinned agent {agent!r}")
        if deck not in world.deck_ids:
            raise ValueError(f"unknown pinned deck {deck!r}")
    for agent in sparse:
        if agent not in world.agent_ids:
            raise ValueError(f"unknown sparse agent {agent!r}")

    reserved_total = sum(sparse.values()) + pinned_reserved * len(pinned)
    if ordinary is None:
        ordinary = world.clusters - reserved_total
    if ordinary < 0:
        raise ValueError("world has fewer clusters than the SC3 reserved clusters")

    context = _context(world)
    allowed_decks = {
        bracket: [
            deck
            for deck in world.deck_ids
            if context.decks.by_id[deck].bracket == bracket and deck not in pinned_decks
        ]
        for bracket in range(world.brackets)
    }

    rows: list[dict[str, object]] = []
    cluster_index = 0
    for sparse_agent, count in sorted(sparse.items()):
        for offset in range(count):
            cluster_id = f"s{cluster_index:03d}"
            rows.extend(
                _sparse_cluster(
                    context,
                    cluster_index,
                    cluster_id,
                    sparse_agent,
                    ordinary_agents,
                    allowed_decks,
                    include_anchor=offset == 0,
                )
            )
            cluster_index += 1
    for pinned_agent, pinned_deck in sorted(pinned.items()):
        for offset in range(pinned_reserved):
            cluster_id = f"s{cluster_index:03d}"
            rows.extend(
                _reserved_cluster(
                    context,
                    cluster_index,
                    cluster_id,
                    pinned_agent,
                    pinned_deck,
                    ordinary_agents,
                    include_anchor=offset == 0,
                )
            )
            cluster_index += 1
    for _ in range(ordinary):
        cluster_id = f"s{cluster_index:03d}"
        rows.extend(
            _ordinary_cluster(context, cluster_index, cluster_id, ordinary_agents, allowed_decks)
        )
        cluster_index += 1
    return rows


def pinned_deck_rows(
    world: SimWorld,
    pinned: Mapping[str, str],
    *,
    reserved: int = 30,
    ordinary: int | None = None,
) -> list[dict[str, object]]:
    """Generate an SC7-style design with exclusive pinned agent/deck pairs.

    Each pinned agent sits on its deck in every rotation, the deck is used by no
    other agent, and no ordinary cluster contains a pinned agent or deck.
    """

    if not pinned:
        raise ValueError("at least one pinned agent/deck pair is required")
    if reserved < 1:
        raise ValueError("reserved must be positive")
    pinned_agents = set(pinned)
    pinned_decks = set(pinned.values())
    free_agents = [agent for agent in world.agent_ids if agent not in pinned_agents]
    if len(free_agents) < 4:
        raise ValueError("pinned-deck design needs at least four free agents")
    for agent, deck in pinned.items():
        if agent not in world.agent_ids:
            raise ValueError(f"unknown pinned agent {agent!r}")
        if deck not in world.deck_ids:
            raise ValueError(f"unknown pinned deck {deck!r}")

    if ordinary is None:
        ordinary = world.clusters - reserved * len(pinned)
    if ordinary < 0:
        raise ValueError("world has fewer clusters than the reserved pinned clusters")

    context = _context(world)
    allowed_decks = {
        bracket: [
            deck
            for deck in world.deck_ids
            if context.decks.by_id[deck].bracket == bracket and deck not in pinned_decks
        ]
        for bracket in range(world.brackets)
    }

    rows: list[dict[str, object]] = []
    cluster_index = 0
    for pinned_agent, pinned_deck in sorted(pinned.items()):
        for offset in range(reserved):
            cluster_id = f"s{cluster_index:03d}"
            rows.extend(
                _reserved_cluster(
                    context,
                    cluster_index,
                    cluster_id,
                    pinned_agent,
                    pinned_deck,
                    free_agents,
                    include_anchor=offset == 0,
                )
            )
            cluster_index += 1
    for _ in range(ordinary):
        cluster_id = f"s{cluster_index:03d}"
        rows.extend(
            _ordinary_cluster(context, cluster_index, cluster_id, free_agents, allowed_decks)
        )
        cluster_index += 1
    return rows


def sc7_rows(world: SimWorld, **kwargs: int) -> list[dict[str, object]]:
    """Generate the SC7 connected-but-confounded design for a reference world."""

    return pinned_deck_rows(world, pinned_roles(world), **kwargs)


def append_crossover_clusters(
    world: SimWorld,
    rows: Sequence[Mapping[str, Any]],
    first: tuple[str, str],
    second: tuple[str, str],
    *,
    count: int = 2,
) -> list[dict[str, object]]:
    """Append SC3b crossover clusters that break the first pinned alias.

    The first pinned agent plays a pod of four non-pinned decks from its pinned
    deck's bracket, with three ordinary agents including the anchor. The second
    pinned agent stays out of the crossover, so its alias survives. Both pinned
    decks remain exclusive.
    """

    if count < 1:
        raise ValueError("count must be positive")
    first_agent, first_deck = first
    second_agent, second_deck = second
    context = _context(world)
    bracket = context.decks.by_id[first_deck].bracket
    pinned_decks = {first_deck, second_deck}
    deck_pool = sorted(
        deck
        for deck in world.deck_ids
        if context.decks.by_id[deck].bracket == bracket and deck not in pinned_decks
    )
    if len(deck_pool) < 4:
        raise ValueError("crossover bracket has fewer than four non-pinned decks")
    free_agents = [
        agent
        for agent in world.agent_ids
        if agent not in {first_agent, second_agent} and agent != "agent_00"
    ]
    if len(free_agents) < 2:
        raise ValueError("crossover design needs two ordinary agents besides the anchor")

    existing = {str(row["cluster_id"]) for row in rows}
    base_index = len(existing)
    result = [dict(row) for row in rows]
    for offset in range(count):
        cluster_index = base_index + offset
        cluster_id = f"x{cluster_index:03d}"
        while cluster_id in existing:  # pragma: no cover - deterministic uniqueness guard
            cluster_index += 1
            cluster_id = f"x{cluster_index:03d}"
        existing.add(cluster_id)

        composition = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(world.seed, spawn_key=(1, cluster_index)))
        )
        others = [
            "agent_00",
            *(str(agent) for agent in composition.choice(free_agents, size=2, replace=False)),
        ]
        ordered_decks = [str(deck) for deck in composition.choice(deck_pool, size=4, replace=False)]
        ordered_agents = [first_agent, *others]

        gumbel_rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(world.seed, spawn_key=(2, cluster_index)))
        )
        base_gumbel = gumbel_rng.gumbel(size=4)
        for rotation in range(4):
            seated_agents = [ordered_agents[(position + rotation) % 4] for position in range(4)]
            utilities = _utilities(context, seated_agents, ordered_decks, bracket)
            outcome, winner = _sampled_outcome(
                context,
                cluster_index,
                rotation,
                utilities,
                base_gumbel,
                gumbel_rng,
                force_decisive=rotation == 0,
            )
            result.append(
                _row(
                    context,
                    cluster_index,
                    cluster_id,
                    rotation,
                    seated_agents,
                    ordered_decks,
                    outcome,
                    winner,
                )
            )
    return result


def crossover_cluster_ids(rows: Sequence[Mapping[str, Any]], first: tuple[str, str]) -> set[str]:
    """Return clusters where the first pinned agent played outside its deck."""

    agent, deck = first
    ids: set[str] = set()
    for row in rows:
        agents = [str(value) for value in row["agents"]]
        if agent not in agents:
            continue
        decks = [str(value) for value in row["decks"]]
        seat = agents.index(agent)
        if decks[seat] != deck:
            ids.add(str(row["cluster_id"]))
    return ids


def _decisive_graph(rows: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for row in rows:
        if row["outcome"] != "win":
            continue
        agents = [str(agent) for agent in row["agents"]]
        for agent in agents:
            graph.setdefault(agent, set())
        for left in agents:
            for right in agents:
                if left != right:
                    graph[left].add(right)
    return graph


def assert_pinned_design(
    world: SimWorld,
    rows: Sequence[Mapping[str, Any]],
    pinned: Mapping[str, str],
    *,
    min_clusters: int = 30,
) -> None:
    """Assert the §11.2 pinned-deck generator invariants (a)–(c).

    Failures are generator bugs and must never be read as identification
    results.
    """

    pinned_agents = set(pinned)
    pinned_decks = set(pinned.values())

    # (a) every game of a used pinned deck has its pinned agent in that seat.
    for row in rows:
        seats = [str(deck) for deck in row["decks"]]
        agents = [str(agent) for agent in row["agents"]]
        for deck, agent in zip(seats, agents, strict=True):
            if deck in pinned_decks:
                owner = next(
                    (candidate for candidate, owned in pinned.items() if owned == deck), None
                )
                if owner is None or agent != owner:
                    raise AssertionError(f"generator bug: pinned deck {deck!r} used by {agent!r}")
            if agent in pinned_agents:
                owner = pinned[agent]
                if deck != owner:
                    raise AssertionError(
                        f"generator bug: pinned agent {agent!r} played deck {deck!r}"
                    )

    # (b) each pinned agent appears in at least ``min_clusters`` decisive clusters.
    decisive_clusters: dict[str, set[str]] = {}
    for row in rows:
        if row["outcome"] != "win":
            continue
        agents = [str(agent) for agent in row["agents"]]
        for agent in pinned_agents:
            if agent in agents:
                decisive_clusters.setdefault(agent, set()).add(str(row["cluster_id"]))
    for agent in pinned_agents:
        count = len(decisive_clusters.get(agent, set()))
        if count < min_clusters:
            raise AssertionError(
                f"generator bug: pinned agent {agent!r} has {count} decisive clusters"
            )

    # (c) the decisive agent graph is connected and contains the anchor.
    graph = _decisive_graph(rows)
    if "agent_00" not in graph:
        raise AssertionError("generator bug: anchor is absent from the decisive graph")
    seen = {"agent_00"}
    stack = ["agent_00"]
    while stack:
        node = stack.pop()
        for neighbour in graph.get(node, set()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    missing = sorted(pinned_agents - seen)
    if missing:
        raise AssertionError(f"generator bug: pinned agents disconnected from anchor: {missing}")


def alias_vector(index: ParameterIndex, agent_id: str, deck_id: str) -> np.ndarray:
    """Return the full-parameter alias direction ``e_theta[a] - e_delta[d]``."""

    result = np.zeros(len(index), dtype=np.float64)
    result[index.index(f"theta[{agent_id}]")] = 1.0
    result[index.index(f"delta[{deck_id}]")] = -1.0
    return result


def assert_alias_in_null_space(patterns: PatternTable, agent_id: str, deck_id: str) -> float:
    """Assert the §11.2 (d) D4 null-space condition and return the quadratic."""

    vector = alias_vector(patterns.index, agent_id, deck_id)
    gram = difference_gram(patterns)
    scale = float(np.linalg.norm(gram, 2))
    if scale == 0.0:
        raise AssertionError("generator bug: D4 Gram matrix is zero")
    quadratic = float(vector @ gram @ vector)
    if quadratic > 1e-12 * scale:
        raise AssertionError(
            f"generator bug: alias {agent_id}/{deck_id} is not in the likelihood "
            f"null space ({quadratic:.3e} > {1e-12 * scale:.3e})"
        )
    return quadratic


__all__ = [
    "alias_vector",
    "assert_alias_in_null_space",
    "assert_pinned_design",
    "pinned_deck_rows",
    "pinned_roles",
    "sc7_rows",
]
