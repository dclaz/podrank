"""Frozen comparison worlds over the standard decks, brackets and opponent pool."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import numpy as np

from podrank.sim.world import SimWorld, reference_world

_WORLDS = Path(__file__).with_name("worlds")
Design = Literal["duplicate", "paired"]
_OPPONENT_POOL = ("agent_00", "agent_01", "agent_02")
_cache: dict[tuple[str, float, Design], tuple[np.ndarray, np.ndarray | None]] = {}
_interaction_cache: dict[tuple[str, float, Design, int], tuple[np.ndarray, np.ndarray | None]] = {}


@lru_cache(maxsize=2)
def _standard_world(name: str) -> SimWorld:
    return reference_world(name)


@dataclass(frozen=True, slots=True)
class CompareCell:
    label: str
    mean: float
    theta_a_duplicate: float
    theta_a_paired: float

    def theta_a(self, design: Design) -> float:
        return self.theta_a_duplicate if design == "duplicate" else self.theta_a_paired


_SUPPORT_PODS = 64


def _pods() -> tuple[tuple[int, tuple[str, str, str, str]], ...]:
    world = _standard_world("standard")
    candidates = [
        (bracket, combination)
        for bracket in range(world.brackets)
        for combination in itertools.combinations(
            sorted(deck for deck in world.deck_ids if deck.startswith(f"deck_b{bracket}_")), 4
        )
    ]
    rng = np.random.default_rng(707)
    selected = sorted(rng.choice(len(candidates), size=_SUPPORT_PODS, replace=False).tolist())
    return tuple(candidates[index] for index in selected)


@lru_cache(maxsize=1)
def support() -> tuple[tuple[int, tuple[str, str, str, str], tuple[str, str, str] | None], ...]:
    items: list[tuple[int, tuple[str, str, str, str], tuple[str, str, str] | None]] = []
    for bracket, decks in _pods():
        items.append((bracket, decks, None))
        for opponents in itertools.product(_OPPONENT_POOL, repeat=3):
            triple: tuple[str, str, str] = (opponents[0], opponents[1], opponents[2])
            items.append((bracket, decks, triple))
    return tuple(items)


def _utilities(
    world_name: str,
    theta_a: float,
    bracket: int,
    decks: tuple[str, str, str, str],
    agents: tuple[str, str, str, str],
    *,
    interactions: dict[tuple[str, str], float] | None = None,
    opponent_effects: dict[tuple[str, str], float] | None = None,
    kingmaker: float = 0.0,
) -> np.ndarray:
    world = _standard_world(world_name)
    theta = world.theta
    mu = world.bracket_effects[bracket]
    delta = world.deck_effects
    values = np.asarray(
        [
            (theta_a if agents[seat] == "agent_00" else theta[agents[seat]])
            + mu
            + delta[decks[seat]]
            + world.seats[seat]
            for seat in range(4)
        ],
        dtype=np.float64,
    )
    if interactions is not None:
        for seat in range(4):
            values[seat] += interactions.get((agents[seat], decks[seat]), 0.0)
    if opponent_effects is not None:
        for seat in range(4):
            values[seat] += sum(
                opponent_effects.get((agents[seat], other), 0.0)
                for other in agents
                if other != agents[seat]
            )
    if kingmaker:
        values[int(np.argmax(values))] -= kingmaker
    return values


def _prob(value: np.ndarray, seat: int) -> float:
    shifted = value - value.max()
    weights = np.exp(shifted)
    return float(weights[seat] / weights.sum())


_array_cache: dict[
    tuple[str, Design], tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]
] = {}


def _design_arrays(
    world_name: str, design: Design
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    key = (world_name, design)
    cached = _array_cache.get(key)
    if cached is not None:
        return cached
    world = _standard_world(world_name)
    theta = world.theta
    delta = world.deck_effects
    items = support()
    if design == "duplicate":
        filtered = [item for item in items if item[2] is None]
    else:
        filtered = [item for item in items if item[2] is not None]
    base = np.empty((len(filtered), 4, 4), dtype=np.float64)
    mask = np.zeros((len(filtered), 4, 4), dtype=np.float64)
    base_b = np.empty((len(filtered), 4, 4), dtype=np.float64) if design == "paired" else None
    mask_b = np.zeros((len(filtered), 4, 4), dtype=np.float64) if design == "paired" else None
    for index, (bracket, decks, opponents) in enumerate(filtered):
        for rotation in range(4):
            if design == "duplicate":
                agents = ["agent_01"] * 4
                agents[rotation] = "agent_00"
            else:
                assert opponents is not None
                agents = ["agent_01"] * 4
                agents[rotation] = "agent_00"
                for offset, opponent in enumerate(opponents, start=1):
                    agents[(rotation + offset) % 4] = opponent
            for seat, agent in enumerate(agents):
                value = (
                    theta[agent]
                    + world.bracket_effects[bracket]
                    + delta[decks[seat]]
                    + world.seats[seat]
                )
                base[index, rotation, seat] = value
                mask[index, rotation, seat] = 1.0 if agent == "agent_00" else 0.0
            if base_b is not None and mask_b is not None:
                agents_b = list(agents)
                agents_b[rotation] = "agent_01"
                for seat, agent in enumerate(agents_b):
                    value = (
                        theta[agent]
                        + world.bracket_effects[bracket]
                        + delta[decks[seat]]
                        + world.seats[seat]
                    )
                    base_b[index, rotation, seat] = value
                    mask_b[index, rotation, seat] = 1.0 if agent == "agent_00" else 0.0
    result: tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] = (
        base,
        mask,
        base_b if base_b is not None else None,
        mask_b if mask_b is not None else None,
    )
    _array_cache[key] = result
    return result


def _support_probabilities(
    world_name: str, theta_a: float, design: Design
) -> tuple[np.ndarray, np.ndarray | None]:
    key = (world_name, float(theta_a), design)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    base, mask, base_b, mask_b = _design_arrays(world_name, design)
    values = base + theta_a * mask
    shifted = values - values.max(axis=2, keepdims=True)
    weights = np.exp(shifted)
    probabilities = weights / weights.sum(axis=2, keepdims=True)
    p_a = np.stack([probabilities[:, rotation, rotation] for rotation in range(4)], axis=1)
    p_b = None
    if base_b is not None and mask_b is not None:
        values_b = base_b + theta_a * mask_b
        shifted_b = values_b - values_b.max(axis=2, keepdims=True)
        weights_b = np.exp(shifted_b)
        probabilities_b = weights_b / weights_b.sum(axis=2, keepdims=True)
        p_b = np.stack([probabilities_b[:, rotation, rotation] for rotation in range(4)], axis=1)
    result = (p_a, p_b)
    _cache[key] = result
    return result


@dataclass(frozen=True, slots=True)
class CompareWorld:
    name: str
    seed: int
    a: str
    b: str
    theta_b: float
    draw_rate: float
    cap_rate: float
    cells: tuple[CompareCell, ...]
    world_name: str = "standard"

    def _marginal(self, theta_a: float, design: Design) -> tuple[np.ndarray, np.ndarray | None]:
        return _support_probabilities(self.world_name, theta_a, design)

    def mean_lift(self, theta_a: float, design: Design) -> float:
        p_a, p_b = self._marginal(theta_a, design)
        q = self.draw_rate + self.cap_rate
        if design == "duplicate":
            return float(p_a.mean() + 0.25 * q - 0.25)
        assert p_b is not None
        return float(0.25 * (p_a - p_b).sum(axis=1).mean())

    def calibrate(self, target: float, design: Design) -> float:
        low, high = -8.0, 8.0
        for _ in range(60):
            middle = (low + high) / 2.0
            if self.mean_lift(middle, design) < target:
                low = middle
            else:
                high = middle
        return (low + high) / 2.0

    def block_probabilities(
        self, theta_a: float, design: Design, blocks: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray | None]:
        p_a, p_b = self._marginal(theta_a, design)
        indices = rng.integers(0, len(p_a), size=blocks)
        gathered_a = p_a[indices]
        gathered_b = None if p_b is None else p_b[indices]
        return gathered_a, gathered_b

    def interaction_probabilities(
        self, theta_a: float, design: Design, cell_index: int
    ) -> tuple[np.ndarray, np.ndarray | None]:
        from podrank.sim.scenarios import sc4_interactions

        key = (self.world_name, float(theta_a), design, cell_index)
        cached = _interaction_cache.get(key)
        if cached is not None:
            return cached
        world = _standard_world(self.world_name)
        agent_deck, agent_opponent = sc4_interactions(world, self.seed, cell_index)
        items = [item for item in support() if (item[2] is None) == (design == "duplicate")]
        p_a = np.empty((len(items), 4), dtype=np.float64)
        p_b = np.empty((len(items), 4), dtype=np.float64) if design == "paired" else None
        for index, (bracket, decks, opponents) in enumerate(items):
            for rotation in range(4):
                if design == "duplicate":
                    agents = ["agent_01"] * 4
                    agents[rotation] = "agent_00"
                else:
                    assert opponents is not None
                    agents = ["agent_01"] * 4
                    agents[rotation] = "agent_00"
                    for offset, opponent in enumerate(opponents, start=1):
                        agents[(rotation + offset) % 4] = opponent
                utilities = _utilities(
                    self.world_name,
                    theta_a,
                    bracket,
                    decks,
                    cast(tuple[str, str, str, str], tuple(agents)),
                    interactions=dict(agent_deck),
                    opponent_effects=dict(agent_opponent),
                    kingmaker=0.2,
                )
                p_a[index, rotation] = _prob(utilities, rotation)
                if p_b is not None:
                    agents_b = list(agents)
                    agents_b[rotation] = "agent_01"
                    utilities = _utilities(
                        self.world_name,
                        theta_a,
                        bracket,
                        decks,
                        cast(tuple[str, str, str, str], tuple(agents_b)),
                        interactions=dict(agent_deck),
                        opponent_effects=dict(agent_opponent),
                        kingmaker=0.2,
                    )
                    p_b[index, rotation] = _prob(utilities, rotation)
        _interaction_cache[key] = (p_a, p_b)
        return p_a, p_b

    def interaction_target(self, theta_a: float, design: Design, cell_index: int) -> float:
        p_a, p_b = self.interaction_probabilities(theta_a, design, cell_index)
        q = self.draw_rate + self.cap_rate
        if design == "duplicate":
            return float(p_a.mean() + 0.25 * q - 0.25)
        assert p_b is not None
        return float(0.25 * (p_a - p_b).sum(axis=1).mean())

    def interaction_block_probabilities(
        self,
        theta_a: float,
        design: Design,
        cell_index: int,
        blocks: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        p_a, p_b = self.interaction_probabilities(theta_a, design, cell_index)
        indices = rng.integers(0, len(p_a), size=blocks)
        return p_a[indices], None if p_b is None else p_b[indices]


def reference_compare_world(name: str) -> CompareWorld:
    path = _WORLDS / f"{name}.json"
    if not path.exists():
        raise ValueError(f"unknown comparison world {name!r}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CompareWorld(
        name=str(raw["name"]),
        seed=int(raw["seed"]),
        a=str(raw["a"]),
        b=str(raw["b"]),
        theta_b=float(raw["theta_b"]),
        draw_rate=float(raw["draw_rate"]),
        cap_rate=float(raw["cap_rate"]),
        cells=tuple(
            CompareCell(
                label=str(cell["label"]),
                mean=float(cell["mean"]),
                theta_a_duplicate=float(cell["theta_a"]["duplicate"]),
                theta_a_paired=float(cell["theta_a"]["paired"]),
            )
            for cell in raw["cells"]
        ),
        world_name=str(raw.get("world", "standard")),
    )


__all__ = ["CompareCell", "CompareWorld", "reference_compare_world", "support"]
