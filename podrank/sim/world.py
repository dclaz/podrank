"""Frozen reference-world definitions and a small deterministic simulator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import norm

from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.rating.field import Field

_WORLDS = Path(__file__).with_name("worlds")


@dataclass(frozen=True, slots=True)
class SimWorld:
    name: str
    seed: int
    agents: int
    brackets: int
    decks_per_bracket: int
    clusters: int
    balanced: bool

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(f"agent_{index:02d}" for index in range(self.agents))

    @property
    def deck_ids(self) -> tuple[str, ...]:
        return tuple(
            f"deck_b{bracket}_{index:02d}"
            for bracket in range(self.brackets)
            for index in range(self.decks_per_bracket)
        )

    @property
    def theta(self) -> dict[str, float]:
        quantiles = sorted(
            0.05 + 0.6 * float(norm.ppf((index - 0.5) / (self.agents - 1)))
            for index in range(1, self.agents)
        )
        return dict(zip(self.agent_ids, [0.0, *quantiles], strict=True))

    @property
    def bracket_effects(self) -> dict[int, float]:
        if self.brackets == 1:
            return {0: 0.0}
        values = np.linspace(-0.3, 0.3, self.brackets)
        return {bracket: float(values[bracket]) for bracket in range(self.brackets)}

    @property
    def deck_effects(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for bracket in range(self.brackets):
            values = sorted(
                float(0.5 * norm.ppf((index - 0.5) / self.decks_per_bracket))
                for index in range(1, self.decks_per_bracket + 1)
            )
            for index, value in enumerate(values):
                result[f"deck_b{bracket}_{index:02d}"] = value
        return result

    @property
    def seats(self) -> tuple[float, float, float, float]:
        return (0.15, 0.05, -0.05, -0.15)

    def agent_registry(self) -> AgentRegistry:
        return AgentRegistry.from_records(
            [
                {
                    "agent_id": agent_id,
                    "kind": "heuristic" if agent_id == "agent_00" else "checkpoint",
                    "policy_version": f"{agent_id}_v1",
                    "is_anchor": agent_id == "agent_00",
                }
                for agent_id in self.agent_ids
            ]
        )

    def deck_registry(self) -> DeckRegistry:
        return DeckRegistry.from_records(
            [
                {
                    "deck_id": deck_id,
                    "bracket": int(deck_id.split("_")[1][1:]),
                    "split": "train",
                    "deck_version": f"{deck_id}_v1",
                }
                for deck_id in self.deck_ids
            ]
        )

    def game_rows(self) -> list[dict[str, object]]:
        agents = self.agent_registry()
        decks = self.deck_registry()
        theta = self.theta
        mu = self.bracket_effects
        delta = self.deck_effects
        rows: list[dict[str, object]] = []
        for cluster in range(self.clusters):
            bracket = cluster % self.brackets
            bracket_decks = sorted(
                deck for deck in self.deck_ids if decks.by_id[deck].bracket == bracket
            )
            shift = (cluster // self.brackets) % self.decks_per_bracket
            ordered_decks = bracket_decks[shift:] + bracket_decks[:shift]
            ordered_agents = [
                self.agent_ids[(cluster + offset) % self.agents] for offset in range(4)
            ]
            for rotation in range(4):
                seated_agents = [ordered_agents[(position + rotation) % 4] for position in range(4)]
                utilities = np.array(
                    [
                        theta[seated_agents[seat]]
                        + mu[bracket]
                        + delta[ordered_decks[seat]]
                        + self.seats[seat]
                        for seat in range(4)
                    ],
                    dtype=np.float64,
                )
                if self.balanced:
                    rng = np.random.Generator(
                        np.random.PCG64(
                            np.random.SeedSequence(self.seed, spawn_key=(1, cluster, rotation))
                        )
                    )
                    winner = int(np.argmax(utilities + rng.gumbel(size=4)))
                    outcome = "win"
                else:
                    outcome = "win"
                    winner = int(np.argmax(utilities))
                rows.append(
                    {
                        "game_id": f"b{cluster:03d}r{rotation}",
                        "played_at": f"2026-01-01T00:00:{cluster % 60:02d}Z",
                        "agents": seated_agents,
                        "decks": ordered_decks,
                        "policy_versions": [
                            agents.by_id[agent].policy_version for agent in seated_agents
                        ],
                        "swap_counts": [0, 0, 0, 0],
                        "outcome": outcome,
                        "winner_seat": winner,
                        "suite": "train",
                        "cluster_id": f"b{cluster:03d}",
                        "engine_version": "world",
                    }
                )
        return rows

    def field(self, n_contexts: int = 24) -> Field:
        decks = self.deck_registry()
        recipe = {
            "field_id": "sim_field",
            "version": 1,
            "recipe": {
                "deck_split": "train",
                "pod": {
                    "bracket_policy": "same_bracket",
                    "bracket_weights": "equal",
                    "min_decks_per_bracket": self.decks_per_bracket,
                    "distinct_decks": True,
                    "swap_counts": 0,
                },
                "opponents": {
                    "pool": ["agent_00", "agent_01", "agent_02"],
                    "draw": "independent_uniform_per_seat",
                },
                "focal_seat": "uniform",
                "n_contexts": n_contexts,
                "seed": self.seed,
            },
        }
        return Field.materialise(recipe, decks=decks)


def reference_world(name: str) -> SimWorld:
    path = _WORLDS / f"{name}.json"
    if not path.exists():
        raise ValueError(f"unknown reference world {name!r}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return SimWorld(
        name=str(raw["name"]),
        seed=int(raw["seed"]),
        agents=int(raw["agents"]),
        brackets=int(raw["brackets"]),
        decks_per_bracket=int(raw["decks_per_bracket"]),
        clusters=int(raw["clusters"]),
        balanced=bool(raw["balanced"]),
    )


__all__ = ["SimWorld", "reference_world"]
