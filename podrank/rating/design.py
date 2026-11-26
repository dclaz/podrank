"""Parameter indexing and compressed design data for R1.

Patterns are stored as compact integer arrays: agent/deck indices by seat, four
winner counts, and swaps only when the swap effect is enabled. Dense designs are
kept only as a small reference representation for fixtures and equivalence tests.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from podrank.data.registry import AgentRegistry, DeckRegistry
from podrank.rating.spec import ModelSpec
from podrank.types import ClusterKey, GameRow


def cluster_key(row: GameRow) -> ClusterKey:
    """Return the tagged C.2 cluster key for a row."""

    if row.cluster_id is None:
        return ("game", row.game_id)
    return ("cluster", row.cluster_id)


@dataclass(frozen=True, slots=True)
class ParameterIndex:
    """Canonical full parameter order (Appendix C.2.1)."""

    names: tuple[str, ...]
    positions: Mapping[str, int]
    supported_agents: tuple[str, ...]
    supported_decks: tuple[str, ...]
    supported_brackets: tuple[int, ...]
    has_eta: bool
    block_slices: Mapping[str, tuple[int, int]]

    @classmethod
    def build(
        cls,
        agents: AgentRegistry,
        decks: DeckRegistry,
        supported_agents: Iterable[str],
        supported_decks: Iterable[str],
        *,
        swap_effect: bool,
    ) -> ParameterIndex:
        agent_ids = tuple(sorted(set(supported_agents)))
        deck_ids = tuple(sorted(set(supported_decks)))
        bracket_ids = tuple(sorted({decks.by_id[deck_id].bracket for deck_id in deck_ids}))

        names: list[str] = []
        blocks: dict[str, tuple[int, int]] = {}
        start = 0
        for agent_id in agent_ids:
            names.append(f"theta[{agent_id}]")
        blocks["theta"] = (start, len(names))
        start = len(names)
        for seat in range(4):
            names.append(f"seat[{seat}]")
        blocks["seat"] = (start, len(names))
        start = len(names)
        for bracket in bracket_ids:
            names.append(f"mu[{bracket}]")
        blocks["mu"] = (start, len(names))
        start = len(names)
        for deck_id in deck_ids:
            names.append(f"delta[{deck_id}]")
        blocks["delta"] = (start, len(names))
        start = len(names)
        if swap_effect:
            names.append("eta")
        blocks["eta"] = (start, len(names))

        return cls(
            names=tuple(names),
            positions=MappingProxyType({name: index for index, name in enumerate(names)}),
            supported_agents=agent_ids,
            supported_decks=deck_ids,
            supported_brackets=bracket_ids,
            has_eta=swap_effect,
            block_slices=MappingProxyType(blocks),
        )

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.positions[name]

    def vector(self, values: Mapping[str, float]) -> np.ndarray:
        result = np.zeros(len(self), dtype=np.float64)
        for name, value in values.items():
            if name in self.positions:
                result[self.positions[name]] = float(value)
        return result

    def block(self, name: str) -> np.ndarray:
        start, stop = self.block_slices[name]
        result = np.zeros(len(self), dtype=np.float64)
        result[start:stop] = 1.0
        return result

    @property
    def reduced_size(self) -> int:
        return (
            len(self.supported_agents) + len(self.supported_decks) + 4 + (1 if self.has_eta else 0)
        )

    @property
    def beta_start(self) -> int:
        return len(self.supported_agents)

    @property
    def reduced_seat_start(self) -> int:
        return len(self.supported_agents) + len(self.supported_decks)

    @property
    def reduced_eta_index(self) -> int | None:
        if not self.has_eta:
            return None
        return len(self.supported_agents) + len(self.supported_decks) + 4


PatternKey = tuple[tuple[str, str, str, str], tuple[str, str, str, str], tuple[int, int, int, int]]


@dataclass(frozen=True, slots=True, eq=False)
class Pattern:
    """Dense reference pattern, used by small fixtures and equivalence tests."""

    key: PatternKey
    design: np.ndarray
    target: np.ndarray
    weight: int


@dataclass(frozen=True, slots=True)
class CompactPatterns:
    """Compact owned representation of the compressed likelihood frame."""

    agent_indices: np.ndarray
    deck_indices: np.ndarray
    targets: np.ndarray
    swap_counts: np.ndarray | None
    deck_mu_positions: np.ndarray
    deck_delta_positions: np.ndarray
    seat_positions: np.ndarray
    eta_position: int | None
    reduction: np.ndarray
    cluster_keys: tuple[ClusterKey, ...]
    cluster_patterns: np.ndarray
    cluster_winner_seats: np.ndarray
    cluster_indices: np.ndarray
    cluster_counts: np.ndarray

    @property
    def n_patterns(self) -> int:
        return int(self.agent_indices.shape[0])

    @property
    def reduced_size(self) -> int:
        return int(self.reduction.shape[1])

    def swap_batch(self, start: int, stop: int) -> np.ndarray | None:
        if self.swap_counts is None:
            return None
        return self.swap_counts[start:stop]


@dataclass(frozen=True, slots=True)
class PatternTable:
    """Compressed likelihood data for one fitted frame."""

    index: ParameterIndex
    patterns: tuple[Pattern, ...] = ()
    compact: CompactPatterns | None = None
    cluster_counts: Mapping[tuple[int, int, ClusterKey], int] = field(default_factory=dict)
    n_games: int = 0

    def __len__(self) -> int:
        if self.compact is not None:
            return self.compact.n_patterns
        return len(self.patterns)

    @property
    def n_patterns(self) -> int:
        return len(self)


def design_row(
    agents: tuple[str, str, str, str],
    decks: tuple[str, str, str, str],
    swaps: tuple[int, int, int, int],
    index: ParameterIndex,
    deck_registry: DeckRegistry,
) -> np.ndarray:
    """Build the four-seat dense design matrix for one fully specified context."""

    design = np.zeros((4, len(index)), dtype=np.float64)
    for seat in range(4):
        deck_id = decks[seat]
        bracket = deck_registry.by_id[deck_id].bracket
        design[seat, index.index(f"theta[{agents[seat]}]")] = 1.0
        design[seat, index.index(f"mu[{bracket}]")] = 1.0
        design[seat, index.index(f"delta[{deck_id}]")] = 1.0
        design[seat, index.index(f"seat[{seat}]")] = 1.0
        if index.has_eta:
            design[seat, index.index("eta")] = float(swaps[seat])
    return design


def _reduction_matrix(
    index: ParameterIndex,
    deck_mu_positions: np.ndarray,
    deck_delta_positions: np.ndarray,
    seat_positions: np.ndarray,
    eta_position: int | None,
) -> np.ndarray:
    """Map reduced ``(theta, beta, seat, eta)`` coordinates to full parameters."""

    reduced = index.reduced_size
    matrix = np.zeros((len(index), reduced), dtype=np.float64)
    theta_start, theta_stop = index.block_slices["theta"]
    for position in range(theta_stop - theta_start):
        matrix[theta_start + position, position] = 1.0
    for deck_position in range(len(index.supported_decks)):
        column = index.beta_start + deck_position
        matrix[int(deck_mu_positions[deck_position]), column] += 1.0
        matrix[int(deck_delta_positions[deck_position]), column] += 1.0
    for seat in range(4):
        matrix[int(seat_positions[seat]), index.reduced_seat_start + seat] = 1.0
    eta_reduced = index.reduced_eta_index
    if eta_position is not None:
        assert eta_reduced is not None
        matrix[eta_position, eta_reduced] = 1.0
    return matrix


def _effective_swaps(row: GameRow, spec: ModelSpec) -> tuple[int, int, int, int]:
    if spec.swap_effect:
        return row.swap_counts
    return (0, 0, 0, 0)


def build_patterns(
    rows: Sequence[GameRow],
    index: ParameterIndex,
    decks: DeckRegistry,
    spec: ModelSpec,
) -> PatternTable:
    """Group likelihood-used rows by canonical compact design row."""

    grouped: dict[PatternKey, list[GameRow]] = defaultdict(list)
    for row in rows:
        key: PatternKey = (row.agents, row.decks, _effective_swaps(row, spec))
        grouped[key].append(row)

    agent_lookup = {agent_id: position for position, agent_id in enumerate(index.supported_agents)}
    deck_lookup = {deck_id: position for position, deck_id in enumerate(index.supported_decks)}
    deck_mu_positions = np.array(
        [index.index(f"mu[{decks.by_id[deck_id].bracket}]") for deck_id in index.supported_decks],
        dtype=np.int32,
    )
    deck_delta_positions = np.array(
        [index.index(f"delta[{deck_id}]") for deck_id in index.supported_decks],
        dtype=np.int32,
    )
    seat_positions = np.array(
        list(range(index.block_slices["seat"][0], index.block_slices["seat"][1])),
        dtype=np.int32,
    )
    eta_position = index.positions.get("eta")

    agent_indices: list[list[int]] = []
    deck_indices: list[list[int]] = []
    target_rows: list[list[int]] = []
    swap_rows: list[list[int]] = []
    cluster_counts: dict[tuple[int, int, ClusterKey], int] = defaultdict(int)

    for pattern_index, key in enumerate(sorted(grouped)):
        pattern_rows = grouped[key]
        agents = [agent_lookup[agent_id] for agent_id in key[0]]
        deck_ids = [deck_lookup[deck_id] for deck_id in key[1]]
        target = [0, 0, 0, 0]
        for row in pattern_rows:
            if row.winner_seat is None:  # guarded by the likelihood frame
                raise ValueError("likelihood rows must have a winner_seat")
            target[row.winner_seat] += 1
            cluster_counts[(pattern_index, row.winner_seat, cluster_key(row))] += 1
        agent_indices.append(agents)
        deck_indices.append(deck_ids)
        target_rows.append(target)
        if spec.swap_effect:
            swap_rows.append(list(key[2]))

    agent_array = np.asarray(agent_indices, dtype=np.int32).reshape(-1, 4)
    deck_array = np.asarray(deck_indices, dtype=np.int32).reshape(-1, 4)
    target_array = np.asarray(target_rows, dtype=np.int64).reshape(-1, 4)
    swap_array = np.asarray(swap_rows, dtype=np.int32).reshape(-1, 4) if spec.swap_effect else None

    cluster_keys = tuple(sorted({cluster for (_, _, cluster) in cluster_counts}))
    cluster_index = {cluster: position for position, cluster in enumerate(cluster_keys)}
    ordered_clusters = sorted(cluster_counts)
    cluster_patterns = np.array([item[0] for item in ordered_clusters], dtype=np.int32)
    cluster_winner_seats = np.array([item[1] for item in ordered_clusters], dtype=np.int8)
    cluster_indices = np.array(
        [cluster_index[item[2]] for item in ordered_clusters], dtype=np.int32
    )
    cluster_count_values = np.array(
        [cluster_counts[item] for item in ordered_clusters], dtype=np.int64
    )

    reduction = _reduction_matrix(
        index,
        deck_mu_positions,
        deck_delta_positions,
        seat_positions,
        eta_position,
    )
    compact = CompactPatterns(
        agent_indices=agent_array,
        deck_indices=deck_array,
        targets=target_array,
        swap_counts=swap_array,
        deck_mu_positions=deck_mu_positions,
        deck_delta_positions=deck_delta_positions,
        seat_positions=seat_positions,
        eta_position=eta_position,
        reduction=reduction,
        cluster_keys=cluster_keys,
        cluster_patterns=cluster_patterns,
        cluster_winner_seats=cluster_winner_seats,
        cluster_indices=cluster_indices,
        cluster_counts=cluster_count_values,
    )
    return PatternTable(
        index=index,
        compact=compact,
        cluster_counts=MappingProxyType({}),
        n_games=int(target_array.sum()),
    )


__all__ = [
    "CompactPatterns",
    "ParameterIndex",
    "Pattern",
    "PatternKey",
    "PatternTable",
    "build_patterns",
    "cluster_key",
    "design_row",
]
