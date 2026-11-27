"""Prescribed-replication S5 state-variant counts."""

from __future__ import annotations

import itertools
import json
import tempfile
from dataclasses import replace
from pathlib import Path

from podrank.compare import BlockDescriptor, Comparison, PodDistribution, Superiority
from podrank.data import AgentRegistry, DeckRegistry

A, B = "agent_00", "agent_01"
AGENTS = AgentRegistry.from_records(
    [
        {"agent_id": A, "kind": "heuristic", "policy_version": "1", "is_anchor": True},
        {"agent_id": B, "kind": "checkpoint", "policy_version": "2", "is_anchor": False},
    ]
)
DECKS = DeckRegistry.from_records(
    [
        {"deck_id": f"d{index}", "bracket": 0, "split": "train", "deck_version": f"v{index}"}
        for index in range(1, 5)
    ]
)
PODS = PodDistribution.from_support(
    "pods",
    1,
    "duplicate",
    [{"decks": ("d1", "d2", "d3", "d4"), "opponents": None, "weight": 1.0}],
)


def descriptor(index: int) -> BlockDescriptor:
    return BlockDescriptor(
        block_id=f"block {index}",
        block_index=index,
        design="duplicate",
        deal_id=f"deal-{index}",
        decks=("d1", "d2", "d3", "d4"),
        deck_versions=("v1", "v2", "v3", "v4"),
        swap_counts=(0, 0, 0, 0),
        opponents=None,
        policy_versions={A: "1", B: "2"},
        engine_version="engine",
        pod_distribution_id=PODS.pod_distribution_id,
        pod_distribution_version=PODS.version,
        pod_distribution_hash=PODS.pod_distribution_hash,
    )


def row(
    block: int, rotation: int, game_id: str, *, outcome: str = "win", winner: int | None = None
):
    agents = [B, B, B, B]
    agents[rotation] = A
    if outcome == "win" and winner is None:
        winner = rotation
    if outcome != "win":
        winner = None
    return {
        "game_id": game_id,
        "played_at": "2026-01-01T00:00:00Z",
        "agents": tuple(agents),
        "decks": ("d1", "d2", "d3", "d4"),
        "policy_versions": tuple({A: "1", B: "2"}[agent] for agent in agents),
        "deck_versions": ("v1", "v2", "v3", "v4"),
        "swap_counts": (0, 0, 0, 0),
        "outcome": outcome,
        "winner_seat": winner,
        "suite": "train",
        "cluster_id": f"block {block}",
        "block_id": f"block {block}",
        "rotation": rotation,
        "role": None,
        "deal_id": f"deal-{block}",
        "engine_version": "engine",
    }


def comparison(*, max_blocks: int = 1, replay: str = "stochastic", alpha: float = 0.99):
    return Comparison(
        a=A,
        b=B,
        design="duplicate",
        agents=AGENTS,
        decks=DECKS,
        engine_version="engine",
        pod_distribution=PODS,
        hypothesis=Superiority(0.0),
        max_blocks=max_blocks,
        alpha=alpha,
        replay=replay,  # type: ignore[arg-type]
    )


def block_rows(block: int) -> list[dict[str, object]]:
    return [row(block, rotation, f"b{block}r{rotation}") for rotation in range(4)]


def stochastic_retry() -> bool:
    cmp = comparison(replay="stochastic")
    cmp.declare_blocks([descriptor(0)])
    cmp.ingest([row(0, 1, "fail", outcome="failed")])
    cmp.ingest([row(0, 0, "r0"), row(0, 2, "r2"), row(0, 3, "r3")])
    before = cmp.rating_rows()[0]
    cmp.ingest([row(0, 1, "recover", winner=1)])
    return (
        cmp.report().blocks_consumed == 1
        and cmp.rating_rows()[0].to_records() == before.to_records()
    )


def deterministic_recovery() -> bool:
    cmp = comparison(replay="deterministic")
    cmp.declare_blocks([descriptor(0)])
    cmp.ingest([row(0, 1, "fail", outcome="failed")])
    cmp.ingest([row(0, 0, "r0"), row(0, 2, "r2"), row(0, 3, "r3")])
    if cmp.report().blocks_consumed != 0:
        return False
    cmp.ingest([row(0, 1, "recover", winner=1)])
    return cmp.report().blocks_consumed == 1


def late_same_outcome() -> bool:
    cmp = comparison(replay="deterministic")
    cmp.declare_blocks([descriptor(0)])
    cmp.ingest(block_rows(0))
    revision = cmp.rating_rows()[1].export_revision
    cmp.ingest([row(0, 2, "late", winner=2)])
    return cmp.rating_rows()[1].export_revision == revision and cmp.evidence_valid


def late_different_stochastic() -> bool:
    cmp = comparison(replay="stochastic")
    cmp.declare_blocks([descriptor(0)])
    cmp.ingest(block_rows(0))
    cmp.ingest([row(0, 2, "late", winner=3)])
    return cmp.evidence_valid


def late_different_deterministic() -> bool:
    cmp = comparison(replay="deterministic", max_blocks=1)
    cmp.declare_blocks([descriptor(0)])
    cmp.ingest(block_rows(0))
    cmp.ingest([row(0, 2, "late", winner=3)])
    return not cmp.evidence_valid and cmp.status() == "INCONCLUSIVE_INVALID"


def arrival_invariance() -> bool:
    reports = []
    for order in itertools.permutations(range(3)):
        cmp = comparison(max_blocks=3, alpha=0.2)
        cmp.declare_blocks([descriptor(0), descriptor(1), descriptor(2)])
        for block in order:
            cmp.ingest(block_rows(block))
        reports.append(replace(cmp.report(), comparison_id="same"))
    return all(report == reports[0] for report in reports)


def resume_equality() -> bool:
    full = comparison(max_blocks=2, replay="deterministic")
    full.declare_blocks([descriptor(0), descriptor(1)])
    full.ingest(block_rows(0))
    full.ingest(block_rows(1))
    resumed = comparison(max_blocks=2, replay="deterministic")
    resumed.declare_blocks([descriptor(0), descriptor(1)])
    resumed.ingest(block_rows(0))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "comparison.json"
        resumed.save(path)
        resumed = Comparison.load(path)
        resumed.ingest(block_rows(1))
    table_a, export_a = full.rating_rows()
    table_b, export_b = resumed.rating_rows()
    return (
        replace(full.report(), comparison_id="same")
        == replace(resumed.report(), comparison_id="same")
        and table_a.to_records() == table_b.to_records()
        and replace(export_a, comparison_id="same") == replace(export_b, comparison_id="same")
    )


def main() -> None:
    reps = 10_000
    functions = [
        ("stochastic_retry", stochastic_retry),
        ("deterministic_recovery", deterministic_recovery),
        ("late_same_outcome", late_same_outcome),
        ("late_different_stochastic", late_different_stochastic),
        ("late_different_deterministic", late_different_deterministic),
        ("arrival_invariance", arrival_invariance),
    ]
    results = {}
    for name, function in functions:
        passed = sum(1 for _ in range(reps) if function())
        results[name] = {"replications": reps, "passed": passed}
    resume_reps = 10_000
    results["resume_equality"] = {
        "replications": resume_reps,
        "passed": sum(1 for _ in range(resume_reps) if resume_equality()),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
