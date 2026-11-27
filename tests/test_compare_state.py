"""Comparison state-machine fixtures F12, F16 and F19."""

from __future__ import annotations

import pytest

from podrank.compare import (
    BlockDescriptor,
    Comparison,
    PodDistribution,
    Superiority,
)
from podrank.data import AgentRegistry, DeckRegistry

AGENTS = AgentRegistry.from_records(
    [
        {"agent_id": "agent_00", "kind": "heuristic", "policy_version": "1", "is_anchor": True},
        {"agent_id": "agent_01", "kind": "checkpoint", "policy_version": "2", "is_anchor": False},
    ]
)
DECKS = DeckRegistry.from_records(
    [
        {"deck_id": deck, "bracket": 0, "split": "train", "deck_version": f"v{index}"}
        for index, deck in enumerate(("d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8"), start=1)
    ]
)
PODS = PodDistribution.from_support(
    "pods",
    1,
    "duplicate",
    [{"decks": ("d1", "d2", "d3", "d4"), "opponents": None, "weight": 1.0}],
)
A = "agent_00"
B = "agent_01"
POLICIES = {A: "1", B: "2"}


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
        policy_versions=POLICIES,
        engine_version="engine",
        pod_distribution_id=PODS.pod_distribution_id,
        pod_distribution_version=PODS.version,
        pod_distribution_hash=PODS.pod_distribution_hash,
    )


def row(
    block: int,
    rotation: int,
    *,
    game_id: str,
    outcome: str = "win",
    winner_seat: int | None = None,
) -> dict[str, object]:
    agents = [B, B, B, B]
    agents[rotation] = A
    if outcome != "win":
        winner_seat = None
    elif winner_seat is None:
        winner_seat = rotation
    return {
        "game_id": game_id,
        "played_at": "2026-01-01T00:00:00Z",
        "agents": tuple(agents),
        "decks": ("d1", "d2", "d3", "d4"),
        "policy_versions": tuple(POLICIES[agent] for agent in agents),
        "deck_versions": ("v1", "v2", "v3", "v4"),
        "swap_counts": (0, 0, 0, 0),
        "outcome": outcome,
        "winner_seat": winner_seat,
        "suite": "train",
        "cluster_id": f"block {block}",
        "block_id": f"block {block}",
        "rotation": rotation,
        "role": None,
        "deal_id": f"deal-{block}",
        "engine_version": "engine",
    }


def make_comparison(*, max_blocks: int, alpha: float = 0.01, replay: str = "stochastic"):
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


def complete_block(block: int) -> list[dict[str, object]]:
    return [row(block, rotation, game_id=f"b{block}r{rotation}") for rotation in range(4)]


def test_f19_budget_and_atomic_failure() -> None:
    comparison = make_comparison(max_blocks=2)
    comparison.declare_blocks([descriptor(index) for index in range(4)])
    comparison.ingest(complete_block(0))
    comparison.ingest(complete_block(1))
    report = comparison.report()
    assert report.blocks_consumed == 2
    assert report.status == "BUDGET_EXHAUSTED"

    comparison.ingest(complete_block(2))
    assert comparison.report().blocks_consumed == 2
    with pytest.raises(ValueError, match="exceed"):
        comparison.extend_budget(2)
    comparison.extend_budget(3)
    assert comparison.report().blocks_consumed == 3

    comparison.extend_budget(4)
    comparison.ingest([row(3, 0, game_id="b3r0")])
    comparison.mark_block_failed("block 3")
    report = comparison.report()
    assert report.blocks_consumed == 4
    assert len(comparison.failure_records()) == 3
    table, export = comparison.rating_rows()
    block_three = [item for item in table if item["cluster_id"] == "block 3"]
    assert len(block_three) == 1
    assert block_three[0]["game_id"] == "b3r0"
    assert export.evidence_valid


def test_f19_batch_and_single_event_logs_match() -> None:
    batch = make_comparison(max_blocks=2)
    batch.declare_blocks([descriptor(0), descriptor(1)])
    batch.ingest(complete_block(1) + complete_block(0))

    single = make_comparison(max_blocks=2)
    single.declare_blocks([descriptor(0), descriptor(1)])
    for item in complete_block(1) + complete_block(0):
        single.ingest([item])

    assert batch.report().blocks_consumed == single.report().blocks_consumed == 2
    assert [event.kind for event in batch._events] == [event.kind for event in single._events]


def test_f16_export_revision_and_late_replay_violation() -> None:
    comparison = make_comparison(max_blocks=1, replay="deterministic")
    comparison.declare_blocks([descriptor(0)])
    comparison.ingest([row(0, 1, game_id="e1", outcome="failed")])
    comparison.ingest(
        [
            row(0, 0, game_id="e2"),
            row(0, 2, game_id="e3"),
            row(0, 3, game_id="e4"),
        ]
    )
    comparison.ingest([row(0, 1, game_id="e5", winner_seat=1)])
    assert comparison.report().blocks_consumed == 1
    table, export = comparison.rating_rows()
    assert len(table) == 4
    assert export.export_revision == 1
    assert export.evidence_valid

    comparison.ingest([row(0, 1, game_id="e6", winner_seat=1)])
    assert comparison.rating_rows()[1].export_revision == 1

    comparison.ingest([row(0, 2, game_id="e7", winner_seat=3)])
    table, export = comparison.rating_rows()
    assert export.export_revision == 2
    assert len(table) == 3
    assert export.omissions == (("block 0", 2, None, "replay_violation"),)
    assert not export.evidence_valid
    assert comparison.status() == "INCONCLUSIVE_INVALID"
    assert comparison.report().decision_at_stop == "INCONCLUSIVE"


def test_f12_late_contradiction_preserves_above() -> None:
    comparison = make_comparison(max_blocks=2, alpha=0.9, replay="deterministic")
    comparison.declare_blocks([descriptor(0), descriptor(1)])
    comparison.ingest(complete_block(0))
    assert comparison.report().decision_at_stop == "ABOVE"

    comparison.ingest([row(0, 2, game_id="late", winner_seat=3)])
    report = comparison.report()
    assert report.decision_at_stop == "ABOVE"
    assert report.evidence_valid is False
    assert comparison.status() == "INCONCLUSIVE_INVALID"


def test_mark_failed_and_integrity_conflict() -> None:
    comparison = make_comparison(max_blocks=1)
    comparison.declare_blocks([descriptor(0)])
    comparison.ingest([row(0, 0, game_id="same")])
    comparison.ingest([row(0, 0, game_id="same", winner_seat=3)])
    assert comparison.status() == "INCONCLUSIVE_INVALID"
    assert comparison.report().blocks_consumed == 0
    assert len(comparison.rating_rows()[0]) == 0


def _replace_descriptor(index: int, **changes: object) -> BlockDescriptor:
    from dataclasses import replace

    return replace(descriptor(index), **changes)  # type: ignore[arg-type]


def test_constructor_validation() -> None:
    from dataclasses import replace

    base = make_comparison(max_blocks=1)
    cases = [
        ({"a": ""}, "a must be a non-empty string"),
        ({"a": "missing"}, "a is not in the agent snapshot"),
        ({"b": base.a}, "a and b must differ"),
        ({"design": "paired"}, "design does not match"),
        ({"engine_version": ""}, "engine_version"),
        ({"max_blocks": 0}, "max_blocks"),
        ({"replay": "bad"}, "replay"),
        ({"min_blocks_for_failure_rule": -1}, "min_blocks"),
        ({"max_failed_fraction": 2}, "max_failed_fraction"),
        ({"alpha_up": 0.0}, "alpha_up"),
        ({"alpha_low": 1.0}, "alpha_low"),
        ({"cs_alpha": 0.0}, "cs_alpha"),
    ]
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace(base, **changes)  # type: ignore[arg-type]


def test_declaration_validation() -> None:
    comparison = make_comparison(max_blocks=1)
    with pytest.raises(ValueError, match="sequence"):
        comparison.declare_blocks("x")
    with pytest.raises(ValueError, match="at least one"):
        comparison.declare_blocks([])

    gap = make_comparison(max_blocks=1)
    with pytest.raises(ValueError, match="contiguous"):
        gap.declare_blocks([descriptor(1)])
    duplicate = make_comparison(max_blocks=1)
    with pytest.raises(ValueError, match="unique"):
        duplicate.declare_blocks([descriptor(0), _replace_descriptor(1, block_id="block 0")])

    cases = [
        (_replace_descriptor(0, design="paired", opponents=(B, B, B)), "design does not match"),
        (_replace_descriptor(0, engine_version="other"), "engine_version"),
        (_replace_descriptor(0, pod_distribution_hash="x"), "pod distribution"),
        (_replace_descriptor(0, swap_counts=(1, 0, 0, 0)), "swap counts"),
        (_replace_descriptor(0, decks=("zz", "d2", "d3", "d4")), "unknown deck"),
        (_replace_descriptor(0, deck_versions=("x", "v2", "v3", "v4")), "deck version"),
        (
            _replace_descriptor(
                0, decks=("d5", "d6", "d7", "d8"), deck_versions=("v5", "v6", "v7", "v8")
            ),
            "outside the pod distribution",
        ),
        (_replace_descriptor(0, policy_versions={A: "9", B: "2"}), "policy version"),
    ]
    for bad, message in cases:
        candidate = make_comparison(max_blocks=1)
        with pytest.raises(ValueError, match=message):
            candidate.declare_blocks([bad])

    declared = make_comparison(max_blocks=1)
    declared.declare_blocks([descriptor(0)])
    with pytest.raises(ValueError, match="already been declared"):
        declared.declare_blocks([descriptor(0)])


def test_ingest_classification_and_match_errors() -> None:
    comparison = make_comparison(max_blocks=1)
    with pytest.raises(ValueError, match="declare_blocks"):
        comparison.ingest([row(0, 0, game_id="x")])
    comparison.declare_blocks([descriptor(0)])

    report = comparison.ingest([{"bad": 1}])
    assert report.rejected == 1
    report = comparison.ingest([{**row(0, 0, game_id="x"), "game_id": 1}])
    assert report.rejected == 1
    report = comparison.ingest([row(9, 0, game_id="x")])
    assert report.rejected == 1

    base = row(0, 0, game_id="base")
    comparison.ingest([base])
    report = comparison.ingest([base])
    assert report.transport_duplicates == 1

    metadata = {**base, "suite": "other"}
    report = comparison.ingest([metadata])
    assert report.id_conflicts == 1
    assert comparison.status() == "CONTINUE"

    mutations = [
        ({"cluster_id": None}, "declared block"),
        ({"block_id": "other"}, "block_id"),
        ({"deal_id": "other"}, "deal_id"),
        ({"engine_version": "other"}, "engine_version"),
        ({"decks": ("d2", "d1", "d3", "d4")}, "decks"),
        ({"deck_versions": (None, None, None, None)}, "non-null"),
        ({"deck_versions": ("x", "v2", "v3", "v4")}, "deck versions"),
        ({"swap_counts": (1, 0, 0, 0)}, "swap counts"),
        ({"policy_versions": ("9", "2", "2", "2")}, "policy version"),
        ({"rotation": None}, "rotation"),
        ({"role": "a"}, "duplicate rows"),
        ({"agents": (B, A, B, B)}, "agents"),
    ]
    for change, message in mutations:
        report = comparison.ingest([{**row(0, 0, game_id=f"bad-{message}"), **change}])
        assert report.rejected == 1, message


def test_mark_and_extend_validation() -> None:
    comparison = make_comparison(max_blocks=1)
    comparison.declare_blocks([descriptor(0)])
    with pytest.raises(ValueError, match="unknown block"):
        comparison.mark_failed("missing", 0)
    with pytest.raises(ValueError, match="rotation"):
        comparison.mark_failed("block 0", 4)
    with pytest.raises(ValueError, match="null role"):
        comparison.mark_failed("block 0", 0, "a")
    comparison.mark_failed("block 0", 0)
    with pytest.raises(ValueError, match="unknown block"):
        comparison.mark_block_failed("missing")
    with pytest.raises(ValueError, match="integer"):
        comparison.extend_budget(2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exceed"):
        comparison.extend_budget(1)


def test_below_and_failure_rule() -> None:
    below = make_comparison(max_blocks=1, alpha=0.99)
    below.declare_blocks([descriptor(0)])
    below.ingest(
        [
            row(0, rotation, game_id=f"b{rotation}", winner_seat=(rotation + 1) % 4)
            for rotation in range(4)
        ]
    )
    assert below.status() == "BELOW"

    failures = make_comparison(max_blocks=1, alpha=0.99)
    failures.min_blocks_for_failure_rule = 1
    failures.max_failed_fraction = 0.0
    failures.declare_blocks([descriptor(0)])
    failures.ingest(
        [row(0, rotation, game_id=f"f{rotation}", outcome="failed") for rotation in range(4)]
    )
    assert failures.status() == "INCONCLUSIVE"


def test_comparison_save_load_round_trip(tmp_path) -> None:
    comparison = make_comparison(max_blocks=1, replay="deterministic")
    comparison.declare_blocks([descriptor(0)])
    comparison.ingest([row(0, 1, game_id="e1", outcome="failed")])
    comparison.ingest([row(0, 0, game_id="e2"), row(0, 2, game_id="e3"), row(0, 3, game_id="e4")])
    comparison.ingest([row(0, 1, game_id="e5", winner_seat=1)])
    path = tmp_path / "comparison.json"
    comparison.save(path)
    loaded = Comparison.load(path)

    assert loaded.comparison_id == comparison.comparison_id
    assert loaded.status() == comparison.status()
    assert loaded.report() == comparison.report()
    assert loaded.failure_records() == comparison.failure_records()
    assert loaded.rating_rows() == comparison.rating_rows()
    assert loaded._events == comparison._events

    loaded.mark_failed("block 0", 1)
    assert loaded.rating_rows() == comparison.rating_rows()


def test_comparison_load_rejects_bad_envelopes(tmp_path) -> None:
    comparison = make_comparison(max_blocks=1)
    comparison.declare_blocks([descriptor(0)])
    path = tmp_path / "comparison.json"
    comparison.save(path)
    envelope = path.read_text(encoding="utf-8")

    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="comparison"):
        Comparison.load(path)

    path.write_text(
        '{"kind":"podrank.comparison","schema_version":1,"package_version":"x","payload":{},"extra":1}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="envelope"):
        Comparison.load(path)

    path.write_text(
        '{"kind":"other","schema_version":1,"package_version":"x","payload":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported"):
        Comparison.load(path)

    path.write_text(
        envelope.replace(
            '"kind":"podrank.comparison"', '"kind":"podrank.comparison","kind":"podrank.comparison"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        Comparison.load(path)


def test_f9_retry_recovery_freezes_scores() -> None:
    stochastic = make_comparison(max_blocks=1, replay="stochastic")
    stochastic.declare_blocks([descriptor(0)])
    stochastic.ingest([row(0, 1, game_id="e1", outcome="failed")])
    stochastic.ingest(
        [
            row(0, 0, game_id="e2", winner_seat=0),
            row(0, 2, game_id="e3", outcome="draw"),
            row(0, 3, game_id="e4", winner_seat=0),
        ]
    )
    report = stochastic.report()
    assert report.blocks_consumed == 1
    assert report.mean_lift_bounds == pytest.approx((0.0625, 0.3125))
    stochastic.ingest([row(0, 1, game_id="e5", winner_seat=1)])
    table, _ = stochastic.rating_rows()
    assert len(table) == 4
    assert next(item for item in table if item["rotation"] == 1)["outcome"] == "failed"

    deterministic = make_comparison(max_blocks=1, replay="deterministic")
    deterministic.declare_blocks([descriptor(0)])
    deterministic.ingest([row(0, 1, game_id="d1", outcome="failed")])
    deterministic.ingest(
        [
            row(0, 0, game_id="d2", winner_seat=0),
            row(0, 2, game_id="d3", outcome="draw"),
            row(0, 3, game_id="d4", winner_seat=0),
        ]
    )
    assert deterministic.report().blocks_consumed == 0
    deterministic.ingest([row(0, 1, game_id="d5", winner_seat=1)])
    assert deterministic.report().mean_lift_bounds == pytest.approx((0.3125, 0.3125))


def test_e5_order_invariance() -> None:
    import itertools
    from dataclasses import replace

    def ingest_order(order):
        comparison = make_comparison(max_blocks=3, alpha=0.2)
        comparison.declare_blocks([descriptor(0), descriptor(1), descriptor(2)])
        for block in order:
            comparison.ingest(complete_block(block))
        return comparison

    reports = [
        replace(ingest_order(order).report(), comparison_id="same")
        for order in itertools.permutations(range(3))
    ]
    assert all(report == reports[0] for report in reports)
    assert reports[0].blocks_consumed == 3
    assert reports[0].confidence_sequence is not None


PAIRED_PODS = PodDistribution.from_support(
    "pods",
    1,
    "paired",
    [{"decks": ("d1", "d2", "d3", "d4"), "opponents": (B, B, B), "weight": 1.0}],
)


def paired_descriptor() -> BlockDescriptor:
    return BlockDescriptor(
        block_id="block 0",
        block_index=0,
        design="paired",
        deal_id="deal-0",
        decks=("d1", "d2", "d3", "d4"),
        deck_versions=("v1", "v2", "v3", "v4"),
        swap_counts=(0, 0, 0, 0),
        opponents=(B, B, B),
        policy_versions=POLICIES,
        engine_version="engine",
        pod_distribution_id=PAIRED_PODS.pod_distribution_id,
        pod_distribution_version=PAIRED_PODS.version,
        pod_distribution_hash=PAIRED_PODS.pod_distribution_hash,
    )


def paired_row(
    rotation: int, role: str, *, game_id: str, outcome: str = "win"
) -> dict[str, object]:
    focal = A if role == "a" else B
    agents = [B, B, B, B]
    agents[rotation] = focal
    winner = rotation if outcome == "win" else None
    return {
        "game_id": game_id,
        "played_at": "2026-01-01T00:00:00Z",
        "agents": tuple(agents),
        "decks": ("d1", "d2", "d3", "d4"),
        "policy_versions": tuple(POLICIES[agent] for agent in agents),
        "deck_versions": ("v1", "v2", "v3", "v4"),
        "swap_counts": (0, 0, 0, 0),
        "outcome": "draw" if outcome == "draw" else outcome,
        "winner_seat": winner,
        "suite": "train",
        "cluster_id": "block 0",
        "block_id": "block 0",
        "rotation": rotation,
        "role": role,
        "deal_id": "deal-0",
        "engine_version": "engine",
    }


def test_paired_pairing_report_matched_denominator() -> None:
    comparison = Comparison(
        a=A,
        b=B,
        design="paired",
        agents=AGENTS,
        decks=DECKS,
        engine_version="engine",
        pod_distribution=PAIRED_PODS,
        hypothesis=Superiority(0.0),
        max_blocks=1,
        alpha=0.99,
    )
    comparison.declare_blocks([paired_descriptor()])
    comparison.ingest(
        [
            paired_row(0, "a", game_id="a0", outcome="draw"),
            paired_row(0, "b", game_id="b0"),
            paired_row(1, "a", game_id="a1"),
            paired_row(1, "b", game_id="b1", outcome="draw"),
            paired_row(2, "a", game_id="a2", outcome="failed"),
            paired_row(2, "b", game_id="b2"),
            paired_row(3, "a", game_id="a3", outcome="cap"),
            paired_row(3, "b", game_id="b3"),
        ]
    )
    pairing = comparison.pairing_report()
    assert pairing.design == "paired"
    assert pairing.matched_positions == 3
    assert pairing.draw_cap_difference == pytest.approx(1.0 / 3.0)
    assert pairing.var_x is None
    assert pairing.variance_ratio is None


@pytest.mark.parametrize("replay", ["stochastic", "deterministic"])
def test_s5_retry_recovery_variant(replay) -> None:
    comparison = make_comparison(max_blocks=1, replay=replay)
    comparison.declare_blocks([descriptor(0)])
    comparison.ingest([row(0, 1, game_id="retry-fail", outcome="failed")])
    comparison.ingest(
        [
            row(0, 0, game_id="r0"),
            row(0, 2, game_id="r2"),
            row(0, 3, game_id="r3"),
        ]
    )
    if replay == "stochastic":
        assert comparison.report().blocks_consumed == 1
        assert comparison.report().mean_lift_bounds == pytest.approx((0.5, 0.75))
    else:
        assert comparison.report().blocks_consumed == 0
        comparison.ingest([row(0, 1, game_id="retry-win", winner_seat=1)])
        assert comparison.report().mean_lift_bounds == pytest.approx((0.75, 0.75))


def test_s5_late_recovery_variant() -> None:
    deterministic = make_comparison(max_blocks=1, replay="deterministic")
    deterministic.declare_blocks([descriptor(0)])
    deterministic.ingest(complete_block(0))
    before = deterministic.rating_rows()
    deterministic.ingest([row(0, 2, game_id="late-same", winner_seat=2)])
    assert deterministic.rating_rows() == before
    assert deterministic.status() != "INCONCLUSIVE_INVALID"

    stochastic = make_comparison(max_blocks=1, replay="stochastic")
    stochastic.declare_blocks([descriptor(0)])
    stochastic.ingest(complete_block(0))
    stochastic.ingest([row(0, 2, game_id="late-different", winner_seat=3)])
    assert stochastic.status() != "INCONCLUSIVE_INVALID"


def test_s5_resume_matches_uninterrupted(tmp_path) -> None:
    uninterrupted = make_comparison(max_blocks=2, replay="deterministic")
    uninterrupted.declare_blocks([descriptor(0), descriptor(1)])
    uninterrupted.ingest(complete_block(0))
    uninterrupted.ingest(complete_block(1))

    resumed = make_comparison(max_blocks=2, replay="deterministic")
    resumed.declare_blocks([descriptor(0), descriptor(1)])
    resumed.ingest(complete_block(0))
    path = tmp_path / "s5-resume.json"
    resumed.save(path)
    resumed = Comparison.load(path)
    resumed.ingest(complete_block(1))
    from dataclasses import replace

    assert replace(resumed.report(), comparison_id="same") == replace(
        uninterrupted.report(), comparison_id="same"
    )
    resumed_table, resumed_export = resumed.rating_rows()
    full_table, full_export = uninterrupted.rating_rows()
    assert resumed_table == full_table
    assert replace(resumed_export, comparison_id="same") == replace(
        full_export, comparison_id="same"
    )
