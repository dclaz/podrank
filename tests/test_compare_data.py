"""R2 data surfaces: pod distributions, descriptors, block scoring (C1–C3)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from podrank.compare import (
    BlockDescriptor,
    PodDistribution,
    ScoredBlockInput,
    focal_points,
    score_block,
    summarise_blocks,
)
from podrank.errors import PersistenceError


def test_f2_duplicate_and_paired_scores_agree() -> None:
    r_a = {0: 1.0, 1: 0.0, 2: 0.25, 3: 0.0}
    r_b = {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}
    duplicate = score_block(
        "duplicate", {(rotation, None): value for rotation, value in r_a.items()}
    )
    paired = score_block(
        "paired",
        {
            **{(rotation, "a"): value for rotation, value in r_a.items()},
            **{(rotation, "b"): value for rotation, value in r_b.items()},
        },
    )
    assert duplicate.x_up == pytest.approx(0.0625)
    assert duplicate.x_low == pytest.approx(0.0625)
    assert paired.x_up == pytest.approx(0.0625)
    assert paired.x_low == pytest.approx(0.0625)
    assert duplicate.complete == pytest.approx(0.0625)
    assert not duplicate.failed


def test_f4_directional_scores_and_f21_summary() -> None:
    points = {(0, None): 1.0, (1, None): None, (2, None): 0.25, (3, None): 0.0}
    scores = score_block("duplicate", points)
    assert scores.x_up == pytest.approx(0.0625)
    assert scores.x_low == pytest.approx(0.3125)
    assert scores.complete is None
    assert scores.failed

    summary = summarise_blocks([ScoredBlockInput("block 0", points)], design="duplicate")
    assert summary.n_blocks == 1
    assert summary.n_failed_blocks == 1
    assert summary.failed_block_fraction == 1.0
    assert summary.mean_lift_bounds == pytest.approx((0.0625, 0.3125))

    empty = summarise_blocks([], design="duplicate")
    assert (empty.n_blocks, empty.n_failed_blocks) == (0, 0)
    assert empty.failed_block_fraction is None
    assert empty.mean_lift_bounds is None

    with pytest.raises(ValueError, match="positions mismatch"):
        summarise_blocks([ScoredBlockInput("block 0", {(0, None): 1.0})], design="duplicate")


def test_paired_directional_bounds_with_both_roles_failed() -> None:
    points = {
        (0, "a"): None,
        (0, "b"): None,
        (1, "a"): 1.0,
        (1, "b"): 0.0,
        (2, "a"): 0.25,
        (2, "b"): 0.25,
        (3, "a"): 0.0,
        (3, "b"): 1.0,
    }
    scores = score_block("paired", points)
    # Up replaces A with 0 and B with 1; low replaces A with 1 and B with 0.
    assert scores.x_up == pytest.approx(-0.25)
    assert scores.x_low == pytest.approx(0.25)
    assert scores.complete is None
    assert scores.failed


def test_focal_points_mapping() -> None:
    assert focal_points("win", 2, 2) == 1.0
    assert focal_points("win", 2, 1) == 0.0
    assert focal_points("draw", None, 0) == 0.25
    assert focal_points("cap", None, 3) == 0.25
    assert focal_points("failed", None, 0) is None
    with pytest.raises(ValueError, match="winner_seat"):
        focal_points("win", None, 0)


def _duplicate_support() -> list[dict[str, object]]:
    return [
        {"decks": ("d1", "d2", "d3", "d4"), "opponents": None, "weight": 1.0},
        {"decks": ("d1", "d2", "d3", "d5"), "opponents": None, "weight": 3.0},
    ]


def test_pod_distribution_canonicalises_and_samples() -> None:
    distribution = PodDistribution.from_support("pods", 1, "duplicate", _duplicate_support())
    assert [item.weight for item in distribution.support] == [0.25, 0.75]
    assert distribution.payload()["support"][0] == {
        "decks": ["d1", "d2", "d3", "d4"],
        "opponents": None,
        "weight": 0.25,
    }

    reordered = PodDistribution.from_support(
        "pods",
        1,
        "duplicate",
        [
            {"decks": ("d1", "d2", "d3", "d5"), "opponents": None, "weight": 2.0},
            {"decks": ("d1", "d2", "d3", "d4"), "opponents": None, "weight": 0.5},
            {"decks": ("d1", "d2", "d3", "d5"), "opponents": None, "weight": 1.0},
            {"decks": ("d1", "d2", "d3", "d4"), "opponents": None, "weight": 0.5},
        ],
    )
    assert reordered.pod_distribution_hash == distribution.pod_distribution_hash

    full = distribution.sample(10, seed=12345)
    split = distribution.sample(6, seed=12345, start_index=4)
    assert [(draw.block_index, draw.decks) for draw in split] == [
        (draw.block_index, draw.decks) for draw in full[4:]
    ]
    for index, draw in enumerate(full):
        assert draw.block_index == index
        assert draw.deal_id == f"{distribution.pod_distribution_hash[:12]}:12345:{index}"
        assert draw.pod_distribution_hash == distribution.pod_distribution_hash


def test_pod_distribution_save_load_and_tamper(tmp_path) -> None:
    distribution = PodDistribution.from_support("pods", 1, "duplicate", _duplicate_support())
    first = tmp_path / "pods.json"
    second = tmp_path / "pods-copy.json"
    distribution.save(first)
    loaded = PodDistribution.load(first)
    assert loaded.pod_distribution_hash == distribution.pod_distribution_hash
    assert loaded.support == distribution.support
    loaded.save(second)
    assert first.read_bytes() == second.read_bytes()

    envelope = json.loads(first.read_text(encoding="utf-8"))
    envelope["payload"]["support"][0]["weight"] = 0.5
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(PersistenceError, match="sum to one"):
        PodDistribution.load(tampered)

    envelope = json.loads(first.read_text(encoding="utf-8"))
    envelope["payload"]["version"] = 2
    tampered_hash = tmp_path / "tampered-hash.json"
    tampered_hash.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(PersistenceError, match="hash"):
        PodDistribution.load(tampered_hash)

    extra = json.loads(first.read_text(encoding="utf-8"))
    extra["extra"] = 1
    bad_schema = tmp_path / "bad-schema.json"
    bad_schema.write_text(json.dumps(extra), encoding="utf-8")
    with pytest.raises(PersistenceError, match="envelope"):
        PodDistribution.load(bad_schema)

    duplicate_key = tmp_path / "duplicate-key.json"
    duplicate_key.write_text(
        '{"payload":{},"payload":{},"pod_distribution_hash":"x","provenance":{}}'
    )
    with pytest.raises(PersistenceError, match="duplicate JSON key"):
        PodDistribution.load(duplicate_key)


def test_pod_distribution_rejects_bad_support() -> None:
    with pytest.raises(ValueError, match="distinct"):
        PodDistribution.from_support(
            "pods",
            1,
            "duplicate",
            [{"decks": ("d1", "d1", "d3", "d4"), "opponents": None, "weight": 1.0}],
        )
    with pytest.raises(ValueError, match="three opponents"):
        PodDistribution.from_support(
            "pods",
            1,
            "paired",
            [{"decks": ("d1", "d2", "d3", "d4"), "opponents": None, "weight": 1.0}],
        )
    with pytest.raises(ValueError, match="null opponents"):
        PodDistribution.from_support(
            "pods",
            1,
            "duplicate",
            [
                {
                    "decks": ("d1", "d2", "d3", "d4"),
                    "opponents": ("a", "b", "c"),
                    "weight": 1.0,
                }
            ],
        )
    with pytest.raises(ValueError, match="positive"):
        PodDistribution.from_support(
            "pods",
            1,
            "duplicate",
            [{"decks": ("d1", "d2", "d3", "d4"), "opponents": None, "weight": 0.0}],
        )


def _descriptor(**overrides: object) -> BlockDescriptor:
    values: dict[str, object] = {
        "block_id": "block 0",
        "block_index": 0,
        "design": "duplicate",
        "deal_id": "deal-0",
        "decks": ("d1", "d2", "d3", "d4"),
        "deck_versions": ("v1", "v2", "v3", "v4"),
        "swap_counts": (0, 0, 0, 0),
        "opponents": None,
        "policy_versions": {"a": "1", "b": "1"},
        "engine_version": "engine",
        "pod_distribution_id": "pods",
        "pod_distribution_version": 1,
        "pod_distribution_hash": "a" * 64,
    }
    values.update(overrides)
    return BlockDescriptor(**values)  # type: ignore[arg-type]


def test_block_descriptor_positions_and_validation() -> None:
    duplicate = _descriptor()
    assert duplicate.positions == ((0, None), (1, None), (2, None), (3, None))
    paired = _descriptor(design="paired", opponents=("b", "b", "b"))
    assert len(paired.positions) == 8
    assert paired.positions[0] == (0, "a")
    assert paired.positions[1] == (0, "b")

    with pytest.raises(ValueError, match="distinct"):
        _descriptor(decks=("d1", "d1", "d3", "d4"))
    with pytest.raises(ValueError, match="null opponents"):
        _descriptor(opponents=("b", "b", "b"))
    with pytest.raises(ValueError, match="three opponents"):
        _descriptor(design="paired")
    with pytest.raises(ValueError, match="policy_versions"):
        _descriptor(policy_versions={})


def test_block_descriptor_from_draw() -> None:
    distribution = PodDistribution.from_support("pods", 1, "duplicate", _duplicate_support())
    draw = distribution.sample(1, seed=7)[0]
    descriptor = BlockDescriptor.from_draw(
        draw,
        block_id="block 0",
        design="duplicate",
        engine_version="engine",
        policy_versions={"a": "1", "b": "1"},
        deck_versions={deck: "v" for deck in draw.decks},
    )
    assert descriptor.block_index == draw.block_index
    assert descriptor.deal_id == draw.deal_id
    assert descriptor.pod_distribution_hash == distribution.pod_distribution_hash
    with pytest.raises(ValueError, match="deck version"):
        BlockDescriptor.from_draw(
            draw,
            block_id="block 0",
            design="duplicate",
            engine_version="engine",
            policy_versions={"a": "1", "b": "1"},
            deck_versions={},
        )


def test_compare_package_is_lazy() -> None:
    code = (
        "import sys, podrank; "
        "assert 'podrank.compare' not in sys.modules; "
        "podrank.compare; "
        "assert 'podrank.compare' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
