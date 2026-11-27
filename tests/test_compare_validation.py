"""Validation branches for the R2 data surfaces."""

from __future__ import annotations

import json
import math

import pytest

from podrank.compare import (
    BlockDescriptor,
    PodDistribution,
    PodSupport,
    ScoredBlockInput,
    Superiority,
    Wealth,
    score_block,
    summarise_blocks,
)
from podrank.errors import PersistenceError

DECKS = ("d1", "d2", "d3", "d4")
PROVENANCE = {"recipe": None, "deck_snapshot_id": None, "materialised_with": "test"}


def _support(weight: float = 1.0, decks: tuple[str, str, str, str] = DECKS) -> PodSupport:
    return PodSupport(decks=decks, opponents=None, weight=weight)


def _distribution() -> PodDistribution:
    return PodDistribution.from_support("pods", 1, "duplicate", [_support()])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decks": "d1"}, "sequence"),
        ({"decks": ("d1", "d2", "d3")}, "length 4"),
        ({"decks": ("d1", "d1", "d3", "d4")}, "distinct"),
        ({"opponents": "abc"}, "sequence"),
        ({"opponents": ("a", "b")}, "length 3"),
        ({"weight": True}, "positive"),
        ({"weight": math.inf}, "positive"),
        ({"weight": 0}, "positive"),
    ],
)
def test_pod_support_validation(kwargs, message) -> None:
    values = {"decks": DECKS, "opponents": None, "weight": 1.0}
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        PodSupport(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pod_distribution_id": ""}, "pod_distribution_id"),
        ({"version": -1}, "version"),
        ({"design": "other"}, "design"),
        ({"support": ()}, "support must not be empty"),
        (
            {"support": (_support(), _support()), "pod_distribution_hash": "hash"},
            "keys must be unique",
        ),
        ({"support": (_support(), _support(2.0, ("d5", "d6", "d7", "d8")))}, "sum to one"),
        ({"pod_distribution_hash": ""}, "pod_distribution_hash"),
        ({"provenance": {}}, "provenance"),
    ],
)
def test_pod_distribution_direct_validation(overrides, message) -> None:
    values = {
        "pod_distribution_id": "pods",
        "version": 1,
        "design": "duplicate",
        "support": (_support(),),
        "pod_distribution_hash": "hash",
        "provenance": PROVENANCE,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        PodDistribution(**values)


def test_pod_distribution_order_and_design_validation() -> None:
    first = _support(0.5)
    second = _support(0.5, ("d5", "d6", "d7", "d8"))
    with pytest.raises(ValueError, match="canonical order"):
        PodDistribution("pods", 1, "duplicate", (second, first), "hash", PROVENANCE)
    with pytest.raises(ValueError, match="null opponents"):
        PodDistribution(
            "pods",
            1,
            "duplicate",
            (PodSupport(DECKS, ("a", "b", "c"), 1.0),),
            "hash",
            PROVENANCE,
        )
    with pytest.raises(ValueError, match="three opponents"):
        PodDistribution("pods", 1, "paired", (_support(),), "hash", PROVENANCE)


def test_from_support_validation() -> None:
    with pytest.raises(ValueError, match="design"):
        PodDistribution.from_support("pods", 1, "other", [_support()])
    with pytest.raises(ValueError, match="PodSupport or mapping"):
        PodDistribution.from_support("pods", 1, "duplicate", [object()])
    with pytest.raises(ValueError, match="exactly decks"):
        PodDistribution.from_support("pods", 1, "duplicate", [{"decks": DECKS}])
    with pytest.raises(ValueError, match="support must not be empty"):
        PodDistribution.from_support("pods", 1, "duplicate", [])
    with pytest.raises(ValueError, match="provenance"):
        PodDistribution.from_support("pods", 1, "duplicate", [_support()], provenance={})


def test_sample_validation() -> None:
    distribution = _distribution()
    with pytest.raises(ValueError, match="n_blocks"):
        distribution.sample(0, 1)
    with pytest.raises(ValueError, match="seed"):
        distribution.sample(1, -1)
    with pytest.raises(ValueError, match="start_index"):
        distribution.sample(1, 1, -1)
    assert distribution.weights().tolist() == [1.0]


def _write(path, payload, *, hash_value="hash", provenance=None) -> None:
    envelope = {
        "payload": payload,
        "pod_distribution_hash": hash_value,
        "provenance": PROVENANCE if provenance is None else provenance,
    }
    path.write_text(json.dumps(envelope), encoding="utf-8")


def test_load_validation(tmp_path) -> None:
    with pytest.raises(PersistenceError, match="could not read"):
        PodDistribution.load(tmp_path / "missing.json")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(PersistenceError, match="invalid pod distribution JSON"):
        PodDistribution.load(invalid)

    base_support = {"decks": list(DECKS), "opponents": None, "weight": 1.0}
    payload = {
        "design": "duplicate",
        "pod_distribution_id": "pods",
        "version": 1,
        "support": [base_support],
    }
    cases = [
        ({**payload, "design": "other"}, "design"),
        ({**payload, "support": "x"}, "must be a list"),
        ({**payload, "support": []}, "must not be empty"),
        ({**payload, "support": [{}]}, "support schema"),
        ({**payload, "support": [{**base_support, "decks": ["d1"]}]}, "invalid pod support"),
        ({**payload, "support": [base_support, base_support]}, "unique"),
    ]
    for index, (bad_payload, message) in enumerate(cases):
        path = tmp_path / f"bad-{index}.json"
        _write(path, bad_payload)
        with pytest.raises(PersistenceError, match=message):
            PodDistribution.load(path)

    reversed_support = [
        {"decks": ["d5", "d6", "d7", "d8"], "opponents": None, "weight": 0.5},
        {"decks": list(DECKS), "opponents": None, "weight": 0.5},
    ]
    path = tmp_path / "order.json"
    _write(path, {**payload, "support": reversed_support})
    with pytest.raises(PersistenceError, match="canonical order"):
        PodDistribution.load(path)

    path = tmp_path / "weights.json"
    _write(path, {**payload, "support": [{**base_support, "weight": 0.5}]})
    with pytest.raises(PersistenceError, match="sum to one"):
        PodDistribution.load(path)

    valid = _distribution()
    path = tmp_path / "valid.json"
    valid.save(path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["provenance"] = {}
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(PersistenceError, match="provenance"):
        PodDistribution.load(path)


def test_block_descriptor_validation() -> None:
    values = {
        "block_id": "block 0",
        "block_index": 0,
        "design": "duplicate",
        "deal_id": "deal",
        "decks": DECKS,
        "deck_versions": ("v1", "v2", "v3", "v4"),
        "swap_counts": (0, 0, 0, 0),
        "opponents": None,
        "policy_versions": {"a": "1"},
        "engine_version": "engine",
        "pod_distribution_id": "pods",
        "pod_distribution_version": 1,
        "pod_distribution_hash": "hash",
    }
    cases = [
        ({"block_id": ""}, "block_id"),
        ({"block_index": -1}, "block_index"),
        ({"design": "other"}, "design"),
        ({"deal_id": ""}, "deal_id"),
        ({"decks": "d1"}, "sequence"),
        ({"deck_versions": ("v1",)}, "length 4"),
        ({"swap_counts": "x"}, "sequence"),
        ({"opponents": "abc"}, "sequence"),
        ({"policy_versions": []}, "mapping"),
        ({"engine_version": ""}, "engine_version"),
        ({"pod_distribution_id": ""}, "pod_distribution_id"),
        ({"pod_distribution_version": -1}, "pod_distribution_version"),
        ({"pod_distribution_hash": ""}, "pod_distribution_hash"),
    ]
    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            BlockDescriptor(**{**values, **overrides})  # type: ignore[arg-type]


def test_block_scoring_validation() -> None:
    with pytest.raises(ValueError, match="positions mismatch"):
        score_block("duplicate", {(0, None): 1.0})
    points = {(0, None): 1.0, (1, None): 0.5, (2, None): 0.0, (3, None): 0.0}
    with pytest.raises(ValueError, match="focal points"):
        score_block("duplicate", points)
    with pytest.raises(ValueError, match="block_id"):
        ScoredBlockInput("", {})
    with pytest.raises(ValueError, match="mapping"):
        ScoredBlockInput("block", [])  # type: ignore[arg-type]
    block = ScoredBlockInput(
        "block", {(0, None): 1.0, (1, None): 0.0, (2, None): 0.0, (3, None): 0.0}
    )
    with pytest.raises(ValueError, match="ScoredBlockInput"):
        summarise_blocks([object()], design="duplicate")  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unique"):
        summarise_blocks([block, block], design="duplicate")


def test_wealth_validation() -> None:
    with pytest.raises(ValueError, match="grid"):
        Wealth("duplicate", Superiority(0.0), grid=0)
    with pytest.raises(ValueError, match="grid"):
        Wealth("duplicate", Superiority(0.0), grid=True)
    wealth = Wealth("duplicate", Superiority(0.0))
    assert wealth.threshold_up == pytest.approx(20.0)
    assert wealth.threshold_low == pytest.approx(20.0)
