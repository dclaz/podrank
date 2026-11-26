"""Canonical data bytes and ``data_hash`` behaviour (C.2, F22)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from podrank import GameTable, Inclusion
from podrank.data import inclusion as inclusion_module
from podrank.data._canonical import canonical_json
from podrank.data.inclusion import run_pipeline
from podrank.data.table import canonical_row_json, canonical_row_payload, normalize_game_row
from podrank.errors import DuplicateRowWarning
from tests._fixtures import f7_rows, make_game, registries


def run(rows, *, inclusion: Inclusion | None = None):
    agents, decks = registries()
    return run_pipeline(
        GameTable.from_records(rows),
        agents,
        decks,
        inclusion=inclusion if inclusion is not None else Inclusion(),
    )


def test_fast_canonical_row_json_matches_generic_serializer() -> None:
    record = make_game(
        "g1",
        agents=("É", 'a"b', "c\\d", "e\nf"),
        decks=("d1", "d2", "d3", "d4"),
        policy_versions=("p😀", "p2", "p3", "p4"),
        deck_versions=("v1", None, "v3", None),
        swap_counts=(0, 1, 2, 3),
        outcome="draw",
        winner_seat=None,
        suite='suite,"x"',
        cluster_id="c\n",
        block_id="b",
        rotation=-2,
        role="a",
        deal_id="d",
        spec_hash="s",
        audit={"z": "é", "a": [1, {"b": "😀"}]},
    )
    row = normalize_game_row(record)
    assert canonical_row_json(row) == canonical_json(canonical_row_payload(row))


def test_data_hash_matches_an_independent_canonical_json_oracle() -> None:
    result = run([make_game("g1")])
    expected_rows = [
        {
            "game_id": "g1",
            "played_at": "2026-01-01T00:00:00.000000Z",
            "agents": ["H", "A", "B", "C"],
            "decks": ["d1", "d2", "d3", "d4"],
            "policy_versions": ["H1", "A1", "B1", "C1"],
            "deck_versions": [None, None, None, None],
            "swap_counts": [0, 0, 0, 0],
            "outcome": "win",
            "winner_seat": 0,
            "suite": "train",
            "cluster_id": None,
            "block_id": None,
            "rotation": None,
            "role": None,
            "deal_id": None,
            "spec_hash": None,
            "engine_version": "e1",
            "audit": None,
        }
    ]
    encoded = json.dumps(
        expected_rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    expected = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    assert result.data_hash == expected


def test_data_hash_ignores_order_duplicates_and_inclusion_filters() -> None:
    base = run(f7_rows())
    reordered = run(list(reversed(f7_rows())))
    assert reordered.data_hash == base.data_hash

    with pytest.warns(DuplicateRowWarning):
        duplicated = run([*f7_rows(), make_game("g1", cluster_id="b1")])
    assert duplicated.data_hash == base.data_hash

    upper = datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
    filtered = run(f7_rows(), inclusion=Inclusion(played_at=(None, upper)))
    assert filtered.data_hash == base.data_hash


def test_data_hash_changes_for_an_excluded_audited_row() -> None:
    upper = datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
    inclusion = Inclusion(played_at=(None, upper))
    before = run(f7_rows(), inclusion=inclusion)

    rows = f7_rows()
    rows[3]["audit"] = {"note": 1}
    after = run(rows, inclusion=inclusion)

    assert after.counts.eligible == before.counts.eligible
    assert after.data_hash != before.data_hash


def test_dedup_encodes_only_repeated_game_ids(monkeypatch) -> None:
    calls: list[str] = []
    original = inclusion_module.canonical_row_json

    def spy(row):
        calls.append(row.game_id)
        return original(row)

    monkeypatch.setattr(inclusion_module, "canonical_row_json", spy)
    run(f7_rows())
    assert calls == []

    calls.clear()
    with pytest.warns(DuplicateRowWarning):
        run([make_game("g1"), make_game("g1")])
    assert calls == ["g1", "g1"]


def test_data_hash_uses_canonical_utc_timestamps() -> None:
    one = run([make_game("g1", played_at="2026-01-01T00:00:00+00:00")])
    two = run([make_game("g1", played_at="2026-01-01T01:00:00+01:00")])
    assert one.data_hash == two.data_hash
