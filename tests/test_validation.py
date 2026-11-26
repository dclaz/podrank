"""Boundary validation and shared I0 steps 1–3."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import pytest

from podrank import DeckRegistry, GameTable, Inclusion, validate
from podrank.data.inclusion import run_pipeline
from podrank.errors import (
    DuplicateGameError,
    DuplicateRowWarning,
    IdentityMismatchError,
    UnknownEntityError,
    ValidationError,
)
from tests._fixtures import deck_rows, make_game, registries


def test_valid_preview_reports_identity_and_hash() -> None:
    agents, decks = registries()
    report = validate(GameTable.from_records([make_game("g1")]), agents, decks)

    assert report.ok
    assert report.received == 1
    assert report.exact_duplicates == 0
    assert len(report.valid_rows) == 1
    assert report.agent_snapshot_id == agents.snapshot_id
    assert report.deck_snapshot_id == decks.snapshot_id


def test_exact_duplicates_are_removed_with_a_warning() -> None:
    agents, decks = registries()
    rows = [make_game("g1"), make_game("g1")]

    with pytest.warns(DuplicateRowWarning):
        report = validate(GameTable.from_records(rows), agents, decks)

    assert report.exact_duplicates == 1
    assert len(report.valid_rows) == 1


def test_conflicting_game_ids_raise() -> None:
    agents, decks = registries()
    rows = [make_game("g1", suite="train"), make_game("g1", suite="other")]

    with pytest.raises(DuplicateGameError):
        validate(GameTable.from_records(rows), agents, decks)


def test_equivalent_offsets_are_one_canonical_row_not_a_conflict() -> None:
    agents, decks = registries()
    rows = [
        make_game("g1", played_at="2026-01-01T00:00:00+00:00"),
        make_game("g1", played_at="2026-01-01T01:00:00+01:00"),
    ]

    with pytest.warns(DuplicateRowWarning):
        report = validate(GameTable.from_records(rows), agents, decks)

    assert report.exact_duplicates == 1
    assert len(report.valid_rows) == 1
    assert report.valid_rows[0].played_at.isoformat() == "2026-01-01T00:00:00+00:00"


@pytest.mark.parametrize(
    "played_at",
    [
        "2026-01-01T00:00:00",
        "2026-01-01T00:00:00.1234567Z",
        datetime(2026, 1, 1),
    ],
)
def test_naive_or_sub_microsecond_timestamps_are_rejected(played_at: object) -> None:
    agents, decks = registries()
    row = make_game("g1", played_at=played_at)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        validate(GameTable.from_records([row]), agents, decks)


def test_trailing_zero_sub_microseconds_are_accepted() -> None:
    agents, decks = registries()
    row = make_game("g1", played_at="2026-01-01T00:00:00.1234560Z")
    report = validate(GameTable.from_records([row]), agents, decks)

    assert report.valid_rows[0].played_at.microsecond == 123456


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(agents=["H", "A", "B"]),
        lambda row: row.update(decks=["d1", "d2", "d3", "d4", "d5"]),
        lambda row: row.update(policy_versions=["H1", "A1", "B1"]),
        lambda row: row.update(deck_versions=[None, None, None]),
        lambda row: row.update(swap_counts=[0, 0, 0]),
        lambda row: row.update(outcome="loss"),
        lambda row: row.update(role="focal"),
        lambda row: row.update(swap_counts=[-1, 0, 0, 0]),
        lambda row: row.update(swap_counts=[True, 0, 0, 0]),
        lambda row: row.update(winner_seat=4),
        lambda row: row.update(winner_seat=-1),
        lambda row: row.update(winner_seat=None),
        lambda row: row.update(outcome="draw", winner_seat=0),
        lambda row: row.update(unexpected=1),
        lambda row: row.update(audit={"rate": float("nan")}),
        lambda row: row.update(game_id=""),
    ],
)
def test_shape_and_enum_boundaries(mutate: Callable[[dict[str, object]], None]) -> None:
    agents, decks = registries()
    row = make_game("g1")
    mutate(row)

    with pytest.raises(ValidationError):
        validate(GameTable.from_records([row]), agents, decks)


def test_identity_errors_are_distinct() -> None:
    agents, decks = registries()

    unknown_agent = make_game(
        "g1",
        agents=("Z", "A", "B", "C"),
        policy_versions=("Z1", "A1", "B1", "C1"),
    )
    with pytest.raises(UnknownEntityError, match="unknown agent"):
        validate(GameTable.from_records([unknown_agent]), agents, decks)

    unknown_deck = make_game("g1", decks=("dx", "d2", "d3", "d4"))
    with pytest.raises(UnknownEntityError, match="unknown deck"):
        validate(GameTable.from_records([unknown_deck]), agents, decks)

    wrong_policy = make_game("g1", policy_versions=("WRONG", "A1", "B1", "C1"))
    with pytest.raises(IdentityMismatchError, match="policy version"):
        validate(GameTable.from_records([wrong_policy]), agents, decks)

    wrong_deck = make_game("g1", deck_versions=("wrong", "d2v1", "d3v1", "d4v1"))
    with pytest.raises(IdentityMismatchError, match="deck version"):
        validate(GameTable.from_records([wrong_deck]), agents, decks)


@pytest.mark.parametrize("anchors", [(), ("H", "A")])
def test_exactly_one_anchor_is_required(anchors: tuple[str, ...]) -> None:
    agents, decks = registries(anchors=anchors)

    with pytest.raises(IdentityMismatchError, match="exactly one anchor"):
        validate(GameTable.from_records([make_game("g1")]), agents, decks)


def test_mirror_seats_are_opt_in() -> None:
    agents, decks = registries()
    row = make_game(
        "g1",
        agents=("A", "A", "B", "C"),
        policy_versions=("A1", "A1", "B1", "C1"),
    )
    table = GameTable.from_records([row])

    assert validate(table, agents, decks).ok
    with pytest.raises(ValidationError, match="mirror seats"):
        validate(table, agents, decks, allow_mirror=False)
    with pytest.raises(ValidationError, match="mirror seats"):
        run_pipeline(
            table,
            agents,
            decks,
            inclusion=Inclusion(allow_mirror=False),
        )


def test_split_policy_is_opt_in() -> None:
    agents, decks = registries()
    row = make_game("g5", decks=("d1", "d2", "d3", "d5"), suite="eval")
    table = GameTable.from_records([row])

    assert validate(table, agents, decks, split_policy={"train": {"train", "eval"}}).ok
    with pytest.raises(ValidationError, match="not allowed"):
        validate(table, agents, decks, split_policy={"train": {"train"}})
    with pytest.raises(ValidationError, match="not allowed"):
        validate(table, agents, decks, split_policy={"holdout": {"train"}})


def test_tag_schema_applies_to_preview_and_full_pipeline() -> None:
    schema = {"colors": "wubrg", "required": ["colors"]}
    agents, decks = registries()
    table = GameTable.from_records([make_game("g1")])
    assert validate(table, agents, decks, tag_schema=schema).ok

    invalid = deck_rows(colors={"d1": "x"})
    bad_decks = DeckRegistry.from_records(invalid)
    with pytest.raises(ValidationError, match="must use only wubrg"):
        validate(table, agents, bad_decks, tag_schema=schema)

    missing_rows = deck_rows()
    for row in missing_rows:
        if row["deck_id"] == "d2":
            del row["tags"]["colors"]  # type: ignore[union-attr]
    missing_decks = DeckRegistry.from_records(missing_rows)
    with pytest.raises(ValidationError, match="missing required tag"):
        validate(table, agents, missing_decks, tag_schema=schema)
    with pytest.raises(ValidationError, match="missing required tag"):
        run_pipeline(
            table,
            agents,
            missing_decks,
            inclusion=Inclusion(),
            tag_schema=schema,
        )


def test_non_strict_preview_collects_errors_and_keeps_valid_rows() -> None:
    agents, decks = registries()
    bad_length = make_game("bad-length", agents=("H", "A", "B"))
    unknown_agent = make_game(
        "bad-agent",
        agents=("Z", "A", "B", "C"),
        policy_versions=("Z1", "A1", "B1", "C1"),
    )
    table = GameTable.from_records([make_game("g1"), bad_length, unknown_agent])

    report = validate(table, agents, decks, strict=False)

    assert not report.ok
    assert report.received == 3
    assert len(report.valid_rows) == 1
    assert {issue.game_id for issue in report.errors} == {"bad-length", "bad-agent"}
    expected = validate(GameTable.from_records([make_game("g1")]), agents, decks).data_hash
    assert report.data_hash == expected
