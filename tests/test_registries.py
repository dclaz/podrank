"""Registry snapshots are immutable, ordered and content-hashed."""

from dataclasses import FrozenInstanceError

import pytest

from podrank import AgentRegistry, DeckRegistry
from podrank.errors import ValidationError
from tests._fixtures import agent_rows, deck_rows


def test_agent_snapshot_is_order_independent_and_content_addressed() -> None:
    rows = agent_rows()
    first = AgentRegistry.from_records(rows)
    second = AgentRegistry.from_records(reversed(rows))

    assert first.snapshot_id == second.snapshot_id
    assert [row.agent_id for row in first] == ["A", "B", "C", "H"]

    edited = [dict(row) for row in rows]
    edited[0]["policy_version"] = "changed"
    assert AgentRegistry.from_records(edited).snapshot_id != first.snapshot_id


def test_deck_snapshot_is_order_independent_and_content_addressed() -> None:
    rows = deck_rows()
    first = DeckRegistry.from_records(rows)
    second = DeckRegistry.from_records(reversed(rows))

    assert first.snapshot_id == second.snapshot_id
    assert [row.deck_id for row in first] == ["d1", "d2", "d3", "d4", "d5"]

    edited = [dict(row) for row in rows]
    edited[0]["deck_version"] = "changed"
    assert DeckRegistry.from_records(edited).snapshot_id != first.snapshot_id


def test_registry_is_immutable_and_defensive_about_input() -> None:
    rows = agent_rows()
    registry = AgentRegistry.from_records(rows)
    rows[0]["tags"].append("mutated")  # type: ignore[union-attr]
    rows[0]["metadata"]["source"] = "mutated"  # type: ignore[index]

    assert "mutated" not in registry["C"].tags
    assert registry["C"].metadata["source"] == "fixture"

    with pytest.raises(FrozenInstanceError):
        registry.rows = ()  # type: ignore[misc]


def test_agent_rows_reject_duplicates_and_bad_fields() -> None:
    duplicate = agent_rows() + [agent_rows()[0]]
    with pytest.raises(ValidationError, match="duplicate agent_id"):
        AgentRegistry.from_records(duplicate)

    bad_kind = agent_rows()
    bad_kind[0]["kind"] = "oracle"
    with pytest.raises(ValidationError, match="kind must be"):
        AgentRegistry.from_records(bad_kind)

    unknown = agent_rows()
    unknown[0]["weight"] = 1
    with pytest.raises(ValidationError, match="unknown agent fields"):
        AgentRegistry.from_records(unknown)


def test_deck_rows_reject_duplicates_and_bad_fields() -> None:
    duplicate = deck_rows() + [deck_rows()[0]]
    with pytest.raises(ValidationError, match="duplicate deck_id"):
        DeckRegistry.from_records(duplicate)

    bad_split = deck_rows()
    bad_split[0]["split"] = "test"
    with pytest.raises(ValidationError, match="split must be"):
        DeckRegistry.from_records(bad_split)

    bad_bracket = deck_rows()
    bad_bracket[0]["bracket"] = True
    with pytest.raises(ValidationError, match="bracket must be an integer"):
        DeckRegistry.from_records(bad_bracket)
