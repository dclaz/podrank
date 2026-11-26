"""GameTable preserves input order and defers all validation and deduplication."""

import pytest

from podrank import GameTable
from podrank.errors import ValidationError
from tests._fixtures import make_game


def test_table_preserves_order_and_does_not_deduplicate() -> None:
    rows = [make_game("g2"), make_game("g1"), make_game("g2")]
    table = GameTable.from_records(rows)

    assert len(table) == 3
    assert [row["game_id"] for row in table] == ["g2", "g1", "g2"]


def test_table_copies_input_and_exposes_immutable_rows() -> None:
    rows = [make_game("g1")]
    table = GameTable.from_records(rows)

    rows[0]["suite"] = "mutated"
    assert table.rows[0]["suite"] == "train"

    with pytest.raises(TypeError):
        table.rows[0]["suite"] = "mutated"  # type: ignore[index]


def test_table_copies_nested_containers() -> None:
    rows = [make_game("g1")]
    table = GameTable.from_records(rows)

    agents = rows[0]["agents"]
    assert isinstance(agents, list)
    agents[0] = "mutated"
    assert table.rows[0]["agents"][0] == "H"  # type: ignore[index]

    audit = {"nested": {"value": 1}}
    rows[0]["audit"] = audit
    table = GameTable.from_records(rows)
    audit["nested"]["value"] = 2
    assert table.rows[0]["audit"]["nested"]["value"] == 1  # type: ignore[index]


def test_table_rejects_non_mapping_rows() -> None:
    with pytest.raises(ValidationError, match="must be a mapping"):
        GameTable.from_records([["not", "a", "mapping"]])  # type: ignore[list-item]
