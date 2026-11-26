"""Focused error-path and helper coverage for the P1 data layer."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from podrank import (
    AgentRegistry,
    DeckRegistry,
    GameTable,
    Inclusion,
    OutcomeCounts,
    OutcomeRates,
    validate,
)
from podrank.data.inclusion import run_pipeline
from podrank.data.table import format_timestamp
from podrank.errors import InclusionError, ValidationError
from tests._fixtures import agent_rows, deck_rows, make_game, registries


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allow_mirror": "yes"}, "allow_mirror"),
        ({"clusters": "other"}, "clusters"),
        ({"played_at": [None, None]}, "played_at must be"),
        (
            {
                "played_at": (
                    datetime(2026, 1, 2, tzinfo=UTC),
                    datetime(2026, 1, 1, tzinfo=UTC),
                )
            },
            "lower bound",
        ),
        ({"suites": "train"}, "suites must be"),
        ({"suites": {""}}, "non-empty"),
        ({"splits": {"test"}}, "subset"),
        ({"clusters": "complete_only", "manifest": []}, "manifest must be a mapping"),
        ({"clusters": "complete_only", "manifest": {}}, "must not be empty"),
        ({"clusters": "complete_only", "manifest": {"": {"g1"}}}, "cluster IDs"),
        ({"clusters": "complete_only", "manifest": {"b1": []}}, "at least one"),
        ({"clusters": "complete_only", "manifest": {"b1": "g1"}}, "iterable"),
        ({"clusters": "complete_only", "manifest": {"b1": {""}}}, "game IDs"),
    ],
)
def test_inclusion_configuration_errors(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(InclusionError, match=message):
        Inclusion(**kwargs)


def test_split_policy_and_tag_schema_configuration_errors() -> None:
    agents, decks = registries()
    table = GameTable.from_records([make_game("g1")])

    for policy in ({"test": {"train"}}, {"train": "train"}, {"train": {""}}):
        with pytest.raises(InclusionError):
            run_pipeline(table, agents, decks, inclusion=Inclusion(), split_policy=policy)

    for schema in (
        {"required": "colors"},
        {"": {"x"}},
        {"colors": ""},
        {"colors": {""}},
    ):
        with pytest.raises(ValidationError):
            run_pipeline(table, agents, decks, inclusion=Inclusion(), tag_schema=schema)

    with pytest.raises(ValidationError, match="must be one of"):
        run_pipeline(
            table,
            agents,
            decks,
            inclusion=Inclusion(),
            tag_schema={"speed": {"slow"}},
        )
    assert (
        run_pipeline(
            table,
            agents,
            decks,
            inclusion=Inclusion(),
            tag_schema={"speed": "fast"},
        ).counts.eligible
        == 1
    )


def test_registry_rejects_bad_mappings_and_metadata() -> None:
    with pytest.raises(ValidationError, match="agent record must be a mapping"):
        AgentRegistry.from_records([1])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="keys must be strings"):
        AgentRegistry.from_records([{1: "x"}])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="missing required agent fields"):
        AgentRegistry.from_records([{}])

    bad_anchor = agent_rows()
    bad_anchor[0]["is_anchor"] = "yes"
    with pytest.raises(ValidationError, match="is_anchor must be a boolean"):
        AgentRegistry.from_records(bad_anchor)

    bad_id = agent_rows()
    bad_id[0]["agent_id"] = ""
    with pytest.raises(ValidationError, match="non-empty string"):
        AgentRegistry.from_records(bad_id)

    bad_tags = agent_rows()
    bad_tags[0]["tags"] = "anchor"
    with pytest.raises(ValidationError, match="sequence of strings"):
        AgentRegistry.from_records(bad_tags)

    bad_metadata = agent_rows()
    bad_metadata[0]["metadata"] = {1: "x"}
    with pytest.raises(ValidationError, match="keys must be strings"):
        AgentRegistry.from_records(bad_metadata)

    bad_metadata_value = agent_rows()
    bad_metadata_value[0]["metadata"] = {"x": object()}
    with pytest.raises(ValidationError, match="non-JSON value"):
        AgentRegistry.from_records(bad_metadata_value)

    with pytest.raises(ValidationError, match="deck record must be a mapping"):
        DeckRegistry.from_records([1])  # type: ignore[list-item]
    with pytest.raises(ValidationError, match="missing required deck fields"):
        DeckRegistry.from_records([{}])

    no_tags = deck_rows()
    del no_tags[0]["tags"]
    registry = DeckRegistry.from_records(no_tags)
    assert registry[no_tags[0]["deck_id"]].tags == {}  # type: ignore[index]
    assert len(registry) == 5
    assert len(list(registry)) == 5

    bad_deck_tags = deck_rows()
    bad_deck_tags[0]["tags"] = {"": "wu"}
    with pytest.raises(ValidationError, match="keys must be non-empty"):
        DeckRegistry.from_records(bad_deck_tags)

    bad_tag_value = deck_rows()
    bad_tag_value[0]["tags"] = {"colors": 1}
    with pytest.raises(ValidationError, match="non-empty string"):
        DeckRegistry.from_records(bad_tag_value)


def test_game_defaults_and_raw_row_errors() -> None:
    agents, decks = registries()
    row = make_game("g1")
    del row["deck_versions"]
    del row["swap_counts"]
    del row["suite"]
    del row["engine_version"]
    report = validate(GameTable.from_records([row]), agents, decks)
    normalized = report.valid_rows[0]
    assert normalized.deck_versions == (None, None, None, None)
    assert normalized.swap_counts == (0, 0, 0, 0)
    assert normalized.suite == ""
    assert normalized.engine_version == ""

    table = GameTable.from_records([row])
    assert table.to_records()[0]["game_id"] == "g1"

    with pytest.raises(ValidationError, match="keys must be strings"):
        GameTable.from_records([{1: "x"}])  # type: ignore[list-item]

    with pytest.raises(ValidationError, match="missing required game fields"):
        validate(GameTable.from_records([{}]), agents, decks)

    for mutate in (
        lambda item: item.update(played_at=123),
        lambda item: item.update(played_at="2026-13-01T00:00:00Z"),
        lambda item: item.update(agents=1),
        lambda item: item.update(swap_counts=["x", 0, 0, 0]),
        lambda item: item.update(audit=1),
        lambda item: item.update(rotation="0"),
        lambda item: item.update(game_id=1),
        lambda item: item.update(suite=1),
    ):
        broken = make_game("g1")
        mutate(broken)
        with pytest.raises(ValidationError):
            validate(GameTable.from_records([broken]), agents, decks)

    with pytest.raises(ValidationError, match="timezone-aware"):
        format_timestamp(datetime(2026, 1, 1))


def test_outcome_rates_and_report_helpers() -> None:
    counts = OutcomeCounts(win=1, cap=1, failed=1)
    rates = OutcomeRates(counts=counts)
    assert counts.total == 3
    assert rates.win == pytest.approx(1 / 3)
    assert rates.draw == 0.0
    assert rates.cap == pytest.approx(1 / 3)
    assert rates.failed == pytest.approx(1 / 3)

    agents, decks = registries()
    report = validate(GameTable.from_records([make_game("g1")]), agents, decks)
    assert report.deduplicated == 1


def test_non_strict_pipeline_collects_duplicate_and_anchor_errors() -> None:
    agents, decks = registries(anchors=())
    rows = [make_game("g1", suite="train"), make_game("g1", suite="other")]
    report = run_pipeline(
        GameTable.from_records(rows),
        agents,
        decks,
        inclusion=Inclusion(),
        validate_only=True,
        strict=False,
    )

    assert not report.ok
    assert {issue.error_type for issue in report.errors} == {
        "DuplicateGameError",
        "IdentityMismatchError",
    }
