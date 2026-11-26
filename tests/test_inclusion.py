"""I0–I5 pipeline tests, including the F7 and F22 cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest

from podrank import AgentRegistry, DeckRegistry, GameTable, Inclusion
from podrank.data.inclusion import run_pipeline
from podrank.errors import (
    DuplicateRowWarning,
    InclusionError,
    ManifestMismatchError,
    NoRankableEvidence,
    ValidationError,
)
from tests._fixtures import f7_manifest, f7_rows, make_game, registries


def run(
    rows: Sequence[Mapping[str, object]],
    *,
    inclusion: Inclusion | None = None,
    agents: AgentRegistry | None = None,
    decks: DeckRegistry | None = None,
    split_policy: Mapping[str, Sequence[str]] | None = None,
    tag_schema: Mapping[str, object] | None = None,
):
    if agents is None or decks is None:
        default_agents, default_decks = registries()
        if agents is None:
            agents = default_agents
        if decks is None:
            decks = default_decks
    return run_pipeline(
        GameTable.from_records(rows),
        agents,
        decks,
        inclusion=inclusion if inclusion is not None else Inclusion(),
        split_policy=split_policy,
        tag_schema=tag_schema,
    )


def test_f7_cluster_filter_and_counts() -> None:
    upper = datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
    result = run(f7_rows(), inclusion=Inclusion(played_at=(None, upper)))

    assert result.received == 4
    assert result.exact_duplicates == 0
    assert [row.game_id for row in result.eligible_rows] == ["g1", "g2", "g3"]
    assert result.counts.eligible == 3
    assert result.counts.likelihood_used == 2
    assert result.counts.likelihood_bearing_clusters == 1
    assert dict(result.partial_clusters) == {"b1": 1}
    assert result.counts.excluded["played_at"] == 1
    assert result.counts.reconciles
    assert result.pre_inclusion_rates.counts.win == 3
    assert result.pre_inclusion_rates.counts.draw == 1
    assert result.eligible_rates.counts.win == 2
    assert result.eligible_rates.counts.draw == 1


def test_f7_complete_only_requires_a_manifest() -> None:
    with pytest.raises(InclusionError, match="manifest is required"):
        Inclusion(clusters="complete_only")
    with pytest.raises(InclusionError, match="manifest must be absent"):
        Inclusion(clusters="use_available", manifest=f7_manifest())


def test_f7_complete_only_partial_cluster_is_dropped() -> None:
    upper = datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
    inclusion = Inclusion(
        clusters="complete_only",
        manifest=f7_manifest(),
        played_at=(None, upper),
    )
    rows = [*f7_rows(), make_game("g8", played_at="2026-01-01T00:00:00Z")]
    result = run(rows, inclusion=inclusion)

    assert result.counts.eligible == 1
    assert [row.game_id for row in result.eligible_rows] == ["g8"]
    assert result.counts.excluded["played_at"] == 1
    assert result.counts.excluded["cluster_filtered"] == 3
    assert dict(result.cluster_reasons) == {"b1": "filtered"}
    assert result.counts.reconciles


def test_f7_row_filters_and_per_entity_counts() -> None:
    g5 = make_game(
        "g5",
        decks=("d1", "d2", "d3", "d5"),
        agents=("A", "B", "B", "C"),
        outcome="win",
        winner_seat=0,
    )

    split_result = run(
        [g5, make_game("g1")],
        inclusion=Inclusion(splits=frozenset({"train"})),
    )
    assert split_result.counts.eligible == 1
    assert split_result.counts.excluded["split"] == 1

    agent_result = run([g5], inclusion=Inclusion(agents=frozenset({"A"})))
    assert agent_result.counts.eligible == 1
    assert agent_result.counts.agents["B"].games == 1
    assert agent_result.counts.agents["B"].seat_appearances == 2
    assert agent_result.counts.reconciles

    g6 = make_game(
        "g6",
        agents=("B", "B", "C", "C"),
        outcome="win",
        winner_seat=0,
    )
    agent_result = run(
        [g6, make_game("g1")],
        inclusion=Inclusion(agents=frozenset({"A"})),
    )
    assert agent_result.counts.excluded["agent"] == 1
    assert agent_result.counts.eligible == 1


def test_f7_duplicate_count_reconciles() -> None:
    rows = [*f7_rows(), make_game("g1", played_at="2026-01-01T00:00:00Z", cluster_id="b1")]
    upper = datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
    with pytest.warns(DuplicateRowWarning):
        result = run(rows, inclusion=Inclusion(played_at=(None, upper)))

    assert result.received == 5
    assert result.exact_duplicates == 1
    assert result.counts.eligible == 3
    assert result.counts.reconciles


@pytest.mark.parametrize(
    "rows, manifest",
    [
        (
            [*f7_rows(), make_game("g9", cluster_id="b1")],
            f7_manifest(),
        ),
        (
            f7_rows(),
            {"b1": {"g1", "g2"}, "b2": {"g1", "g3", "g4"}},
        ),
        (
            [
                make_game("g1", cluster_id="b1"),
                make_game("g2", cluster_id="b1"),
                make_game("g3", cluster_id="b1"),
                make_game("g4", played_at="2026-01-01T00:00:03Z", cluster_id=None),
            ],
            f7_manifest(),
        ),
        (
            [
                make_game("g1", cluster_id="b1"),
                make_game("g2", cluster_id="b1"),
                make_game("g3", cluster_id="b1"),
                make_game("g4", played_at="2026-01-01T00:00:03Z", cluster_id="b2"),
            ],
            f7_manifest(),
        ),
        (
            [*f7_rows(), make_game("g10", cluster_id="b7")],
            f7_manifest(),
        ),
    ],
)
def test_manifest_membership_failures(
    rows: Sequence[Mapping[str, object]], manifest: Mapping[str, set[str]]
) -> None:
    with pytest.raises(ManifestMismatchError):
        run(rows, inclusion=Inclusion(clusters="complete_only", manifest=manifest))


def test_complete_only_missing_game_drops_cluster() -> None:
    rows = [
        *f7_rows()[:-1],
        make_game("g8", played_at="2026-01-01T00:00:04Z"),
    ]
    result = run(
        rows,
        inclusion=Inclusion(clusters="complete_only", manifest=f7_manifest()),
    )

    assert result.counts.eligible == 1
    assert [row.game_id for row in result.eligible_rows] == ["g8"]
    assert result.counts.excluded["cluster_missing"] == 3
    assert dict(result.cluster_reasons) == {"b1": "missing"}
    assert result.counts.reconciles


def test_complete_only_failed_game_drops_cluster() -> None:
    rows = f7_rows()
    rows[2] = make_game(
        "g3",
        played_at="2026-01-01T00:00:02Z",
        outcome="failed",
        cluster_id="b1",
    )
    rows.append(make_game("g8", played_at="2026-01-01T00:00:04Z"))
    result = run(
        rows,
        inclusion=Inclusion(clusters="complete_only", manifest=f7_manifest()),
    )

    assert result.counts.eligible == 1
    assert [row.game_id for row in result.eligible_rows] == ["g8"]
    assert result.counts.excluded["cluster_failed"] == 4
    assert dict(result.cluster_reasons) == {"b1": "failed"}
    assert result.counts.reconciles


def test_complete_only_keeps_null_singleton_and_failed_singleton_is_dropped() -> None:
    rows = [
        *f7_rows(),
        make_game("g8", played_at="2026-01-01T00:00:04Z", cluster_id=None),
    ]
    result = run(
        rows,
        inclusion=Inclusion(clusters="complete_only", manifest=f7_manifest()),
    )
    assert result.counts.eligible == 5
    assert result.counts.likelihood_used == 4
    assert result.counts.likelihood_bearing_clusters == 2

    failed_rows = [
        *f7_rows(),
        make_game(
            "g8",
            played_at="2026-01-01T00:00:04Z",
            outcome="failed",
            cluster_id=None,
        ),
    ]
    result = run(
        failed_rows,
        inclusion=Inclusion(clusters="complete_only", manifest=f7_manifest()),
    )
    assert result.counts.eligible == 4
    assert result.counts.excluded["cluster_failed"] == 1


def test_use_available_keeps_failed_rows_for_reporting() -> None:
    rows = [
        make_game("g1", cluster_id="b1"),
        make_game("g2", played_at="2026-01-01T00:00:01Z", outcome="failed", cluster_id="b1"),
    ]
    result = run(rows)

    assert result.counts.eligible == 2
    assert result.counts.likelihood_used == 1
    assert result.pre_inclusion_rates.counts.failed == 1
    assert result.eligible_rates.counts.failed == 1


@pytest.mark.parametrize(
    "rows",
    [
        [make_game("g1", outcome="draw", cluster_id="b1")],
        [make_game("g1", outcome="cap", cluster_id="b1")],
        [make_game("g1", outcome="failed", cluster_id="b1")],
    ],
)
def test_no_likelihood_evidence_raises(rows: Sequence[Mapping[str, object]]) -> None:
    with pytest.raises(NoRankableEvidence):
        run(rows)


def test_unrankable_saturated_and_null_cluster_keys() -> None:
    agents, decks = registries(extra_ids=("D",))
    rows = [
        make_game("g1", agents=("A", "B", "C", "H"), cluster_id=None),
        make_game("g2", agents=("A", "B", "C", "H"), cluster_id=None),
        make_game(
            "g3",
            agents=("A", "B", "C", "H"),
            winner_seat=2,
            cluster_id="b1",
        ),
        make_game(
            "g4",
            played_at="2026-01-01T00:00:03Z",
            agents=("H", "A", "B", "C"),
            decks=("d1", "d2", "d3", "d5"),
            outcome="draw",
            cluster_id=None,
        ),
    ]
    result = run(rows, agents=agents, decks=decks)

    assert result.supported_agents == frozenset({"A", "B", "C", "H"})
    assert result.unrankable_agents == frozenset({"D"})
    assert result.saturated_agents == frozenset({"B", "H"})
    assert result.supported_decks == frozenset({"d1", "d2", "d3", "d4"})
    assert result.unrankable_decks == frozenset({"d5"})
    assert result.counts.likelihood_bearing_clusters == 3


def test_all_wins_marks_an_agent_saturated() -> None:
    rows = [
        make_game("g1", agents=("A", "B", "C", "H"), cluster_id=None),
        make_game("g2", agents=("A", "B", "C", "H"), cluster_id=None),
    ]
    result = run(rows)
    assert result.saturated_agents == frozenset({"A", "B", "C", "H"})


def test_none_disables_a_filter_and_empty_selects_nothing() -> None:
    assert run(f7_rows(), inclusion=Inclusion(suites=None)).counts.eligible == 4
    with pytest.raises(NoRankableEvidence):
        run(f7_rows(), inclusion=Inclusion(suites=frozenset()))
    with pytest.raises(NoRankableEvidence):
        run(f7_rows(), inclusion=Inclusion(splits=frozenset()))
    with pytest.raises(NoRankableEvidence):
        run(f7_rows(), inclusion=Inclusion(agents=frozenset()))
    with pytest.raises(NoRankableEvidence):
        run(f7_rows(), inclusion=Inclusion(engine_versions=frozenset()))


def test_filters_use_i3_order_and_inclusive_time_bounds() -> None:
    g5 = make_game(
        "g5",
        decks=("d1", "d2", "d3", "d5"),
        agents=("A", "B", "C", "H"),
        suite="other",
    )
    g1 = make_game("g1", suite="other")
    result = run(
        [g5, g1],
        inclusion=Inclusion(splits=frozenset({"train"}), suites=frozenset({"other"})),
    )
    assert result.counts.excluded["split"] == 1
    assert result.counts.eligible == 1
    assert result.counts.excluded["suite"] == 0

    g6 = make_game("g6", agents=("B", "B", "C", "C"), suite="other")
    result = run(
        [g6, g1],
        inclusion=Inclusion(agents=frozenset({"A"}), suites=frozenset({"other"})),
    )
    assert result.counts.excluded["agent"] == 1
    assert result.counts.eligible == 1

    lower = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)
    upper = datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC)
    bounded = run(f7_rows(), inclusion=Inclusion(played_at=(lower, upper)))
    assert [row.game_id for row in bounded.eligible_rows] == ["g2", "g3", "g4"]
    assert bounded.counts.excluded["played_at"] == 1


def test_r1_does_not_check_block_layout() -> None:
    row = make_game("g1", cluster_id="b1", rotation=2)
    result = run(
        [row],
        inclusion=Inclusion(clusters="complete_only", manifest={"b1": {"g1"}}),
    )
    assert result.counts.eligible == 1


def test_split_policy_is_applied_on_the_full_path() -> None:
    row = make_game("g5", decks=("d1", "d2", "d3", "d5"), suite="eval")
    with pytest.raises(ValidationError, match="not allowed"):
        run([row], split_policy={"train": ("train",)})
