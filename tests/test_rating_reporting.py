"""Field canonicalisation, persistence, SDWP and FD6 fixtures (F8, F8b, F8c, F15)."""

from __future__ import annotations

import json
import math
from itertools import permutations
from types import MappingProxyType

import numpy as np
import pytest

from podrank import (
    AgentRegistry,
    CountsReport,
    DeckRegistry,
    Field,
    FieldContext,
    FieldIntegrityError,
    GameTable,
    Inclusion,
    ModelSpec,
    OutcomeRatesReport,
    Provenance,
    fit,
)
from podrank.rating.design import ParameterIndex, PatternTable
from podrank.rating.identification import (
    ComponentsReport,
    ContrastContext,
    build_eligibility,
    null_space_basis,
)
from podrank.rating.posterior import Posterior
from tests._fixtures import make_game


def _manual_posterior(
    theta: dict[str, float],
    seats: tuple[float, float, float, float],
    *,
    anchor: str = "A",
) -> Posterior:
    agent_ids = sorted(theta)
    deck_ids = [f"d{number}" for number in range(1, 5)]
    agents = AgentRegistry.from_records(
        [
            {
                "agent_id": agent_id,
                "kind": "heuristic" if agent_id == anchor else "checkpoint",
                "policy_version": f"{agent_id}1",
                "is_anchor": agent_id == anchor,
            }
            for agent_id in agent_ids
        ]
    )
    decks = DeckRegistry.from_records(
        [
            {
                "deck_id": deck_id,
                "bracket": 1,
                "split": "train",
                "deck_version": f"{deck_id}v1",
            }
            for deck_id in deck_ids
        ]
    )
    index = ParameterIndex.build(agents, decks, agent_ids, deck_ids, swap_effect=False)
    mode = np.zeros(len(index), dtype=np.float64)
    for agent_id, value in theta.items():
        mode[index.index(f"theta[{agent_id}]")] = value
    for seat, value in enumerate(seats):
        mode[index.index(f"seat[{seat}]")] = value
    mode[index.index("mu[1]")] = 0.0
    for deck_id in deck_ids:
        mode[index.index(f"delta[{deck_id}]")] = 0.0

    precision = np.zeros(len(index), dtype=np.float64)
    for name in index.names:
        if name.startswith("theta["):
            precision[index.index(name)] = 1 / 9
        elif name.startswith("seat["):
            precision[index.index(name)] = 4.0
        elif name.startswith("mu["):
            precision[index.index(name)] = 1.0
        elif name.startswith("delta["):
            precision[index.index(name)] = 4.0

    lower = np.eye(len(index))
    row_space = np.eye(len(index))
    components = ComponentsReport(
        anchor_id=anchor,
        anchor_component=tuple(agent_ids),
        other_components=(),
    )
    component_of = {agent_id: anchor for agent_id in agent_ids}
    context = ContrastContext(
        mode=mode,
        lower_factor=lower,
        precision=precision,
        row_space=row_space,
        index=index,
        spec=ModelSpec(),
    )
    eligibility = build_eligibility(agents, index, context, components, component_of)
    assert isinstance(eligibility, tuple)
    counts = CountsReport(
        received=0,
        exact_duplicates=0,
        eligible=0,
        excluded=MappingProxyType({}),
        likelihood_used=0,
        likelihood_bearing_clusters=0,
        agents=MappingProxyType({}),
        decks=MappingProxyType({}),
    )
    return Posterior(
        mode=mode,
        lower_factor=lower,
        parameter_index=index,
        spec=ModelSpec(),
        inclusion=Inclusion(),
        _counts=counts,
        _agents=agents,
        _decks=decks,
        data_hash="0" * 64,
        patterns=PatternTable(
            patterns=(),
            index=index,
            cluster_counts=MappingProxyType({}),
            n_games=0,
        ),
        precision=precision,
        row_space=row_space,
        null_space=null_space_basis(row_space),
        data_hessian=np.zeros((len(index), len(index))),
        _components=components,
        component_of=component_of,
        _eligibility=eligibility,
        pooled_variants=False,
        _provenance=Provenance(
            package_version="test",
            data_hash="0" * 64,
            agent_snapshot_id=agents.snapshot_id,
            deck_snapshot_id=decks.snapshot_id,
        ),
        _outcome_rates=OutcomeRatesReport(rows=()),
        cluster_table={},
        cluster_strata={},
        stratum_crossings={},
    )


def _context(focal_seat: int, *, decks=("d1", "d2", "d3", "d4"), opponents=("B", "B", "B")):
    return FieldContext(
        decks=decks,
        swap_counts=(0, 0, 0, 0),
        opponents=opponents,
        focal_seat=focal_seat,
        weight=0.5,
    )


def test_f8_sdwp_map_and_laplace() -> None:
    post = _manual_posterior({"A": 0.5, "B": 0.0}, (0.2, 0.0, 0.0, -0.2))
    field = Field.from_contexts("f", 1, [_context(0), _context(3)])

    report = post.sdwp(field)
    rows = {row.agent_id: row for row in report.rows}
    assert rows["A"].sdwp == pytest.approx(0.356002, abs=2e-6)
    assert rows["A"].field_label == "field_identified"
    assert report.rows[0].agent_id == "A"

    laplace = post.sdwp(field, method="laplace", n_draws=100, seed=7)
    laplace_rows = {row.agent_id: row for row in laplace.rows}
    assert laplace_rows["A"].sdwp == rows["A"].sdwp
    assert laplace_rows["A"].sdwp_posterior_mean is not None
    assert laplace_rows["A"].interval is not None
    assert laplace_rows["A"].interval.method == "laplace"
    assert laplace_rows["A"].interval.lo <= laplace_rows["A"].interval.hi

    with pytest.raises(ValueError, match="requires an explicit seed"):
        post.sdwp(field, method="laplace")
    with pytest.raises(ValueError, match="method"):
        post.sdwp(field, method="posterior")  # type: ignore[arg-type]


def test_f8b_canonical_field_hash() -> None:
    ctx_a = _context(0, opponents=("B", "B", "B"))
    ctx_b = _context(3, opponents=("B", "B", "B"))
    field = Field.from_contexts("f", 1, [(ctx_a, 1.0), (ctx_b, 1.0), (ctx_a, 2.0)])

    assert [context.weight for context in field.contexts] == [0.75, 0.25]
    assert field.field_hash == ("74c61ecb69d988879d4c89221a3d76414cff99807425113575aedb111850c610")

    reordered = Field.from_contexts("f", 1, [(ctx_b, 1.0), (ctx_a, 2.0), (ctx_a, 1.0)])
    assert reordered.field_hash == field.field_hash
    assert reordered.payload() == field.payload()


def test_f8c_save_load_does_not_renormalise(tmp_path) -> None:
    contexts = [_context(seat, opponents=("H", "H", "H")) for seat in range(3)]
    field = Field.from_contexts(
        "f", 1, [(contexts[0], 1.0), (contexts[1], 6.0), (contexts[2], 15.0)]
    )
    assert [context.weight for context in field.contexts] == [
        0.045454545454545456,
        0.2727272727272727,
        0.6818181818181818,
    ]
    assert math.fsum(context.weight for context in field.contexts) == pytest.approx(
        0.9999999999999999, abs=0
    )

    path = tmp_path / "field.json"
    field.save(path)
    loaded = Field.load(path)
    assert loaded.payload() == field.payload()
    assert loaded.field_hash == field.field_hash
    assert dict(loaded.provenance) == dict(field.provenance)
    assert [context.weight for context in loaded.contexts] == [
        context.weight for context in field.contexts
    ]

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["contexts"][0]["weight"] = 0.5
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(FieldIntegrityError):
        Field.load(path)

    field.save(path)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["provenance"]["materialised_with"] = "edited"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    loaded = Field.load(path)
    assert loaded.field_hash == field.field_hash
    assert loaded.provenance["materialised_with"] == "edited"


def test_field_load_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "field.json"
    path.write_text(
        '{"field_hash":"x","field_hash":"y","payload":{},"provenance":{}}',
        encoding="utf-8",
    )
    with pytest.raises(FieldIntegrityError, match="duplicate JSON key"):
        Field.load(path)


def test_f15_field_identification_label() -> None:
    agents = AgentRegistry.from_records(
        [
            {
                "agent_id": agent_id,
                "kind": "heuristic" if agent_id == "H" else "checkpoint",
                "policy_version": f"{agent_id}1",
                "is_anchor": agent_id == "H",
            }
            for agent_id in ("H", "A", "B")
        ]
    )
    decks = DeckRegistry.from_records(
        [
            {
                "deck_id": f"d{number}",
                "bracket": 1,
                "split": "train",
                "deck_version": f"d{number}v1",
            }
            for number in range(1, 9)
        ]
    )
    rows: list[dict[str, object]] = []
    for group in (("d1", "d2", "d3", "d4"), ("d5", "d6", "d7", "d8")):
        for seating in sorted(set(permutations(["H", "A", "B", "B"]))):
            for deck_order in permutations(group):
                index = len(rows)
                rows.append(
                    make_game(
                        f"g{index}",
                        agents=seating,
                        decks=deck_order,
                        outcome="win",
                        winner_seat=index % 4,
                        cluster_id=f"b{index}",
                    )
                )
    post = fit(GameTable.from_records(rows), agents, decks)

    same_group = Field.from_contexts(
        "same", 1, [_context(0, decks=("d1", "d2", "d3", "d4"), opponents=("H", "H", "H"))]
    )
    mixed_group = Field.from_contexts(
        "mixed", 1, [_context(0, decks=("d1", "d2", "d5", "d6"), opponents=("H", "H", "H"))]
    )
    same_rows = {row.agent_id: row for row in post.sdwp(same_group).rows}
    mixed_rows = {row.agent_id: row for row in post.sdwp(mixed_group).rows}
    assert same_rows["A"].field_label == "field_identified"
    assert same_rows["A"].failing_contexts == ()
    assert mixed_rows["A"].field_label == "prior_dependent_field"
    assert mixed_rows["A"].failing_contexts == (0,)


def test_field_materialise_matches_f23_seeded_draws() -> None:
    decks = DeckRegistry.from_records(
        [
            {
                "deck_id": f"b{bracket}d{number}",
                "bracket": bracket,
                "split": "train",
                "deck_version": f"b{bracket}d{number}v1",
            }
            for bracket in range(2)
            for number in range(4)
        ]
    )
    recipe = {
        "field_id": "f",
        "version": 1,
        "recipe": {
            "deck_split": "train",
            "pod": {
                "bracket_policy": "same_bracket",
                "bracket_weights": "equal",
                "min_decks_per_bracket": 4,
                "distinct_decks": True,
                "swap_counts": 0,
            },
            "opponents": {
                "pool": ["p1", "p0"],
                "draw": "independent_uniform_per_seat",
            },
            "focal_seat": "uniform",
            "n_contexts": 3,
            "seed": 12345,
        },
    }
    field = Field.materialise(recipe, decks=decks)

    expected = {
        (
            1,
            ("b0d1", "b0d2", "b0d0", "b0d3"),
            (0, 0, 0, 0),
            ("p0", "p1", "p0"),
        ),
        (
            0,
            ("b1d0", "b1d1", "b1d2", "b1d3"),
            (0, 0, 0, 0),
            ("p0", "p0", "p0"),
        ),
        (
            0,
            ("b0d1", "b0d2", "b0d3", "b0d0"),
            (0, 0, 0, 0),
            ("p0", "p0", "p1"),
        ),
    }
    assert {
        (context.focal_seat, context.decks, context.swap_counts, context.opponents)
        for context in field.contexts
    } == expected
    assert all(context.weight == pytest.approx(1 / 3) for context in field.contexts)
    assert field.provenance["deck_snapshot_id"] == decks.snapshot_id
    assert field.provenance["recipe"] == recipe


def test_field_materialise_validation() -> None:
    deck_rows = [
        {
            "deck_id": f"d{number}",
            "bracket": 1,
            "split": "train",
            "deck_version": f"d{number}v1",
        }
        for number in range(1, 5)
    ]
    decks = DeckRegistry.from_records(deck_rows)
    base = {
        "field_id": "f",
        "version": 1,
        "recipe": {
            "deck_split": "train",
            "pod": {
                "bracket_policy": "same_bracket",
                "bracket_weights": "equal",
                "min_decks_per_bracket": 5,
                "distinct_decks": True,
                "swap_counts": 0,
            },
            "opponents": {
                "pool": ["H"],
                "draw": "independent_uniform_per_seat",
            },
            "focal_seat": "uniform",
            "n_contexts": 1,
            "seed": 1,
        },
    }
    from podrank.errors import NoEligiblePodsError

    with pytest.raises(NoEligiblePodsError):
        Field.materialise(base, decks=decks)

    duplicate_pool = json.loads(json.dumps(base))
    duplicate_pool["recipe"]["pod"]["min_decks_per_bracket"] = 4
    duplicate_pool["recipe"]["opponents"]["pool"] = ["H", "H"]
    with pytest.raises(ValueError, match="duplicates"):
        Field.materialise(duplicate_pool, decks=decks)

    unknown = json.loads(json.dumps(base))
    unknown["unknown"] = 1
    with pytest.raises(ValueError, match="unknown recipe keys"):
        Field.materialise(unknown, decks=decks)

    bad_policy = json.loads(json.dumps(base))
    bad_policy["recipe"]["pod"]["bracket_policy"] = "mixed"
    with pytest.raises(ValueError, match="bracket_policy"):
        Field.materialise(bad_policy, decks=decks)


def test_field_context_version_fingerprint() -> None:
    agent_rows = [
        {
            "agent_id": "H",
            "kind": "heuristic",
            "policy_version": "H1",
            "is_anchor": True,
        },
        {
            "agent_id": "A",
            "kind": "checkpoint",
            "policy_version": "A1",
            "is_anchor": False,
        },
    ]
    deck_rows = [
        {
            "deck_id": f"d{number}",
            "bracket": 1,
            "split": "train",
            "deck_version": f"d{number}v1",
        }
        for number in range(1, 5)
    ]
    agents = AgentRegistry.from_records(agent_rows)
    decks = DeckRegistry.from_records(deck_rows)
    field = Field.from_contexts(
        "f",
        1,
        [
            FieldContext(
                decks=("d1", "d2", "d3", "d4"),
                swap_counts=(0, 0, 0, 0),
                opponents=("H", "H", "H"),
                focal_seat=0,
                weight=1.0,
            )
        ],
    )
    baseline = field.context_version_fingerprint(agents, decks)

    changed_agent = AgentRegistry.from_records(
        [{**agent_rows[0], "policy_version": "H2"}, agent_rows[1]]
    )
    assert field.context_version_fingerprint(changed_agent, decks) != baseline

    changed_deck = DeckRegistry.from_records(
        [{**deck_rows[0], "deck_version": "changed"}, *deck_rows[1:]]
    )
    assert field.context_version_fingerprint(agents, changed_deck) != baseline

    unrelated = AgentRegistry.from_records(
        [
            *agent_rows,
            {
                "agent_id": "Z",
                "kind": "checkpoint",
                "policy_version": "Z1",
                "is_anchor": False,
            },
        ]
    )
    assert field.context_version_fingerprint(unrelated, decks) == baseline


def test_sample_is_seeded() -> None:
    post = _manual_posterior({"A": 0.5, "B": 0.0}, (0.2, 0.0, 0.0, -0.2))
    first = post.sample(5, seed=11)
    second = post.sample(5, seed=11)
    assert first.shape == (5, len(post.parameter_index))
    assert np.array_equal(first, second)
    with pytest.raises(ValueError):
        post.sample(0, seed=11)
