"""Cluster-bootstrap smoke tests (B1–B6)."""

from __future__ import annotations

import multiprocessing
from dataclasses import replace

import numpy as np
import pytest

from podrank import (
    DeckRegistry,
    Field,
    FieldContext,
    GameTable,
    fit,
    uncertainty,
)
from podrank.errors import ConvergenceError
from podrank.rating.identification import agent_vector
from podrank.uncertainty import bootstrap as bootstrap_module
from podrank.uncertainty.bootstrap import _refit_weights
from tests._fixtures import deck_rows, make_game, registries


def _posterior(cluster_strata=None):
    agents, decks = registries()
    rng = np.random.default_rng(17)
    rows = []
    for cluster in range(60):
        agent_order = rng.permutation(["H", "A", "B", "C"])
        deck_order = rng.permutation(["d1", "d2", "d3", "d4"])
        for rotation in range(4):
            seats = [None] * 4
            for position, agent in enumerate(agent_order):
                seats[(position + rotation) % 4] = str(agent)
            rows.append(
                make_game(
                    f"g{cluster:03d}r{rotation}",
                    agents=tuple(seats),
                    decks=tuple(str(deck) for deck in deck_order),
                    winner_seat=(cluster + rotation) % 4,
                    cluster_id=f"b{cluster:03d}",
                )
            )
    return fit(
        GameTable.from_records(rows),
        agents,
        decks,
        cluster_strata=cluster_strata,
    )


def _field() -> Field:
    return Field.from_contexts(
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


def test_cluster_bootstrap_reports_and_diagnostics() -> None:
    post = _posterior()
    boot = uncertainty.cluster_bootstrap(post, n=3, seed=7, min_clusters=1, min_entity_clusters=1)
    assert boot.n == 3
    assert boot.n_converged + boot.n_failed == 3

    result = boot.contrast("A", "H")
    assert result.status in {"weak", "identified", "aliased"}
    if result.status in {"weak", "identified"}:
        assert result.interval is not None
        assert result.interval.method == "cluster_bootstrap"
        assert result.interval.lo <= result.interval.hi

    zero = boot.contrast("H", "H")
    assert zero.status == "zero"
    zero_vector = boot.contrast_vector(np.zeros(len(post.parameter_index)))
    assert zero_vector.status == "zero"
    nonzero_vector = boot.contrast_vector(np.ones(len(post.parameter_index)))
    assert nonzero_vector.status in {"aliased", "weak", "identified"}
    agent_difference = agent_vector(post.parameter_index, "A") - agent_vector(
        post.parameter_index, "H"
    )
    identified_vector = boot.contrast_vector(agent_difference)
    assert identified_vector.status in {"weak", "identified"}
    assert identified_vector.interval is not None

    worth = boot.log_worth()
    assert isinstance(worth, tuple)
    seats = boot.seats()
    assert len(seats) == 4
    decks = boot.decks()
    assert decks.baseline_status == "available"
    field = _field()
    sdwp = boot.sdwp(field)
    assert not isinstance(sdwp, type(None))
    assert all(
        row.interval is None or row.interval.method == "cluster_bootstrap" for row in sdwp.rows
    )
    diagnostics = boot.diagnostics()
    assert diagnostics
    assert ("contrast", "A", "H") in diagnostics
    assert ("log_worth", "H") in diagnostics
    assert ("sdwp", field.field_hash, "H") in diagnostics
    assert all(item.n_converged == boot.n_converged for item in diagnostics.values())


def test_batched_sdwp_matches_per_draw() -> None:
    post = _posterior()
    field = _field()
    agents = [row.agent_id for row in post.sdwp(field).rows]
    modes = np.vstack([post.mode, post.mode + 0.01 * np.arange(len(post.parameter_index))])
    matrix = post._sdwp_matrix(field, agents, modes)
    assert matrix.shape == (2, len(agents))
    for row_index, mode in enumerate(modes):
        for column, agent in enumerate(agents):
            assert matrix[row_index, column] == pytest.approx(
                post._sdwp_for_draw(field, agent, mode), rel=1e-12, abs=1e-12
            )


def test_bootstrap_sdwp_uses_batched_mode_values() -> None:
    post = _posterior()
    boot = uncertainty.cluster_bootstrap(post, n=4, seed=7, min_clusters=1, min_entity_clusters=1)
    field = _field()
    report = boot.sdwp(field)
    for row in report.rows:
        if row.interval is None:
            continue
        values = post._sdwp_matrix(field, [row.agent_id], boot.replicate_modes)[:, 0]
        lo, hi = np.quantile(values, [0.025, 0.975], method="linear")
        assert row.interval.lo == pytest.approx(float(lo))
        assert row.interval.hi == pytest.approx(float(hi))


def _assert_bootstrap_parity(first, second) -> None:
    assert second.failed_fraction == first.failed_fraction
    assert np.array_equal(second.replicate_weights, first.replicate_weights)
    assert np.array_equal(second.replicate_modes, first.replicate_modes)
    first_result = first.contrast("A", "H")
    second_result = second.contrast("A", "H")
    assert second_result.status == first_result.status
    if first_result.interval is not None:
        assert second_result.interval is not None
        assert second_result.interval.lo == pytest.approx(first_result.interval.lo)
        assert second_result.interval.hi == pytest.approx(first_result.interval.hi)
    assert second.diagnostics() == first.diagnostics()


def test_serial_parallel_bootstrap_parity() -> None:
    post = _posterior()
    serial = uncertainty.cluster_bootstrap(
        post, n=4, seed=13, min_clusters=1, min_entity_clusters=1, workers=1
    )
    parallel = uncertainty.cluster_bootstrap(
        post, n=4, seed=13, min_clusters=1, min_entity_clusters=1, workers=2
    )
    _assert_bootstrap_parity(serial, parallel)


def test_spawn_start_method_bootstrap_parity(monkeypatch) -> None:
    """The spawn start method (default on some POSIX platforms) must agree."""
    spawn = multiprocessing.get_context("spawn")
    monkeypatch.setattr(bootstrap_module, "_process_context", lambda: spawn)
    post = _posterior()
    serial = uncertainty.cluster_bootstrap(
        post, n=3, seed=21, min_clusters=1, min_entity_clusters=1, workers=1
    )
    parallel = uncertainty.cluster_bootstrap(
        post, n=3, seed=21, min_clusters=1, min_entity_clusters=1, workers=2
    )
    _assert_bootstrap_parity(serial, parallel)


def test_bootstrap_workers_validation_and_serial_no_pool(monkeypatch) -> None:
    post = _posterior()
    for value in (True, False, 0, -1, 1.5):
        with pytest.raises(ValueError, match="workers"):
            uncertainty.cluster_bootstrap(
                post, n=1, seed=1, workers=value, min_clusters=1, min_entity_clusters=1
            )

    def fail_pool(*args, **kwargs):
        raise AssertionError("workers=1 must not create a process pool")

    monkeypatch.setattr(bootstrap_module, "ProcessPoolExecutor", fail_pool)
    serial = uncertainty.cluster_bootstrap(
        post, n=1, seed=1, workers=1, min_clusters=1, min_entity_clusters=1
    )
    assert serial.n == 1


def test_bootstrap_custom_strata() -> None:
    strata = {
        "custom": {
            ("cluster", f"b{cluster:03d}"): ("even" if cluster % 2 == 0 else "odd")
            for cluster in range(60)
        }
    }
    post = _posterior(cluster_strata=strata)
    boot = uncertainty.cluster_bootstrap(
        post,
        n=2,
        seed=5,
        strata="custom",
        min_clusters=1,
        min_entity_clusters=1,
    )
    assert boot.strata == "custom"
    assert boot.n_converged == 2


def test_bootstrap_gate_and_stratification() -> None:
    post = _posterior()
    boot = uncertainty.cluster_bootstrap(
        post, n=2, seed=11, strata="suite", min_clusters=1, min_entity_clusters=1
    )
    assert boot.strata == "suite"

    gated = uncertainty.cluster_bootstrap(
        post, n=2, seed=11, min_clusters=1000, min_entity_clusters=1
    )
    result = gated.contrast("A", "H")
    assert result.interval is None
    assert "insufficient_clusters" in result.labels

    with pytest.raises(ValueError, match="unknown cluster stratum"):
        uncertainty.cluster_bootstrap(
            post, n=1, seed=1, strata="missing", min_clusters=1, min_entity_clusters=1
        )

    entity_gated = uncertainty.cluster_bootstrap(
        post, n=1, seed=1, min_clusters=1, min_entity_clusters=1000
    )
    assert entity_gated.contrast("A", "H").interval is None
    for kwargs in (
        {"n": 0, "seed": 1},
        {"n": 1, "seed": -1},
        {"n": 1, "seed": 1, "min_clusters": 0},
        {"n": 1, "seed": 1, "min_entity_clusters": 0},
    ):
        with pytest.raises(ValueError):
            uncertainty.cluster_bootstrap(post, **kwargs)


def test_f6_frozen_index_and_prior_only_entity() -> None:
    agents, decks = registries(extra_ids=("X",))
    rows = [
        make_game("g1", cluster_id="c1", winner_seat=0),
        make_game("g2", played_at="2026-01-01T00:00:01Z", cluster_id="c2", winner_seat=1),
        make_game(
            "g3",
            played_at="2026-01-01T00:00:02Z",
            agents=("H", "A", "B", "X"),
            policy_versions=("H1", "A1", "B1", "X1"),
            cluster_id="c3",
            winner_seat=2,
        ),
    ]
    post = fit(GameTable.from_records(rows), agents, decks)
    weights = np.array([2, 1, 0], dtype=np.int64)
    mode = _refit_weights(post, weights)
    boot_base = uncertainty.cluster_bootstrap(
        post, n=1, seed=0, min_clusters=1, min_entity_clusters=1
    )
    boot = replace(
        boot_base,
        n=1,
        replicate_weights=np.asarray([weights], dtype=np.int64),
        replicate_modes=np.asarray([mode]),
        failed_fraction=0.0,
    )
    assert boot.replicate_modes.shape[1] == len(post.parameter_index)
    x_position = post.parameter_index.index("theta[X]")
    anchor_position = post.parameter_index.index("theta[H]")
    assert boot.replicate_modes[0, x_position] == pytest.approx(0.0, abs=1e-12)
    assert boot.agent_clusters["X"] == frozenset({2})
    assert boot.replicate_modes[0, x_position] - boot.replicate_modes[0, anchor_position] == (
        pytest.approx(-boot.replicate_modes[0, anchor_position], abs=1e-12)
    )
    passed, labels, n_zero = boot._gate(("entity", "X"), [boot.agent_clusters["X"]])
    assert not passed
    assert "insufficient_clusters" in labels
    assert n_zero == 1


def test_f18_gate_precedence_with_controlled_weights() -> None:
    post = _posterior()
    boot = uncertainty.cluster_bootstrap(post, n=1, seed=7, min_clusters=1, min_entity_clusters=1)
    frame_size = len(boot.frame_keys)
    support_index = min(boot.agent_clusters["A"])
    support = frozenset({support_index})
    modes = np.zeros((97, len(post.parameter_index)))
    q1_weights = np.ones((97, frame_size), dtype=np.int64)
    q1_weights[:6, support_index] = 0
    q1 = replace(
        boot,
        n=100,
        replicate_weights=q1_weights,
        replicate_modes=modes,
        failed_fraction=0.03,
    )
    passed, labels, n_zero = q1._gate(("q", 1), [support])
    assert not passed
    assert "insufficient_clusters" in labels
    assert n_zero == 6

    q2_weights = np.ones((97, frame_size), dtype=np.int64)
    q2_weights[:4, support_index] = 0
    q2 = replace(
        boot,
        n=100,
        replicate_weights=q2_weights,
        replicate_modes=modes,
        failed_fraction=0.03,
    )
    interval, q2_labels, diagnostics = q2._interval(("q", 2), np.zeros(97), [support], ())
    assert interval is not None
    assert "unstable_bootstrap" in q2_labels
    assert diagnostics.n_zero_support == 4
    assert diagnostics.n_used == 97

    q3 = replace(
        boot,
        n=100,
        replicate_weights=q2_weights,
        replicate_modes=modes,
        failed_fraction=0.06,
    )
    passed, labels, n_zero = q3._gate(("q", 3), [support])
    assert not passed
    assert "unstable_bootstrap" in labels
    assert n_zero is None

    # Exactly 5% failed is not > 5%; exactly 5% zero support is not > 5%.
    exact_five_weights = np.ones((95, frame_size), dtype=np.int64)
    exact_five = replace(
        boot,
        n=100,
        replicate_weights=exact_five_weights,
        replicate_modes=np.zeros((95, len(post.parameter_index))),
        failed_fraction=0.05,
    )
    passed, labels, n_zero = exact_five._gate(("q", "five"), [support])
    assert passed
    assert labels == ()
    assert n_zero == 0

    boundary_weights = np.ones((100, frame_size), dtype=np.int64)
    boundary_weights[:5, support_index] = 0
    boundary = replace(
        boot,
        n=100,
        replicate_weights=boundary_weights,
        replicate_modes=np.zeros((100, len(post.parameter_index))),
        failed_fraction=0.0,
    )
    passed, labels, n_zero = boundary._gate(("q", "boundary"), [support])
    assert passed
    assert labels == ()
    assert n_zero == 5


def test_bootstrap_failed_replicates_are_suppressed(monkeypatch) -> None:
    post = _posterior()

    def fail(*args, **kwargs):
        raise ConvergenceError("forced")

    monkeypatch.setattr("podrank.uncertainty.bootstrap.solve_patterns", fail)
    boot = uncertainty.cluster_bootstrap(post, n=1, seed=5, min_clusters=1, min_entity_clusters=1)
    result = boot.contrast("A", "H")
    assert result.interval is None
    assert "unstable_bootstrap" in result.labels


def _deck_support_posterior():
    agents, _ = registries()
    decks = DeckRegistry.from_records(
        [
            {"deck_id": f"d{index}", "bracket": 0, "split": "train", "deck_version": f"d{index}v1"}
            for index in range(10)
        ]
    )
    rows = []
    for cluster in range(60):
        if cluster < 9:
            chosen = ["d0", "d1", "d2", "d3"]
        elif cluster < 30:
            chosen = ["d0", "d2", "d3", "d4"]
        else:
            chosen = ["d2", "d3", "d4", "d5"]
        order = ["H", "A", "B", "C"]
        for rotation in range(4):
            seats = [order[(position + rotation) % 4] for position in range(4)]
            rows.append(
                make_game(
                    f"b{cluster:03d}r{rotation}",
                    agents=tuple(seats),
                    decks=tuple(chosen),
                    winner_seat=(cluster + rotation) % 4,
                    cluster_id=f"b{cluster:03d}",
                )
            )
    return fit(GameTable.from_records(rows), agents, decks)


def test_f18_extension_b4_and_b5_suppression(monkeypatch) -> None:
    post = _deck_support_posterior()
    boot = uncertainty.cluster_bootstrap(post, n=2, seed=4, min_clusters=1, min_entity_clusters=10)
    decks = boot.decks()
    target = next(row for row in decks.rows if row.deck_id == "d0")
    baseline = next(row for row in decks.rows if row.deck_id == "d1")
    assert target.value is not None
    assert target.interval is None
    assert "insufficient_clusters" in target.labels
    assert baseline.interval is None
    anchor = boot.contrast("H", "H")
    assert anchor.interval is not None
    assert (anchor.interval.lo, anchor.interval.hi) == (0.0, 0.0)

    def fail(*args, **kwargs):
        raise ConvergenceError("forced")

    monkeypatch.setattr("podrank.uncertainty.bootstrap.solve_patterns", fail)
    failed = uncertainty.cluster_bootstrap(
        post, n=2, seed=4, min_clusters=1, min_entity_clusters=10
    )
    seat = failed.seats()[0]
    assert seat.interval is None
    assert "unstable_bootstrap" in seat.labels
    exact = failed.contrast("H", "H")
    assert exact.interval is not None
    assert (exact.interval.lo, exact.interval.hi) == (0.0, 0.0)


def test_single_train_deck_centred_contrast_is_zero() -> None:
    agents, _ = registries()
    decks = DeckRegistry.from_records(
        [{"deck_id": "only", "bracket": 0, "split": "train", "deck_version": "onlyv1"}]
    )
    rows = [
        make_game(
            f"g{cluster}r{rotation}",
            agents=tuple(["H", "A", "B", "C"][(position + rotation) % 4] for position in range(4)),
            decks=("only", "only", "only", "only"),
            winner_seat=(cluster + rotation) % 4,
            cluster_id=f"b{cluster:03d}",
        )
        for cluster in range(10)
        for rotation in range(4)
    ]
    post = fit(GameTable.from_records(rows), agents, decks)
    boot = uncertainty.cluster_bootstrap(post, n=1, seed=2, min_clusters=1, min_entity_clusters=1)
    report = boot.decks()
    assert report.baseline_status == "available"
    assert report.reference_deck_ids == ("only",)
    row = report.rows[0]
    assert row.status == "zero"
    assert row.interval is not None
    assert (row.interval.lo, row.interval.hi) == (0.0, 0.0)
    assert boot.diagnostics()[("deck", "only")].n_used == 0


def test_bootstrap_anchor_unsupported_passthrough() -> None:
    agents, decks = registries()
    rows = []
    for index, permutation in enumerate(
        sorted(
            {
                ("A", "B", "C", "C"),
                ("A", "C", "B", "C"),
                ("B", "A", "C", "C"),
                ("C", "A", "B", "C"),
            }
        )
    ):
        rows.append(
            make_game(
                f"g{index}",
                agents=permutation,
                winner_seat=index % 4,
                cluster_id=f"b{index}",
            )
        )
    rows.append(make_game("drawH", agents=("H", "A", "B", "C"), outcome="draw"))
    post = fit(GameTable.from_records(rows), agents, decks)
    boot = uncertainty.cluster_bootstrap(post, n=1, seed=2, min_clusters=1, min_entity_clusters=1)
    assert isinstance(boot.log_worth(), uncertainty.bootstrap.AnchorUnsupported)
    assert isinstance(boot.sdwp(_field()), uncertainty.bootstrap.AnchorUnsupported)


def test_bootstrap_deck_nuisance_unavailable() -> None:
    agents, _ = registries()
    holdout_rows = []
    for row in deck_rows():
        updated = dict(row)
        updated["split"] = "holdout"
        holdout_rows.append(updated)
    decks = DeckRegistry.from_records(holdout_rows)
    post = fit(
        GameTable.from_records([make_game("g1", agents=("H", "A", "B", "C"), winner_seat=0)]),
        agents,
        decks,
    )
    boot = uncertainty.cluster_bootstrap(post, n=1, seed=3, min_clusters=1, min_entity_clusters=1)
    assert boot.decks().baseline_status == "unavailable"


def test_bootstrap_crossing_stratum_rejected() -> None:
    agents, decks = registries()
    rows = [
        make_game("g1", suite="train", winner_seat=0, cluster_id="bx"),
        make_game(
            "g2",
            played_at="2026-01-01T00:00:01Z",
            suite="eval",
            winner_seat=1,
            cluster_id="bx",
        ),
    ]
    post = fit(GameTable.from_records(rows), agents, decks)
    with pytest.raises(ValueError, match="crosses clusters"):
        uncertainty.cluster_bootstrap(
            post, n=1, seed=1, strata="suite", min_clusters=1, min_entity_clusters=1
        )
    boot = uncertainty.cluster_bootstrap(post, n=1, seed=1, min_clusters=1, min_entity_clusters=1)
    assert boot.n == 1
