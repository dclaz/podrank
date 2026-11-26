"""SC1/SC2 study harness smoke checks."""

from __future__ import annotations

import warnings

import pytest

from podrank import sim
from podrank.sim.studies import (
    StudyKind,
    StudyQuantity,
    StudyReport,
    run_study,
    scoring_contrasts,
    truth_sdwp,
)


def _quantity(
    key: tuple[str, ...],
    kind: StudyKind,
    truth: float,
    n_scored: int,
    n_emitted: int,
    n_hits: int,
    mean_bias: float,
) -> StudyQuantity:
    coverage = n_hits / n_emitted if n_emitted else None
    return StudyQuantity(
        key=key,
        kind=kind,
        truth=truth,
        n_scored=n_scored,
        n_emitted=n_emitted,
        n_hits=n_hits,
        coverage=coverage,
        mean_bias=mean_bias,
    )


def test_study_report_target_diagnostics() -> None:
    empty = StudyReport(
        "s",
        "w",
        1,
        1,
        0,
        (
            _quantity(("contrast", "a", "b"), "log_worth", 1.0, 1, 0, 0, 0.0),
            _quantity(("sdwp", "a"), "sdwp", 0.5, 1, 0, 0, 0.0),
        ),
    )
    assert empty.n_emitted == 0
    assert empty.emission_rate == 0.0
    assert empty.pooled_coverage is None
    assert empty.low_coverage() == ()
    assert any("emission rate" in miss for miss in empty.target_misses())

    report = StudyReport(
        "s",
        "w",
        2,
        1,
        0,
        (
            _quantity(("contrast", "a", "b"), "log_worth", 1.0, 2, 2, 1, 0.2),
            _quantity(("sdwp", "a"), "sdwp", 0.5, 2, 2, 2, 0.005),
        ),
    )
    assert report.n_emitted == 4
    assert report.emission_rate == 1.0
    assert report.pooled_coverage == 0.75
    assert report.low_coverage() == (("contrast", "a", "b"),)
    misses = report.target_misses()
    assert any("pooled coverage" in miss for miss in misses)
    assert any("bias" in miss for miss in misses)


def test_study_validation() -> None:
    with pytest.raises(ValueError, match="replications"):
        run_study("x", world_name="doc_tiny", replications=0, bootstrap_replicates=1)
    with pytest.raises(ValueError, match="bootstrap_replicates"):
        run_study("x", world_name="doc_tiny", replications=1, bootstrap_replicates=0)


def test_scoring_contrasts_are_deduplicated() -> None:
    world = sim.reference_world("ci_smoke")
    contrasts = scoring_contrasts(world)
    pairs = [(left, right) for _, left, right in contrasts]
    assert len(pairs) == len(set(pairs)) == 11
    assert ("top_vs_second", "agent_09", "agent_08") in contrasts
    assert ("agent_09", "agent_00") in pairs


def test_truth_sdwp_is_a_probability() -> None:
    world = sim.reference_world("doc_tiny")
    field = world.field(24)
    for agent in world.agent_ids:
        value = truth_sdwp(world, field, agent)
        assert 0.0 < value < 1.0


def test_study_harness_runs_on_doc_tiny() -> None:
    report = run_study(
        "doc-smoke",
        world_name="doc_tiny",
        replications=1,
        bootstrap_replicates=2,
        seed=7,
    )
    assert report.replications == 1
    assert report.failed_replications == 0
    assert len(report.quantities) == len(scoring_contrasts(sim.reference_world("doc_tiny"))) + 6
    assert 0 < report.n_scored <= len(report.quantities)
    for quantity in report.quantities:
        assert quantity.n_scored in {0, 1}
        assert quantity.n_emitted <= quantity.n_scored

    frozen = run_study(
        "doc-smoke",
        world_name="doc_tiny",
        replications=1,
        bootstrap_replicates=1,
        seed=7,
        redraw_delta=False,
        rho_div=0.0,
    )
    assert frozen.failed_replications == 0


@pytest.mark.stat
def test_sc1_study_smoke() -> None:
    report = run_study(
        "SC1-smoke",
        world_name="ci_smoke",
        replications=2,
        bootstrap_replicates=5,
        seed=11,
    )
    assert report.replications == 2
    assert report.failed_replications == 0
    world = sim.reference_world("ci_smoke")
    expected = len(scoring_contrasts(world)) + len(world.agent_ids)
    assert len(report.quantities) == expected
    assert report.n_scored == 2 * expected
    assert report.n_emitted == 2 * expected
    assert report.emission_rate == pytest.approx(1.0)
    assert report.pooled_coverage is not None
    assert 0.0 <= report.pooled_coverage <= 1.0
    for quantity in report.quantities:
        assert quantity.n_scored == 2
        assert quantity.n_emitted == 2
        assert quantity.coverage in {0.0, 0.5, 1.0}
        assert quantity.mean_bias is not None
        assert abs(quantity.mean_bias) < 10.0
    assert isinstance(report.target_misses(), tuple)
    assert isinstance(report.low_coverage(), tuple)


@pytest.mark.stat
@pytest.mark.parametrize(("scenario", "rho_div"), [("SC1", None), ("SC2", 0.0)])
def test_ci_smoke_study_reports(scenario: str, rho_div: float | None) -> None:
    report = run_study(
        scenario,
        world_name="ci_smoke",
        replications=50,
        bootstrap_replicates=100,
        seed=202,
        rho_div=rho_div,
    )
    assert report.replications == 50
    assert report.failed_replications == 0
    assert report.n_scored > 0
    assert report.n_emitted > 0
    assert 0.0 <= (report.pooled_coverage or 0.0) <= 1.0
    misses = report.target_misses()
    assert isinstance(misses, tuple)
    if misses:
        warnings.warn(
            f"{scenario} ci_smoke release-review misses: {misses}",
            UserWarning,
            stacklevel=1,
        )
