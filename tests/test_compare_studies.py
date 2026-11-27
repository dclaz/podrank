"""Reduced S5/S11 runner smoke tests."""

from __future__ import annotations

from podrank.compare import NonInferiority, Superiority
from podrank.sim.compare_studies import (
    S5Cell,
    block_scores,
    run_s5_cell,
    run_s11_cell,
    s5_cells,
)


def test_s5_cells_cover_the_declared_dimensions() -> None:
    cells = s5_cells()
    assert len(cells) == 72
    assert {cell.design for cell in cells} == {"duplicate", "paired"}
    assert {cell.rho for cell in cells} == {0.0, 0.3, 0.6, 1.0}
    assert {cell.failure for cell in cells} == {"none", "random", "outcome"}


def test_block_scores_shapes_and_failures() -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    up, low, failed = block_scores(
        rng, design="duplicate", mean=0.02, rho=0.0, failure="none", blocks=10
    )
    assert up.shape == low.shape == failed.shape == (10,)
    assert not failed.any()
    up, low, failed = block_scores(
        rng, design="duplicate", mean=0.02, rho=1.0, failure="outcome", blocks=10
    )
    assert ((up <= low) | failed).all()
    up, low, failed = block_scores(
        rng, design="paired", mean=0.02, rho=0.3, failure="random", blocks=10
    )
    assert up.shape == low.shape == failed.shape == (10,)
    assert up.min() >= -1.0 and up.max() <= 1.0


def test_run_s5_and_s11_smoke() -> None:
    cell = S5Cell(
        "superiority:duplicate",
        Superiority(0.0),
        0.0,
        "duplicate",
        0.0,
        "none",
    )
    result = run_s5_cell(cell, replications=3, blocks=60, seed=1)
    assert result.replications == 3
    assert result.false_above + result.false_below <= 3

    noninferiority = S5Cell(
        "noninferiority:paired",
        NonInferiority(0.02),
        -0.02,
        "paired",
        0.3,
        "random",
        check_above=True,
        check_below=True,
    )
    result = run_s5_cell(noninferiority, replications=2, blocks=60, seed=2)
    assert result.replications == 2

    s11 = run_s11_cell(design="paired", mean=0.02, rho=0.0, replications=3, blocks=60, seed=3)
    assert s11.replications == 3
    assert s11.blocks == 60
    assert abs(s11.truth - 0.02) < 0.05
    assert s11.bias == s11.mean - s11.truth
