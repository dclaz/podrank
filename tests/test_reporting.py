"""Markdown, plots and CLI reporting surfaces."""

from __future__ import annotations

import pytest

from podrank import GameTable, fit, sim
from podrank.cli import main
from podrank.diagnostics import forward_split, posterior_predictive, predictive_quality
from podrank.report import calibration_markdown, forward_markdown, ppc_markdown


@pytest.fixture(scope="module")
def world_and_post():
    world = sim.reference_world("doc_tiny")
    post = fit(
        GameTable.from_records(world.game_rows()),
        world.agent_registry(),
        world.deck_registry(),
    )
    return world, post


def test_markdown_reports(world_and_post) -> None:
    world, post = world_and_post
    calibration = calibration_markdown(predictive_quality(post, world.game_rows()))
    assert "# Calibration" in calibration
    assert "| Model | Log-loss | Brier |" in calibration
    assert "## Reliability" in calibration

    ppc = ppc_markdown(posterior_predictive(post, world.game_rows()))
    assert "# Posterior predictive checks" in ppc
    assert "| Group | Key | Observed |" in ppc

    forward = forward_markdown(
        forward_split(
            GameTable.from_records(world.game_rows()),
            world.agent_registry(),
            world.deck_registry(),
            n_windows=2,
            n_burn=2,
        )
    )
    assert "# Forward split" in forward
    assert "not worse" in forward


def test_cli_version_and_calibrate(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "podrank" in capsys.readouterr().out
    assert main(["calibrate", "--world", "doc_tiny"]) == 0
    output = capsys.readouterr().out
    assert "# Calibration" in output


def test_plots_if_matplotlib_is_installed(world_and_post) -> None:
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from podrank.report.plots import ppc_plot, reliability_plot

    world, post = world_and_post
    figure = reliability_plot(predictive_quality(post, world.game_rows()))
    assert figure is not None
    figure.clf()
    figure = ppc_plot(posterior_predictive(post, world.game_rows()))
    assert figure is not None
    figure.clf()
