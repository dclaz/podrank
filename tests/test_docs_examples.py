"""Pin the numeric examples shown in the R1 documentation.

The values come from the frozen ``doc_tiny`` world. The tolerance is the FD5
documentation-pin tolerance (1e-6 within one environment), not a statistical
acceptance test.
"""

from __future__ import annotations

import pytest

from podrank import AnchorUnsupported, GameTable, fit, sim, uncertainty

LOG_WORTH = {
    "agent_00": 0.0,
    "agent_01": -0.599385,
    "agent_02": -0.190913,
    "agent_03": 0.326982,
    "agent_04": 0.373443,
    "agent_05": 0.614616,
}

SDWP = {
    "agent_05": 0.454114,
    "agent_04": 0.400392,
    "agent_03": 0.390259,
    "agent_00": 0.321843,
    "agent_02": 0.284809,
    "agent_01": 0.214389,
}


def test_quickstart_outputs_match_docs() -> None:
    world = sim.reference_world("doc_tiny")
    post = fit(
        GameTable.from_records(world.game_rows()),
        world.agent_registry(),
        world.deck_registry(),
    )

    counts = post.counts()
    assert counts.received == 240
    assert counts.eligible == 240
    assert counts.likelihood_used == 223
    assert counts.likelihood_bearing_clusters == 60

    log_worth = {row.agent_id: row.value for row in post.log_worth()}
    for agent_id, expected in LOG_WORTH.items():
        assert log_worth[agent_id] == pytest.approx(expected, abs=1e-6)

    field = world.field(24)
    assert field.field_hash.startswith("203392dd51855ad9")
    sdwp = {row.agent_id: row.sdwp for row in post.sdwp(field).rows}
    assert list(sdwp) == list(SDWP)
    for agent_id, expected in SDWP.items():
        assert sdwp[agent_id] == pytest.approx(expected, abs=1e-6)


def test_documented_bootstrap_intervals() -> None:
    world = sim.reference_world("doc_tiny")
    post = fit(
        GameTable.from_records(world.game_rows()),
        world.agent_registry(),
        world.deck_registry(),
    )
    field = world.field(24)
    boot = uncertainty.cluster_bootstrap(post, n=100, seed=101)

    contrast = boot.contrast("agent_04", "agent_00").interval
    assert contrast is not None
    assert contrast.method == "cluster_bootstrap"
    assert contrast.lo == pytest.approx(0.100170, abs=1e-6)
    assert contrast.hi == pytest.approx(0.694306, abs=1e-6)

    sdwp = boot.sdwp(field)
    assert not isinstance(sdwp, AnchorUnsupported)
    row = next(row for row in sdwp.rows if row.agent_id == "agent_05")
    assert row.interval is not None
    assert row.interval.lo == pytest.approx(0.381084, abs=1e-6)
    assert row.interval.hi == pytest.approx(0.529913, abs=1e-6)
