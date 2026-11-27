"""R2 fixed-mixture wealth and thresholds (E1, F3)."""

from __future__ import annotations

import numpy as np
import pytest

from podrank.compare import NonInferiority, Superiority, Wealth, thresholds


def test_f3_fixed_mixture_update() -> None:
    wealth = Wealth("duplicate", Superiority(0.0), alpha_up=0.05, alpha_low=0.05, grid=4)
    assert wealth.lam_up == pytest.approx([0.5, 1.0, 1.5, 2.0])
    assert wealth.lam_low == pytest.approx([0.5 / 0.75 * k / 4 for k in range(1, 5)])

    wealth.update(0.5, 0.5)
    assert wealth.factors_up() == pytest.approx([1.25, 1.5, 1.75, 2.0])
    assert wealth.mixture_up() == pytest.approx(1.625)

    wealth.update(-0.25, -0.25)
    assert wealth.factors_up() == pytest.approx([1.09375, 1.125, 1.09375, 1.0])
    assert wealth.mixture_up() == pytest.approx(1.078125)

    wealth.update(0.25, 0.25)
    assert wealth.factors_up() == pytest.approx([1.23046875, 1.40625, 1.50390625, 1.5])
    assert wealth.mixture_up() == pytest.approx(1.41015625)


def test_thresholds_and_alpha_validation() -> None:
    superiority = thresholds(Superiority(0.02), "duplicate")
    assert (superiority.m_low, superiority.m_up) == (0.0, 0.02)
    paired = thresholds(NonInferiority(0.02), "paired")
    assert (paired.m_low, paired.m_up) == (-0.02, -0.02)

    with pytest.raises(ValueError, match="invalid thresholds"):
        thresholds(Superiority(0.75), "duplicate")
    with pytest.raises(ValueError, match="margin must be > 0"):
        NonInferiority(0.0)
    with pytest.raises(ValueError, match="alpha_up"):
        Wealth("duplicate", Superiority(0.0), alpha_up=1.0)
    with pytest.raises(ValueError, match="alpha_low"):
        Wealth("duplicate", Superiority(0.0), alpha_low=0.0)


def test_crossing_uses_strict_inclusive_threshold() -> None:
    wealth = Wealth("duplicate", Superiority(0.0), alpha_up=0.4, alpha_low=0.4, grid=1)
    assert not wealth.crossed_up()
    wealth.update(0.75, 0.75)
    assert wealth.factors_up()[0] == pytest.approx(2.5)
    assert wealth.crossed_up()


def test_log_wealth_stays_finite_for_long_runs() -> None:
    wealth = Wealth("paired", Superiority(0.05), grid=16)
    for _ in range(5_000):
        wealth.update(-1.0, 1.0)
    assert np.all(np.isfinite(wealth.log_wealth_up))
    assert np.all(np.isfinite(wealth.log_wealth_low))
    assert 0.0 < wealth.mixture_up() < np.inf
