"""Confidence-sequence grid and bracket checks (E6, F13)."""

from __future__ import annotations

import pytest

from podrank.compare import confidence_sequence, grid_points


def test_f13_grid_points() -> None:
    duplicate = grid_points("duplicate")
    assert len(duplicate) == 399
    assert duplicate[0] == pytest.approx(-0.2475)
    assert duplicate[-1] == pytest.approx(0.7475)
    assert -0.25 not in duplicate
    assert 0.75 not in duplicate
    paired = grid_points("paired")
    assert len(paired) == 799
    assert paired[0] == pytest.approx(-0.9975)
    assert paired[-1] == pytest.approx(0.9975)


def test_empty_and_invalid_inputs() -> None:
    assert confidence_sequence([], [], design="duplicate", cs_alpha=0.05) is None
    with pytest.raises(ValueError, match="same length"):
        confidence_sequence([0.0], [], design="duplicate", cs_alpha=0.05)
    with pytest.raises(ValueError, match="cs_alpha"):
        confidence_sequence([0.0], [0.0], design="duplicate", cs_alpha=1.0)


def test_bracket_and_alpha_direction() -> None:
    scores = [0.75] * 20
    narrow = confidence_sequence(scores, scores, design="duplicate", cs_alpha=0.2)
    wide = confidence_sequence(scores, scores, design="duplicate", cs_alpha=0.01)
    assert narrow is not None and wide is not None
    assert narrow.lo >= wide.lo
    assert narrow.hi == wide.hi == pytest.approx(0.75)
    assert not narrow.cs_conflict

    neutral = confidence_sequence([-0.25] * 5, [0.75] * 5, design="duplicate", cs_alpha=0.05)
    assert neutral is not None
    assert neutral.lo == pytest.approx(-0.25)
    assert neutral.hi == pytest.approx(0.75)
