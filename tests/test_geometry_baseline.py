"""Check that the temporary geometry baseline locates simple wall deviations."""

import numpy as np
import pytest

from infradetect.geometry.baseline import score_wall_points


def test_plane_baseline_marks_a_bump():
    """A reproducible synthetic wall illustrates score alignment, not accuracy."""

    rng = np.random.default_rng(4)
    xy = rng.uniform(-1, 1, (500, 2))
    flat = np.column_stack((xy, rng.normal(0, 0.001, 500)))
    bump = flat[:30].copy()
    bump[:, 2] += 0.08
    result = score_wall_points(np.vstack((flat, bump)), iterations=100)
    assert result["candidate"][-30:].mean() > 0.9
    assert result["candidate"][:500].mean() < 0.1
    assert len(result["scores"]) == 530


def test_baseline_rejects_bad_threshold_order():
    """An invalid threshold configuration must fail before writing results."""

    with pytest.raises(ValueError):
        score_wall_points(np.eye(3), fit_fraction=0.01, candidate_fraction=0.005)
