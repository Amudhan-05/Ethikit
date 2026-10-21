"""The dependency-free numpy paths must agree with the library paths."""
import numpy as np
import pytest

from ethikit.evaluators.fairness import FairnessEvaluator

fairlearn = pytest.importorskip("fairlearn")


def test_classification_numpy_matches_fairlearn(rng):
    n = 800
    groups = rng.integers(0, 3, n)
    y_true = rng.integers(0, 2, n)
    y_pred = (rng.random(n) < 0.3 + 0.15 * groups).astype(int)
    ev = FairnessEvaluator()
    np_vals = ev._via_numpy(y_true, y_pred, groups)
    fl_vals = ev._via_fairlearn(y_true, y_pred, groups)
    assert np_vals == pytest.approx(fl_vals, abs=1e-9)


def test_regression_numpy_matches_fairlearn(rng):
    n = 600
    groups = rng.integers(0, 2, n)
    y_true = rng.normal(size=n)
    y_pred = y_true + rng.normal(scale=0.2 + 0.3 * groups)
    ev = FairnessEvaluator()
    assert ev._via_numpy_regression(y_true, y_pred, groups) == pytest.approx(
        ev._via_fairlearn_regression(y_true, y_pred, groups), abs=1e-9)
