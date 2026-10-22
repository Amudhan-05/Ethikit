import numpy as np
import pandas as pd
import pytest

from ethikit.scoring import normalize, objective_matrix, pareto_front, weighted_score


def _raw(rows):
    return pd.DataFrame(rows, columns=["model", "evaluator", "principle", "target", "metric",
                                       "value", "higher_is_better"])


def test_normalize_orients_every_metric_so_one_is_best():
    raw = _raw([("a", "t", "fairness", "predictions", "gap", 0.1, False),
                ("b", "t", "fairness", "predictions", "gap", 0.5, False),
                ("a", "t", "safety", "behavior", "stab", 0.9, True),
                ("b", "t", "safety", "behavior", "stab", 0.7, True)])
    n = normalize(raw).set_index(["model", "metric"])["norm"]
    assert n["a", "gap"] == 1.0 and n["b", "gap"] == 0.0      # lower-is-better flipped
    assert n["a", "stab"] == 1.0 and n["b", "stab"] == 0.0
    assert n.between(0, 1).all()


def test_normalize_constant_metric_is_all_ones_not_nan():
    raw = _raw([("a", "t", "safety", "behavior", "m", 0.5, True),
                ("b", "t", "safety", "behavior", "m", 0.5, True)])
    assert normalize(raw)["norm"].tolist() == [1.0, 1.0]


def test_normalize_drops_error_rows():
    raw = _raw([("a", "t", "safety", "error", "ERROR:X", np.nan, True),
                ("a", "t", "safety", "behavior", "m", 1.0, True)])
    assert len(normalize(raw)) == 1


def test_pareto_front_contains_no_dominated_model():
    prof = pd.DataFrame({"fair": [1, 0.5, 0.2], "util": [0.2, 0.5, 0.1]}, index=["a", "b", "c"])
    front = pareto_front(prof)
    assert "c" not in front and set(front) == {"a", "b"}


def test_pareto_front_keeps_all_ties():
    prof = pd.DataFrame({"x": [1, 1]}, index=["a", "b"])
    assert set(pareto_front(prof)) == {"a", "b"}


def test_weighted_score_normalises_weights_and_ignores_unknown_keys():
    prof = pd.DataFrame({"fair": [1.0, 0.0], "safe": [0.0, 1.0]}, index=["a", "b"])
    s1 = weighted_score(prof, {"fair": 3, "safe": 1, "nonsense": 99})
    s2 = weighted_score(prof, {"fair": 0.75, "safe": 0.25})
    assert s1.round(6).to_dict() == s2.round(6).to_dict()
    assert s1["a"] == pytest.approx(0.75) and s1["b"] == pytest.approx(0.25)


def test_objective_matrix_has_utility_and_excludes_it_from_principles(tiny_ds):
    from ethikit.benchmark import CLASSIFIERS, run_benchmark
    raw = run_benchmark(tiny_ds, {"lr": CLASSIFIERS["linear"]["logistic_regression"]})
    obj = objective_matrix(normalize(raw))
    assert "utility" in obj.columns
    assert obj.drop(columns="utility").notna().any().any()
