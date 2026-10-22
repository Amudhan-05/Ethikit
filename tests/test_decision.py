import pandas as pd
import pytest

from ethikit.decision import (Requirement, check_requirements, choice_stability, parse_requirement,
                              parse_requirements)


def raw(rows):
    return pd.DataFrame(rows, columns=["model", "evaluator", "principle", "target", "metric", "value", "higher_is_better"])


R = raw([("a", "fairness", "fairness", "predictions", "dpd", 0.05, False),
         ("b", "fairness", "fairness", "predictions", "dpd", 0.30, False),
         ("a", "perf", "utility", "predictions", "accuracy", 0.70, True),
         ("b", "perf", "utility", "predictions", "accuracy", 0.90, True),
         ("a", "giskard", "fairness", "behavior", "issues", 1.0, False),
         ("a", "giskard2", "safety", "behavior", "issues", 3.0, False)])


@pytest.mark.parametrize("text,expected", [
    ("dpd<=0.1", Requirement("dpd", "<=", 0.1)),
    (" accuracy >= 0.8 ", Requirement("accuracy", ">=", 0.8)),
    ("aif360.theil_index<0.2", Requirement("theil_index", "<", 0.2, "aif360")),
    ("x>-1e-3", Requirement("x", ">", -0.001)),
])
def test_parse(text, expected):
    assert parse_requirement(text) == expected


@pytest.mark.parametrize("bad", ["dpd=0.1", "<=0.1", "dpd<=", "dpd <= abc", ""])
def test_parse_rejects_with_an_example(bad):
    with pytest.raises(ValueError, match="write it like"):
        parse_requirement(bad)


def test_check_lists_failures_with_margin_and_direction():
    out = check_requirements(R, parse_requirements(["dpd<=0.1", "accuracy>=0.8"]))
    assert out["a"]["passes"] is False and out["b"]["passes"] is False
    a_acc = [x for x in out["a"]["results"] if x["requirement"].startswith("accuracy")][0]
    assert a_acc["margin"] == pytest.approx(-0.1) and "below" in a_acc["reason"]
    b_dpd = [x for x in out["b"]["results"] if x["requirement"].startswith("dpd")][0]
    assert b_dpd["margin"] == pytest.approx(-0.2) and "above" in b_dpd["reason"]
    ok = check_requirements(R, parse_requirements(["dpd<=0.1"]))
    assert ok["a"]["passes"] and not ok["b"]["passes"]


def test_unknown_metric_names_the_available_ones():
    with pytest.raises(ValueError, match="Metrics available: accuracy, dpd, issues"):
        check_requirements(R, [parse_requirement("nope<1")])


def test_ambiguous_metric_asks_for_a_tool_prefix_and_prefix_resolves_it():
    with pytest.raises(ValueError, match="ambiguous.*prefix the tool"):
        check_requirements(R, [parse_requirement("issues<=2")], models=["a"])
    out = check_requirements(R, [parse_requirement("giskard.issues<=2")], models=["a"])
    assert out["a"]["passes"]


def test_unmeasured_metric_counts_as_failure_not_pass():
    out = check_requirements(R, [parse_requirement("issues<=10")], models=["b"])
    assert out["b"]["passes"] is False and out["b"]["results"][0]["reason"] == "not measured for this model"


def _obj():
    return pd.DataFrame({"fairness": [1.0, 0.0, 0.5], "utility": [0.0, 1.0, 0.6]}, index=["fair", "acc", "mid"])


def test_stability_reports_the_switch_point_and_who_takes_over():
    st = choice_stability(_obj(), {"fairness": 0.8, "utility": 0.2})
    assert st["available"] and st["first"] == "fair"
    pf = st["per_principle"]["fairness"]
    assert 0 < pf["stable_from"] <= 0.8 <= pf["stable_to"] == 1.0
    assert pf["taken_over_below_by"] in {"mid", "acc"} and "takes over" in pf["sentence"]
    assert 0 <= st["nearby_weights_keep_first"] <= 1


def test_a_dominant_model_is_stable_for_any_weight():
    obj = pd.DataFrame({"fairness": [1.0, 0.2], "utility": [1.0, 0.1]}, index=["best", "worse"])
    st = choice_stability(obj, {"fairness": 0.5, "utility": 0.5})
    assert st["nearby_weights_keep_first"] == 1.0
    assert all("any" in v["sentence"] for v in st["per_principle"].values())


def test_stability_handles_single_candidate_and_no_weights():
    assert choice_stability(_obj(), {"fairness": 1}, candidates=["fair"])["available"] is False
    assert "no candidates or no weighted" in choice_stability(_obj(), {"privacy": 1})["reason"]


def test_stability_is_reproducible():
    a = choice_stability(_obj(), {"fairness": 0.5, "utility": 0.5}, seed=3)
    b = choice_stability(_obj(), {"fairness": 0.5, "utility": 0.5}, seed=3)
    assert a == b
