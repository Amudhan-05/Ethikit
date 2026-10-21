import pandas as pd
import pytest

from ethikit import session
from ethikit.benchmark import CLASSIFIERS, estimator_from_path, run_benchmark, select_evaluators, with_params


def test_events_arrive_in_a_sensible_order(tiny_ds):
    ev = []
    run_benchmark(tiny_ds, {"dt": CLASSIFIERS["tree_based"]["decision_tree"]},
                  on_event=ev.append, evaluators=select_evaluators(["performance", "fairness"]))
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "run_start" and kinds[-1] == "run_done"
    assert kinds.index("model_fit_start") < kinds.index("model_fit_done") < kinds.index("tool_start")
    done = [e for e in ev if e["type"] == "tool_done"]
    assert {e["tool"] for e in done} == {"performance", "fairness"}
    assert all(e["results"] for e in done)


def test_traces_are_collected_per_model_and_tool(tiny_ds):
    df = run_benchmark(tiny_ds, {"dt": CLASSIFIERS["tree_based"]["decision_tree"]},
                       evaluators=select_evaluators(["fairness"]))
    t = df.attrs["traces"]["dt"]["fairness"]
    assert t["formula"] and t["trace"]["steps"] and t["trace"]["tables"]


def test_a_tool_that_raises_does_not_break_the_sweep(tiny_ds):
    from ethikit.evaluators.base import Evaluator

    class Boom(Evaluator):
        name = "boom"; principle_ids = ["safety"]
        def evaluate(self, ctx): raise RuntimeError("kaput")

    ev = []
    df = run_benchmark(tiny_ds, {"dt": CLASSIFIERS["tree_based"]["decision_tree"]},
                       on_event=ev.append, evaluators=[Boom()] + select_evaluators(["performance"]))
    assert "performance" in set(df["evaluator"])
    assert any(e["type"] == "tool_error" and "kaput" in e["error"] for e in ev)
    assert (df["target"] == "error").any()


def test_prefit_model_is_evaluated_as_is_not_refit(tiny_ds):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(max_iter=300).fit(tiny_ds.X_train[:50], tiny_ds.y_train[:50])
    coef_before = m.coef_.copy()
    run_benchmark(tiny_ds, {}, prefit={"mine": m}, evaluators=select_evaluators(["performance"]))
    assert (m.coef_ == coef_before).all()


def test_select_evaluators_rejects_unknown_tools():
    with pytest.raises(KeyError, match="unknown or unavailable"):
        select_evaluators(["definitely_not_a_tool"])


def test_with_params_overrides_and_reports_bad_keys():
    f = with_params(CLASSIFIERS["tree_based"]["decision_tree"], {"max_depth": "2"})
    assert f().get_params()["max_depth"] == 2
    with pytest.raises(KeyError, match="valid:"):
        with_params(CLASSIFIERS["tree_based"]["decision_tree"], {"nope": 1})()


def test_any_estimator_by_import_path():
    f = estimator_from_path("sklearn.ensemble.ExtraTreesClassifier", {"n_estimators": "7"})
    assert f().get_params()["n_estimators"] == 7


def test_session_end_to_end_and_ordering_follows_weights():
    cfg = {"data": {"source": "synthetic"}, "models": [{"name": "logistic_regression"}, {"name": "decision_tree"}],
           "tools": ["performance", "fairness", "robustness"], "weights": {"fairness": 1.0, "safety": 0.0001}}
    out = session.execute(session.prepare(cfg))
    s = out["summary"]
    assert set(s["objective"]) == {"logistic_regression", "decision_tree"}
    best_on_fairness = max(s["objective"], key=lambda m: s["objective"][m]["fairness"])
    assert s["ranking"][0] == best_on_fairness           # weights drive the order
    assert set(s["pareto"]) <= set(s["objective"])


def test_session_regression_and_custom_synthetic():
    out = session.execute(session.prepare({"data": {"source": "regression"},
          "models": [{"name": "ridge"}], "tools": ["performance", "fairness"]}))
    assert out["summary"]["task"] == "regression"
    out = session.execute(session.prepare({"data": {"source": "synthetic_custom",
          "synthetic": {"n": 600, "minority_share": 0.2}}, "models": [{"name": "naive_bayes"}],
          "tools": ["performance"]}))
    assert "minority=0.2" in out["summary"]["dataset"]


def test_unknown_model_for_task_gives_a_helpful_error():
    with pytest.raises(KeyError, match="unknown model"):
        session.prepare({"data": {"source": "regression"}, "models": [{"name": "random_forest"}]})


def test_csv_source_requires_target():
    with pytest.raises(ValueError, match="target"):
        session.prepare({"data": {"source": "csv", "path": "x.csv"}})


def test_catalogue_describes_every_tool():
    cat = session.catalogue()
    assert {t["name"] for t in cat["tools"]} >= {"performance", "fairness", "aif360", "membership_inference"}
    assert all("formula" in t and "params" in t for t in cat["tools"])
    assert len(cat["checklists"]) == 4
