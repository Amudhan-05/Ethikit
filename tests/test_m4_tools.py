"""M4: each new tool is checked against a case whose right answer is known."""
import types

import numpy as np
import pytest

from ethikit import principles as P
from ethikit.benchmark import CLASSIFIERS
from ethikit.benchmark.runner import _predict_all
from ethikit.core.types import Dataset, EvalContext, TaskType
from ethikit.data import make_synthetic_raw, split_dataset
from ethikit.evaluators.attribute_inference import AttributeInferenceEvaluator
from ethikit.evaluators.base import all_evaluators, available_evaluators
from ethikit.evaluators.calibration import CalibrationEvaluator
from ethikit.evaluators.permutation import PermutationImportanceEvaluator
from ethikit.evaluators.subgroup import SubgroupPerformanceEvaluator
from ethikit.explain import explain_row
from ethikit.mitigation import MitigationError, mitigate


class FixedProba:
    """A 'model' whose probabilities we choose, to test calibration exactly."""
    classes_ = np.array([0, 1])

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])

    def predict(self, X):
        return (self.p >= 0.5).astype(int)


def _ctx_with(p, y, groups):
    n = len(y)
    ds = Dataset("t", np.zeros((n, 1)), np.zeros((n, 1)), y, y, TaskType.CLASSIFICATION,
                 groups, groups, "g", ["x"])
    m = FixedProba(p)
    return EvalContext(model=m, dataset=ds, y_pred=m.predict(None), y_proba=m.predict_proba(None))


def test_calibration_is_near_zero_for_a_calibrated_model_and_large_for_an_overconfident_one():
    rng = np.random.default_rng(0); n = 20000
    p = rng.uniform(size=n); y = (rng.uniform(size=n) < p).astype(int)
    g = rng.choice(["a", "b"], n)
    calibrated = {r.metric: r.value for r in CalibrationEvaluator().evaluate(_ctx_with(p, y, g))}
    over = np.where(p > 0.5, 0.99, 0.01)                       # says 99% when it is ~75%
    overconf = {r.metric: r.value for r in CalibrationEvaluator().evaluate(_ctx_with(over, y, g))}
    assert calibrated["expected_calibration_error"] < 0.02
    assert overconf["expected_calibration_error"] > 0.2
    assert calibrated["calibration_gap_between_groups"] < 0.03


def test_calibration_gap_detects_a_group_that_is_miscalibrated():
    rng = np.random.default_rng(1); n = 20000
    p = rng.uniform(size=n); g = rng.choice(["a", "b"], n)
    truth = np.where(g == "a", p, np.clip(p - 0.3, 0, 1))     # group b: scores 0.3 too high
    y = (rng.uniform(size=n) < truth).astype(int)
    r = {x.metric: x.value for x in CalibrationEvaluator().evaluate(_ctx_with(p, y, g))}
    assert r["calibration_gap_between_groups"] > 0.15


@pytest.fixture(scope="module")
def biased():
    ds = split_dataset(make_synthetic_raw(n=3000, bias=2.0, minority_share=0.3, seed=2), seed=2)
    model = CLASSIFIERS["linear"]["logistic_regression"]().fit(ds.X_train, ds.y_train)
    return ds, model, _predict_all(model, ds)


def test_subgroup_worst_accuracy_matches_a_manual_computation(biased):
    ds, model, ctx = biased
    r = {x.metric: x.value for x in SubgroupPerformanceEvaluator().evaluate(ctx)}
    y, yp, g = np.asarray(ds.y_test), ctx.y_pred, np.asarray(ds.sensitive_test).astype(str)
    accs = [np.mean(yp[g == k] == y[g == k]) for k in np.unique(g)]
    assert r["worst_group_accuracy"] == pytest.approx(min(accs))
    assert r["accuracy_gap_between_groups"] == pytest.approx(max(accs) - min(accs))


def test_permutation_sees_direct_use_of_the_sensitive_attribute_and_not_when_it_is_excluded():
    with_s = split_dataset(make_synthetic_raw(n=3000, bias=3.0, seed=3), seed=3)
    m = CLASSIFIERS["linear"]["logistic_regression"]().fit(with_s.X_train, with_s.y_train)
    r = {x.metric: x.value for x in PermutationImportanceEvaluator().evaluate(_predict_all(m, with_s))}
    assert r["sensitive_feature_reliance"] > 0.1 and 0 < r["importance_concentration"] <= 1
    raw = make_synthetic_raw(n=3000, bias=3.0, seed=3)
    keep = [i for i, f in enumerate(raw.feature_names) if f != "group"]
    raw.X, raw.feature_names = raw.X[:, keep], [raw.feature_names[i] for i in keep]
    without = split_dataset(raw, seed=3)
    m2 = CLASSIFIERS["linear"]["logistic_regression"]().fit(without.X_train, without.y_train)
    r2 = {x.metric: x.value for x in PermutationImportanceEvaluator().evaluate(_predict_all(m2, without))}
    assert r2["sensitive_feature_reliance"] == 0.0


def test_attribute_inference_is_large_when_the_output_reveals_the_attribute(biased):
    ds, model, ctx = biased
    honest = AttributeInferenceEvaluator().evaluate(ctx)[0].value
    leak = types.SimpleNamespace(**vars(ctx))
    g = (np.asarray(ds.sensitive_test).astype(str) == "1").astype(float)
    leak.y_proba = np.column_stack([1 - g, g])                  # output literally encodes the group
    leaky = AttributeInferenceEvaluator().evaluate(leak)[0].value
    assert 0 <= honest < leaky and leaky > 0.15


def test_every_tool_addressable_principle_has_two_independent_tools_and_every_tool_states_its_limits():
    tools = {}
    for ev in available_evaluators():
        for p in ev.principle_ids:
            tools.setdefault(p, set()).add(ev.name)
    for p in P.tool_addressable():
        assert len(tools.get(p.id, ())) >= 2, f"{p.id}: {tools.get(p.id)}"
    for ev in all_evaluators():
        assert ev.limits.strip(), f"{ev.name} must say what it does not measure"


def test_art_trace_states_its_budget(biased):
    pytest.importorskip("art")
    from ethikit.evaluators.art_robustness import ARTRobustnessEvaluator
    ds, model, ctx = biased
    steps = ARTRobustnessEvaluator(sample_size=4).evaluate(ctx)[0].details["trace"]["steps"]
    assert steps[0].startswith("attack budget:") and "max_iter=" in steps[0]


# ------------------------------------------------------------------ mitigation what-if
@pytest.mark.parametrize("method", ["exponentiated_gradient", "threshold_optimizer"])
def test_mitigation_closes_the_targeted_gap_and_reports_the_cost(biased, method):
    pytest.importorskip("fairlearn")
    ds, _, _ = biased
    r = mitigate(ds, CLASSIFIERS["linear"]["logistic_regression"], method, "demographic_parity")
    assert r["after"]["demographic_parity_diff"] < r["before"]["demographic_parity_diff"] * 0.5
    assert r["label"].startswith("counterfactual") and r["caveats"]
    assert ("prediction time" in r["uses_sensitive_attribute"]) == (method == "threshold_optimizer")
    assert mitigate(ds, CLASSIFIERS["linear"]["logistic_regression"], method)["after"] == r["after"]   # reproducible


def test_mitigation_refuses_clearly(biased):
    pytest.importorskip("fairlearn")
    ds, _, _ = biased
    with pytest.raises(MitigationError, match="use method='threshold_optimizer' instead"):
        mitigate(ds, CLASSIFIERS["instance_based"]["knn"], "exponentiated_gradient")
    with pytest.raises(MitigationError, match="method must be"):
        mitigate(ds, CLASSIFIERS["linear"]["logistic_regression"], "magic")


def test_whatif_never_touches_the_audit():
    pytest.importorskip("fairlearn")
    from ethikit import session
    cfg = {"data": {"source": "synthetic"}, "models": [{"name": "logistic_regression"}], "tools": ["performance", "fairness"],
           "save": False}
    run = session.prepare(cfg)
    before = session.execute(run)["summary"]["objective"]
    mitigate(run.dataset, run.factories["logistic_regression"])
    assert session.execute(run)["summary"]["objective"] == before


# ------------------------------------------------------------------ local explanation
def test_local_explanation_finds_the_feature_that_decides(biased):
    ds, model, _ = biased
    e = explain_row(model, ds, 0)
    assert e["contributions"][0]["feature"] in {"proxy_income", "group", "feature_2"}
    assert 0 <= e["score"] <= 1 and e["group"] is not None
    with pytest.raises(IndexError):
        explain_row(model, ds, 10 ** 6)


def test_occluding_a_feature_the_model_ignores_contributes_nothing():
    rng = np.random.default_rng(0); n = 600
    X = rng.normal(size=(n, 3)); y = (X[:, 0] > 0).astype(int)
    from sklearn.tree import DecisionTreeClassifier
    m = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    ds = Dataset("t", X, X, y, y, TaskType.CLASSIFICATION, None, None, None, ["a", "b", "c"])
    contrib = {c["feature"]: c["contribution"] for c in explain_row(m, ds, 5)["contributions"]}
    if m.feature_importances_[1] == 0:
        assert contrib["b"] == 0.0
