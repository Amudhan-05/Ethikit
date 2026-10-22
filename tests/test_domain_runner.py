"""Suite runner + aggregation, on offline synthetic suites (no network)."""
import numpy as np
import pandas as pd
import pytest

from ethikit.data import make_synthetic_raw
from ethikit.domains import (DatasetUnavailable, compare_domains, run_domain, shortlist)
from ethikit.domains.runner import DomainError, KEY_METRICS
from ethikit.domains.spec import DatasetSpec, Domain

MODELS = ["logistic_regression", "decision_tree", "naive_bayes", "knn"]
TOOLS = ["performance", "fairness", "robustness"]


def toy_domain(n_datasets=6, name="toy", bad=(), **knobs):
    def mk(i):
        if i in bad:
            def boom():
                raise DatasetUnavailable(f"syn{i}: simulated download failure")
            return DatasetSpec(f"{name}{i}", f"{name} {i}", boom, "x", "group", needs="none")
        return DatasetSpec(f"{name}{i}", f"{name} {i}",
                           lambda i=i: make_synthetic_raw(n=600, seed=10 + i, minority_share=0.2 + 0.1 * (i % 3), **knobs),
                           "synthetic", "group", needs="none")
    return Domain(name, name.title(), "toy suite", [mk(i) for i in range(n_datasets)], ["toy caveat"])


@pytest.fixture(scope="module")
def result():
    return run_domain(toy_domain(), models=MODELS, tools=TOOLS, use_cache=False)


def test_events_cover_every_dataset(tmp_path):
    ev = []
    run_domain(toy_domain(3), models=MODELS[:3], tools=["performance"], use_cache=False, on_event=ev.append)
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "domain_start" and kinds[-1] == "domain_done"
    assert kinds.count("dataset_start") == kinds.count("dataset_done") == 3


def test_profiles_are_normalised_and_every_dataset_contributes(result):
    assert set(result.per_dataset["dataset"]) == {f"toy{i}" for i in range(6)}
    assert result.per_dataset["score"].between(0, 1).all()
    assert {"fairness", "utility", "safety"} <= set(result.per_dataset["principle"])


def test_family_table_averages_models_within_a_family_first():
    """A family with more members must not count more: knn (1 model) vs tree (2 models)."""
    res = run_domain(toy_domain(4), models=["decision_tree", "random_forest", "knn", "logistic_regression"],
                     tools=["performance"], use_cache=False)
    pd_ = res.per_dataset
    manual = (pd_[(pd_.principle == "utility")].groupby(["dataset", "family"])["score"].mean()
              .groupby("family").mean())
    got = res.family_table["mean"].xs("utility", level="principle")
    assert got.round(3).to_dict() == manual.round(3).to_dict()


def test_leading_family_and_agreement_are_consistent(result):
    for p, fam in result.top_family.items():
        assert fam in set(result.per_dataset["family"])
        assert 0.0 <= result.agreement[p] <= 1.0


def test_absolute_table_has_real_units(result):
    ab = result.absolute_table
    assert {"accuracy", "demographic_parity_diff", "perturbation_stability"} <= set(ab.columns)
    assert ab["accuracy"].between(0.4, 1).all()
    assert set(ab.columns) <= set(KEY_METRICS)


def test_significance_is_tested_when_there_is_enough_data_and_skipped_with_a_reason_when_not(result):
    assert "p_value" in result.significance["utility"]           # 6 datasets, 4 models
    small = run_domain(toy_domain(2), models=MODELS, tools=["performance"], use_cache=False)
    assert "needs >= 3 datasets" in small.significance["utility"]["skipped"]
    two_models = run_domain(toy_domain(5), models=MODELS[:2], tools=["performance"], use_cache=False)
    assert "skipped" in two_models.significance["utility"]


def test_a_failing_dataset_is_skipped_and_reported_not_fatal():
    ev = []
    res = run_domain(toy_domain(5, bad=(1, 3)), models=MODELS[:3], tools=["performance"], use_cache=False,
                     on_event=ev.append)
    assert len(res.datasets) == 3 and [s["id"] for s in res.skipped] == ["toy1", "toy3"]
    assert all("simulated download failure" in s["reason"] for s in res.skipped)
    assert any(e["type"] == "dataset_skip" for e in ev)
    assert any("skipped" in n for n in res.notes)


def test_when_nothing_loads_the_error_lists_every_reason():
    with pytest.raises(DomainError, match="(?s)toy0: .*toy1: "):
        run_domain(toy_domain(2, bad=(0, 1)), models=MODELS[:3], tools=["performance"], use_cache=False)


def test_empty_suite_says_to_bring_your_own_data():
    with pytest.raises(DomainError, match="your own data"):
        run_domain(Domain("hiring", "Hiring", "x", [], ["Use your own historical data"]))


def test_unknown_model_is_rejected():
    with pytest.raises(DomainError, match="unknown model"):
        run_domain(toy_domain(2), models=["quantum_forest"], use_cache=False)


def test_thin_suites_say_so():
    one = run_domain(toy_domain(1), models=MODELS[:3], tools=["performance"], use_cache=False)
    assert one.strength == "single" and any("indicative" in n for n in one.notes)


def test_max_datasets_limits_the_sweep():
    res = run_domain(toy_domain(6), models=MODELS[:3], tools=["performance"], max_datasets=2, use_cache=False)
    assert [d["id"] for d in res.datasets] == ["toy0", "toy1"]


def test_cv_mode_runs_and_is_reproducible():
    kw = dict(models=MODELS[:3], tools=["performance"], cv=(3, 1), max_datasets=2, use_cache=False)
    a, b = run_domain(toy_domain(2), **kw), run_domain(toy_domain(2), **kw)
    assert a.cv == [3, 1]
    assert a.per_dataset["score"].round(9).tolist() == b.per_dataset["score"].round(9).tolist()


def test_shortlist_always_weighs_utility_and_shows_its_weights(result):
    sl = shortlist(result, weights={"fairness": 1.0})
    assert sl["weights_used"]["utility"] > 0 and abs(sum(sl["weights_used"].values()) - 1) < 1e-6
    assert sl["models"] and all(m in MODELS for m in sl["models"])
    assert "own data" in sl["caveat"]
    only_u = shortlist(result, weights={"utility": 1.0, "fairness": 0.0})
    assert only_u["families"][0] == result.top_family["utility"]


def test_compare_domains_reports_agreement_not_a_winner():
    a = run_domain(toy_domain(4, name="a"), models=MODELS, tools=["performance", "fairness"], use_cache=False)
    b = run_domain(toy_domain(4, name="b", bias=2.5), models=MODELS, tools=["performance", "fairness"], use_cache=False)
    cmp = compare_domains({"a": a, "b": b})
    assert {"principle", "domain_a", "domain_b", "spearman_rho", "same_leading_family"} <= set(cmp.columns)
    assert cmp["spearman_rho"].dropna().between(-1, 1).all()
    assert not any("winner" in c or "best" in c for c in cmp.columns)


def test_to_dict_is_json_serialisable(result):
    import json
    d = result.to_dict()
    json.dumps(d)
    assert d["strength"] == "suite" and d["family_table"] and d["absolute_table"] and d["caveats"] == ["toy caveat"]
