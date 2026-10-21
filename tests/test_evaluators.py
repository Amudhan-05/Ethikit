"""The evaluator contract. Every registered tool, on a tiny dataset, must:
 - emit finite values,
 - carry a formula, a mechanism and a trace with at least one worked step,
 - tag a known principle (or the utility axis),
 - be deterministic for a fixed seed.
A new tool that breaks any of these fails here."""
import math

import numpy as np
import pytest

from ethikit import principles as P
from ethikit.evaluators.base import all_evaluators, available_evaluators

KNOWN = {p.id for p in P.PRINCIPLES} | {"utility"}
AVAILABLE = [e for e in available_evaluators()]


def _applicable(ev, ctx):
    return ev.applies_to(ctx)


@pytest.mark.parametrize("ev", AVAILABLE, ids=lambda e: e.name)
def test_contract(ev, fitted_ctx):
    if not _applicable(ev, fitted_ctx):
        pytest.skip(f"{ev.name} not applicable to this dataset")
    results = ev.evaluate(fitted_ctx)
    assert results, "tool returned nothing"
    assert ev.formula.strip(), "tool must document its formula"
    assert ev.mechanism.strip() and "no mechanism" not in ev.mechanism
    for r in results:
        assert math.isfinite(r.value), f"{r.metric} not finite"
        assert r.principle_id in KNOWN, f"unknown principle {r.principle_id}"
        assert r.mechanism.strip(), f"{r.metric} has no run-specific mechanism"
        trace = r.details.get("trace")
        assert trace and trace["steps"], f"{r.metric} has no worked steps"


@pytest.mark.parametrize("ev", AVAILABLE, ids=lambda e: e.name)
def test_deterministic_for_fixed_seed(ev, fitted_ctx):
    if not _applicable(ev, fitted_ctx):
        pytest.skip("not applicable")
    a = {r.metric: r.value for r in ev.evaluate(fitted_ctx)}
    b = {r.metric: r.value for r in ev.evaluate(fitted_ctx)}
    assert a == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_every_tool_addressable_principle_has_a_tool():
    from ethikit.coverage import principle_to_tools
    covered = set(principle_to_tools())
    for p in P.tool_addressable():
        if p.id == "transparency":   # served by explainability's surrogate; listed by coverage
            assert "transparency" in covered
        else:
            assert p.id in covered, f"no registered tool for {p.id}"


def test_unavailable_tools_say_how_to_enable_them():
    for ev in all_evaluators():
        if not ev.available():
            assert ev.install_hint, f"{ev.name} is unavailable but gives no install hint"


def test_tool_params_are_overridable_and_cast():
    from ethikit.evaluators.robustness import RobustnessEvaluator
    ev = RobustnessEvaluator().configured(sigma="0.3", n_repeats="2")
    assert ev.sigma == 0.3 and ev.n_repeats == 2 and isinstance(ev.n_repeats, int)
    with pytest.raises(KeyError, match="no tunable param"):
        RobustnessEvaluator().configured(nope=1)


def test_more_noise_never_makes_the_model_more_stable(fitted_ctx):
    from ethikit.evaluators.robustness import RobustnessEvaluator
    lo = RobustnessEvaluator(sigma=0.05).evaluate(fitted_ctx)[0].value
    hi = RobustnessEvaluator(sigma=2.0).evaluate(fitted_ctx)[0].value
    assert hi <= lo


def test_regression_dataset_runs_applicable_tools(tiny_reg_ds):
    from ethikit.benchmark import REGRESSORS, run_benchmark
    df = run_benchmark(tiny_reg_ds, REGRESSORS["linear"])
    assert not (df["target"] == "error").any()
    assert {"r2", "group_mae_gap"} <= set(df["metric"])
