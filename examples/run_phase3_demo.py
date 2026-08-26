"""Phase 3 demo: privacy, adversarial robustness, and drift.

Three new evaluators, all registered automatically (nothing else in the
pipeline changes -- see ethikit/evaluators/base.py's registry):

  1. Membership inference (ethikit.evaluators.privacy) -- a REAL black-box
     attack (Yeom et al. 2018 confidence/loss-threshold style) replacing the
     Phase-1 train/test-gap proxy. TensorFlow Privacy itself has no
     Python-3.12-compatible release (confirmed by hand; see the module
     docstring), so the attack algorithm is implemented directly against
     scikit-learn instead of pulling that package in.
  2. Adversarial robustness (ethikit.evaluators.art_robustness) -- a real
     black-box evasion attack (HopSkipJump) via the Adversarial Robustness
     Toolbox, complementing the existing Gaussian-noise robustness check
     with an adversarially-SEARCHED worst case.
  3. Drift detection (ethikit.evaluators.drift) -- Alibi Detect's
     TabularDrift statistical test against a manufactured covariate shift,
     checking both whether the detector catches it and how much the model's
     own performance degrades when it happens.

Run:  python -m examples.run_phase3_demo
Needs adversarial-robustness-toolbox + alibi-detect
(pip install -r requirements.txt already includes both, or
pip install -e ".[phase3]"). ART's HopSkipJump attack is query-heavy against
tree ensembles -- this script uses a small model zoo and the default tight
attack budget (see art_robustness.py) so it finishes in a couple of minutes.
"""
from __future__ import annotations

import pandas as pd

from ethikit.data import make_synthetic_fairness_dataset
from ethikit.benchmark import run_benchmark, flat_classifiers
from ethikit.coverage import coverage_table, principle_to_tools, tool_info

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# SVC(probability=True) is a documented slowdown (HANDOFF.md) that gets much
# worse once an adversarial attack starts querying it hundreds of times per
# sample -- excluded here the same way the Phase 2 demo excludes it from the
# meta-benchmark.
FAST_CLASSIFIERS = {k: v for k, v in flat_classifiers().items() if k != "svc"}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def part0_coverage():
    section("0. Updated tool coverage for 'privacy' and 'safety' (reverse mapping)")
    p2t = principle_to_tools()
    for pid in ("privacy", "safety"):
        print(f"\n{pid}:")
        for name in p2t.get(pid, []):
            info = tool_info(name)
            flag = "" if info["available"] else "  [unavailable in this environment]"
            print(f"  - {name}{flag}")
            print(f"      {info['mechanism']}")


def part1_run():
    section("1. Running the battery (incl. the three new Phase 3 evaluators)")
    ds = make_synthetic_fairness_dataset(n=2000, seed=0)
    raw = run_benchmark(ds, FAST_CLASSIFIERS)
    phase3_rows = raw[raw["evaluator"].isin(
        ["membership_inference", "art_robustness", "drift_detection"])]
    print(phase3_rows[["model", "evaluator", "metric", "value", "mechanism"]]
         .to_string(index=False))
    return raw


def part2_privacy_ranking(raw: pd.DataFrame):
    section("2. Which models are most vulnerable to membership inference?")
    adv = raw[(raw["evaluator"] == "membership_inference") &
             (raw["metric"] == "membership_inference_advantage")]
    ranked = adv[["model", "value"]].sort_values("value", ascending=False)
    ranked = ranked.rename(columns={"value": "attack_advantage (0=no leakage detected)"})
    print(ranked.to_string(index=False))
    print("\nA model that memorizes its training data (e.g. an unconstrained "
         "tree ensemble) tends to show a higher attack advantage than a "
         "regularized linear model -- this is the real signal the Phase-1 "
         "train/test-gap proxy was only approximating.")


def part3_safety_profile(raw: pd.DataFrame):
    section("3. Three independent 'safety' angles: noise, adversarial search, drift")
    safety = raw[(raw["principle"] == "safety") &
                 (raw["evaluator"].isin(["robustness", "art_robustness", "drift_detection"]))]
    pivot = safety.pivot_table(index="model", columns=["evaluator", "metric"],
                               values="value").round(3)
    print(pivot.to_string())


def main():
    part0_coverage()
    raw = part1_run()
    part2_privacy_ranking(raw)
    part3_safety_profile(raw)


if __name__ == "__main__":
    main()
