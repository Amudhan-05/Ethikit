"""Phase 2 demo: generalization & statistical significance.

Everything in examples/run_demo.py is a single train/test split on a single
dataset -- useful, but "model A is best" from one split could just be that
split's luck. This script demonstrates the four Phase 2 capabilities added
on top of the same architecture (every evaluator still emits
EvalResult -> as_record(); nothing here required changing that):

  1. Repeated cross-validation (ethikit.benchmark.cv) -- a DISTRIBUTION of
     scores per model instead of one number.
  2. Friedman + Nemenyi significance testing + a critical-difference diagram
     (ethikit.scoring.significance) -- so "best model" claims from (1) are
     backed by a hypothesis test.
  3. A meta-benchmark across multiple PMLB datasets (ethikit.benchmark.meta)
     -- "best model per family, in general" aggregated across datasets, not
     one dataset's anecdote.
  4. Regression fairness (group MAE/MSE gap) -- FairnessEvaluator's new
     regression branch, exercised here with a synthetic biased regression
     dataset + a regressor zoo.

Run:  python -m examples.run_phase2_demo
Needs pmlb + scikit-posthocs (pip install -r requirements.txt already
includes both). Takes a few minutes -- CV alone fits N models x n_splits x
n_repeats times.
"""
from __future__ import annotations

import pandas as pd

from ethikit.data import make_synthetic_fairness_raw, make_synthetic_fairness_regression_dataset
from ethikit.benchmark import (
    run_benchmark, run_benchmark_cv, score_distribution, summarize_cv,
    run_meta_benchmark, aggregate_across_datasets, weighted_score_across_datasets,
    flat_classifiers, flat_regressors,
)
from ethikit.scoring import compare_models, critical_difference_diagram

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

# SVC(probability=True) is fine on the small synthetic dataset used for CV
# below, but its fit/predict time scales badly with row count -- excluded
# from the PMLB meta-benchmark, several of whose datasets run into the
# thousands of rows (see HANDOFF.md's documented SVC bottleneck).
FAST_CLASSIFIERS = {k: v for k, v in flat_classifiers().items() if k != "svc"}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def part1_cv():
    section("1. Repeated cross-validation (replaces a single train/test split)")
    raw = make_synthetic_fairness_raw(n=2000, seed=0)
    cv_raw = run_benchmark_cv(raw, flat_classifiers(), n_splits=5, n_repeats=3, seed=0)
    print(f"{len(cv_raw)} rows across {cv_raw['repeat'].nunique()} repeats x "
         f"{cv_raw['fold'].nunique()} folds = {cv_raw['repeat'].nunique() * cv_raw['fold'].nunique()} samples\n")
    print(summarize_cv(cv_raw).to_string(index=False))
    return cv_raw


def part2_significance(cv_raw):
    section("2. Statistical significance: Friedman + Nemenyi + critical-difference diagram")

    for metric, evaluator, ascending, label in [
        ("accuracy", "performance", False, "accuracy"),
        ("perturbation_stability", "robustness", False, "robustness (safety)"),
    ]:
        dist = score_distribution(cv_raw, metric, evaluator=evaluator)
        result = compare_models(dist, ascending=ascending)
        print(f"\n--- {label} ---")
        print(f"Friedman: chi2={result['statistic']:.3f}  p={result['p_value']:.2e}  "
             f"n_samples={result['n_samples']}  n_models={result['n_models']}  "
             f"significant={result['significant_at_0.05']}  enough_data={result['enough_data']}")
        print("Average ranks (1 = best):")
        print(result["average_ranks"].to_string())
        if result["nemenyi"] is not None:
            out = f"cd_diagram_{metric}.png"
            critical_difference_diagram(dist, out, ascending=ascending,
                                        title=f"Critical difference -- {label}")
            print(f"Critical-difference diagram written to: {out}")
        else:
            print("(Friedman not significant or scikit-posthocs unavailable -- no Nemenyi/CD diagram)")


def part3_meta_benchmark():
    section("3. Meta-benchmark across multiple datasets (PMLB)")
    raw = run_meta_benchmark(FAST_CLASSIFIERS)
    print(f"\n{len(raw)} rows across datasets: {sorted(raw['dataset'].unique())}\n")

    print("Per-principle profile, averaged across all datasets:")
    print(aggregate_across_datasets(raw).to_string())

    weights = {"fairness": 0.3, "explainability": 0.2, "safety": 0.3, "privacy": 0.2}
    print(f"\nBest model per family, in general (weights={weights}):")
    board = weighted_score_across_datasets(raw, weights)
    print(board.to_string())
    print("\nTop model per family (board is already sorted best-first, so the first "
         "row per family group is the winner):")
    for fam, rows in board.groupby("family"):
        top = rows.iloc[0]
        print(f"  {fam:14} -> {top.name:22} "
             f"(mean_weighted_score={top['mean_weighted_score']:.3f}, "
             f"over {top['n_datasets']} datasets)")


def part4_regression_fairness():
    section("4. Regression fairness (group MAE/MSE gap)")
    ds = make_synthetic_fairness_regression_dataset(n=2000, seed=0)
    print(f"Dataset: {ds.name}  |  sensitive attribute: {ds.sensitive_name}  |  task: {ds.task.value}")
    raw = run_benchmark(ds, flat_regressors())
    fairness_rows = raw[raw["evaluator"] == "fairness"]
    print(fairness_rows[["model", "metric", "value", "mechanism"]].to_string(index=False))


def main():
    cv_raw = part1_cv()
    part2_significance(cv_raw)
    part3_meta_benchmark()
    part4_regression_fairness()


if __name__ == "__main__":
    main()
