"""End-to-end demo: load data -> sweep models -> evaluate -> score -> report.

Run:  python -m examples.run_demo
It writes `ethikit_report.html` and prints the comparison table to the console.
Works offline (synthetic data) and online (real Adult dataset).
"""
from __future__ import annotations

import pandas as pd

from ethikit.data import load_adult
from ethikit.benchmark import run_benchmark, flat_classifiers, family_of
from ethikit.scoring import leaderboard, normalize, objective_matrix, pareto_front, principle_profile
from ethikit.report import render_html, render_model_cards
from ethikit.coverage import coverage_table

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def main():
    print("=== Tool & principle coverage (which tool addresses which principle, and how) ===")
    print(coverage_table())

    ds = load_adult()
    print(f"\nDataset: {ds.name}  |  sensitive attribute: {ds.sensitive_name}")
    print(f"train={len(ds.y_train)}  test={len(ds.y_test)}\n")

    raw = run_benchmark(ds, flat_classifiers())

    # A worked example of an EXPLICIT value judgement: fairness-first weights.
    weights = {"fairness": 0.35, "explainability": 0.2, "safety": 0.2,
               "privacy": 0.15, "inclusiveness": 0.1}
    board = leaderboard(raw, weights=weights)
    print("=== Comparison (fairness-weighted order; a selection aid, not a verdict) ===")
    print(board, "\n")

    print("Top-scoring model per family under these weights (a selection aid, not a verdict):")
    board2 = board.copy()
    board2["family"] = [family_of(m) for m in board2.index]
    for fam, rows in board2.groupby("family"):
        top = rows.sort_values("weighted_score", ascending=False).iloc[0]
        print(f"  {fam:14} -> {top.name:22} (score={top['weighted_score']:.3f}, "
              f"utility={top['utility']:.3f})")

    dn = normalize(raw)
    objectives = objective_matrix(dn)   # includes the utility axis
    front = pareto_front(objectives)
    print(f"\nPareto-optimal models (utility vs all principles): {front}")

    out = render_html(ds.name, board, front, "ethikit_report.html")
    print(f"\nHTML report written to: {out}")

    profile = principle_profile(dn)
    card_paths = render_model_cards(ds.name, raw, profile, front,
                                     out_dir="model_cards", sensitive_name=ds.sensitive_name)
    print(f"Model cards written to: {', '.join(card_paths)}")


if __name__ == "__main__":
    main()
