"""Permutation-importance evaluator: a second, independent view of what the model relies on.

Shuffle one feature's column in the test set and measure how much the model's accuracy drops:
a feature the model leans on hurts a lot when scrambled. Unlike SHAP or a surrogate tree this
needs no approximation of the model -- it asks the real model -- so it is a useful cross-check
on the explainability tool.

Two signals:
  * importance_concentration (explainability): share of the total importance carried by the top
    three features. A model whose behaviour is concentrated in a few features is easier to
    explain and to audit. (A heuristic, not a law: some honest tasks need many features.)
  * sensitive_feature_reliance (transparency): share of the total importance carried by the
    sensitive attribute's own columns. It answers "does the model use the protected attribute
    directly?" -- it cannot see use through proxies (the data profile's proxy screen does that).
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


class PermutationImportanceEvaluator(Evaluator):
    name = "permutation_importance"
    principle_ids = ["explainability", "transparency"]
    supports = [TaskType.CLASSIFICATION, TaskType.REGRESSION]
    mechanism = (
        "sklearn.inspection.permutation_importance on the held-out test set: each feature column "
        "is shuffled n_repeats times and the drop in the model's own score (accuracy, or R² for "
        "regression) is recorded. Treats the model as a black box. Reports how concentrated the "
        "model's reliance is (top-3 share of total importance) and how much of it falls on the "
        "sensitive attribute's own columns."
    )
    formula = (
        "imp_j = score(X) − mean_r score(X with column j shuffled, repeat r)\n"
        "share_j = max(imp_j, 0) / Σ_k max(imp_k, 0)\n"
        "importance_concentration = Σ_{j in top-3} share_j\n"
        "sensitive_feature_reliance = Σ_{j in sensitive columns} share_j"
    )
    limits = ("Correlated features share or hide importance (shuffling one leaves its twin); "
              "global only (see inspect's local explanation for one row); does not detect use of "
              "the sensitive attribute through proxies.")

    def __init__(self, n_repeats: int = 5, max_rows: int = 2000, seed: int = 0):
        self.n_repeats = n_repeats
        self.max_rows = max_rows
        self.seed = seed

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        from sklearn.inspection import permutation_importance
        ds = ctx.dataset
        X = np.asarray(ds.X_test, dtype=float)
        y = np.asarray(ds.y_test)
        rng = np.random.default_rng(self.seed)
        if len(y) > self.max_rows:
            idx = rng.choice(len(y), size=self.max_rows, replace=False)
            X, y = X[idx], y[idx]
        names = ds.feature_names or [f"f{i}" for i in range(X.shape[1])]
        res = permutation_importance(ctx.model, X, y, n_repeats=self.n_repeats, random_state=self.seed)
        imp = np.maximum(res.importances_mean, 0.0)
        total = float(imp.sum())
        share = imp / total if total > 0 else np.zeros_like(imp)
        order = np.argsort(share)[::-1]
        conc = float(share[order[:3]].sum()) if total > 0 else 0.0
        sname = (ds.sensitive_name or "").lower()
        sens_idx = [i for i, n in enumerate(names) if sname and (n.lower() == sname or n.lower().startswith(sname + "="))]
        reliance = float(share[sens_idx].sum()) if sens_idx else 0.0
        top = [[names[i], round(float(res.importances_mean[i]), 5), round(float(res.importances_std[i]), 5),
                round(float(share[i]), 4)] for i in order[:12]]
        mechanism = (f"permutation_importance over {len(y)} test rows x {self.n_repeats} repeats: top-3 features "
                     f"carry {conc:.1%} of the importance; the sensitive attribute's own column(s) "
                     + (f"({', '.join(names[i] for i in sens_idx)}) carry {reliance:.1%}." if sens_idx
                        else "are not model inputs, so direct reliance is 0 (proxies are not covered)."))
        trace = make_trace(
            steps=[f"importance_concentration = top-3 share = {conc:.4f}",
                   f"sensitive_feature_reliance = {reliance:.4f}"
                   + ("" if sens_idx else "  (sensitive attribute is not a model input)")],
            tables=[{"title": "Permutation importance (drop in score when the column is shuffled)",
                     "columns": ["feature", "mean drop", "std", "share"], "rows": top}],
            charts=[{"title": "Share of total importance", "type": "bars", "labels": [t[0] for t in top],
                     "series": [{"name": "share", "values": [t[3] for t in top]}]}],
        )
        d = {"trace": trace, "sensitive_columns": [names[i] for i in sens_idx]}
        return [EvalResult(self.name, "explainability", InspectionTarget.MODEL_INTERNALS,
                           "importance_concentration", conc, True, mechanism, d),
                EvalResult(self.name, "transparency", InspectionTarget.MODEL_INTERNALS,
                           "sensitive_feature_reliance", reliance, False, mechanism, d)]


register(PermutationImportanceEvaluator())
