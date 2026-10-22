"""Utility / performance metrics.

Not an ethics principle on its own, but every ethical trade-off is measured
*against* utility, so we always compute it. Tagged under 'transparency' only
so it appears in the profile; the aggregator treats it as the utility axis.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn import metrics as skm

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


class PerformanceEvaluator(Evaluator):
    name = "performance"
    principle_ids = ["utility"]  # not a canonical principle; the utility axis
    mechanism = (
        "Standard scikit-learn metrics (accuracy/F1/ROC-AUC for classification, "
        "R2/RMSE/MAE for regression) computed by comparing the model's own "
        "predict()/predict_proba() output against the true labels on the held-out "
        "test set. Not an ethics evaluator -- provides the utility axis every "
        "trade-off in the profile is measured against."
    )
    limits = ('Measures utility only; says nothing about who bears the errors. A single held-out split (use --cv for a spread).')
    formula = (
        "accuracy = (TP + TN) / n\n"
        "f1 (macro) = mean_c 2·precision_c·recall_c / (precision_c + recall_c)\n"
        "roc_auc = P(score(x⁺) > score(x⁻))\n"
        "regression: r2 = 1 − Σ(y−ŷ)² / Σ(y−ȳ)²,  rmse = √mean((y−ŷ)²)"
    )

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        y_true = np.asarray(ctx.dataset.y_test)
        y_pred = np.asarray(ctx.y_pred)
        n = len(y_true)
        out: List[EvalResult] = []

        if ctx.dataset.task == TaskType.CLASSIFICATION:
            labels = sorted(np.unique(np.concatenate([y_true, y_pred])), key=str)
            cm = skm.confusion_matrix(y_true, y_pred, labels=labels)
            correct = int(np.trace(cm))
            trace = make_trace(
                steps=[f"accuracy = correct / n = {correct} / {n} = {correct / n:.4f}"],
                tables=[{"title": "Confusion matrix (rows = true, columns = predicted)",
                         "columns": ["true \\ pred"] + [str(l) for l in labels],
                         "rows": [[str(l)] + [int(v) for v in row]
                                  for l, row in zip(labels, cm)]}],
            )
            out.append(self._r("accuracy", skm.accuracy_score(y_true, y_pred), True, n, trace))
            out.append(self._r("f1", skm.f1_score(y_true, y_pred, average="macro"), True, n, trace))
            if ctx.y_proba is not None:
                try:
                    proba = np.asarray(ctx.y_proba)
                    score = proba[:, 1] if proba.ndim == 2 else proba
                    out.append(self._r("roc_auc", skm.roc_auc_score(y_true, score), True, n, trace))
                except Exception:
                    pass
        else:
            resid = y_true - y_pred
            hist, edges = np.histogram(resid, bins=12)
            trace = make_trace(
                steps=[f"r2 = 1 − SS_res/SS_tot = 1 − {float(np.sum(resid ** 2)):.3f} / "
                       f"{float(np.sum((y_true - y_true.mean()) ** 2)):.3f}"],
                charts=[{"title": "Residual distribution (y − ŷ)", "type": "hist",
                         "labels": [f"{e:.2f}" for e in edges[:-1]],
                         "series": [{"name": "rows", "values": [int(h) for h in hist]}]}],
            )
            out.append(self._r("r2", skm.r2_score(y_true, y_pred), True, n, trace))
            rmse = float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
            out.append(self._r("rmse", rmse, False, n, trace))
            out.append(self._r("mae", float(skm.mean_absolute_error(y_true, y_pred)), False, n, trace))
        return out

    def _r(self, metric: str, value: float, higher: bool, n: int, trace: dict) -> EvalResult:
        return EvalResult(
            evaluator=self.name, principle_id="utility",
            target=InspectionTarget.PREDICTIONS, metric=metric,
            value=float(value), higher_is_better=higher,
            mechanism=f"sklearn.metrics.{metric}(y_test, y_pred) over {n} held-out test rows.",
            details={"axis": "utility", "n_test": n, "trace": trace},
        )


register(PerformanceEvaluator())
