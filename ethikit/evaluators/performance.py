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
from .base import Evaluator, register


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

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        y_true = np.asarray(ctx.dataset.y_test)
        y_pred = np.asarray(ctx.y_pred)
        n = len(y_true)
        out: List[EvalResult] = []

        if ctx.dataset.task == TaskType.CLASSIFICATION:
            out.append(self._r("accuracy", skm.accuracy_score(y_true, y_pred), True, n))
            out.append(self._r("f1", skm.f1_score(y_true, y_pred, average="macro"), True, n))
            if ctx.y_proba is not None:
                try:
                    proba = np.asarray(ctx.y_proba)
                    score = proba[:, 1] if proba.ndim == 2 else proba
                    out.append(self._r("roc_auc", skm.roc_auc_score(y_true, score), True, n))
                except Exception:
                    pass
        else:
            out.append(self._r("r2", skm.r2_score(y_true, y_pred), True, n))
            rmse = float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))
            out.append(self._r("rmse", rmse, False, n))
            out.append(self._r("mae", float(skm.mean_absolute_error(y_true, y_pred)), False, n))
        return out

    def _r(self, metric: str, value: float, higher: bool, n: int) -> EvalResult:
        return EvalResult(
            evaluator=self.name, principle_id="utility",
            target=InspectionTarget.PREDICTIONS, metric=metric,
            value=float(value), higher_is_better=higher,
            mechanism=f"sklearn.metrics.{metric}(y_test, y_pred) over {n} held-out test rows.",
            details={"axis": "utility", "n_test": n},
        )


register(PerformanceEvaluator())
