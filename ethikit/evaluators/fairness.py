"""Fairness evaluator.

Inspects PREDICTIONS grouped by a sensitive attribute. Prefers Fairlearn's
MetricFrame; if Fairlearn is not installed it computes the same core metrics
directly with numpy so the demo still runs. In Claude Code, `pip install
fairlearn aif360` unlocks the richer implementations (AIF360 has 70+ metrics
and mitigation algorithms you can wire in as further evaluators).
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, register


def _try_fairlearn():
    try:
        import fairlearn  # noqa: F401
        return True
    except Exception:
        return False


class FairnessEvaluator(Evaluator):
    name = "fairness"
    principle_ids = ["fairness", "inclusiveness"]
    supports = [TaskType.CLASSIFICATION]  # extend to regression via group MAE later
    mechanism = (
        "Groups the model's test-set predictions by a declared sensitive attribute "
        "and compares outcome rates across groups -- selection rate (demographic "
        "parity), true/false-positive rate gap (equalized odds) -- using "
        "fairlearn.metrics.MetricFrame, or the equivalent numpy computation if "
        "fairlearn isn't installed. Inspects PREDICTIONS only: it never looks at "
        "the model's weights or training process, so it can flag *that* a gap "
        "exists between groups but not *why*."
    )

    def applies_to(self, ctx: EvalContext) -> bool:
        return (
            ctx.dataset.task == TaskType.CLASSIFICATION
            and ctx.dataset.sensitive_test is not None
        )

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        y_true = np.asarray(ctx.dataset.y_test)
        y_pred = np.asarray(ctx.y_pred)
        groups = np.asarray(ctx.dataset.sensitive_test)
        group_labels = sorted(str(g) for g in np.unique(groups))

        if _try_fairlearn():
            dpd, eod, dir_ = self._via_fairlearn(y_true, y_pred, groups)
            backend = "fairlearn.metrics.MetricFrame"
        else:
            dpd, eod, dir_ = self._via_numpy(y_true, y_pred, groups)
            backend = "numpy fallback (equivalent hand-rolled computation)"

        mechanism = (
            f"Computed via {backend}, grouping {len(y_true)} test predictions by "
            f"sensitive attribute '{ctx.dataset.sensitive_name}' into "
            f"{len(group_labels)} groups ({', '.join(group_labels)})."
        )
        details = {"backend": backend, "sensitive": ctx.dataset.sensitive_name,
                   "groups": group_labels}
        return [
            EvalResult(self.name, "fairness", InspectionTarget.PREDICTIONS,
                       "demographic_parity_diff", dpd, False, mechanism, details),
            EvalResult(self.name, "fairness", InspectionTarget.PREDICTIONS,
                       "equalized_odds_diff", eod, False, mechanism, details),
            EvalResult(self.name, "inclusiveness", InspectionTarget.PREDICTIONS,
                       "min_group_selection_rate", dir_, True, mechanism, details),
        ]

    # --- fairlearn path -----------------------------------------------------
    def _via_fairlearn(self, y_true, y_pred, groups):
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference,
            MetricFrame,
            selection_rate,
        )
        dpd = float(demographic_parity_difference(y_true, y_pred, sensitive_features=groups))
        eod = float(equalized_odds_difference(y_true, y_pred, sensitive_features=groups))
        mf = MetricFrame(metrics=selection_rate, y_true=y_true, y_pred=y_pred,
                         sensitive_features=groups)
        return dpd, eod, float(mf.by_group.min())

    # --- numpy fallback -----------------------------------------------------
    def _via_numpy(self, y_true, y_pred, groups):
        classes = np.unique(groups)
        sel_rates, tpr, fpr = [], [], []
        for g in classes:
            mask = groups == g
            sel_rates.append(float(np.mean(y_pred[mask] == 1)))
            pos = mask & (y_true == 1)
            neg = mask & (y_true == 0)
            tpr.append(float(np.mean(y_pred[pos] == 1)) if pos.any() else 0.0)
            fpr.append(float(np.mean(y_pred[neg] == 1)) if neg.any() else 0.0)
        dpd = float(max(sel_rates) - min(sel_rates))
        eod = float(max(max(tpr) - min(tpr), max(fpr) - min(fpr)))
        return dpd, eod, float(min(sel_rates))


register(FairnessEvaluator())
