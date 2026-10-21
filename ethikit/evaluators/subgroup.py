"""Subgroup performance evaluator.

Fairness metrics compare *decisions* between groups; this one compares *how good the model is*
for each group. A model can have equal selection rates and still be far less accurate for a
small group -- the classic "works well on average, fails the minority" pattern. Reports the
worst group's accuracy (inclusiveness: nobody is left with a bad model) and the gap between the
best- and worst-served groups (fairness).
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


class SubgroupPerformanceEvaluator(Evaluator):
    name = "subgroup_performance"
    principle_ids = ["inclusiveness", "fairness"]
    supports = [TaskType.CLASSIFICATION]
    mechanism = (
        "Splits the held-out test set by the sensitive attribute and computes accuracy, and the "
        "true-positive and true-negative rates, for every group separately. The worst group's "
        "accuracy (-> inclusiveness) and the best-minus-worst accuracy gap (-> fairness) are "
        "reported; groups smaller than a minimum size are listed but not scored. Inspects "
        "PREDICTIONS against ground truth."
    )
    formula = (
        "acc(g) = mean_{i: A_i=g} 1[ŷ_i = y_i]\n"
        "worst_group_accuracy = min_g acc(g)\n"
        "accuracy_gap_between_groups = max_g acc(g) − min_g acc(g)"
    )
    limits = ("Only the declared sensitive attribute is sliced (not intersections such as "
              "sex x age); small groups give noisy numbers; says nothing about why a group is served worse.")

    def __init__(self, min_group_rows: int = 20):
        self.min_group_rows = min_group_rows

    def applies_to(self, ctx: EvalContext) -> bool:
        return ctx.dataset.task in self.supports and ctx.dataset.sensitive_test is not None

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        y = np.asarray(ctx.dataset.y_test)
        yp = np.asarray(ctx.y_pred)
        g = np.asarray(ctx.dataset.sensitive_test).astype(str)
        pos = sorted(np.unique(y), key=str)[-1]
        rows, accs = [], {}
        for grp in sorted(set(g)):
            m = g == grp
            n = int(m.sum())
            acc = float(np.mean(yp[m] == y[m]))
            tpr = float(np.mean(yp[m & (y == pos)] == pos)) if (m & (y == pos)).any() else float("nan")
            tnr = float(np.mean(yp[m & (y != pos)] != pos)) if (m & (y != pos)).any() else float("nan")
            scored = n >= self.min_group_rows
            if scored:
                accs[grp] = acc
            rows.append([grp, n, round(acc, 4), round(tpr, 4), round(tnr, 4), "yes" if scored else "too small"])
        if not accs:
            accs = {r[0]: r[2] for r in rows}
        worst = min(accs, key=accs.get)
        best = max(accs, key=accs.get)
        gap = accs[best] - accs[worst]
        mechanism = (f"accuracy per '{ctx.dataset.sensitive_name}' group on {len(y)} test rows: "
                     + ", ".join(f"{k}={v:.3f}" for k, v in accs.items())
                     + f" -> worst {worst} ({accs[worst]:.4f}), gap {gap:.4f}.")
        trace = make_trace(
            steps=[f"worst_group_accuracy = acc({worst}) = {accs[worst]:.4f}",
                   f"accuracy_gap = acc({best}) − acc({worst}) = {accs[best]:.4f} − {accs[worst]:.4f} = {gap:.4f}"],
            tables=[{"title": "Performance per group", "columns": ["group", "rows", "accuracy", "TPR", "TNR", "scored"],
                     "rows": rows}],
            charts=[{"title": "Accuracy per group", "type": "bars", "labels": [r[0] for r in rows],
                     "series": [{"name": "accuracy", "values": [r[2] for r in rows]}]}],
        )
        d = {"trace": trace, "min_group_rows": self.min_group_rows}
        return [EvalResult(self.name, "inclusiveness", InspectionTarget.PREDICTIONS,
                           "worst_group_accuracy", accs[worst], True, mechanism, d),
                EvalResult(self.name, "fairness", InspectionTarget.PREDICTIONS,
                           "accuracy_gap_between_groups", gap, False, mechanism, d)]


register(SubgroupPerformanceEvaluator())
