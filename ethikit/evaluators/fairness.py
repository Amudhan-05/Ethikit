"""Fairness evaluator.

Inspects PREDICTIONS grouped by a sensitive attribute. Prefers Fairlearn's
MetricFrame; if Fairlearn is not installed it computes the same core metrics
directly with numpy so the demo still runs. In Claude Code, `pip install
fairlearn aif360` unlocks the richer implementations (AIF360 has 70+ metrics
and mitigation algorithms you can wire in as further evaluators).

Classification: demographic parity diff, equalized odds diff, and the
worst-off group's selection rate. Regression (Phase 2): the same idea
applied to error instead of selection rate -- group MAE/MSE gap, and the
worst-off group's own MAE -- since "selection rate" and TPR/FPR aren't
defined for a continuous target.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


def _try_fairlearn():
    try:
        import fairlearn  # noqa: F401
        return True
    except Exception:
        return False


class FairnessEvaluator(Evaluator):
    name = "fairness"
    principle_ids = ["fairness", "inclusiveness"]
    supports = [TaskType.CLASSIFICATION, TaskType.REGRESSION]
    mechanism = (
        "Groups the model's test-set predictions by a declared sensitive attribute "
        "and compares outcomes across groups using fairlearn.metrics.MetricFrame (or "
        "the equivalent numpy computation if fairlearn isn't installed): for "
        "classification, selection rate (demographic parity) and TPR/FPR gap "
        "(equalized odds); for regression, MAE/MSE gap between groups, since "
        "selection rate isn't defined for a continuous target. Inspects PREDICTIONS "
        "only: it never looks at the model's weights or training process, so it can "
        "flag *that* a gap exists between groups but not *why*."
    )
    limits = ('Group fairness on one declared attribute only (no intersections); demographic parity and equalized odds can conflict, so both cannot generally be zero; says nothing about why a gap exists.')
    formula = (
        "selection_rate(g) = P(ŷ=1 | A=g)\n"
        "TPR(g) = P(ŷ=1 | y=1, A=g)    FPR(g) = P(ŷ=1 | y=0, A=g)\n"
        "demographic_parity_diff = max_g selection_rate(g) − min_g selection_rate(g)\n"
        "equalized_odds_diff = max( max_g TPR − min_g TPR , max_g FPR − min_g FPR )\n"
        "min_group_selection_rate = min_g selection_rate(g)\n"
        "regression: group_mae_gap = max_g MAE(g) − min_g MAE(g)"
    )

    def applies_to(self, ctx: EvalContext) -> bool:
        return (
            ctx.dataset.task in self.supports
            and ctx.dataset.sensitive_test is not None
        )

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        y_true = np.asarray(ctx.dataset.y_test)
        y_pred = np.asarray(ctx.y_pred)
        groups = np.asarray(ctx.dataset.sensitive_test)
        group_labels = sorted(str(g) for g in np.unique(groups))
        n_groups = len(group_labels)
        sensitive = ctx.dataset.sensitive_name

        if ctx.dataset.task == TaskType.REGRESSION:
            return self._evaluate_regression(y_true, y_pred, groups, group_labels, sensitive)

        if _try_fairlearn():
            dpd, eod, dir_ = self._via_fairlearn(y_true, y_pred, groups)
            backend = "fairlearn.metrics.MetricFrame"
        else:
            dpd, eod, dir_ = self._via_numpy(y_true, y_pred, groups)
            backend = "numpy fallback (equivalent hand-rolled computation)"

        mechanism = (
            f"Computed via {backend}, grouping {len(y_true)} test predictions by "
            f"sensitive attribute '{sensitive}' into {n_groups} groups "
            f"({', '.join(group_labels)})."
        )
        details = {"backend": backend, "sensitive": sensitive, "groups": group_labels,
                   "trace": self._classification_trace(y_true, y_pred, groups, sensitive,
                                                       dpd, eod, dir_)}
        return [
            EvalResult(self.name, "fairness", InspectionTarget.PREDICTIONS,
                       "demographic_parity_diff", dpd, False, mechanism, details),
            EvalResult(self.name, "fairness", InspectionTarget.PREDICTIONS,
                       "equalized_odds_diff", eod, False, mechanism, details),
            EvalResult(self.name, "inclusiveness", InspectionTarget.PREDICTIONS,
                       "min_group_selection_rate", dir_, True, mechanism, details),
        ]

    # --- trace: the per-group table every headline number is read off ---------
    def _classification_trace(self, y_true, y_pred, groups, sensitive, dpd, eod, dir_):
        rows, sel, tprs, fprs = [], {}, {}, {}
        for g in sorted(np.unique(groups), key=str):
            m = groups == g
            pos, neg = m & (y_true == 1), m & (y_true == 0)
            sel[g] = float(np.mean(y_pred[m] == 1))
            tprs[g] = float(np.mean(y_pred[pos] == 1)) if pos.any() else 0.0
            fprs[g] = float(np.mean(y_pred[neg] == 1)) if neg.any() else 0.0
            rows.append([str(g), int(m.sum()), round(float(np.mean(y_true[m] == 1)), 4),
                         round(sel[g], 4), round(tprs[g], 4), round(fprs[g], 4)])
        hi, lo = max(sel, key=sel.get), min(sel, key=sel.get)
        tpr_gap = max(tprs.values()) - min(tprs.values())
        fpr_gap = max(fprs.values()) - min(fprs.values())
        return make_trace(
            steps=[
                f"demographic_parity_diff = sel({hi}) − sel({lo}) = "
                f"{sel[hi]:.4f} − {sel[lo]:.4f} = {dpd:.4f}",
                f"equalized_odds_diff = max(TPR gap {tpr_gap:.4f}, FPR gap {fpr_gap:.4f}) = {eod:.4f}",
                f"min_group_selection_rate = sel({lo}) = {dir_:.4f}",
            ],
            tables=[{"title": f"Per-group outcomes on the test set (A = {sensitive})",
                     "columns": ["group", "n", "base_rate P(y=1)", "selection_rate",
                                 "TPR", "FPR"],
                     "rows": rows}],
            charts=[{"title": "Selection rate by group", "type": "bars",
                     "labels": [r[0] for r in rows],
                     "series": [{"name": "selection_rate", "values": [r[3] for r in rows]},
                                {"name": "base_rate", "values": [r[2] for r in rows]}]}],
        )

    # --- classification: fairlearn path --------------------------------------
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

    # --- classification: numpy fallback ---------------------------------------
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

    # --- regression: group MAE/MSE gap ----------------------------------------
    def _evaluate_regression(self, y_true, y_pred, groups, group_labels, sensitive):
        n_groups = len(group_labels)
        if _try_fairlearn():
            mae_gap, mse_gap, worst_mae = self._via_fairlearn_regression(y_true, y_pred, groups)
            backend = "fairlearn.metrics.MetricFrame (mean_absolute_error, mean_squared_error)"
        else:
            mae_gap, mse_gap, worst_mae = self._via_numpy_regression(y_true, y_pred, groups)
            backend = "numpy fallback (equivalent hand-rolled computation)"

        mechanism = (
            f"Computed via {backend}, grouping {len(y_true)} test predictions by "
            f"sensitive attribute '{sensitive}' into {n_groups} groups "
            f"({', '.join(group_labels)}); gap = max(group error) - min(group error)."
        )
        rows = []
        for g in sorted(np.unique(groups), key=str):
            err = y_pred[groups == g] - y_true[groups == g]
            rows.append([str(g), int((groups == g).sum()), round(float(np.mean(np.abs(err))), 4),
                         round(float(np.mean(err ** 2)), 4), round(float(np.mean(err)), 4)])
        trace = make_trace(
            steps=[f"group_mae_gap = max_g MAE − min_g MAE = {mae_gap:.4f}",
                   f"group_mse_gap = {mse_gap:.4f}", f"worst_group_mae = {worst_mae:.4f}"],
            tables=[{"title": f"Per-group error (A = {sensitive})",
                     "columns": ["group", "n", "MAE", "MSE", "mean signed error"],
                     "rows": rows}],
            charts=[{"title": "MAE by group", "type": "bars", "labels": [r[0] for r in rows],
                     "series": [{"name": "MAE", "values": [r[2] for r in rows]}]}],
        )
        details = {"backend": backend, "sensitive": sensitive, "groups": group_labels,
                   "trace": trace}
        return [
            EvalResult(self.name, "fairness", InspectionTarget.PREDICTIONS,
                       "group_mae_gap", mae_gap, False, mechanism, details),
            EvalResult(self.name, "fairness", InspectionTarget.PREDICTIONS,
                       "group_mse_gap", mse_gap, False, mechanism, details),
            EvalResult(self.name, "inclusiveness", InspectionTarget.PREDICTIONS,
                       "worst_group_mae", worst_mae, False, mechanism, details),
        ]

    def _via_fairlearn_regression(self, y_true, y_pred, groups):
        from fairlearn.metrics import MetricFrame
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        mf = MetricFrame(metrics={"mae": mean_absolute_error, "mse": mean_squared_error},
                         y_true=y_true, y_pred=y_pred, sensitive_features=groups)
        mae_by_group, mse_by_group = mf.by_group["mae"], mf.by_group["mse"]
        return (float(mae_by_group.max() - mae_by_group.min()),
                float(mse_by_group.max() - mse_by_group.min()),
                float(mae_by_group.max()))

    def _via_numpy_regression(self, y_true, y_pred, groups):
        classes = np.unique(groups)
        maes, mses = [], []
        for g in classes:
            err = y_pred[groups == g] - y_true[groups == g]
            maes.append(float(np.mean(np.abs(err))))
            mses.append(float(np.mean(err ** 2)))
        return (float(max(maes) - min(maes)), float(max(mses) - min(mses)), float(max(maes)))


register(FairnessEvaluator())
