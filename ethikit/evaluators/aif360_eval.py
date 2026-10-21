"""AIF360 fairness evaluator.

IBM's AI Fairness 360 (https://github.com/Trusted-AI/AIF360, Apache-2.0) is the
largest open-source fairness toolkit: 70+ group, individual and
between-group fairness metrics plus pre-/in-/post-processing mitigation
algorithms. Phase 1 left it out because its 0.5.x release failed to build on
Python 3.12; aif360 0.6.1 installs and runs on Python 3.11-3.13 (confirmed by
hand), so it now sits alongside the fairlearn-backed FairnessEvaluator as a
second, independent opinion on the same principle.

What it adds that fairlearn's three numbers don't:
  - disparate impact (the "80% rule" ratio used in US employment law),
  - equal-opportunity and average-odds differences (signed, so you can see
    *which* group is disadvantaged, not just the size of the gap),
  - the Theil index -- an *individual* fairness measure (generalised entropy
    of per-person benefit b_i = ŷ_i − y_i + 1), which looks at inequality
    between individuals rather than between declared groups,
  - between-group Theil, the share of that inequality explained by group.

AIF360 needs a privileged / unprivileged split. ethikit picks the
privileged group automatically as the sensitive value with the highest
base rate P(y=1) in the *training* labels (the group the data already
favours), treats every other value as unprivileged, and says so in the
mechanism text; this is a convention, not a judgement, and the console
shows it so a user can disagree.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


def _import_aif360():
    """Import the bits we use with aif360's import-time chatter silenced (it logs
    'No module named tensorflow/inFairness: X will be unavailable' warnings for
    mitigation algorithms ethikit never calls)."""
    import logging
    logging.disable(logging.WARNING)
    try:
        from aif360.datasets import BinaryLabelDataset
        from aif360.metrics import ClassificationMetric
    finally:
        logging.disable(logging.NOTSET)
    return BinaryLabelDataset, ClassificationMetric


def _has_aif360() -> bool:
    try:
        _import_aif360()
        return True
    except Exception:
        return False


class AIF360Evaluator(Evaluator):
    name = "aif360"
    principle_ids = ["fairness", "inclusiveness"]
    supports = [TaskType.CLASSIFICATION]
    mechanism = (
        "Wraps the test labels and the model's test predictions as two aif360 "
        "BinaryLabelDatasets over a binarised sensitive attribute (privileged = the "
        "value with the highest training base rate, unprivileged = all others), then "
        "reads group metrics off aif360.metrics.ClassificationMetric: disparate impact, "
        "statistical parity, equal-opportunity and average-odds differences, plus the "
        "individual-fairness Theil index and its between-group component. Inspects "
        "PREDICTIONS against ground truth, like fairlearn, but adds the 80%-rule ratio "
        "and an individual-level inequality measure fairlearn doesn't compute."
    )
    install_hint = "pip install aif360"
    limits = ('Binarises the sensitive attribute (privileged = highest training base rate vs everyone else); the 80% rule is a legal heuristic, not an ethical standard.')
    formula = (
        "P = privileged group, U = unprivileged group\n"
        "disparate_impact = P(ŷ=1 | U) / P(ŷ=1 | P)             (1 = parity; < 0.8 fails the 80% rule)\n"
        "disparate_impact_gap = |1 − disparate_impact|\n"
        "statistical_parity_diff = P(ŷ=1 | U) − P(ŷ=1 | P)\n"
        "equal_opportunity_diff = TPR(U) − TPR(P)\n"
        "average_odds_diff = ½[(FPR(U) − FPR(P)) + (TPR(U) − TPR(P))]\n"
        "theil_index = (1/n) Σ_i (b_i/μ) ln(b_i/μ),   b_i = ŷ_i − y_i + 1,  μ = mean(b)"
    )

    def available(self) -> bool:
        return _has_aif360()

    def applies_to(self, ctx: EvalContext) -> bool:
        ds = ctx.dataset
        return (ds.task in self.supports and ds.sensitive_test is not None
                and len(np.unique(ds.y_test)) == 2 and _has_aif360())

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        import pandas as pd
        BinaryLabelDataset, ClassificationMetric = _import_aif360()

        ds = ctx.dataset
        y_true = np.asarray(ds.y_test)
        y_pred = np.asarray(ctx.y_pred)
        pos_label = sorted(np.unique(np.asarray(ds.y_train)), key=str)[-1]

        # privileged = value with the highest training base rate
        g_train = np.asarray(ds.sensitive_train if ds.sensitive_train is not None
                             else ds.sensitive_test)
        y_train = np.asarray(ds.y_train if ds.sensitive_train is not None else ds.y_test)
        rates = {g: float(np.mean(y_train[g_train == g] == pos_label))
                 for g in np.unique(g_train)}
        privileged = max(rates, key=rates.get)
        groups = np.asarray(ds.sensitive_test)
        a = (groups == privileged).astype(float)

        def bld(labels):
            df = pd.DataFrame({"a": a, "y": (labels == pos_label).astype(float)})
            return BinaryLabelDataset(df=df, label_names=["y"], protected_attribute_names=["a"],
                                      favorable_label=1.0, unfavorable_label=0.0)

        truth, pred = bld(y_true), bld(y_pred)
        m = ClassificationMetric(truth, pred, unprivileged_groups=[{"a": 0.0}],
                                 privileged_groups=[{"a": 1.0}])

        di = float(m.disparate_impact())
        if not np.isfinite(di):
            di = 0.0
        spd = float(m.statistical_parity_difference())
        eod = float(m.equal_opportunity_difference())
        aod = float(m.average_odds_difference())
        theil = float(m.theil_index())
        bg_theil = float(m.between_group_theil_index())

        unpriv = sorted({str(g) for g in np.unique(groups) if g != privileged})
        mechanism = (
            f"aif360.metrics.ClassificationMetric over {len(y_true)} test predictions; "
            f"sensitive attribute '{ds.sensitive_name}' binarised as privileged = "
            f"'{privileged}' (highest training base rate, {rates[privileged]:.3f}) vs "
            f"unprivileged = {', '.join(unpriv)}. disparate_impact={di:.4f}, "
            f"statistical_parity_diff={spd:+.4f}, equal_opportunity_diff={eod:+.4f}, "
            f"average_odds_diff={aod:+.4f}, theil_index={theil:.4f}."
        )

        def rate(cond):
            return float(np.mean(y_pred[cond] == pos_label)) if cond.any() else 0.0
        P, U = a == 1, a == 0
        rows = [
            ["privileged", str(privileged), int(P.sum()), round(rate(P), 4),
             round(rate(P & (y_true == pos_label)), 4), round(rate(P & (y_true != pos_label)), 4)],
            ["unprivileged", ", ".join(unpriv), int(U.sum()), round(rate(U), 4),
             round(rate(U & (y_true == pos_label)), 4), round(rate(U & (y_true != pos_label)), 4)],
        ]
        trace = make_trace(
            steps=[
                f"disparate_impact = sel(U) / sel(P) = {rows[1][3]:.4f} / {rows[0][3]:.4f} = {di:.4f}"
                + ("   ✗ fails the 80% rule" if di < 0.8 else "   ✓ passes the 80% rule"),
                f"statistical_parity_diff = {rows[1][3]:.4f} − {rows[0][3]:.4f} = {spd:+.4f}",
                f"equal_opportunity_diff = TPR(U) − TPR(P) = {rows[1][4]:.4f} − {rows[0][4]:.4f} = {eod:+.4f}",
                f"average_odds_diff = {aod:+.4f}",
                f"theil_index = {theil:.4f}  (of which between-group: {bg_theil:.6f})",
            ],
            tables=[
                {"title": "Privileged vs unprivileged, as AIF360 sees them",
                 "columns": ["role", "values", "n", "selection_rate", "TPR", "FPR"],
                 "rows": rows},
                {"title": "Training base rate per sensitive value (how 'privileged' was chosen)",
                 "columns": ["value", "P(y=1) in train"],
                 "rows": [[str(g), round(r, 4)] for g, r in sorted(rates.items(), key=lambda kv: -kv[1])]},
            ],
        )
        details = {"privileged": str(privileged), "unprivileged": unpriv,
                   "disparate_impact": round(di, 4), "trace": trace}
        T = InspectionTarget.PREDICTIONS
        return [
            EvalResult(self.name, "fairness", T, "disparate_impact_gap", abs(1.0 - di), False,
                       mechanism, details),
            EvalResult(self.name, "fairness", T, "abs_equal_opportunity_diff", abs(eod), False,
                       mechanism, details),
            EvalResult(self.name, "fairness", T, "abs_average_odds_diff", abs(aod), False,
                       mechanism, details),
            EvalResult(self.name, "inclusiveness", T, "theil_index", theil, False,
                       mechanism, details),
        ]


register(AIF360Evaluator())
