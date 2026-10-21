"""Calibration evaluator.

When a model says "70% likely", is it right about 70% of the time? Calibration matters for
safety (decisions are made on the probability, not just the label) and for fairness: if a score
of 0.7 means 70% for one group but 50% for another, the same threshold treats them differently
("calibration within groups", Kleinberg et al. 2016 -- one of the criteria that cannot in general
hold together with equal error rates).

Expected calibration error (ECE): bin the predicted probability of the positive class into
equal-width bins, and average |mean predicted - observed positive rate| weighted by bin size.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


def _ece(p: np.ndarray, y: np.ndarray, bins: int):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        rows.append([f"{edges[b]:.1f}-{edges[b + 1]:.1f}", int(m.sum()), round(conf, 4), round(acc, 4)])
    return float(ece), rows


class CalibrationEvaluator(Evaluator):
    name = "calibration"
    principle_ids = ["safety", "fairness"]
    supports = [TaskType.CLASSIFICATION]
    mechanism = (
        "Bins the model's predicted probability of the positive class on the test set into "
        "equal-width bins and compares, in each bin, the average predicted probability with the "
        "share of rows that were actually positive (a reliability diagram). Expected calibration "
        "error is the size-weighted average gap (-> safety). The same computation per sensitive "
        "group gives the spread of calibration between groups (-> fairness: the same score should "
        "mean the same risk for everyone). Inspects PREDICTIONS (probabilities) only."
    )
    formula = (
        "bins B_1..B_k over p = P(ŷ=1 | x)\n"
        "ECE = Σ_b (|B_b| / n) · | mean_{i∈B_b} p_i − mean_{i∈B_b} y_i |\n"
        "calibration_gap_between_groups = max_g ECE(g) − min_g ECE(g)"
    )
    limits = ("Needs predict_proba (skipped otherwise). With few rows per bin the estimate is noisy; "
              "binary targets only. Good calibration says nothing about accuracy or about which "
              "errors a model makes.")

    def __init__(self, bins: int = 10, min_group_rows: int = 30):
        self.bins = bins
        self.min_group_rows = min_group_rows

    def applies_to(self, ctx: EvalContext) -> bool:
        return (ctx.dataset.task in self.supports and ctx.y_proba is not None
                and np.asarray(ctx.y_proba).ndim == 2 and np.asarray(ctx.y_proba).shape[1] == 2)

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        p = np.asarray(ctx.y_proba)[:, 1]
        classes = list(getattr(ctx.model, "classes_", [0, 1]))
        y = (np.asarray(ctx.dataset.y_test) == classes[-1]).astype(float)
        ece, rows = _ece(p, y, self.bins)
        group_rows, gaps = [], {}
        if ctx.dataset.sensitive_test is not None:
            g = np.asarray(ctx.dataset.sensitive_test).astype(str)
            for grp in sorted(set(g)):
                m = g == grp
                if m.sum() >= self.min_group_rows:
                    e, _ = _ece(p[m], y[m], self.bins)
                    gaps[grp] = e
                    group_rows.append([grp, int(m.sum()), round(e, 4)])
        gap = (max(gaps.values()) - min(gaps.values())) if len(gaps) >= 2 else None
        mechanism = (f"{self.bins} equal-width probability bins over {len(p)} test rows -> "
                     f"ECE {ece:.4f}" + (f"; per-group ECE {', '.join(f'{k}={v:.3f}' for k, v in gaps.items())}"
                                         f" -> gap {gap:.4f}" if gap is not None else
                                         "; fewer than two groups with enough rows for a per-group gap"))
        trace = make_trace(
            steps=[f"ECE = Σ (bin share × |mean predicted − observed rate|) = {ece:.4f}"]
                  + ([f"calibration gap = {max(gaps.values()):.4f} − {min(gaps.values()):.4f} = {gap:.4f}"]
                     if gap is not None else []),
            tables=[{"title": "Reliability table (perfect calibration: predicted = observed)",
                     "columns": ["probability bin", "rows", "mean predicted", "observed positive rate"],
                     "rows": rows}]
                   + ([{"title": "ECE per sensitive group", "columns": ["group", "rows", "ECE"],
                        "rows": group_rows}] if group_rows else []),
            charts=[{"title": "Reliability: predicted vs observed per bin", "type": "bars",
                     "labels": [r[0] for r in rows],
                     "series": [{"name": "mean predicted", "values": [r[2] for r in rows]},
                                {"name": "observed", "values": [r[3] for r in rows]}]}],
        )
        details = {"bins": self.bins, "trace": trace}
        out = [EvalResult(self.name, "safety", InspectionTarget.PREDICTIONS,
                          "expected_calibration_error", ece, False, mechanism, details)]
        if gap is not None:
            out.append(EvalResult(self.name, "fairness", InspectionTarget.PREDICTIONS,
                                  "calibration_gap_between_groups", float(gap), False, mechanism, details))
        return out


register(CalibrationEvaluator())
