"""Drift/outlier evaluator via Alibi Detect
(https://github.com/SeldonIO/alibi-detect), HANDOFF.md's third Phase-3 item:
"Add drift/outlier checks (Alibi Detect) for deployment-time behaviour."

The train/test split every other evaluator uses is, by construction, the
SAME distribution (a random split) -- there is no real drift to detect
there. So this evaluator manufactures a synthetic covariate shift (a
per-feature mean shift, scaled by the training distribution's own spread,
simulating "deployment population differs systematically from training
population") and checks two independent things against it:

  1. Does Alibi Detect's own statistical drift test (TabularDrift, a
     per-feature Kolmogorov-Smirnov test with Bonferroni correction across
     features) actually catch the injected shift, and correctly NOT flag
     the unshifted test set as drifted? This is a real, off-the-shelf
     deployment-monitoring tool actually running -- not a hand-rolled
     approximation of one.
  2. How much does the model's own performance degrade once that shift is
     present? A model that degrades gracefully is safer to deploy behind
     drift monitoring than one that silently collapses.

Both angles feed the `safety` principle, alongside RobustnessEvaluator's
random-noise check and ARTRobustnessEvaluator's adversarial search -- three
different, complementary "what happens when the input isn't exactly what
training saw" questions, corroborating each other rather than duplicating.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn import metrics as skm

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


def _has_alibi_detect() -> bool:
    try:
        import alibi_detect  # noqa: F401
        return True
    except Exception:
        return False


class DriftEvaluator(Evaluator):
    name = "drift_detection"
    principle_ids = ["safety"]
    mechanism = (
        "Manufactures a synthetic covariate shift by adding shift_scale x "
        "(per-feature training std) to every feature of the test set "
        "(simulating a deployment population that has systematically "
        "drifted from the training population), then runs Alibi Detect's "
        "TabularDrift (per-feature Kolmogorov-Smirnov test, Bonferroni-"
        "corrected across features, reference = training data) against both "
        "the shifted and the original (unshifted) test set. Scores the "
        "detector's own reliability on this dataset (did it correctly flag "
        "the shifted set as drifted AND correctly leave the unshifted set "
        "alone?) and separately measures how much the model's own held-out "
        "performance metric degrades once that shift is present."
    )
    install_hint = "pip install alibi-detect   (needs Python 3.12 or earlier)"
    limits = ('A manufactured, uniform shift -- real-world drift is rarely that tidy; Kolmogorov-Smirnov per feature misses shifts in feature relationships.')
    formula = (
        "x_shift = x_test + s ⊙ shift_scale · std_train,   s_j ∈ {−1, +1}\n"
        "per feature j: KS statistic D_j = sup_t |F_ref,j(t) − F_test,j(t)|  → p_j\n"
        "is_drift = min_j p_j < p_val / d          (Bonferroni over d features)\n"
        "drift_detection_score = (1[is_drift(shifted)] + 1[¬is_drift(original)]) / 2\n"
        "performance_retention = clip(1 − (perf_orig − perf_shift)/|perf_orig|, 0, 1)"
    )

    def __init__(self, shift_scale: float = 1.0, p_val: float = 0.05, seed: int = 0):
        self.shift_scale = shift_scale
        self.p_val = p_val
        self.seed = seed

    def available(self) -> bool:
        return _has_alibi_detect()

    def applies_to(self, ctx: EvalContext) -> bool:
        return ctx.dataset.task in self.supports and _has_alibi_detect()

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        from alibi_detect.cd import TabularDrift

        ds = ctx.dataset
        Xtr = np.asarray(ds.X_train, dtype=float)
        Xte = np.asarray(ds.X_test, dtype=float)
        y_true = np.asarray(ds.y_test)

        stds = Xtr.std(axis=0)
        stds[stds == 0] = 1.0
        rng = np.random.default_rng(self.seed)
        shift = rng.choice([-1.0, 1.0], size=Xtr.shape[1]) * self.shift_scale * stds
        Xte_shift = Xte + shift

        cd = TabularDrift(x_ref=Xtr, p_val=self.p_val)
        res_orig = cd.predict(Xte)
        res_shift = cd.predict(Xte_shift)
        drift_orig = bool(res_orig["data"]["is_drift"])
        drift_shift = bool(res_shift["data"]["is_drift"])
        # 1.0 = correctly flagged shifted set AND correctly left original alone;
        # 0.5 = got one of the two right; 0.0 = got both wrong.
        detection_score = float(((drift_shift is True) + (drift_orig is False)) / 2.0)

        y_pred_shift = ctx.model.predict(Xte_shift)
        if ds.task == TaskType.CLASSIFICATION:
            metric_name = "accuracy"
            perf_orig = skm.accuracy_score(y_true, ctx.y_pred)
            perf_shift = skm.accuracy_score(y_true, y_pred_shift)
        else:
            metric_name = "r2"
            perf_orig = skm.r2_score(y_true, ctx.y_pred)
            perf_shift = skm.r2_score(y_true, y_pred_shift)
        drop = max(0.0, float(perf_orig) - float(perf_shift))
        retention = float(np.clip(1.0 - drop / (abs(perf_orig) + 1e-9), 0.0, 1.0))

        names = ds.feature_names or [f"f{i}" for i in range(Xtr.shape[1])]
        p_orig = np.atleast_1d(np.asarray(res_orig["data"]["p_val"], dtype=float))
        p_shift = np.atleast_1d(np.asarray(res_shift["data"]["p_val"], dtype=float))
        threshold = self.p_val / max(1, Xtr.shape[1])
        trace = make_trace(
            steps=[f"Bonferroni threshold = {self.p_val} / {Xtr.shape[1]} = {threshold:.5f}",
                   f"is_drift(shifted) = {drift_shift}, is_drift(original) = {drift_orig} "
                   f"→ detection_score = {detection_score:.2f}",
                   f"retention = 1 − ({float(perf_orig):.4f} − {float(perf_shift):.4f}) / "
                   f"{abs(float(perf_orig)):.4f} = {retention:.4f}"],
            tables=[{"title": "Per-feature Kolmogorov–Smirnov p-values (reference = train)",
                     "columns": ["feature", "p (original test)", "p (shifted test)",
                                 "flagged after shift"],
                     "rows": [[str(nm), round(float(a), 5), round(float(b), 5),
                               bool(b < threshold)]
                              for nm, a, b in zip(names, p_orig, p_shift)]}],
        )
        mechanism = (
            f"Shift = {self.shift_scale} x per-feature training std, random sign per feature "
            f"(seed={self.seed}). TabularDrift(x_ref=train, p_val={self.p_val}, "
            f"Bonferroni-corrected over {Xtr.shape[1]} features): "
            f"is_drift(shifted test)={drift_shift}, is_drift(original test)={drift_orig} "
            f"-> detection_score={detection_score:.2f}. Model {metric_name}: "
            f"{perf_orig:.4f} (original) -> {perf_shift:.4f} (shifted) -> "
            f"retention={retention:.4f}."
        )
        details = {
            "shift_scale": self.shift_scale, "p_val": self.p_val,
            "drift_detected_on_shifted": drift_shift, "drift_detected_on_original": drift_orig,
            "metric": metric_name, "perf_original": round(float(perf_orig), 4),
            "perf_shifted": round(float(perf_shift), 4), "trace": trace,
        }
        return [
            EvalResult(self.name, "safety", InspectionTarget.DATA,
                       "drift_detection_score", detection_score, True, mechanism, details),
            EvalResult(self.name, "safety", InspectionTarget.BEHAVIOR,
                       "performance_retention_under_drift", retention, True, mechanism, details),
        ]


register(DriftEvaluator())
