"""Attribute-inference evaluator: a second privacy attack, alongside membership inference.

Membership inference asks "was this person in the training data?". Attribute inference asks
"does the model's output help an attacker recover a person's *sensitive attribute*?" (Yeom et
al. 2018 analyse both). We measure the model's *own contribution*: an attacker who knows the
other features can already guess the attribute to some degree (that is the data's proxy
problem, reported by the data profile); the privacy question is how much better they do once
they also see the model's output.

  gain = AUC(attacker using features + model output) − AUC(attacker using features only)

Both attackers are the same small logistic regression, scored out-of-fold, so the difference is
the information the model's predictions add.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


class AttributeInferenceEvaluator(Evaluator):
    name = "attribute_inference"
    principle_ids = ["privacy"]
    mechanism = (
        "Two attackers try to recover the sensitive attribute of held-out rows: one sees only the "
        "non-sensitive features, the other sees the same features plus the model's output "
        "(probabilities, or predictions). Both are logistic regressions scored by k-fold "
        "out-of-fold AUC (the largest group vs the rest). The gain in AUC is the leakage the model "
        "itself adds. Inspects BEHAVIOR (what the outputs reveal)."
    )
    formula = (
        "attacker A₀: features (without the sensitive columns) → P(A = largest group)\n"
        "attacker A₁: features + model output → P(A = largest group)\n"
        "attribute_inference_gain = max(0, AUC(A₁) − AUC(A₀))      (out-of-fold)"
    )
    limits = ("Black-box, test-set rows only; a stronger attacker (more features, more queries, "
              "auxiliary data) could recover more; it does not measure leakage of other attributes.")

    def __init__(self, n_splits: int = 5, seed: int = 0, min_rows: int = 60):
        self.n_splits = n_splits
        self.seed = seed
        self.min_rows = min_rows

    def applies_to(self, ctx: EvalContext) -> bool:
        s = ctx.dataset.sensitive_test
        return s is not None and len(np.unique(np.asarray(s).astype(str))) >= 2 and len(s) >= self.min_rows

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        ds = ctx.dataset
        names = ds.feature_names or [f"f{i}" for i in range(np.asarray(ds.X_test).shape[1])]
        sname = (ds.sensitive_name or "").lower()
        keep = [i for i, n in enumerate(names) if not (sname and (n.lower() == sname or n.lower().startswith(sname + "=")))]
        X = np.asarray(ds.X_test, dtype=float)[:, keep]
        s = np.asarray(ds.sensitive_test).astype(str)
        vals, counts = np.unique(s, return_counts=True)
        ref = vals[np.argmax(counts)]
        t = (s == ref).astype(int)
        out = np.asarray(ctx.y_proba) if ctx.y_proba is not None else np.asarray(ctx.y_pred, dtype=float).reshape(-1, 1)
        out = out.reshape(len(t), -1)
        k = max(2, min(self.n_splits, int(np.bincount(t).min())))
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.seed)

        def auc(features):
            if features.shape[1] == 0:
                return 0.5
            p = cross_val_predict(LogisticRegression(max_iter=1000), features, t, cv=cv, method="predict_proba")[:, 1]
            return float(roc_auc_score(t, p))
        base = auc(X)
        with_out = auc(np.column_stack([X, out]))
        gain = max(0.0, with_out - base)
        mechanism = (f"Recovering '{ds.sensitive_name}' ({ref} vs rest) on {len(t)} test rows, {k}-fold out-of-fold: "
                     f"AUC {base:.4f} from {len(keep)} features alone, {with_out:.4f} with the model's output too "
                     f"-> gain {gain:.4f}.")
        trace = make_trace(
            steps=[f"AUC(features only) = {base:.4f}", f"AUC(features + model output) = {with_out:.4f}",
                   f"attribute_inference_gain = max(0, {with_out:.4f} − {base:.4f}) = {gain:.4f}"],
            charts=[{"title": "Attacker AUC without vs with the model's output", "type": "bars",
                     "labels": ["features only", "features + output"],
                     "series": [{"name": "AUC", "values": [round(base, 4), round(with_out, 4)]}]}],
        )
        return [EvalResult(self.name, "privacy", InspectionTarget.BEHAVIOR, "attribute_inference_gain",
                           gain, False, mechanism, {"trace": trace, "auc_base": base, "auc_with_output": with_out})]


register(AttributeInferenceEvaluator())
