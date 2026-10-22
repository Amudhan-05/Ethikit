"""Mitigation what-if: "what would it cost to close this fairness gap?"

This is a COUNTERFACTUAL, not part of the audit. The audit measures the models you gave it;
this trains a *different*, fairness-constrained version of one of them with fairlearn and puts
the before/after numbers side by side, so the trade-off is visible before anyone decides to
mitigate. Nothing here is ever written into the audit's scores, Pareto front or model cards.

Methods (fairlearn):
  exponentiated_gradient -- in-processing: retrains the estimator under a fairness constraint
                            (Agarwal et al. 2018). The estimator must accept sample_weight.
  threshold_optimizer    -- post-processing: picks group-specific decision thresholds on the
                            existing model's scores (Hardt et al. 2016). Note: it needs the
                            sensitive attribute AT PREDICTION TIME, i.e. it treats groups
                            differently on purpose -- which may be unlawful in some settings.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

METHODS = ("exponentiated_gradient", "threshold_optimizer")
CONSTRAINTS = ("demographic_parity", "equalized_odds")


class MitigationError(ValueError):
    pass


def _metrics(y, yp, s) -> Dict[str, float]:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from sklearn.metrics import accuracy_score, f1_score
    return {"accuracy": round(float(accuracy_score(y, yp)), 4),
            "f1": round(float(f1_score(y, yp, average="macro")), 4),
            "demographic_parity_diff": round(float(demographic_parity_difference(y, yp, sensitive_features=s)), 4),
            "equalized_odds_diff": round(float(equalized_odds_difference(y, yp, sensitive_features=s)), 4)}


def mitigate(dataset, factory: Callable, method: str = "exponentiated_gradient",
             constraint: str = "demographic_parity", seed: int = 0) -> Dict[str, Any]:
    try:
        import fairlearn  # noqa: F401
    except ImportError as exc:
        raise MitigationError("the what-if needs fairlearn: pip install fairlearn") from exc
    from fairlearn.postprocessing import ThresholdOptimizer
    from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient

    if method not in METHODS:
        raise MitigationError(f"method must be one of {METHODS}")
    if constraint not in CONSTRAINTS:
        raise MitigationError(f"constraint must be one of {CONSTRAINTS}")
    if dataset.sensitive_train is None:
        raise MitigationError("this dataset has no sensitive attribute to mitigate on")
    if len(np.unique(dataset.y_train)) != 2:
        raise MitigationError("the what-if supports binary classification only")

    Xtr, Xte = np.asarray(dataset.X_train, dtype=float), np.asarray(dataset.X_test, dtype=float)
    ytr, yte = np.asarray(dataset.y_train), np.asarray(dataset.y_test)
    str_, ste = np.asarray(dataset.sensitive_train).astype(str), np.asarray(dataset.sensitive_test).astype(str)

    base = factory().fit(Xtr, ytr)
    before = _metrics(yte, base.predict(Xte), ste)

    if method == "exponentiated_gradient":
        import inspect
        if "sample_weight" not in inspect.signature(factory().fit).parameters:
            raise MitigationError(f"{type(factory()).__name__} can't be retrained with weights "
                                  f"(no sample_weight); use method='threshold_optimizer' instead")
        cons = DemographicParity() if constraint == "demographic_parity" else EqualizedOdds()
        mit = ExponentiatedGradient(factory(), constraints=cons)
        mit.fit(Xtr, ytr, sensitive_features=str_)
        rng_state = np.random.get_state()
        np.random.seed(seed)               # ExponentiatedGradient predicts by sampling its mixture
        try:
            after_pred = mit.predict(Xte, random_state=seed)
        finally:
            np.random.set_state(rng_state)
        uses_attr = "training only"
    else:
        predict_method = "predict_proba" if hasattr(base, "predict_proba") else "predict"
        mit = ThresholdOptimizer(estimator=base, constraints=constraint, prefit=True,
                                 predict_method=predict_method)
        mit.fit(Xtr, ytr, sensitive_features=str_)
        after_pred = mit.predict(Xte, sensitive_features=ste, random_state=seed)
        uses_attr = "at prediction time (group-specific thresholds)"
    after = _metrics(yte, after_pred, ste)
    delta = {k: round(after[k] - before[k], 4) for k in before}
    return {
        "label": "counterfactual -- not part of the audit",
        "method": method, "constraint": constraint, "before": before, "after": after, "delta": delta,
        "uses_sensitive_attribute": uses_attr,
        "summary": (f"{constraint.replace('_', ' ')} via {method.replace('_', ' ')}: "
                    f"parity gap {before['demographic_parity_diff']:.3f} -> {after['demographic_parity_diff']:.3f}, "
                    f"odds gap {before['equalized_odds_diff']:.3f} -> {after['equalized_odds_diff']:.3f}, "
                    f"accuracy {before['accuracy']:.3f} -> {after['accuracy']:.3f}."),
        "caveats": [
            "A different model from the one audited: re-run the full audit on it before relying on it.",
            "Closing one fairness gap can widen another (they cannot in general all be zero).",
        ] + (["Uses the sensitive attribute at prediction time -- check whether that is lawful where "
              "the model will be used."] if method == "threshold_optimizer" else []),
    }
