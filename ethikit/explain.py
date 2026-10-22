"""Local explanation: why did the model decide *this* row the way it did?

Occlusion, model-agnostic and deterministic: for each feature, replace this row's value with the
training-set mean and see how far the model's score (probability of the positive class, or the
prediction for regression) moves. A large move means the model leaned on that feature for this
person. It complements the global tools (permutation importance, SHAP, the surrogate tree).

Limits: one feature at a time (interactions are missed), "the mean" is an arbitrary reference
point, and one-hot columns of the same category are occluded separately.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .core.types import TaskType


def _score(model, X, task):
    if task == TaskType.CLASSIFICATION and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, -1]
    return np.asarray(model.predict(X), dtype=float)


def explain_row(model, dataset, row: int, top: int = 15) -> Dict[str, Any]:
    X = np.asarray(dataset.X_test, dtype=float)
    if not 0 <= row < len(X):
        raise IndexError(f"row must be between 0 and {len(X) - 1}")
    names = dataset.feature_names or [f"f{i}" for i in range(X.shape[1])]
    base = np.asarray(dataset.X_train, dtype=float).mean(axis=0)
    x = X[row:row + 1]
    s0 = float(_score(model, x, dataset.task)[0])
    probes = np.repeat(x, X.shape[1], axis=0)
    probes[np.arange(X.shape[1]), np.arange(X.shape[1])] = base
    contrib = s0 - _score(model, probes, dataset.task)
    order = np.argsort(-np.abs(contrib))[:top]
    return {
        "row": row, "score": round(s0, 4),
        "score_meaning": ("probability of the positive class" if dataset.task == TaskType.CLASSIFICATION
                          and hasattr(model, "predict_proba") else "prediction"),
        "prediction": np.asarray(model.predict(x)).tolist()[0],
        "actual": np.asarray(dataset.y_test)[row].item() if hasattr(np.asarray(dataset.y_test)[row], "item") else dataset.y_test[row],
        "group": None if dataset.sensitive_test is None else str(np.asarray(dataset.sensitive_test)[row]),
        "contributions": [{"feature": names[i], "contribution": round(float(contrib[i]), 4)} for i in order],
        "method": "occlusion: each feature in turn set to its training mean; contribution = score change",
        "limits": "one feature at a time (interactions missed); the training mean is an arbitrary reference",
    }
