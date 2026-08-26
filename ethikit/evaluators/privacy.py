"""Privacy evaluator (Phase 3: real membership-inference attack).

HANDOFF.md's Phase 3 plan was to swap the Phase-1 train/test-gap proxy for a
real membership-inference attack via TensorFlow Privacy. TensorFlow Privacy's
latest release (0.8.11, effectively unmaintained since ~2023) hard-pins
`pandas~=1.4`, which has no Python 3.12 wheel and fails to build from source
-- confirmed by hand in this environment, the same class of blocker already
hit AIF360 in Phase 1. Rather than block on that, this module implements the
actual attack algorithm directly against scikit-learn: TensorFlow Privacy's
own membership-inference module is, under the hood, exactly this -- extract
a per-sample "how well does the model fit this point" feature (confidence in
the true class, or loss) for known members and known non-members, then fit a
simple attack classifier to distinguish them. There is no TF-specific magic
being skipped; this is the Yeom et al. (2018) confidence/loss-threshold
attack, the same baseline Shokri et al. (2017)'s shadow-model attack reduces
to once you already have the target model's own labelled train/test split
(no separate shadow models needed when, as here, ground-truth membership
labels for the evaluation set are already known).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, register

_EPS = 1e-12


def _class_index_map(model, fallback_labels: np.ndarray) -> Dict[object, int]:
    """model.classes_ gives predict_proba's column order; fall back to the
    sorted unique training labels if the estimator doesn't expose it."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.unique(fallback_labels)
    return {c: i for i, c in enumerate(classes)}


def _classification_features(y_true: np.ndarray, y_pred: np.ndarray,
                             proba: Optional[np.ndarray],
                             cls_map: Dict[object, int]) -> tuple:
    """Per-sample attack feature(s). Prefers confidence-in-true-class + loss
    (2 features) when predict_proba is available; falls back to the
    label-only correctness indicator (Yeom et al.'s weaker baseline attack)
    when it isn't. Returns (features, description)."""
    if proba is not None:
        idx = np.array([cls_map[v] for v in y_true])
        conf_true = proba[np.arange(len(y_true)), idx]
        loss = -np.log(conf_true + _EPS)
        return np.column_stack([conf_true, loss]), "confidence-in-true-class + log-loss"
    correct = (np.asarray(y_pred) == np.asarray(y_true)).astype(float)
    return correct.reshape(-1, 1), "label-only correctness indicator (no predict_proba)"


def _regression_features(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    sq = err ** 2
    return np.column_stack([-sq, -np.abs(err)]), "negative squared error + negative absolute error"


class MembershipInferenceEvaluator(Evaluator):
    name = "membership_inference"
    principle_ids = ["privacy"]
    mechanism = (
        "Black-box membership-inference attack (Yeom et al. 2018 confidence/"
        "loss-threshold style -- the same feature-based approach TensorFlow "
        "Privacy's own attack module implements, run directly via "
        "scikit-learn since tensorflow-privacy has no Python-3.12-compatible "
        "release). Extracts a per-sample 'how well does the model fit this "
        "point' feature for the model's own training rows (known members) "
        "and held-out test rows (known non-members) -- confidence in the "
        "true class + log-loss for classification when predict_proba is "
        "available (else the weaker label-only correctness indicator), or "
        "negative squared/absolute error for regression. Balances member "
        "and non-member counts by subsampling, then fits a LogisticRegression "
        "attack classifier and scores it via k-fold out-of-fold prediction "
        "(so the attack's own accuracy is never inflated by testing on rows "
        "it trained on). Reports attack AUC and attack advantage "
        "(2*(accuracy-0.5), clipped at 0) -- 0 means the attack is no better "
        "than guessing from the class balance alone (no detectable "
        "membership leakage from this feature); higher means the model's "
        "outputs let an attacker tell training rows apart from held-out rows."
    )

    def __init__(self, min_group_size: int = 20, max_splits: int = 5, seed: int = 0):
        self.min_group_size = min_group_size
        self.max_splits = max_splits
        self.seed = seed

    def applies_to(self, ctx: EvalContext) -> bool:
        if ctx.dataset.task not in self.supports or ctx.y_pred_train is None:
            return False
        return min(len(ctx.dataset.y_train), len(ctx.dataset.y_test)) >= self.min_group_size

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        ds = ctx.dataset
        if ds.task == TaskType.CLASSIFICATION:
            cls_map = _class_index_map(ctx.model, ds.y_train)
            feat_train, desc = _classification_features(
                ds.y_train, ctx.y_pred_train, ctx.y_proba_train, cls_map)
            feat_test, _ = _classification_features(
                ds.y_test, ctx.y_pred, ctx.y_proba, cls_map)
        else:
            feat_train, desc = _regression_features(ds.y_train, ctx.y_pred_train)
            feat_test, _ = _regression_features(ds.y_test, ctx.y_pred)

        rng = np.random.default_rng(self.seed)
        n = min(len(feat_train), len(feat_test))
        idx_train = rng.choice(len(feat_train), size=n, replace=False)
        idx_test = rng.choice(len(feat_test), size=n, replace=False)

        X_attack = np.vstack([feat_train[idx_train], feat_test[idx_test]])
        y_attack = np.concatenate([np.ones(n), np.zeros(n)])

        n_splits = max(2, min(self.max_splits, n))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        attack = LogisticRegression(max_iter=1000)
        oof_proba = cross_val_predict(attack, X_attack, y_attack, cv=cv,
                                      method="predict_proba")[:, 1]

        attack_auc = float(roc_auc_score(y_attack, oof_proba))
        attack_acc = float(accuracy_score(y_attack, (oof_proba >= 0.5).astype(int)))
        advantage = float(max(0.0, 2.0 * (attack_acc - 0.5)))

        mechanism = (
            f"Attack features: {desc}. {n} members (train rows) vs {n} "
            f"non-members (test rows, balanced by subsampling from "
            f"{len(feat_train)}/{len(feat_test)} available), "
            f"{n_splits}-fold cross-validated LogisticRegression attack -> "
            f"out-of-fold attack AUC={attack_auc:.4f}, "
            f"attack accuracy={attack_acc:.4f}, advantage={advantage:.4f}."
        )
        details = {
            "n_members_used": n, "feature_set": desc, "n_splits": n_splits,
            "attack_accuracy": round(attack_acc, 4),
        }
        return [
            EvalResult(self.name, "privacy", InspectionTarget.BEHAVIOR,
                       "membership_inference_advantage", advantage, False,
                       mechanism, details),
            EvalResult(self.name, "privacy", InspectionTarget.BEHAVIOR,
                       "membership_inference_auc", attack_auc, False,
                       mechanism, details),
        ]


register(MembershipInferenceEvaluator())
