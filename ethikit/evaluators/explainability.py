"""Explainability evaluator.

Two things matter for explainability:
  1. Can we get feature attributions at all? (availability)
  2. Is a simple, human-readable surrogate faithful to the model? (fidelity)

We report a *surrogate fidelity* score: how well a shallow decision tree can
reproduce the model's own predictions. High fidelity means the model's
behaviour can be summarised by simple rules a human can read. Prefers SHAP
for the attribution details; falls back to permutation importance.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, register


def _has_shap():
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


class ExplainabilityEvaluator(Evaluator):
    name = "explainability"
    principle_ids = ["explainability", "transparency"]

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        X_test = ctx.dataset.X_test
        y_pred = np.asarray(ctx.y_pred)
        fidelity, top = self._surrogate_fidelity(ctx, X_test, y_pred)
        backend = "permutation_importance"

        if _has_shap():
            try:
                shap_top = self._shap_top_features(ctx, X_test)
                if shap_top:
                    top = shap_top
                    backend = "shap"
            except Exception:
                pass  # keep the surrogate-derived ranking if SHAP fails on this model

        details = {"top_features": top, "attribution_backend": backend}
        return [
            EvalResult(self.name, "explainability", InspectionTarget.MODEL_INTERNALS,
                       "surrogate_fidelity", fidelity, True, details),
        ]

    def _shap_top_features(self, ctx, X_test, sample_size: int = 25, background_size: int = 15):
        """Mean |SHAP value| per feature, ranked. Model-agnostic via shap.Explainer's
        auto backend (TreeExplainer for trees, permutation-based otherwise) so it
        works for every model in the zoo without per-model wiring."""
        import shap

        model = ctx.model
        Xt = np.asarray(X_test, dtype=float)
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(len(Xt), size=min(background_size, len(Xt)), replace=False)
        sample_idx = rng.choice(len(Xt), size=min(sample_size, len(Xt)), replace=False)
        background, sample = Xt[bg_idx], Xt[sample_idx]

        predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
        # Force the permutation algorithm with the minimum viable evaluation budget
        # instead of "auto" (which picks Exact -- 2^n_features model calls -- for small
        # feature counts, punishingly slow for models with an expensive predict, e.g.
        # SVC(probability=True)). Approximate attributions are fine for a top-5 ranking.
        n_features = background.shape[1]
        max_evals = max(2 * n_features + 1, 15)
        explainer = shap.Explainer(predict_fn, background, algorithm="permutation")
        sv = explainer(sample, max_evals=max_evals, silent=True)

        values = np.asarray(sv.values)
        if values.ndim == 3:  # (n_samples, n_features, n_classes) -> positive class
            values = values[:, :, -1]
        mean_abs = np.abs(values).mean(axis=0)

        names = ctx.dataset.feature_names or [f"f{i}" for i in range(Xt.shape[1])]
        order = np.argsort(mean_abs)[::-1][:5]
        return [(names[i], round(float(mean_abs[i]), 4)) for i in order if mean_abs[i] > 0]

    def _surrogate_fidelity(self, ctx, X_test, y_pred):
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.metrics import accuracy_score, r2_score

        Xt = np.asarray(X_test, dtype=float)
        if ctx.dataset.task == TaskType.CLASSIFICATION:
            surrogate = DecisionTreeClassifier(max_depth=3, random_state=0)
            surrogate.fit(Xt, y_pred)
            fidelity = float(accuracy_score(y_pred, surrogate.predict(Xt)))
        else:
            surrogate = DecisionTreeRegressor(max_depth=3, random_state=0)
            surrogate.fit(Xt, y_pred)
            fidelity = float(max(0.0, r2_score(y_pred, surrogate.predict(Xt))))

        names = ctx.dataset.feature_names or [f"f{i}" for i in range(Xt.shape[1])]
        imp = surrogate.feature_importances_
        order = np.argsort(imp)[::-1][:5]
        top = [(names[i], round(float(imp[i]), 3)) for i in order if imp[i] > 0]
        return fidelity, top


register(ExplainabilityEvaluator())
