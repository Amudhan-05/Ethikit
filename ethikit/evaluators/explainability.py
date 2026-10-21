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
from .base import Evaluator, make_trace, register


def _has_shap():
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


class ExplainabilityEvaluator(Evaluator):
    name = "explainability"
    principle_ids = ["explainability", "transparency"]
    mechanism = (
        "Two independent, model-agnostic signals, both treating the model as a "
        "black-box predict()/predict_proba() function -- neither inspects real "
        "internal weights: (1) surrogate fidelity -- a depth-3 decision tree is "
        "trained to mimic the model's own predictions; how accurately that tiny, "
        "human-readable tree reproduces them is the fidelity score. (2) feature "
        "attribution -- when SHAP is installed, mean |SHAP value| per feature from "
        "a permutation-based explainer run over a random sample against a "
        "background sample; falls back to the surrogate tree's own "
        "feature_importances_ if SHAP errors or isn't installed."
    )
    formula = (
        "surrogate T = DecisionTree(max_depth=3) fit on (X_test, f(X_test))\n"
        "surrogate_fidelity = mean_i 1[T(x_i) = f(x_i)]       (regression: max(0, R²(f(x), T(x))))\n"
        "SHAP: φ_j(x) = Σ_{S ⊆ F∖{j}} |S|!(|F|−|S|−1)!/|F|! · [f(S ∪ {j}) − f(S)]\n"
        "attribution_j = mean_i |φ_j(x_i)|     (permutation-sampled approximation)"
    )

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        X_test = ctx.dataset.X_test
        y_pred = np.asarray(ctx.y_pred)
        fidelity, top, rules, agree = self._surrogate_fidelity(ctx, X_test, y_pred)
        backend = "permutation_importance"
        shap_meta = None

        if _has_shap():
            try:
                shap_top, shap_meta = self._shap_top_features(ctx, X_test)
                if shap_top:
                    top = shap_top
                    backend = "shap"
            except Exception:
                pass  # keep the surrogate-derived ranking if SHAP fails on this model

        if backend == "shap":
            mechanism = (
                f"SHAP permutation explainer (max_evals={shap_meta['max_evals']}) over "
                f"{shap_meta['sample_size']} sampled test rows against a background of "
                f"{shap_meta['background_size']} rows, ranked by mean |SHAP value|; "
                f"surrogate fidelity from a depth-3 decision tree trained on "
                f"{len(y_pred)} test predictions."
            )
        else:
            mechanism = (
                f"SHAP unavailable or failed on this model -- fell back to a depth-3 "
                f"decision tree trained to mimic {len(y_pred)} test predictions; "
                f"fidelity = how often the tree agrees with the model, features ranked "
                f"by the tree's own feature_importances_."
            )

        trace = make_trace(
            steps=[f"surrogate_fidelity = rows where tree agrees with model / n = "
                   f"{agree} / {len(y_pred)} = {fidelity:.4f}"],
            charts=[{"title": f"Top feature attributions ({backend})", "type": "bars",
                     "labels": [f for f, _ in top],
                     "series": [{"name": "importance", "values": [v for _, v in top]}]}],
            text=[{"title": "The depth-3 surrogate tree a human would read (sklearn export_text)",
                   "body": rules}],
        )
        details = {"top_features": top, "attribution_backend": backend, "trace": trace}
        return [
            EvalResult(self.name, "explainability", InspectionTarget.MODEL_INTERNALS,
                       "surrogate_fidelity", fidelity, True, mechanism, details),
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
        top = [(names[i], round(float(mean_abs[i]), 4)) for i in order if mean_abs[i] > 0]
        meta = {"sample_size": len(sample), "background_size": len(background),
                "max_evals": max_evals}
        return top, meta

    def _surrogate_fidelity(self, ctx, X_test, y_pred):
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from sklearn.metrics import accuracy_score, r2_score

        Xt = np.asarray(X_test, dtype=float)
        if ctx.dataset.task == TaskType.CLASSIFICATION:
            surrogate = DecisionTreeClassifier(max_depth=3, random_state=0)
            surrogate.fit(Xt, y_pred)
            fidelity = float(accuracy_score(y_pred, surrogate.predict(Xt)))
            agree = int(np.sum(surrogate.predict(Xt) == y_pred))
        else:
            surrogate = DecisionTreeRegressor(max_depth=3, random_state=0)
            surrogate.fit(Xt, y_pred)
            fidelity = float(max(0.0, r2_score(y_pred, surrogate.predict(Xt))))
            agree = int(np.sum(np.isclose(surrogate.predict(Xt), y_pred)))

        names = ctx.dataset.feature_names or [f"f{i}" for i in range(Xt.shape[1])]
        imp = surrogate.feature_importances_
        order = np.argsort(imp)[::-1][:5]
        top = [(names[i], round(float(imp[i]), 3)) for i in order if imp[i] > 0]
        from sklearn.tree import export_text
        rules = export_text(surrogate, feature_names=[str(n) for n in names], decimals=3)
        return fidelity, top, rules, agree


register(ExplainabilityEvaluator())
