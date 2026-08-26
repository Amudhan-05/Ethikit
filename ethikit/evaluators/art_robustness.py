"""Adversarial robustness evaluator via the Adversarial Robustness Toolbox
(ART, https://github.com/Trusted-AI/adversarial-robustness-toolbox).

HANDOFF.md's Phase 3 plan: "add ART as a robustness evaluator (real
attacks)" to complement the existing RobustnessEvaluator's Gaussian-noise
proxy (ethikit/evaluators/robustness.py), which measures stability under
*random* perturbation but says nothing about a model's worst case under a
perturbation an adversary deliberately searches for.

This evaluator wraps the fitted model as a black-box label oracle (ART's
SklearnClassifier) and runs HopSkipJump (Chen et al. 2019) -- a
decision-based attack that only needs predict(), no gradients or model
internals, matching how an external attacker actually queries a deployed
model. Query/iteration budgets are kept deliberately small (see __init__
defaults): decision-based attacks are query-hungry, and tree ensembles in
particular can take 10-40s per sample at ART's own defaults -- confirmed by
hand-timing HopSkipJump against RandomForestClassifier/GradientBoosting in
this environment. The tightened budget keeps a full run_benchmark() sweep
over the sklearn model zoo in the tens of seconds to a couple of minutes,
at the cost of a somewhat weaker (but still real) attack.

Classification only -- HopSkipJump's decision-boundary search assumes a
discrete predicted label to flip, which doesn't translate to regression.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, register


def _has_art() -> bool:
    try:
        import art  # noqa: F401
        return True
    except Exception:
        return False


class ARTRobustnessEvaluator(Evaluator):
    name = "art_robustness"
    principle_ids = ["safety"]
    supports = [TaskType.CLASSIFICATION]
    mechanism = (
        "Runs a real black-box adversarial evasion attack (HopSkipJump, Chen "
        "et al. 2019, via the Adversarial Robustness Toolbox) against the "
        "fitted model treated as a label-only oracle -- no gradients or "
        "model internals are used, matching how an external attacker would "
        "actually query a deployed model. Wraps the model in "
        "art.estimators.classification.SklearnClassifier, then for a small "
        "sample of correctly-classified test rows searches for the smallest "
        "L2 perturbation that flips the prediction, within a tight "
        "iteration/query budget kept low so a full model-zoo sweep finishes "
        "in reasonable time (see this module's docstring for measured "
        "per-model timings). Reports the fraction of attacked rows the "
        "search could NOT flip within budget (higher = more robust) and, "
        "among the ones it did flip, the median perturbation size needed "
        "(higher = the attacker needed a bigger, more detectable change -- "
        "also more robust). Complements the existing RobustnessEvaluator's "
        "random-noise stability check with a worst-case, adversarially-"
        "searched one."
    )

    def __init__(self, sample_size: int = 15, max_iter: int = 10, max_eval: int = 300,
                init_eval: int = 15, init_size: int = 15, seed: int = 0):
        self.sample_size = sample_size
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.seed = seed

    def available(self) -> bool:
        return _has_art()

    def applies_to(self, ctx: EvalContext) -> bool:
        return ctx.dataset.task in self.supports and _has_art()

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        from art.estimators.classification import SklearnClassifier
        from art.attacks.evasion import HopSkipJump

        ds = ctx.dataset
        Xte = np.asarray(ds.X_test, dtype=float)
        Xtr = np.asarray(ds.X_train, dtype=float)
        n_features = Xte.shape[1]
        base_pred = np.asarray(ctx.y_pred)
        y_true = np.asarray(ds.y_test)

        correct_idx = np.where(base_pred == y_true)[0]
        rng = np.random.default_rng(self.seed)
        n = min(self.sample_size, len(correct_idx))
        sel = rng.choice(correct_idx, size=n, replace=False) if n > 0 else np.array([], dtype=int)

        lo = float(min(Xtr.min(), Xte.min()))
        hi = float(max(Xtr.max(), Xte.max()))

        success_rate = 0.0
        median_norm = None
        n_flipped = 0
        if n > 0:
            clf = SklearnClassifier(model=ctx.model, clip_values=(lo, hi))
            atk = HopSkipJump(classifier=clf, max_iter=self.max_iter, max_eval=self.max_eval,
                              init_eval=self.init_eval, init_size=self.init_size, verbose=False)
            adv = atk.generate(x=Xte[sel])
            adv_pred = ctx.model.predict(adv)
            flipped = adv_pred != base_pred[sel]
            success_rate = float(np.mean(flipped))
            n_flipped = int(np.sum(flipped))
            if n_flipped > 0:
                norms = np.linalg.norm(adv[flipped] - Xte[sel][flipped], axis=1) / np.sqrt(n_features)
                median_norm = float(np.median(norms))

        robustness = 1.0 - success_rate
        mechanism = (
            f"HopSkipJump L2 black-box attack against {n} correctly-classified test rows "
            f"(max_iter={self.max_iter}, max_eval={self.max_eval}, clip_values=({lo:.3f}, {hi:.3f})): "
            f"{success_rate * 100:.1f}% of attacked rows had their prediction flipped within budget "
            f"-> adversarial_robustness = 1 - success_rate = {robustness:.4f}."
        )
        results = [
            EvalResult(self.name, "safety", InspectionTarget.BEHAVIOR,
                       "adversarial_robustness", robustness, True, mechanism,
                       {"n_attacked": int(n), "success_rate": round(success_rate, 4),
                        "n_flipped": n_flipped}),
        ]
        if median_norm is not None:
            pert_mechanism = (
                f"Among the {n_flipped} of {n} attacked rows HopSkipJump successfully flipped, "
                f"median L2 perturbation norm (scaled by 1/sqrt(n_features)={1 / np.sqrt(n_features):.4f}) "
                f"needed = {median_norm:.4f}; larger means a bigger, more detectable change was required "
                f"to fool the model."
            )
            results.append(EvalResult(
                self.name, "safety", InspectionTarget.BEHAVIOR,
                "adversarial_perturbation_size", median_norm, True, pert_mechanism,
                {"n_flipped": n_flipped, "n_attacked": int(n)}))
        return results


register(ARTRobustnessEvaluator())
