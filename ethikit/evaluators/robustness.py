"""Robustness / safety evaluator.

Inspects BEHAVIOR: we jitter the numeric test features with small Gaussian
noise and measure how often predictions stay the same. A robust model keeps
its answers under mild perturbation. This is the lightweight, dependency-free
version; in Claude Code, Giskard's scan (`pip install giskard`) and the
Adversarial Robustness Toolbox add far richer perturbation and attack suites
you can register as additional evaluators.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


class RobustnessEvaluator(Evaluator):
    name = "robustness"
    principle_ids = ["safety"]
    mechanism = (
        "Adds Gaussian noise (sigma x per-feature standard deviation) to every "
        "numeric test feature, repeated several times, and measures the fraction "
        "of predictions that stay the same (classification) or the relative "
        "change in predicted value (regression). Inspects BEHAVIOR under input "
        "perturbation only -- it says nothing about *why* a model is or isn't "
        "stable, and never touches categorical/text features."
    )
    limits = ("Random Gaussian noise only -- a weak test; worst-case (adversarial) inputs are the art_robustness tool's job; numeric features only.")
    formula = (
        "x̃ = x + ε,  ε_j ~ N(0, (σ · std_j)²)   for every feature j\n"
        "agreement_r = mean_i 1[f(x̃_i) = f(x_i)]          (repeat r = 1..R)\n"
        "perturbation_stability = mean_r agreement_r\n"
        "regression: agreement_r = max(0, 1 − mean|f(x̃)−f(x)| / mean|f(x)|)"
    )

    def __init__(self, sigma: float = 0.1, n_repeats: int = 5, seed: int = 0):
        self.sigma = sigma
        self.n_repeats = n_repeats
        self.seed = seed

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        X = np.asarray(ctx.dataset.X_test, dtype=float)
        base = np.asarray(ctx.y_pred)
        stds = X.std(axis=0, keepdims=True)
        stds[stds == 0] = 1.0

        rng = np.random.default_rng(self.seed)
        agreements = []
        for _ in range(self.n_repeats):
            noise = rng.normal(0, self.sigma, size=X.shape) * stds
            pert = ctx.model.predict(X + noise)
            if ctx.dataset.task == TaskType.CLASSIFICATION:
                agreements.append(float(np.mean(pert == base)))
            else:
                denom = np.abs(base).mean() or 1.0
                rel = np.abs(pert - base).mean() / denom
                agreements.append(float(max(0.0, 1.0 - rel)))

        stability = float(np.mean(agreements))
        mechanism = (
            f"Perturbed all {X.shape[1]} numeric features of {len(X)} test rows with "
            f"N(0, {self.sigma}*feature_std) noise, repeated {self.n_repeats}x, and "
            f"measured prediction agreement against the unperturbed baseline each time "
            f"(mean stability = {stability:.4f} across the {self.n_repeats} repeats)."
        )
        return [
            EvalResult(self.name, "safety", InspectionTarget.BEHAVIOR,
                       "perturbation_stability", stability, True, mechanism,
                       {"sigma": self.sigma, "n_repeats": self.n_repeats,
                        "per_repeat_agreement": [round(a, 4) for a in agreements],
                        "trace": make_trace(
                            steps=[f"stability = mean({', '.join(f'{a:.4f}' for a in agreements)})"
                                   f" = {stability:.4f}"],
                            charts=[{"title": f"Prediction agreement per noise repeat (σ={self.sigma})",
                                     "type": "bars",
                                     "labels": [f"r{i + 1}" for i in range(len(agreements))],
                                     "series": [{"name": "agreement",
                                                 "values": [round(a, 4) for a in agreements]}]}],
                        )}),
        ]


register(RobustnessEvaluator())
