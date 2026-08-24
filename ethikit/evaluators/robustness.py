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
from .base import Evaluator, register


class RobustnessEvaluator(Evaluator):
    name = "robustness"
    principle_ids = ["safety"]

    def __init__(self, sigma: float = 0.1, n_repeats: int = 5, seed: int = 0):
        self.sigma = sigma
        self.n_repeats = n_repeats
        self.rng = np.random.default_rng(seed)

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        X = np.asarray(ctx.dataset.X_test, dtype=float)
        base = np.asarray(ctx.y_pred)
        stds = X.std(axis=0, keepdims=True)
        stds[stds == 0] = 1.0

        agreements = []
        for _ in range(self.n_repeats):
            noise = self.rng.normal(0, self.sigma, size=X.shape) * stds
            pert = ctx.model.predict(X + noise)
            if ctx.dataset.task == TaskType.CLASSIFICATION:
                agreements.append(float(np.mean(pert == base)))
            else:
                denom = np.abs(base).mean() or 1.0
                rel = np.abs(pert - base).mean() / denom
                agreements.append(float(max(0.0, 1.0 - rel)))

        stability = float(np.mean(agreements))
        return [
            EvalResult(self.name, "safety", InspectionTarget.BEHAVIOR,
                       "perturbation_stability", stability, True,
                       {"sigma": self.sigma, "n_repeats": self.n_repeats}),
        ]


register(RobustnessEvaluator())
