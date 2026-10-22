"""Integrated-gradients attributions (Captum) for PyTorch models.

For neural networks, gradient-based attributions read the model's actual internals instead of
probing it as a black box. Integrated gradients (Sundararajan et al. 2017) integrates the
gradient of the positive-class probability along a straight path from a baseline (the training
mean) to the input; the attributions then satisfy *completeness*: they add up to
f(x) − f(baseline). We report how well that holds in practice (a check on the explanation
itself) and how concentrated the model's attributions are.

Applies only to models backed by a torch.nn.Module (ethikit's TorchMLPClassifier, or a
pre-trained PyTorch model loaded with --model-file x.pt).
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register


def _has_captum() -> bool:
    try:
        import captum  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _module(model):
    return getattr(model, "module_", None)


class CaptumIGEvaluator(Evaluator):
    name = "captum_ig"
    principle_ids = ["explainability"]
    supports = [TaskType.CLASSIFICATION]
    install_hint = "pip install torch captum"
    mechanism = (
        "Captum IntegratedGradients on the PyTorch network itself (not a black-box probe): for a "
        "sample of test rows, the gradient of the positive-class probability is integrated along "
        "the straight path from the training-mean baseline to each row. Reports the completeness "
        "error -- how far the attributions' sum is from f(x) − f(baseline), which should be ~0 -- "
        "and the share of mean |attribution| carried by the top three features."
    )
    formula = (
        "IG_j(x) = (x_j − x'_j) · ∫₀¹ ∂f(x' + α(x − x')) / ∂x_j dα     (x' = training mean)\n"
        "completeness: Σ_j IG_j(x) = f(x) − f(x')\n"
        "ig_completeness_error = mean_x |Σ_j IG_j(x) − (f(x) − f(x'))|\n"
        "ig_concentration = top-3 share of mean_x |IG_j(x)|"
    )
    limits = ("PyTorch models only; depends on the baseline chosen (training mean); a Riemann "
              "approximation of the integral (n_steps); attributions explain the model, not the world.")

    def __init__(self, sample_size: int = 100, n_steps: int = 64, seed: int = 0):
        self.sample_size = sample_size
        self.n_steps = n_steps
        self.seed = seed

    def available(self) -> bool:
        return _has_captum()

    def applies_to(self, ctx: EvalContext) -> bool:
        return ctx.dataset.task in self.supports and _has_captum() and _module(ctx.model) is not None

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        import torch
        from captum.attr import IntegratedGradients
        ds = ctx.dataset
        X = np.asarray(ds.X_test, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(self.sample_size, len(X)), replace=False)
        x = torch.from_numpy(X[idx])
        base = torch.from_numpy(np.asarray(ds.X_train, dtype=np.float32).mean(axis=0, keepdims=True))
        module = _module(ctx.model)
        module.eval()
        pos = len(getattr(ctx.model, "classes_", [0, 1])) - 1

        def prob(inp):
            return torch.softmax(module(inp), dim=1)[:, pos]
        ig = IntegratedGradients(prob)
        attr, delta = ig.attribute(x, baselines=base.expand_as(x), n_steps=self.n_steps,
                                   return_convergence_delta=True)
        attr = attr.detach().numpy()
        err = float(np.mean(np.abs(delta.detach().numpy())))
        mean_abs = np.abs(attr).mean(axis=0)
        total = float(mean_abs.sum())
        share = mean_abs / total if total > 0 else np.zeros_like(mean_abs)
        order = np.argsort(share)[::-1]
        conc = float(share[order[:3]].sum())
        names = ds.feature_names or [f"f{i}" for i in range(X.shape[1])]
        with torch.no_grad():
            fx, fb = prob(x).numpy(), float(prob(base).numpy()[0])
        top = [[names[i], round(float(mean_abs[i]), 5), round(float(share[i]), 4)] for i in order[:12]]
        mechanism = (f"Captum IntegratedGradients ({self.n_steps} steps) on {len(idx)} test rows, baseline = "
                     f"training mean (f(baseline) = {fb:.4f}); mean completeness error {err:.2e}; top-3 "
                     f"features carry {conc:.1%} of the attribution.")
        trace = make_trace(
            steps=[f"f(baseline) = {fb:.4f}; mean f(x) over the sample = {float(fx.mean()):.4f}",
                   f"completeness: mean |Σ IG − (f(x) − f(x'))| = {err:.2e}  (≈ 0 means the attributions add up)",
                   f"ig_concentration = {conc:.4f}"],
            tables=[{"title": "Mean |integrated gradient| per feature", "columns": ["feature", "mean |IG|", "share"],
                     "rows": top}],
            charts=[{"title": "Share of attribution (integrated gradients)", "type": "bars",
                     "labels": [t[0] for t in top], "series": [{"name": "share", "values": [t[2] for t in top]}]}],
        )
        d = {"trace": trace, "n_steps": self.n_steps}
        return [EvalResult(self.name, "explainability", InspectionTarget.MODEL_INTERNALS,
                           "ig_completeness_error", err, False, mechanism, d),
                EvalResult(self.name, "explainability", InspectionTarget.MODEL_INTERNALS,
                           "ig_concentration", conc, True, mechanism, d)]


register(CaptumIGEvaluator())
