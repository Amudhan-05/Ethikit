"""Privacy evaluator (Phase 1 proxy).

A full privacy assessment needs membership-inference attacks and/or
differential-privacy training (TensorFlow Privacy, Opacus) -- that is Phase 3.
As a lightweight, honest Phase-1 signal we report the train-vs-test
performance gap: a large gap means the model has memorised its training data,
which is both an overfitting and a privacy-leakage red flag.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn import metrics as skm

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, register


class PrivacyProxyEvaluator(Evaluator):
    name = "privacy_proxy"
    principle_ids = ["privacy"]

    def applies_to(self, ctx: EvalContext) -> bool:
        return ctx.y_pred_train is not None

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        if ctx.dataset.task == TaskType.CLASSIFICATION:
            train = skm.accuracy_score(ctx.dataset.y_train, ctx.y_pred_train)
            test = skm.accuracy_score(ctx.dataset.y_test, ctx.y_pred)
        else:
            train = skm.r2_score(ctx.dataset.y_train, ctx.y_pred_train)
            test = skm.r2_score(ctx.dataset.y_test, ctx.y_pred)
        gap = float(abs(train - test))
        # convert to a 0..1 "privacy safety" score (small gap -> closer to 1)
        safety = float(max(0.0, 1.0 - gap))
        return [
            EvalResult(self.name, "privacy", InspectionTarget.BEHAVIOR,
                       "memorization_safety", safety, True,
                       {"train_score": round(train, 4), "test_score": round(test, 4),
                        "note": "proxy only; replace with membership-inference in Phase 3"}),
        ]


register(PrivacyProxyEvaluator())
