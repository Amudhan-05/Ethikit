"""Benchmark runner: the train -> evaluate -> collect loop.

Given a dataset and a dict of model factories, it fits each model, builds an
EvalContext, runs every *available & applicable* evaluator, and returns a
tidy list of records (one row per model x metric) that the scoring and report
layers consume.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from ..core.types import Dataset, EvalContext
from ..evaluators.base import available_evaluators
# importing the evaluators package registers all built-in evaluators
from .. import evaluators as _evaluators  # noqa: F401


def _predict_all(model, ds: Dataset) -> EvalContext:
    y_pred = model.predict(ds.X_test)
    y_pred_train = model.predict(ds.X_train)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(ds.X_test)
        except Exception:
            y_proba = None
    return EvalContext(model=model, dataset=ds, y_pred=y_pred,
                       y_pred_train=y_pred_train, y_proba=y_proba)


def run_benchmark(dataset: Dataset,
                  model_factories: Dict[str, Callable]) -> pd.DataFrame:
    records: List[dict] = []
    evaluators = available_evaluators()

    for model_name, factory in model_factories.items():
        model = factory()
        model.fit(dataset.X_train, dataset.y_train)
        ctx = _predict_all(model, dataset)

        for ev in evaluators:
            if not ev.applies_to(ctx):
                continue
            try:
                for res in ev.evaluate(ctx):
                    records.append(res.as_record(model_name))
            except Exception as exc:  # never let one tool break the sweep
                records.append({
                    "model": model_name, "evaluator": ev.name,
                    "principle": (ev.principle_ids or ["?"])[0], "target": "error",
                    "metric": f"ERROR:{type(exc).__name__}", "value": float("nan"),
                    "higher_is_better": True,
                    "mechanism": f"{ev.name} raised {type(exc).__name__}: {exc}",
                })
    return pd.DataFrame.from_records(records)
