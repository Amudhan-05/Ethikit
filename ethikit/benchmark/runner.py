"""Benchmark runner: the train -> evaluate -> collect loop.

Given a dataset and a dict of model factories, it fits each model, builds an
EvalContext, runs every *available & applicable* evaluator, and returns a
tidy list of records (one row per model x metric) that the scoring and report
layers consume.

Two extras exist for the live console / CLI (ethikit/cli.py, ethikit/server.py):
  - `on_event(event_dict)` is called at every step (run start, model fit,
    each tool starting/finishing with its results), so a UI can animate the
    flow graph and stream a log while the sweep is still running.
  - every tool's intermediate artefacts (details["trace"]: per-group tables,
    attack histograms, surrogate-tree rules, the formula filled in with this
    run's numbers) are kept in `df.attrs["traces"][model][tool]`.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..core.types import Dataset, EvalContext
from ..evaluators.base import Evaluator, available_evaluators
# importing the evaluators package registers all built-in evaluators
from .. import evaluators as _evaluators  # noqa: F401

Event = Dict[str, Any]


def _predict_all(model, ds: Dataset) -> EvalContext:
    y_pred = model.predict(ds.X_test)
    y_pred_train = model.predict(ds.X_train)
    y_proba = None
    y_proba_train = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(ds.X_test)
        except Exception:
            y_proba = None
        try:
            y_proba_train = model.predict_proba(ds.X_train)
        except Exception:
            y_proba_train = None
    return EvalContext(model=model, dataset=ds, y_pred=y_pred,
                       y_pred_train=y_pred_train, y_proba=y_proba,
                       y_proba_train=y_proba_train)


def jsonable(obj: Any) -> Any:
    """numpy/pandas scalars & arrays -> plain JSON types (for events and traces)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def select_evaluators(tools: Optional[Iterable[str]] = None,
                      tool_params: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Evaluator]:
    """Available evaluators, optionally restricted to `tools` and with
    per-tool param overrides applied ({tool: {param: value}})."""
    evs = available_evaluators()
    if tools is not None:
        wanted = set(tools)
        unknown = wanted - {e.name for e in evs}
        if unknown:
            raise KeyError(f"unknown or unavailable tool(s): {', '.join(sorted(unknown))}")
        evs = [e for e in evs if e.name in wanted]
    tool_params = tool_params or {}
    return [e.configured(**tool_params[e.name]) if e.name in tool_params else e for e in evs]


def run_benchmark(dataset: Dataset,
                  model_factories: Dict[str, Callable],
                  on_event: Optional[Callable[[Event], None]] = None,
                  evaluators: Optional[List[Evaluator]] = None,
                  prefit: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Fit every factory (or take an already-fitted model from `prefit`), run
    the evaluator battery, return tidy records.

    `prefit` maps model_name -> an already-trained estimator that is evaluated
    as-is (no .fit()) -- "evaluate *my* model on *my* data".
    """
    emit = on_event or (lambda e: None)
    records: List[dict] = []
    traces: Dict[str, Dict[str, Any]] = {}
    evaluators = evaluators if evaluators is not None else available_evaluators()
    prefit = prefit or {}
    names = list(model_factories) + [m for m in prefit if m not in model_factories]

    emit({"type": "run_start", "dataset": dataset.name, "task": dataset.task.value,
          "n_train": int(len(dataset.y_train)), "n_test": int(len(dataset.y_test)),
          "n_features": int(np.asarray(dataset.X_test).shape[1]),
          "sensitive": dataset.sensitive_name, "models": names,
          "tools": [{"name": e.name, "principles": list(e.principle_ids)} for e in evaluators]})

    for model_name in names:
        t0 = time.perf_counter()
        if model_name in prefit:
            model = prefit[model_name]
            emit({"type": "model_fit_start", "model": model_name, "prefit": True})
        else:
            emit({"type": "model_fit_start", "model": model_name, "prefit": False})
            model = model_factories[model_name]()
            model.fit(dataset.X_train, dataset.y_train)
        ctx = _predict_all(model, dataset)
        emit({"type": "model_fit_done", "model": model_name,
              "seconds": round(time.perf_counter() - t0, 3)})
        traces[model_name] = {}

        for ev in evaluators:
            if not ev.applies_to(ctx):
                emit({"type": "tool_skip", "model": model_name, "tool": ev.name,
                      "reason": "not applicable to this task / dataset"})
                continue
            emit({"type": "tool_start", "model": model_name, "tool": ev.name,
                  "principles": list(ev.principle_ids)})
            t1 = time.perf_counter()
            try:
                results = ev.evaluate(ctx)
                new = [res.as_record(model_name) for res in results]
                records.extend(new)
                trace = next((r.details.get("trace") for r in results
                              if r.details.get("trace")), None)
                traces[model_name][ev.name] = jsonable({
                    "trace": trace, "formula": ev.formula,
                    "params": ev.params(),
                    "mechanisms": sorted({r.mechanism for r in results}),
                })
                emit({"type": "tool_done", "model": model_name, "tool": ev.name,
                      "seconds": round(time.perf_counter() - t1, 3),
                      "results": jsonable([{k: r[k] for k in
                                            ("principle", "metric", "value", "higher_is_better")}
                                           for r in new])})
            except Exception as exc:  # never let one tool break the sweep
                records.append({
                    "model": model_name, "evaluator": ev.name,
                    "principle": (ev.principle_ids or ["?"])[0], "target": "error",
                    "metric": f"ERROR:{type(exc).__name__}", "value": float("nan"),
                    "higher_is_better": True,
                    "mechanism": f"{ev.name} raised {type(exc).__name__}: {exc}",
                })
                emit({"type": "tool_error", "model": model_name, "tool": ev.name,
                      "error": f"{type(exc).__name__}: {exc}"})

    df = pd.DataFrame.from_records(records)
    df.attrs["traces"] = traces
    emit({"type": "run_done", "n_records": len(records)})
    return df
