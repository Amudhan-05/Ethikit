"""One run configuration, shared by the CLI (`ethikit run`) and the console server.

A config is a plain dict (JSON-serialisable), so the console can show the exact
equivalent CLI command and a recorded run can be replayed:

{
  "data":  {"source": "synthetic" | "synthetic_custom" | "adult" | "german_credit"
                      | "regression" | "csv" | "upload",
            "path": "file.csv",            # source=csv
            "upload_id": "...",            # source=upload (console)
            "target": "y", "sensitive": "sex", "exclude_sensitive": false,
            "positive_label": null, "sensitive_threshold": null,
            "synthetic": {"n": 4000, "bias": 0.9, ...}},   # source=synthetic_custom
  "split": {"test_size": 0.3, "seed": 0, "scale": true},
  "models": [{"name": "random_forest", "params": {"max_depth": 5}},
             {"name": "extra_trees", "estimator": "sklearn.ensemble.ExtraTreesClassifier"},
             {"name": "mine", "file": "model.joblib"}],          # pre-fitted, evaluated as-is
  "tools": ["fairness", "aif360", ...] | null,                  # null = every available tool
  "tool_params": {"robustness": {"sigma": 0.2}},
  "weights": {"fairness": 0.35, ...},
  "checklist": {"checklist": "deon", "answers": {...}} | null
}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from . import checklists as checklists_mod
from . import principles as principles_mod
from .benchmark.models import (
    CLASSIFIERS, REGRESSORS, default_params, estimator_from_path, family_of,
    flat_classifiers, flat_regressors, load_fitted_model, with_params,
)
from .benchmark.runner import jsonable, run_benchmark, select_evaluators
from .core.types import TaskType
from .data import loaders
from .evaluators.base import all_evaluators
from .scoring import normalize, objective_matrix, pareto_front, weighted_score

DEFAULT_WEIGHTS = {"fairness": 0.35, "explainability": 0.20, "safety": 0.20,
                   "privacy": 0.15, "inclusiveness": 0.10}

DATASETS = {
    "synthetic": "Synthetic biased classification data, offline (4000 rows; group leaks into the label).",
    "synthetic_custom": "Synthetic data with knobs: size, bias, minority share, class balance, noise, categoricals.",
    "regression": "Synthetic biased regression data, offline.",
    "adult": "UCI Adult income (OpenML, needs network; sensitive = sex). Falls back to synthetic offline.",
    "german_credit": "Statlog German Credit (OpenML, needs network; sensitive = age >= 25).",
    "csv": "Your own CSV file: choose the target and the sensitive column.",
}

SYNTHETIC_KNOBS = {"n": 4000, "n_features": 4, "bias": 0.9, "proxy_strength": 0.8,
                   "minority_share": 0.5, "positive_rate": 0.5, "noise": 1.0,
                   "n_categorical": 0, "seed": 0}


@dataclass
class PreparedRun:
    dataset: Any
    factories: Dict[str, Callable]
    prefit: Dict[str, Any]
    evaluators: list
    hyperparameters: Dict[str, Dict[str, Any]]
    origins: Dict[str, str]
    weights: Dict[str, float]
    checklist: Optional[dict]
    notes: List[str] = field(default_factory=list)


def load_raw(data: Dict[str, Any], uploads: Optional[Dict[str, pd.DataFrame]] = None):
    src = data.get("source", "synthetic")
    common = {k: data[k] for k in ("exclude_sensitive", "positive_label", "sensitive_threshold")
              if data.get(k) not in (None, "")}
    if src == "synthetic":
        return loaders.make_synthetic_fairness_raw()
    if src == "synthetic_custom":
        knobs = {**SYNTHETIC_KNOBS, **(data.get("synthetic") or {})}
        knobs = {k: type(SYNTHETIC_KNOBS[k])(v) for k, v in knobs.items() if k in SYNTHETIC_KNOBS}
        return loaders.make_synthetic_raw(**knobs, task=data.get("task") or "classification")
    if src == "regression":
        return loaders.make_synthetic_fairness_regression_raw()
    if src == "adult":
        return loaders.load_adult_raw()
    if src == "german_credit":
        return loaders.load_german_credit_raw()
    if src in ("csv", "upload"):
        if not data.get("target"):
            raise ValueError("custom data needs a target column (--target)")
        if src == "csv":
            return loaders.load_csv(data["path"], data["target"], data.get("sensitive") or None,
                                    **common)
        df = (uploads or {}).get(data.get("upload_id"))
        if df is None:
            raise KeyError("uploaded dataset not found -- upload the CSV again")
        return loaders.prepare_frame(df, data["target"], data.get("sensitive") or None,
                                     name=data.get("name") or "uploaded", **common)
    raise KeyError(f"unknown data source {src!r}; choose from {', '.join(DATASETS)}")


def prepare(config: Dict[str, Any], uploads: Optional[Dict[str, pd.DataFrame]] = None,
            uploaded_models: Optional[Dict[str, Any]] = None) -> PreparedRun:
    raw = load_raw(config.get("data") or {}, uploads)
    split = {"test_size": 0.3, "seed": 0, "scale": True, **(config.get("split") or {})}
    ds = loaders.split_dataset(raw, seed=int(split["seed"]), test_size=float(split["test_size"]),
                               scale=bool(split["scale"]))

    zoo = flat_classifiers() if ds.task == TaskType.CLASSIFICATION else flat_regressors()
    factories: Dict[str, Callable] = {}
    prefit: Dict[str, Any] = {}
    hyper: Dict[str, Dict[str, Any]] = {}
    origins: Dict[str, str] = {}
    notes: List[str] = []
    models = config.get("models") or [{"name": n} for n in zoo]
    for spec in models:
        name, params = spec["name"], spec.get("params") or {}
        if spec.get("file") or spec.get("upload_model_id"):
            model = (uploaded_models or {}).get(spec.get("upload_model_id")) \
                if spec.get("upload_model_id") else load_fitted_model(spec["file"])
            if model is None:
                raise KeyError(f"uploaded model for {name!r} not found -- upload it again")
            prefit[name] = model
            hyper[name] = model.get_params(deep=False) if hasattr(model, "get_params") else {}
            origins[name] = (f"A pre-trained {type(model).__name__} supplied by the user and "
                             f"evaluated as-is (not refit) on {ds.name}.")
            continue
        if spec.get("estimator"):
            factory = estimator_from_path(spec["estimator"], params)
            origins[name] = (f"A user-specified estimator <code>{spec['estimator']}</code>, "
                             f"trained and evaluated on {ds.name}.")
        elif name in zoo:
            factory = with_params(zoo[name], params)
            if params:
                origins[name] = (f"ethikit zoo model <code>{name}</code> ({family_of(name)} family) "
                                 f"with user-set hyperparameters {', '.join(params)}, trained and "
                                 f"evaluated on {ds.name}.")
        else:
            raise KeyError(f"unknown model {name!r} for a {ds.task.value} task; choose from "
                           f"{', '.join(zoo)} or give an estimator path")
        factories[name] = factory
        try:
            hyper[name] = factory().get_params(deep=False)
        except Exception as exc:
            raise ValueError(f"model {name!r}: {exc}") from exc

    evaluators = select_evaluators(config.get("tools"), config.get("tool_params"))
    weights = {**DEFAULT_WEIGHTS, **(config.get("weights") or {})}
    checklist = config.get("checklist")
    if checklist and checklist.get("checklist") not in checklists_mod.CHECKLISTS:
        raise KeyError(f"unknown checklist {checklist.get('checklist')!r}")
    if raw.name.startswith("synthetic") and (config.get("data") or {}).get("source") == "adult":
        notes.append("UCI Adult was unreachable (no network); fell back to synthetic data.")
    return PreparedRun(ds, factories, prefit, evaluators, hyper, origins, weights,
                       checklist, notes)


def execute(run: PreparedRun, on_event: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
    """Run the sweep and score it. Returns everything a UI needs, JSON-ready
    except `raw` (the DataFrame, kept for rendering cards/reports)."""
    raw = run_benchmark(run.dataset, run.factories, on_event=on_event,
                        evaluators=run.evaluators, prefit=run.prefit)
    dn = normalize(raw)
    obj = objective_matrix(dn)
    front = pareto_front(obj)
    ranked = obj.copy()
    principles = [c for c in run.weights if c in ranked.columns and run.weights[c] > 0]
    if principles:
        ranked["weighted_score"] = weighted_score(ranked, {p: run.weights[p] for p in principles}).round(3)
        ranked = ranked.sort_values("weighted_score", ascending=False)
    records = raw.to_dict(orient="records")
    return {
        "raw": raw,
        "objective": obj,
        "pareto": front,
        "summary": jsonable({
            "dataset": run.dataset.name, "task": run.dataset.task.value,
            "sensitive": run.dataset.sensitive_name,
            "objective": {m: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                          for m, row in obj.iterrows()},
            "ranking": list(ranked.index),
            "pareto": front, "weights": run.weights,
            "records": records, "traces": raw.attrs.get("traces", {}),
            "families": {m: family_of(m) for m in obj.index},
            "notes": run.notes,
        }),
    }


def catalogue() -> Dict[str, Any]:
    """Everything a UI or `ethikit list` needs to offer choices."""
    from . import evaluators as _ev  # noqa: F401  (registers evaluators)
    tools = []
    for ev in all_evaluators():
        tools.append({"name": ev.name, "principles": list(ev.principle_ids),
                      "available": ev.available(), "mechanism": ev.mechanism,
                      "formula": ev.formula, "params": jsonable(ev.params()),
                      "supports": [t.value for t in ev.supports]})
    def zoo(reg):
        return {fam: {m: jsonable({k: (v if isinstance(v, (bool, int, float, str, type(None)))
                                       else repr(v)) for k, v in default_params(m).items()})
                      for m in models}
                for fam, models in reg.items()}
    return {
        "datasets": DATASETS, "synthetic_knobs": SYNTHETIC_KNOBS,
        "classifiers": zoo(CLASSIFIERS), "regressors": zoo(REGRESSORS),
        "tools": tools, "default_weights": DEFAULT_WEIGHTS,
        "principles": [{"id": p.id, "name": p.name, "plain": p.plain,
                        "tool_addressable": p.tool_addressable} for p in principles_mod.PRINCIPLES],
        "checklists": [checklists_mod.as_dict(c) for c in checklists_mod.CHECKLISTS.values()],
    }
