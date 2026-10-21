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
    raw: Any = None                                # the unsplit RawData (needed for repeated CV)
    cv: Optional[tuple] = None                     # (folds, repeats) or None
    dataset_seed: int = 0
    requirements: list = field(default_factory=list)   # decision.Requirement
    profile: Optional[dict] = None                 # dataset profile (computed before any model)
    config: Optional[dict] = None                  # the config this run came from (for history)
    save: bool = True


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
    from .decision import parse_requirements
    from .profile import profile_raw
    requirements = parse_requirements(config.get("requirements"))   # fail fast on a typo
    raw = load_raw(config.get("data") or {}, uploads)
    data_cfg = config.get("data") or {}
    frame = None
    if data_cfg.get("source") == "upload":
        frame = (uploads or {}).get(data_cfg.get("upload_id"))
    elif data_cfg.get("source") == "csv":
        frame = pd.read_csv(data_cfg["path"])
    profile = profile_raw(raw, frame, seed=int((config.get("split") or {}).get("seed") or 0))
    split = {"test_size": 0.3, "seed": 0, "scale": True, **(config.get("split") or {})}
    cv = split.get("cv")
    cv = (int(cv[0]), int(cv[1])) if cv else None
    if cv and (cv[0] < 2 or cv[1] < 1):
        raise ValueError("cv needs at least 2 folds and 1 repeat, e.g. 5x3")
    ds = loaders.split_dataset(raw, seed=int(split["seed"]), test_size=float(split["test_size"]),
                               scale=bool(split["scale"]))

    if cv and any(m.get("file") or m.get("upload_model_id") for m in (config.get("models") or [])):
        raise ValueError("pre-trained models can't be cross-validated (they are not refit); "
                         "remove --cv or the --model-file")
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
                       checklist, notes, raw=raw, cv=cv, dataset_seed=int(split["seed"]),
                       requirements=requirements, profile=profile, config=config,
                       save=bool(config.get("save", True)))


def execute(run: PreparedRun, on_event: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
    """Run the sweep and score it. Returns everything a UI needs, JSON-ready
    except `raw` (the DataFrame, kept for rendering cards/reports)."""
    emit = on_event or (lambda e: None)
    if run.profile:
        emit({"type": "profile", **jsonable(run.profile)})
    obj_std = None
    if run.cv:
        raw, obj, obj_std = _execute_cv(run, on_event)
    else:
        raw = run_benchmark(run.dataset, run.factories, on_event=on_event,
                            evaluators=run.evaluators, prefit=run.prefit)
        obj = objective_matrix(normalize(raw))
    front = pareto_front(obj)
    ranked = obj.copy()
    principles = [c for c in run.weights if c in ranked.columns and run.weights[c] > 0]
    if principles:
        ranked["weighted_score"] = weighted_score(ranked, {p: run.weights[p] for p in principles}).round(3)
        ranked = ranked.sort_values("weighted_score", ascending=False)
    records = raw.to_dict(orient="records")
    from .decision import check_requirements, choice_stability
    req = None
    if run.requirements:
        by_model = check_requirements(raw, run.requirements, list(obj.index))
        feasible = [m for m in obj.index if by_model[m]["passes"]]
        req = {"list": [r.text for r in run.requirements], "by_model": by_model, "feasible": feasible}
    weights_used = {p: run.weights[p] for p in principles}
    stability = choice_stability(obj, weights_used,
                                 candidates=(req["feasible"] if req else None) or None) \
        if principles and (not req or req["feasible"]) else \
        {"available": False, "reason": "no model meets every requirement" if req else "no weights"}
    out = {
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
            "cv": list(run.cv) if run.cv else None,
            "objective_std": None if obj_std is None else {
                m: {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                for m, row in obj_std.iterrows()},
            "notes": run.notes,
            "requirements": req, "stability": stability, "profile": run.profile,
        }),
    }
    if run.save and run.config is not None:
        from .runs import save_run
        out["summary"]["history_id"] = save_run(run.config, out["summary"])
    return out


def _execute_cv(run: PreparedRun, on_event):
    """Repeated k-fold: every fold is a full tool sweep. The console's graph animates the
    FIRST fold only (later folds just advance a counter) so it isn't redrawn 15 times; the
    reported profile is the mean over folds with the standard deviation alongside."""
    from .benchmark.cv import run_benchmark_cv
    emit = on_event or (lambda e: None)
    folds, repeats = run.cv
    total, state = folds * repeats, {"fold": 0}

    def inner(e):
        t = e["type"]
        if t == "run_start":
            state["fold"] += 1
            if state["fold"] == 1:
                emit({**e, "cv": [folds, repeats]})
            emit({"type": "cv_fold", "i": state["fold"], "n": total})
        elif t == "run_done":
            return
        elif state["fold"] == 1 or t == "tool_error":
            emit(e)

    cvdf = run_benchmark_cv(run.raw, run.factories, n_splits=folds, n_repeats=repeats,
                            seed=run.dataset_seed,
                            evaluators=run.evaluators, on_event=inner)
    per = [objective_matrix(normalize(g.drop(columns=["repeat", "fold", "seed"])))
           for _, g in cvdf.groupby(["repeat", "fold"])]
    stack = pd.concat(per)
    obj, std = stack.groupby(level=0).mean().round(3), stack.groupby(level=0).std().round(3)
    key = ["model", "evaluator", "principle", "target", "metric", "higher_is_better"]
    agg = (cvdf.groupby(key, dropna=False)["value"].mean().reset_index()
           .merge(cvdf.drop_duplicates(key)[key + ["mechanism"]], on=key, how="left"))
    agg.attrs["traces"] = cvdf.attrs.get("traces", {})
    emit({"type": "run_done", "n_records": len(agg)})
    return agg, obj, std


def domains_catalogue() -> list:
    from .domains import get_domains, STRENGTH_TEXT
    out = []
    for d in get_domains().values():
        out.append({"id": d.id, "title": d.title, "blurb": d.blurb, "for_whom": d.for_whom,
                    "strength": d.strength, "strength_text": STRENGTH_TEXT[d.strength],
                    "caveats": d.caveats,
                    "datasets": [{"id": s.id, "title": s.title, "source": s.source,
                                  "sensitive": s.sensitive, "needs": s.needs, "note": s.note,
                                  "verified": s.verified} for s in d.datasets]})
    return out


def catalogue() -> Dict[str, Any]:
    """Everything a UI or `ethikit list` needs to offer choices."""
    from . import evaluators as _ev  # noqa: F401  (registers evaluators)
    tools = []
    for ev in all_evaluators():
        tools.append({"name": ev.name, "principles": list(ev.principle_ids),
                      "available": ev.available(), "install_hint": ev.install_hint,
                      "mechanism": ev.mechanism,
                      "formula": ev.formula, "params": jsonable(ev.params()), "limits": ev.limits,
                      "supports": [t.value for t in ev.supports]})
    def zoo(reg):
        return {fam: {m: jsonable({k: (v if isinstance(v, (bool, int, float, str, type(None)))
                                       else repr(v)) for k, v in default_params(m).items()})
                      for m in models}
                for fam, models in reg.items()}
    return {
        "datasets": DATASETS, "synthetic_knobs": SYNTHETIC_KNOBS,
        "optional_models": {m: hint for m, hint in __import__("ethikit.benchmark.models", fromlist=["x"]).OPTIONAL_MODELS.items()
                            if m not in CLASSIFIERS.get("deep", {})},
        "classifiers": zoo(CLASSIFIERS), "regressors": zoo(REGRESSORS),
        "tools": tools, "default_weights": DEFAULT_WEIGHTS,
        "principles": [{"id": p.id, "name": p.name, "plain": p.plain,
                        "tool_addressable": p.tool_addressable} for p in principles_mod.PRINCIPLES],
        "checklists": [checklists_mod.as_dict(c) for c in checklists_mod.CHECKLISTS.values()],
        "domains": domains_catalogue(),
    }
