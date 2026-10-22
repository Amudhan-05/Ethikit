"""Run history: every run is saved locally so it can be re-opened, compared and reproduced.

Layout (default folder ./ethikit_runs, override with ETHIKIT_RUNS):

    ethikit_runs/<run_id>/config.json    the exact run config (feed it back: ethikit run --config)
    ethikit_runs/<run_id>/summary.json   scores, Pareto front, requirements, stability, profile
    ethikit_runs/<run_id>/meta.json      when, which dataset, which ethikit version

run_id = UTC timestamp + short hash of the config, so ids sort by time and two runs of the same
config on the same second can't collide silently.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .benchmark.runner import jsonable


def runs_dir() -> Path:
    p = Path(os.environ.get("ETHIKIT_RUNS") or "ethikit_runs")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _portable(config: dict) -> dict:
    """Config minus session-only handles (upload ids, in-memory model ids) that can't be replayed."""
    c = json.loads(json.dumps(config, default=str))
    data = c.get("data") or {}
    if data.get("source") == "upload":
        data["note"] = ("this run used a CSV uploaded in the console; to re-run, set source='csv' "
                        "and 'path' to that file")
    for m in c.get("models") or []:
        if m.get("upload_model_id"):
            m["note"] = "pre-trained model uploaded in the console; set 'file' to its path to re-run"
    return c


def save_run(config: dict, summary: dict) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    digest = hashlib.sha1(json.dumps(_portable(config), sort_keys=True).encode()).hexdigest()[:6]
    run_id = f"{stamp}-{digest}"
    d = runs_dir() / run_id
    n = 1
    while d.exists():                      # same config, same second
        n += 1
        d = runs_dir() / f"{run_id}-{n}"
    d.mkdir(parents=True)
    from . import __version__
    (d / "config.json").write_text(json.dumps(_portable(config), indent=2))
    (d / "summary.json").write_text(json.dumps(jsonable(summary)))
    (d / "meta.json").write_text(json.dumps({
        "run_id": d.name, "created_utc": stamp, "dataset": summary.get("dataset"),
        "created_ns": time.time_ns(),
        "models": sorted((summary.get("objective") or {}).keys()), "cv": summary.get("cv"),
        "ethikit_version": __version__}, indent=2))
    return d.name


def list_runs(limit: int = 50) -> List[dict]:
    """Saved runs, newest first (the run id only has one-second resolution, so ``created_ns`` breaks ties)."""
    found = []
    for d in runs_dir().iterdir():
        meta = d / "meta.json"
        if d.is_dir() and meta.exists():
            try:
                m = json.loads(meta.read_text())
            except Exception:
                continue
            found.append(((m.get("created_ns") or meta.stat().st_mtime_ns, d.name), m))
    found.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in found[:limit]]



def load_run(run_id: str) -> Dict[str, Any]:
    d = runs_dir() / run_id
    if not d.is_dir():
        raise KeyError(f"no saved run {run_id!r} in {runs_dir()} (see: ethikit runs list)")
    return {"config": json.loads((d / "config.json").read_text()),
            "summary": json.loads((d / "summary.json").read_text()),
            "meta": json.loads((d / "meta.json").read_text())}


def _flat(d: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    if isinstance(d, list) and d and all(isinstance(x, dict) for x in d):
        return {prefix: json.dumps(d, sort_keys=True)}
    return {prefix: d}


def diff_runs(a_id: str, b_id: str, tol: float = 1e-9) -> Dict[str, Any]:
    """What changed between two runs: config keys, then every model x principle score and every
    raw metric (models present in both), then models added/removed and the Pareto front."""
    a, b = load_run(a_id), load_run(b_id)
    fa, fb = _flat(a["config"]), _flat(b["config"])
    config_changes = [{"key": k, "a": fa.get(k), "b": fb.get(k)}
                      for k in sorted(set(fa) | set(fb)) if fa.get(k) != fb.get(k)]
    oa, ob = a["summary"].get("objective") or {}, b["summary"].get("objective") or {}
    common = sorted(set(oa) & set(ob))
    score_changes = []
    for m in common:
        for p in sorted(set(oa[m]) | set(ob[m])):
            va, vb = oa[m].get(p), ob[m].get(p)
            if va is None or vb is None or abs(va - vb) > tol:
                score_changes.append({"model": m, "principle": p, "a": va, "b": vb,
                                      "delta": None if va is None or vb is None else round(vb - va, 4)})

    def metrics(s):
        return {(r["model"], r["evaluator"], r["metric"]): r["value"]
                for r in s.get("records", []) if r.get("target") != "error"}
    ma, mb = metrics(a["summary"]), metrics(b["summary"])
    metric_changes = []
    for k in sorted(set(ma) | set(mb)):
        if k[0] not in common:
            continue
        va, vb = ma.get(k), mb.get(k)
        if va is None or vb is None or abs(float(va) - float(vb)) > tol:
            metric_changes.append({"model": k[0], "tool": k[1], "metric": k[2], "a": va, "b": vb,
                                   "delta": None if va is None or vb is None else round(float(vb) - float(va), 4)})
    return {"a": a["meta"], "b": b["meta"], "config_changes": config_changes,
            "models_only_in_a": sorted(set(oa) - set(ob)), "models_only_in_b": sorted(set(ob) - set(oa)),
            "score_changes": score_changes, "metric_changes": metric_changes,
            "pareto_a": a["summary"].get("pareto"), "pareto_b": b["summary"].get("pareto"),
            "identical": not (config_changes or score_changes or metric_changes)}
