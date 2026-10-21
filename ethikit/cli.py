"""ethikit command line.

    python -m ethikit run [options]        evaluate models, stream the run, write report + cards
    python -m ethikit list <what>          datasets | models | tools | params <model|tool> | checklists
    python -m ethikit checklist <id>       write a blank answers template (or --markdown)
    python -m ethikit serve                start the live web console
    python -m ethikit export-site [dir]    copy the console + theme into the docs site, with a replay

Every flag maps 1:1 onto the run config in ethikit/session.py, which is also
what the web console builds -- the console shows the equivalent command line
for whatever you click, so anything shown in a demo can be re-run here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

# ---- terminal styling -------------------------------------------------------
_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


G = lambda s: _c("32", s)   # noqa: E731
Y = lambda s: _c("33", s)   # noqa: E731
R = lambda s: _c("31", s)   # noqa: E731
D = lambda s: _c("2", s)    # noqa: E731
B = lambda s: _c("1", s)    # noqa: E731


def _kv_pairs(items: Optional[List[str]], what: str) -> Dict[str, Dict[str, str]]:
    """['rf.max_depth=5', 'rf.n_estimators=50'] -> {'rf': {'max_depth': '5', ...}}"""
    out: Dict[str, Dict[str, str]] = {}
    for item in items or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise SystemExit(f"--{what} expects <name>.<param>=<value>, got {item!r}")
        key, value = item.split("=", 1)
        name, param = key.split(".", 1)
        out.setdefault(name, {})[param] = value
    return out


def _weights(s: Optional[str]) -> Dict[str, float]:
    if not s:
        return {}
    out = {}
    for part in s.split(","):
        k, _, v = part.partition("=")
        out[k.strip()] = float(v)
    return out


def config_from_args(a: argparse.Namespace) -> dict:
    from . import session
    from .benchmark.models import CLASSIFIERS, REGRESSORS

    data: dict = {}
    if a.data.endswith(".csv"):
        data = {"source": "csv", "path": a.data}
    else:
        data = {"source": a.data}
    for k in ("target", "sensitive", "positive_label", "sensitive_threshold"):
        if getattr(a, k) is not None:
            data[k] = getattr(a, k)
    if a.exclude_sensitive:
        data["exclude_sensitive"] = True
    if a.synthetic:
        data["source"] = "synthetic_custom"
        data["synthetic"] = {k: v for k, _, v in (p.partition("=") for p in a.synthetic)}

    params = _kv_pairs(a.model_param, "model-param")
    names: List[str] = []
    registry = REGRESSORS if data["source"] == "regression" else CLASSIFIERS
    for fam in (a.family or "").split(","):
        if fam:
            if fam not in registry:
                raise SystemExit(f"unknown family {fam!r}; choose from {', '.join(registry)}")
            names += list(registry[fam])
    names += [m for m in (a.models or "").split(",") if m]
    models = [{"name": n, "params": params.get(n, {})} for n in dict.fromkeys(names)]
    for spec in a.estimator or []:
        name, _, path = spec.partition("=")
        models.append({"name": name, "estimator": path, "params": params.get(name, {})})
    for spec in a.model_file or []:
        name, _, path = spec.partition("=")
        models.append({"name": name, "file": path})

    cfg = {
        "data": data,
        "split": {"test_size": a.test_size, "seed": a.seed, "scale": not a.no_scale},
        "models": models or None,
        "tools": a.tools.split(",") if a.tools else None,
        "tool_params": _kv_pairs(a.tool_param, "tool-param"),
        "weights": {**session.DEFAULT_WEIGHTS, **_weights(a.weights)},
    }
    if a.checklist:
        from . import checklists
        cfg["checklist"] = checklists.load_answers(a.checklist)
    return cfg


class LiveLog:
    """Prints run events the way the console's log pane shows them."""

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.t0 = time.perf_counter()
        self.total = self.done = 0

    def __call__(self, e: dict):
        t = D(f"{time.perf_counter() - self.t0:7.2f}s")
        kind = e["type"]
        if kind == "run_start":
            self.total = len(e["models"]) * len(e["tools"])
            print(f"{t} {B('run')}   dataset={G(e['dataset'])} task={e['task']} "
                  f"train={e['n_train']} test={e['n_test']} features={e['n_features']} "
                  f"sensitive={e['sensitive']}")
            print(f"{t}       models: {', '.join(e['models'])}")
            print(f"{t}       tools:  {', '.join(x['name'] for x in e['tools'])}")
        elif kind == "model_fit_start":
            print(f"{t} {B('fit')}   {e['model']}{D(' (pre-fitted, not refit)') if e['prefit'] else ''}")
        elif kind == "model_fit_done":
            print(f"{t}       {G('ok')} {e['model']} in {e['seconds']:.2f}s")
        elif kind == "tool_start" and self.verbose:
            print(f"{t} {D('exec')}  {e['tool']} → {e['model']}")
        elif kind == "tool_done":
            self.done += 1
            vals = "  ".join(f"{r['metric']}={r['value']:.4f}" if isinstance(r["value"], float)
                             else f"{r['metric']}={r['value']}" for r in e["results"])
            print(f"{t} {G('✓')} {e['tool']:<22}{D(f'[{self.done}/{self.total}]')} {vals}")
        elif kind == "tool_skip":
            self.done += 1
            if self.verbose:
                print(f"{t} {Y('skip')}  {e['tool']} → {e['model']}: {e['reason']}")
        elif kind == "tool_error":
            self.done += 1
            print(f"{t} {R('✗')} {e['tool']} → {e['model']}: {e['error']}")


def cmd_run(a: argparse.Namespace) -> int:
    from . import session
    from .report import render_html, render_model_cards

    cfg = config_from_args(a)
    prepared = session.prepare(cfg)
    for note in prepared.notes:
        print(Y(f"note: {note}"))
    out = session.execute(prepared, on_event=LiveLog(a.verbose))
    s = out["summary"]

    obj = out["objective"].copy()
    import pandas as pd
    from .scoring import weighted_score
    w = {k: v for k, v in prepared.weights.items() if k in obj.columns and v > 0}
    if w:
        obj["weighted_score"] = weighted_score(obj, w).round(3)
        obj = obj.sort_values("weighted_score", ascending=False)
    print("\n" + B("leaderboard") + D("  (0-1 per column, 1 = best in this run)"))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(obj.round(3).to_string())
    print("\n" + B("pareto front") + ": " + ", ".join(G(m) for m in s["pareto"]))

    os.makedirs(a.out, exist_ok=True)
    report = render_html(s["dataset"], obj, s["pareto"], os.path.join(a.out, "report.html"))
    cards = render_model_cards(s["dataset"], out["raw"], out["objective"], s["pareto"],
                               out_dir=os.path.join(a.out, "model_cards"),
                               sensitive_name=s["sensitive"], checklist=prepared.checklist,
                               hyperparameters=prepared.hyperparameters,
                               origins=prepared.origins)
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": cfg, **s}, fh)
    print(f"\n{D('wrote')} {report}\n{D('wrote')} {len(cards)} model cards in "
          f"{os.path.join(a.out, 'model_cards')}\n{D('wrote')} {os.path.join(a.out, 'results.json')}")
    if a.trace:
        for model, tools in s["traces"].items():
            for tool, t in tools.items():
                print("\n" + B(f"── {model} · {tool}"))
                print(D(t.get("formula") or ""))
                for step in (t.get("trace") or {}).get("steps", []):
                    print("   " + step)
    return 0


def cmd_list(a: argparse.Namespace) -> int:
    from . import session
    cat = session.catalogue()
    what = a.what
    if what == "datasets":
        for k, v in cat["datasets"].items():
            print(f"{G(k):<28} {v}")
        print(D("\nsynthetic knobs (--synthetic key=value): ")
              + ", ".join(f"{k}={v}" for k, v in cat["synthetic_knobs"].items()))
    elif what == "models":
        for label, reg in (("classification", cat["classifiers"]), ("regression", cat["regressors"])):
            print(B(label))
            for fam, models in reg.items():
                print(f"  {fam:<16} {', '.join(models)}")
        print(D("\nany estimator: --estimator name=sklearn.ensemble.ExtraTreesClassifier"))
    elif what == "tools":
        for t in cat["tools"]:
            flag = G("available") if t["available"] else Y("not installed")
            print(f"{B(t['name']):<30} {flag:<20} principles: {', '.join(t['principles'])}")
    elif what == "params":
        if not a.name:
            raise SystemExit("usage: ethikit list params <model-or-tool>")
        tool = next((t for t in cat["tools"] if t["name"] == a.name), None)
        if tool:
            print(B(f"{a.name} (tool) -- set with --tool-param {a.name}.<key>=<value>"))
            for k, v in tool["params"].items():
                print(f"  {k} = {v!r}")
            print(D("\n" + tool["formula"]))
            return 0
        for reg in (cat["classifiers"], cat["regressors"]):
            for models in reg.values():
                if a.name in models:
                    print(B(f"{a.name} (model) -- set with --model-param {a.name}.<key>=<value>"))
                    for k, v in models[a.name].items():
                        print(f"  {k} = {v!r}")
                    return 0
        raise SystemExit(f"no model or tool named {a.name!r}")
    elif what == "checklists":
        for c in cat["checklists"]:
            print(f"{G(c['id']):<22} {c['name']} ({len(c['items'])} items) -- {c['url']}")
    return 0


def cmd_checklist(a: argparse.Namespace) -> int:
    from . import checklists
    if a.markdown:
        text = checklists.to_markdown(a.id)
    else:
        text = json.dumps(checklists.blank_answers(a.id), indent=2)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {a.out}  (fill each answer with yes / partial / no / n/a, "
              f"then pass --checklist {a.out} to ethikit run)")
    else:
        print(text)
    return 0


def cmd_serve(a: argparse.Namespace) -> int:
    from .server import serve
    serve(a.host, a.port, a.open)
    return 0


def cmd_export_site(a: argparse.Namespace) -> int:
    """Copy the console + theme into the docs site and record a replay of a real
    run, so the static site (GitHub Pages) can show the console without Python."""
    from . import session
    from .server import WEB
    dest = os.path.join(a.dir, "console")
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(WEB):
        shutil.copy(os.path.join(WEB, name), os.path.join(dest, name))
    for name in ("theme.css", "theme.js"):
        shutil.copy(os.path.join(WEB, name), os.path.join(a.dir, name))

    from .report import model_card_html
    config = {"data": {"source": "synthetic"}, "split": {"test_size": 0.3, "seed": 0, "scale": True},
              "models": [{"name": m} for m in a.models.split(",")], "tools": None,
              "tool_params": {}, "weights": session.DEFAULT_WEIGHTS}
    events: List[dict] = []
    t0 = time.perf_counter()

    def rec(e):
        events.append({**e, "t": round(time.perf_counter() - t0, 3)})
    prepared = session.prepare(config)
    out = session.execute(prepared, on_event=rec)
    s = out["summary"]
    cards = {m: model_card_html(m, s["dataset"], out["raw"], out["objective"], s["pareto"],
                                sensitive_name=s["sensitive"],
                                hyperparameters=prepared.hyperparameters.get(m))
             for m in out["objective"].index}
    replay = {"meta": {**session.catalogue(), "live": False}, "config": config,
              "events": events,
              "result": {"type": "result", "run_id": "replay", **s,
                         "hyperparameters": {m: {k: repr(v) for k, v in p.items()}
                                             for m, p in prepared.hyperparameters.items()}},
              "cards": cards}
    from .benchmark.runner import jsonable
    with open(os.path.join(dest, "replay.json"), "w", encoding="utf-8") as fh:
        json.dump(jsonable(replay), fh)
    print(f"exported console to {dest} with a replay of {len(events)} events")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ethikit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate models on a dataset")
    r.add_argument("--data", default="synthetic",
                   help="synthetic | adult | german_credit | regression | path/to/file.csv")
    r.add_argument("--target", help="target column (CSV)")
    r.add_argument("--sensitive", help="sensitive-attribute column whose groups are compared (CSV)")
    r.add_argument("--sensitive-threshold", type=float,
                   help="split a numeric sensitive column (e.g. age) at this value")
    r.add_argument("--positive-label", help="which target label is the favourable class=1")
    r.add_argument("--exclude-sensitive", action="store_true",
                   help="do not give the sensitive column to the model as a feature")
    r.add_argument("--synthetic", nargs="+", metavar="KEY=VALUE",
                   help="tunable synthetic data, e.g. n=2000 minority_share=0.2 bias=1.5")
    r.add_argument("--models", help="comma-separated zoo models (see: ethikit list models)")
    r.add_argument("--family", help="comma-separated families, e.g. linear,tree_based")
    r.add_argument("--model-param", action="append", metavar="MODEL.KEY=VALUE",
                   help="override any hyperparameter, e.g. random_forest.max_depth=5 (repeatable)")
    r.add_argument("--estimator", action="append", metavar="NAME=IMPORT.PATH",
                   help="add any estimator class, e.g. xt=sklearn.ensemble.ExtraTreesClassifier")
    r.add_argument("--model-file", action="append", metavar="NAME=PATH",
                   help="evaluate your own pre-trained model (joblib/pickle) as-is")
    r.add_argument("--tools", help="comma-separated subset of tools (default: all available)")
    r.add_argument("--tool-param", action="append", metavar="TOOL.KEY=VALUE",
                   help="override a tool's parameter, e.g. robustness.sigma=0.2 (repeatable)")
    r.add_argument("--weights", help="e.g. fairness=0.5,privacy=0.2 (others keep defaults)")
    r.add_argument("--test-size", type=float, default=0.3)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--no-scale", action="store_true",
                   help="don't standardise features (use with --model-file trained on raw data)")
    r.add_argument("--checklist", help="answers JSON from `ethikit checklist <id> --out`")
    r.add_argument("--out", default="ethikit_out", help="output directory")
    r.add_argument("--trace", action="store_true", help="print every tool's formula + worked steps")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(fn=cmd_run)

    ls = sub.add_parser("list", help="list datasets, models, tools, params or checklists")
    ls.add_argument("what", choices=["datasets", "models", "tools", "params", "checklists"])
    ls.add_argument("name", nargs="?")
    ls.set_defaults(fn=cmd_list)

    c = sub.add_parser("checklist", help="governance checklist template")
    c.add_argument("id", choices=["deon", "altai", "datasheets", "nist_ai_rmf"])
    c.add_argument("--out")
    c.add_argument("--markdown", action="store_true")
    c.set_defaults(fn=cmd_checklist)

    s = sub.add_parser("serve", help="start the live web console")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action="store_true", help="open a browser tab")
    s.set_defaults(fn=cmd_serve)

    e = sub.add_parser("export-site", help="copy the console into the docs site with a replay")
    e.add_argument("dir", nargs="?", default="docs")
    e.add_argument("--models", default="logistic_regression,decision_tree,random_forest,naive_bayes,knn")
    e.set_defaults(fn=cmd_export_site)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.ERROR)
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(R(f"error: {exc}"), file=sys.stderr)
        return 2
