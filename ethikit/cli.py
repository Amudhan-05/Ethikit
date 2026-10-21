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
        "split": {"test_size": a.test_size, "seed": a.seed, "scale": not a.no_scale,
                  "cv": list(_parse_cv(a.cv)) if a.cv else None},
        "models": models or None,
        "tools": a.tools.split(",") if a.tools else None,
        "tool_params": _kv_pairs(a.tool_param, "tool-param"),
        "weights": {**session.DEFAULT_WEIGHTS, **_weights(a.weights)},
    }
    if a.checklist:
        from . import checklists
        cfg["checklist"] = checklists.load_answers(a.checklist)
    if a.require:
        cfg["requirements"] = list(a.require)
    if a.no_save:
        cfg["save"] = False
    return cfg


def load_config_file(path: str) -> dict:
    """A config.json, a results.json (has a "config" key), or a saved run id / folder."""
    from .runs import load_run, runs_dir
    p = path
    if not os.path.exists(p) and (runs_dir() / path).is_dir():
        return load_run(path)["config"]
    if os.path.isdir(p):
        p = os.path.join(p, "config.json")
    with open(p, encoding="utf-8") as fh:
        data = json.load(fh)
    cfg = data.get("config", data) if isinstance(data, dict) else None
    if not isinstance(cfg, dict) or "data" not in cfg:
        raise ValueError(f"{path} doesn't look like an ethikit config (no 'data' section)")
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

    if a.config:
        cfg = load_config_file(a.config)
        if a.no_save:
            cfg["save"] = False
        print(D(f"config loaded from {a.config} (other run flags are ignored)"))
    else:
        cfg = config_from_args(a)
    if a.export_config:
        with open(a.export_config, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"wrote {a.export_config}  (re-run with: ethikit run --config {a.export_config})")
        return 0
    prepared = session.prepare(cfg)
    _print_profile(prepared.profile)
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
    if s.get("objective_std"):
        std = pd.DataFrame(s["objective_std"]).T
        cell = obj.round(3).astype(str).copy()
        for c in std.columns:
            if c in cell.columns:
                cell[c] = [f"{v:.2f} ±{(0 if pd.isna(std.loc[m, c]) else std.loc[m, c]):.2f}" if m in std.index else f"{v:.2f}"
                           for m, v in obj[c].items()]
        obj_show = cell
    else:
        obj_show = obj.round(3)
    print("\n" + B("comparison") + D("  (0-1 per column, 1 = best in this run" +
          (f"; mean ± std over {s['cv'][0]}x{s['cv'][1]} folds" if s.get("cv") else "") + ")"))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(obj_show.to_string())
    print("\n" + B("pareto front") + ": " + ", ".join(G(m) for m in s["pareto"]))
    _print_requirements(s.get("requirements"))
    _print_stability(s.get("stability"))
    if s.get("history_id"):
        h = s["history_id"]
        print(D(f"\nsaved as run {h}  (ethikit runs show {h} · ethikit run --config {h})"))

    os.makedirs(a.out, exist_ok=True)
    report = render_html(s["dataset"], obj, s["pareto"], os.path.join(a.out, "report.html"))
    cards = render_model_cards(s["dataset"], out["raw"], out["objective"], s["pareto"],
                               out_dir=os.path.join(a.out, "model_cards"),
                               sensitive_name=s["sensitive"], checklist=prepared.checklist,
                               hyperparameters=prepared.hyperparameters,
                               origins=prepared.origins, requirements=s.get("requirements"),
                               data_profile=s.get("profile"))
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


def _print_profile(p):
    if not p:
        return
    print(B("data profile") + D(f"  {p['n_rows']} rows · {p['n_features']} features · sensitive: {p['sensitive']}"
          + (f" · positive rate {p['positive_rate']}" if "positive_rate" in p else "")))
    for g in p.get("groups", []):
        extra = f"base rate {g['base_rate']}" if "base_rate" in g else f"mean target {g.get('mean_target')}"
        print(f"  {g['group']:<22} n={g['n']:<7} share={g['share']:<7} {extra}")
    px = p.get("proxy") or {}
    if px.get("auc") is not None:
        print(f"  proxy screen: other features predict the sensitive attribute with AUC {px['auc']}"
              + D("  (0.5 = no information)"))
    for w in p.get("warnings", []):
        print(Y("  ! " + w))
    print()


def _print_requirements(req):
    if not req:
        return
    print("\n" + B("requirements") + ": " + "; ".join(req["list"]))
    for m, v in req["by_model"].items():
        mark = G("meets all") if v["passes"] else R("fails")
        why = "" if v["passes"] else "  " + "; ".join(f"{x['requirement']}: {x['reason']}"
                                                      for x in v["results"] if not x["passes"])
        print(f"  {m:<22} {mark}{why}")
    print(("  feasible: " + ", ".join(G(m) for m in req["feasible"])) if req["feasible"]
          else R("  no model meets every requirement -- relax one, or widen the candidate set"))


def _print_stability(st):
    if not st:
        return
    print("\n" + B("how stable is the order?"))
    if not st.get("available"):
        print(D("  " + st.get("reason", "not available")))
        return
    print(f"  first under your weights: {G(st['first'])}; {st['nearby_weights_keep_first']:.0%} of "
          f"nearby weight settings keep it first" +
          (f" (otherwise: {', '.join(st['main_rivals'])})" if st["main_rivals"] else ""))
    for v in st["per_principle"].values():
        print("  " + v["sentence"])


def cmd_profile(a: argparse.Namespace) -> int:
    from . import session
    from .profile import profile_raw
    cfg = config_from_args(a)
    raw = session.load_raw(cfg["data"])
    frame = None
    if cfg["data"].get("source") == "csv":
        import pandas as pd
        frame = pd.read_csv(cfg["data"]["path"])
    p = profile_raw(raw, frame, seed=a.seed)
    _print_profile(p)
    px = p.get("proxy") or {}
    if px.get("top_features"):
        print(B("strongest single-feature links to the sensitive attribute") + D("  (mutual information)"))
        for f in px["top_features"]:
            print(f"  {f['feature']:<30} {f['mutual_information']}")
    if p.get("missing"):
        print(B("missing values before imputation"))
        for m in p["missing"]:
            print(f"  {m['column']:<30} {m['share_missing']:.1%}")
    return 0


def cmd_runs(a: argparse.Namespace) -> int:
    from .runs import diff_runs, list_runs, load_run, runs_dir
    if a.action == "list":
        runs = list_runs()
        if not runs:
            print(D(f"no saved runs yet in {runs_dir()}"))
        for r in runs:
            print(f"{G(r['run_id'])}  {r.get('dataset')}  models={','.join(r.get('models', []))}"
                  + (f"  cv={r['cv']}" if r.get("cv") else ""))
        return 0
    if a.action == "show":
        if not a.ids:
            raise SystemExit("usage: ethikit runs show <run_id>")
        r = load_run(a.ids[0])
        print(json.dumps({"meta": r["meta"], "config": r["config"],
                          "pareto": r["summary"].get("pareto"), "objective": r["summary"].get("objective"),
                          "requirements": (r["summary"].get("requirements") or {}).get("feasible")}, indent=2))
        return 0
    if a.action == "diff":
        if len(a.ids) != 2:
            raise SystemExit("usage: ethikit runs diff <run_a> <run_b>")
        d = diff_runs(*a.ids)
        if d["identical"]:
            print(G("identical: same config, same scores, same metrics"))
            return 0
        print(B("config changes"))
        for c in d["config_changes"] or [{"key": "(none)", "a": "", "b": ""}]:
            print(f"  {c['key']:<34} {c['a']!s:<20} -> {c['b']}")
        print(B("score changes") + D("  (0-1 profile; models in both runs)"))
        for c in d["score_changes"][:40]:
            print(f"  {c['model']:<22} {c['principle']:<15} {c['a']} -> {c['b']}  ({c['delta']:+})" if c["delta"] is not None
                  else f"  {c['model']:<22} {c['principle']:<15} {c['a']} -> {c['b']}")
        if d["models_only_in_a"] or d["models_only_in_b"]:
            print(B("models") + f"  only in a: {d['models_only_in_a']}  only in b: {d['models_only_in_b']}")
        print(B("pareto front") + f"  a: {d['pareto_a']}   b: {d['pareto_b']}")
        return 0
    return 2


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
            hint = "" if t["available"] or not t["install_hint"] else D(f"   -> {t['install_hint']}")
            print(f"{B(t['name']):<30} {flag:<20} principles: {', '.join(t['principles'])}{hint}")
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


def _parse_cv(s):
    if not s:
        return None
    try:
        folds, repeats = s.lower().split("x")
        return int(folds), int(repeats)
    except ValueError:
        raise SystemExit(f"--cv expects FOLDSxREPEATS, e.g. 5x3 (got {s!r})")


def _family_pivot(result):
    """family x principle table of 'mean +/- std' strings, principles in a stable order."""
    import pandas as pd
    ft = result.family_table
    order = [p for p in ["utility", "fairness", "inclusiveness", "explainability", "transparency",
                         "privacy", "safety"] if p in ft.index.get_level_values("principle")]
    cell = ft.apply(lambda r: f"{r['mean']:.2f} ±" + ("–" if pd.isna(r["std"]) else f"{r['std']:.2f}"), axis=1)
    return cell.unstack("principle")[order]


def cmd_domains(a: argparse.Namespace) -> int:
    import pandas as pd
    from .domains import (STRENGTH_TEXT, compare_domains, get_domains, run_domain, shortlist)
    from .domains.runner import DomainError
    states = tuple(s for s in (a.states or "").split(",") if s) or None
    doms = get_domains(*([states] if states else []))
    if a.action == "list":
        for d in doms.values():
            print(f"{G(d.id):<22} {B(d.title)}  {D('[' + d.strength + ', ' + str(len(d.datasets)) + ' datasets]')}")
            print(f"{'':<14}{d.blurb}")
        print(D("\nethikit domains info <id>  ·  ethikit domains run <id>"))
        return 0
    if not a.id:
        raise SystemExit("usage: ethikit domains " + a.action + " <domain-id>   (see: ethikit domains list)")
    if a.id not in doms:
        raise SystemExit(f"unknown domain {a.id!r}; choose from {', '.join(doms)}")
    d = doms[a.id]
    if a.action == "info":
        print(B(d.title) + D(f"   [{d.strength}: {STRENGTH_TEXT[d.strength]}]"))
        print(d.blurb + ("\nFor: " + d.for_whom if d.for_whom else ""))
        for s in d.datasets:
            flag = G("verified") if s.verified else Y("first run on your machine")
            print(f"  - {B(s.id)}  {s.title}\n      source: {s.source}\n      sensitive: {s.sensitive}  "
                  f"needs: {s.needs}  [{flag}]" + (f"\n      note: {s.note}" if s.note else ""))
        print(Y("\nCaveats:") + "".join(f"\n  ! {c}" for c in d.caveats))
        return 0

    log = LiveLog(False)

    def on_event(e):
        t = D(f"{time.perf_counter() - log.t0:7.1f}s")
        if e["type"] == "domain_start":
            print(f"{t} {B('suite')} {e['domain']}  datasets={len(e['datasets'])} models={','.join(e['models'])}"
                  f" tools={','.join(e['tools'])}" + (f" cv={e['cv'][0]}x{e['cv'][1]}" if e["cv"] else ""))
        elif e["type"] == "dataset_start":
            print(f"{t} {B('data')}  [{e['i']}/{e['n']}] {e['title']}")
        elif e["type"] == "dataset_done":
            print(f"{t}       {G('✓')} {e['n_rows']} rows, positive rate {e['positive_rate']}, "
                  f"{e['seconds']}s" + ("" if e["verified"] else D("  (first real run of this loader)")))
        elif e["type"] == "dataset_skip":
            print(f"{t}       {R('✗ skipped')} {e['reason']}")
    try:
        res = run_domain(d, models=a.models.split(",") if a.models else None,
                         tools=a.tools.split(",") if a.tools else None, max_datasets=a.max_datasets,
                         max_rows=a.max_rows, seed=a.seed, cv=_parse_cv(a.cv), on_event=on_event,
                         use_cache=not a.no_cache)
    except DomainError as exc:
        print(R(f"error: {exc}"), file=sys.stderr)
        return 2
    print("\n" + B(f"family evidence — {res.title}") + D(f"  (mean ± std over {len(res.datasets)} dataset(s); "
          "each column 0-1, relative to the models in this suite)"))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(_family_pivot(res).to_string())
    ab = res.absolute_table
    if not ab.empty:
        from .domains.runner import KEY_METRICS
        print("\n" + B("the same families in real units") + D("  (mean over datasets; accuracy/AUC/stability/fidelity: higher is "
              "better; gaps and MI advantage: lower is better)"))
        print(ab.rename(columns=KEY_METRICS).round(3).to_string())
    print("\n" + B("leading family per principle") + D("  (and on what share of datasets it also leads there)"))
    for p, f in res.top_family.items():
        print(f"  {p:<15} {G(str(f)):<24} " + ("–" if len(res.datasets) < 2 else f"{res.agreement[p]:.0%}"))
    print("\n" + B("does the ordering hold up?") + D("  (Friedman over datasets)"))
    for p, s in res.significance.items():
        print(f"  {p:<15} " + (D("skipped: " + s["skipped"]) if "skipped" in s else
              f"p={s['p_value']}  {'significant' if s['significant_at_0.05'] else 'not significant'}"))
    sl = shortlist(res)
    print("\n" + B("starting shortlist for your own data") + f": {', '.join(G(f) for f in sl['families'])}"
          f"  ({', '.join(sl['models'])})\n  " + D("weights: " + ", ".join(f"{k}={v}" for k, v in sl["weights_used"].items())
          + ". " + sl["caveat"]))
    for n in res.notes:
        print(Y("note: " + n))
    print(Y("\nCaveats:") + "".join(f"\n  ! {c}" for c in res.caveats))
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"domain_{res.domain}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({**res.to_dict(), "shortlist": sl}, fh)
    print(f"\n{D('wrote')} {path}")
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
    r.add_argument("--cv", help="repeated k-fold instead of one split, e.g. 5x3 = 5 folds x 3 repeats; "
                   "every score becomes a mean with a spread")
    r.add_argument("--no-scale", action="store_true",
                   help="don't standardise features (use with --model-file trained on raw data)")
    r.add_argument("--checklist", help="answers JSON from `ethikit checklist <id> --out`")
    r.add_argument("--require", action="append", metavar="METRIC<=VALUE",
                   help="hard requirement on a raw metric, e.g. demographic_parity_diff<=0.1 or "
                        "aif360.theil_index<0.2 (repeatable); failing models are listed with the reason")
    r.add_argument("--config", help="run exactly this config: a config.json, a results.json, or a saved run id")
    r.add_argument("--export-config", metavar="PATH", help="write the config these flags describe and exit")
    r.add_argument("--no-save", action="store_true", help="don't save this run to the local run history")
    r.add_argument("--out", default="ethikit_out", help="output directory")
    r.add_argument("--trace", action="store_true", help="print every tool's formula + worked steps")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(fn=cmd_run)

    pf = sub.add_parser("profile", help="profile a dataset (groups, base rates, proxies, missingness) without training")
    pf.add_argument("--data", default="synthetic")
    pf.add_argument("--target"); pf.add_argument("--sensitive"); pf.add_argument("--sensitive-threshold", type=float)
    pf.add_argument("--positive-label"); pf.add_argument("--exclude-sensitive", action="store_true")
    pf.add_argument("--synthetic", nargs="+", metavar="KEY=VALUE")
    pf.add_argument("--seed", type=int, default=0)
    for k, v in dict(models=None, family=None, model_param=None, estimator=None, model_file=None, tools=None,
                     tool_param=None, weights=None, test_size=0.3, no_scale=False, cv=None, checklist=None,
                     require=None, no_save=True).items():
        pf.set_defaults(**{k: v})
    pf.set_defaults(fn=cmd_profile)

    rh = sub.add_parser("runs", help="saved run history: list, show, diff")
    rh.add_argument("action", choices=["list", "show", "diff"])
    rh.add_argument("ids", nargs="*")
    rh.set_defaults(fn=cmd_runs)

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

    dm = sub.add_parser("domains", help="reference suites: evidence per domain for choosing a starting model")
    dm.add_argument("action", choices=["list", "info", "run"])
    dm.add_argument("id", nargs="?", help="domain id (see `ethikit domains list`)")
    dm.add_argument("--models", help="comma-separated zoo models (default: a representative set)")
    dm.add_argument("--tools", help="comma-separated tools (default: a fast set without the adversarial attack)")
    dm.add_argument("--max-datasets", type=int, help="run only the first N datasets of the suite")
    dm.add_argument("--max-rows", type=int, default=5000, help="cap rows per dataset (default 5000)")
    dm.add_argument("--states", help="census suite: comma-separated US state codes (default CA,TX,NY)")
    dm.add_argument("--cv", help="repeated k-fold per dataset, e.g. 5x3 (slower; steadier numbers)")
    dm.add_argument("--seed", type=int, default=0)
    dm.add_argument("--no-cache", action="store_true", help="re-download instead of using the local cache")
    dm.add_argument("--out", default="ethikit_out")
    dm.set_defaults(fn=cmd_domains)

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
