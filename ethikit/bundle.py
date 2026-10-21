"""Audit bundle: one folder (or zip) a reviewer can use to reproduce or challenge every number.

    config.json            the exact run config; `ethikit run --config config.json` replays it
    results.json           everything the run computed (scores, Pareto front, requirements,
                           stability, data profile, every tool's trace)
    metrics.csv            the raw metrics, one row per model x tool x metric
    traces.json            each tool's formula, worked steps and intermediate tables
    report.html            the comparison report
    model_cards/           one card per model: <model>.html and <model>.mct.json (Model Card Toolkit)
    datasheet.{md,html,json} + datasheet_answers.json   the dataset's datasheet and its open questions
    checklist/             governance checklist answers (.json + .md), if any were given
    environment.json       Python, platform and library versions; which tools could run
    README.md              what all this is and how to reproduce it
    MANIFEST.sha256        a checksum per file, so later edits are detectable

`verify_bundle(path)` re-runs config.json and compares every metric with metrics.csv, and checks
the checksums. Everything is built from JSON (config + summary), so a bundle can be made from a
live run, the console or a saved run alike.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import __version__

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "fairlearn", "aif360", "shap",
            "adversarial-robustness-toolbox", "giskard", "alibi-detect", "torch", "captum", "opacus",
            "folktables"]
METRIC_COLUMNS = ["model", "evaluator", "principle", "target", "metric", "value", "higher_is_better"]


class _Repr(str):
    """A hyperparameter already stored as its repr; keep it as-is when the card repr()s it."""

    def __repr__(self):
        return str(self)


def environment() -> Dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version
    from .evaluators.base import all_evaluators
    pkgs = {}
    for name in PACKAGES:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None
    return {"ethikit": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": pkgs,
            "tools": {ev.name: ("available" if ev.available() else f"not installed ({ev.install_hint})")
                      for ev in all_evaluators()}}


def _metrics_rows(summary: dict) -> List[dict]:
    return [{k: r.get(k) for k in METRIC_COLUMNS} for r in summary.get("records") or []
            if r.get("target") != "error"]


def _frames(summary: dict):
    raw = pd.DataFrame(summary.get("records") or [])
    raw.attrs["traces"] = summary.get("traces") or {}
    obj = pd.DataFrame(summary["objective"]).T.astype(float)
    return raw, obj


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _readme(config: dict, summary: dict, files: List[str], env: dict, sheet_done: dict,
            checklist: Optional[dict]) -> str:
    models = ", ".join(summary.get("objective", {}).keys())
    req = summary.get("requirements") or {}
    cl = (f"`{checklist['checklist']}` checklist, {sum(1 for a in checklist.get('answers', {}).values() if (a or {}).get('answer'))} item(s) answered"
          if checklist else "none attached (answer one with `ethikit checklist deon` or the console's governance tab)")
    lines = [
        f"# ethikit audit bundle: {summary.get('dataset')}", "",
        "Model selection and audit on principles of ethicality -- not a model ranker. This folder holds",
        "everything one run computed and everything needed to reproduce or challenge it.", "",
        f"- **Dataset:** {summary.get('dataset')} ({summary.get('task')}), sensitive attribute: {summary.get('sensitive') or 'none'}",
        f"- **Models:** {models}",
        f"- **Pareto front** (no other candidate at least as good on every principle): {', '.join(summary.get('pareto') or [])}",
        "- **Hard requirements:** " + (", ".join(req.get("list") or []) + " -- met by: " +
                                       (", ".join(req.get("feasible") or []) or "no model") if req else "none"),
        f"- **Governance checklist:** {cl}",
        f"- **Datasheet:** {sheet_done['answered']} of {sheet_done['total']} questions answered "
        f"({sheet_done['by_you']} by you); open ones are in `datasheet_answers.json`",
        f"- **Made with:** ethikit {env['ethikit']}, Python {env['python']}", "",
        "## Reproduce it", "",
        "```bash",
        "pip install -r requirements.txt            # from the ethikit repository, same versions as environment.json",
        "ethikit bundle verify .                     # re-runs config.json and compares every number with metrics.csv",
        "ethikit run --config config.json --out rerun   # or re-run it yourself and compare by hand",
        "```", "",
        "`verify` also checks `MANIFEST.sha256`, so a number edited after the bundle was made is reported.",
        "Tools that were not installed when the bundle was made are listed in `environment.json`; with",
        "them installed, a re-run has more metrics than the bundle (verify reports them as extra, not wrong).", "",
        "## What is inside", "",
        "| File | What it is |", "|---|---|",
    ]
    desc = {
        "config.json": "the exact run config (`ethikit run --config config.json` replays it)",
        "results.json": "everything the run computed: scores, Pareto front, requirements, stability, data profile, traces",
        "metrics.csv": "raw metrics, one row per model x tool x metric (the numbers to challenge)",
        "traces.json": "each tool's formula, worked steps with this run's numbers, intermediate tables",
        "report.html": "the comparison report",
        "datasheet.md": "datasheet for the dataset (after Gebru et al.), pre-filled where ethikit knows the answer",
        "datasheet.html": "the same datasheet, as a page",
        "datasheet.json": "the same datasheet, as data",
        "datasheet_answers.json": "the datasheet's open questions: fill in and pass to `ethikit datasheet --answers`",
        "environment.json": "Python, platform, library versions, which tools could run",
        "MANIFEST.sha256": "a checksum per file",
    }
    for f in files:
        d = desc.get(f) or ("model card (HTML)" if f.endswith(".html") and f.startswith("model_cards/") else
                            "model card in the Model Card Toolkit JSON schema" if f.endswith(".mct.json") else
                            "governance checklist answers" if f.startswith("checklist/") else "")
        lines.append(f"| `{f}` | {d} |")
    return "\n".join(lines) + "\n"


def write_bundle(dest: str, config: dict, summary: dict, checklist: Optional[dict] = None,
                 datasheet_answers: Optional[Dict[str, str]] = None) -> str:
    """Write the bundle into `dest` (a folder; a path ending in .zip makes a zip). Returns the path."""
    from . import checklists as checklists_mod
    from . import datasheet as ds_mod
    from .report import render_model_cards, report_html

    as_zip = str(dest).endswith(".zip")
    root = Path(tempfile.mkdtemp(prefix="ethikit-bundle-")) if as_zip else Path(dest)
    root.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in config.items() if k != "save"}
    if checklist is None:
        checklist = config.get("checklist")
    raw, obj = _frames(summary)

    (root / "config.json").write_text(json.dumps(config, indent=2))
    (root / "results.json").write_text(json.dumps({"config": config, **summary}))
    with open(root / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS)
        w.writeheader()
        w.writerows(_metrics_rows(summary))
    (root / "traces.json").write_text(json.dumps(summary.get("traces") or {}, indent=1))
    (root / "report.html").write_text(report_html(summary["dataset"], obj, summary["pareto"]), encoding="utf-8")
    hp = {m: {k: _Repr(v) for k, v in p.items()} for m, p in (summary.get("hyperparameters") or {}).items()}
    render_model_cards(summary["dataset"], raw, obj, summary["pareto"], out_dir=str(root / "model_cards"),
                       sensitive_name=summary.get("sensitive"), checklist=checklist,
                       hyperparameters=hp or None, origins=summary.get("origins"),
                       requirements=summary.get("requirements"), data_profile=summary.get("profile"))
    sheet = ds_mod.build_datasheet(summary.get("profile") or {"dataset": summary["dataset"]}, config,
                                   summary.get("feature_names"))
    if datasheet_answers:
        sheet = ds_mod.merge_answers(sheet, datasheet_answers)
    (root / "datasheet.md").write_text(ds_mod.to_markdown(sheet), encoding="utf-8")
    (root / "datasheet.html").write_text(ds_mod.to_html(sheet), encoding="utf-8")
    (root / "datasheet.json").write_text(json.dumps(sheet, indent=2))
    (root / "datasheet_answers.json").write_text(json.dumps(ds_mod.blank_answers(sheet), indent=2))
    if checklist and checklist.get("checklist") in checklists_mod.CHECKLISTS:
        (root / "checklist").mkdir(exist_ok=True)
        (root / "checklist" / f"{checklist['checklist']}_answers.json").write_text(json.dumps(checklist, indent=2))
        (root / "checklist" / f"{checklist['checklist']}.md").write_text(
            checklists_mod.to_markdown(checklist["checklist"], checklist.get("answers")), encoding="utf-8")
    env = environment()
    (root / "environment.json").write_text(json.dumps(env, indent=2))

    files = sorted(str(p.relative_to(root)).replace(os.sep, "/") for p in root.rglob("*")
                   if p.is_file() and p.name not in ("README.md", "MANIFEST.sha256"))
    (root / "README.md").write_text(_readme(config, summary, files + ["MANIFEST.sha256"], env,
                                            ds_mod.completeness(sheet), checklist), encoding="utf-8")
    manifest = "".join(f"{_sha(root / f)}  {f}\n" for f in files + ["README.md"])
    (root / "MANIFEST.sha256").write_text(manifest)

    if as_zip:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, Path("ethikit_bundle") / p.relative_to(root))
        shutil.rmtree(root, ignore_errors=True)
        return str(dest)
    return str(root)


def bundle_bytes(config: dict, summary: dict, checklist: Optional[dict] = None,
                 datasheet_answers: Optional[Dict[str, str]] = None) -> bytes:
    """The bundle as zip bytes (for the console's download button)."""
    tmp = Path(tempfile.mkdtemp(prefix="ethikit-zip-")) / "bundle.zip"
    try:
        write_bundle(str(tmp), config, summary, checklist, datasheet_answers)
        return tmp.read_bytes()
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def _open(path: str) -> Path:
    p = Path(path)
    if p.suffix == ".zip":
        out = Path(tempfile.mkdtemp(prefix="ethikit-verify-"))
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():                       # refuse paths escaping the folder
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"unsafe path in zip: {name}")
            z.extractall(out)
        inner = out / "ethikit_bundle"
        return inner if inner.is_dir() else out
    return p


def verify_bundle(path: str, rerun: bool = True, tol: float = 1e-9) -> Dict[str, Any]:
    """Check the checksums, then (rerun=True) re-run config.json and compare every metric."""
    root = _open(path)
    report: Dict[str, Any] = {"bundle": str(path), "tampered": [], "missing_files": [], "mismatches": [],
                              "only_in_bundle": [], "only_in_rerun": [], "compared": 0}
    man = root / "MANIFEST.sha256"
    if not man.exists():
        report["missing_files"].append("MANIFEST.sha256")
    else:
        for line in man.read_text().splitlines():
            digest, _, name = line.partition("  ")
            f = root / name
            if not f.exists():
                report["missing_files"].append(name)
            elif _sha(f) != digest:
                report["tampered"].append(name)
    if rerun:
        from . import session
        config = json.loads((root / "config.json").read_text())
        config["save"] = False
        summary = session.execute(session.prepare(config))["summary"]
        key = lambda r: (r["model"], r["evaluator"], r["metric"])          # noqa: E731
        new = {key(r): r["value"] for r in _metrics_rows(summary)}
        with open(root / "metrics.csv", encoding="utf-8") as fh:
            old = {key(r): r["value"] for r in csv.DictReader(fh)}
        for k in sorted(set(old) | set(new)):
            if k not in new:
                report["only_in_bundle"].append(list(k))
            elif k not in old:
                report["only_in_rerun"].append(list(k))
            else:
                report["compared"] += 1
                a, b = old[k], new[k]
                if a in ("", None) and b in ("", None) or (b is not None and b != b and a in ("", "nan")):
                    continue                                  # both unmeasured (NaN)
                try:
                    same = abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
                except (TypeError, ValueError):
                    same = str(a) == str(b)
                if not same:
                    report["mismatches"].append({"metric": list(k), "bundle": a, "rerun": b})
    report["ok"] = not (report["tampered"] or report["missing_files"] or report["mismatches"]
                        or report["only_in_bundle"])
    return report
