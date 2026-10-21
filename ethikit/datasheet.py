"""Datasheet for the dataset (after Gebru et al., "Datasheets for Datasets", CACM 2021).

A datasheet documents the data a model was trained and evaluated on, in the seven sections of
the original: motivation, composition, collection process, preprocessing, uses, distribution,
maintenance. The questions below are a condensed selection of the original's.

ethikit fills in what it can actually know and leaves the rest to you:
  - from the data profile (computed on the data as loaded): size, label balance, groups and their
    base rates, missing values, the proxy screen, warnings;
  - from the run config: the split, scaling, encoding, whether the sensitive column was a feature;
  - for the bundled public datasets only: short published facts (with the URL to check them).
Motivation, collection, consent, licence and maintenance are questions only the people who made
or chose the data can answer. Each answer records who filled it in ("ethikit" or "you").

    sheet = build_datasheet(profile, config)          # pre-filled
    sheet = merge_answers(sheet, {"motivation.purpose": "..."})
    to_markdown(sheet) / to_html(sheet) / json.dumps(sheet)
"""
from __future__ import annotations

import datetime as _dt
import html
import json
from typing import Any, Dict, List, Optional

SOURCE = ("Gebru, Morgenstern, Vecchione, Wortman Vaughan, Wallach, Daumé III, Crawford. "
          "Datasheets for Datasets. Communications of the ACM 64(12), 2021. arXiv:1803.09010")

# (id, section, question) -- a condensed selection of the original's questions, in its order
QUESTIONS = [
    ("motivation.purpose", "Motivation", "For what purpose was the dataset created? Was there a specific task or gap in mind?"),
    ("motivation.creators", "Motivation", "Who created the dataset, on behalf of which entity, and who funded it?"),
    ("composition.instances", "Composition", "What do the instances represent (people, loans, cases, ...)?"),
    ("composition.count", "Composition", "How many instances are there, and is this a sample of a larger set?"),
    ("composition.label", "Composition", "Is there a label or target for each instance? How is it distributed?"),
    ("composition.features", "Composition", "What data does each instance consist of?"),
    ("composition.missing", "Composition", "Is any information missing from individual instances?"),
    ("composition.subpopulations", "Composition", "Does the dataset identify sub-populations (e.g. by age or sex)? How are they distributed?"),
    ("composition.sensitive", "Composition", "Does it contain data that might be considered sensitive, or that could identify people directly or indirectly?"),
    ("composition.splits", "Composition", "Are there recommended data splits (training / test)?"),
    ("collection.mechanism", "Collection process", "How was the data collected, over what time frame, and how were instances sampled?"),
    ("collection.consent", "Collection process", "Were the people concerned notified, and did they consent? Was an ethical review carried out?"),
    ("preprocessing.steps", "Preprocessing / cleaning / labelling", "What preprocessing, cleaning or labelling was done?"),
    ("preprocessing.raw", "Preprocessing / cleaning / labelling", "Is the raw data kept, and is the preprocessing software available?"),
    ("uses.other_tasks", "Uses", "What other tasks could the dataset be used for, and what has it already been used for?"),
    ("uses.risks", "Uses", "Is there anything about the composition or collection that might lead to unfair treatment of individuals or groups if used naively?"),
    ("uses.not_for", "Uses", "Are there tasks the dataset should NOT be used for?"),
    ("distribution.licence", "Distribution", "Under what licence or terms of use is it distributed?"),
    ("maintenance.owner", "Maintenance", "Who maintains it, how can they be contacted, and will it be updated (errata, deletion requests)?"),
]

# Published facts about the public datasets ethikit can load itself. Short, sourced, and
# marked so a reader knows where to verify them.
KNOWN = {
    "adult": {
        "url": "https://archive.ics.uci.edu/dataset/2/adult",
        "motivation.purpose": "A benchmark for predicting whether a person's income exceeds $50K/year.",
        "motivation.creators": "Extracted by Barry Becker from the 1994 US Census database; donated to the UCI "
                               "Machine Learning Repository by Ronny Kohavi and Barry Becker (1996).",
        "composition.instances": "One adult person from the 1994 US Census extract.",
        "collection.mechanism": "US Census Bureau survey records, filtered by the extractors to 'reasonably "
                                "clean' records (per the UCI description).",
        "distribution.licence": "CC BY 4.0 (UCI Machine Learning Repository).",
        "maintenance.owner": "UCI Machine Learning Repository; ethikit loads it from OpenML ('adult', version 2).",
    },
    "german_credit": {
        "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        "motivation.purpose": "A benchmark for classifying credit applicants as good or bad credit risks.",
        "motivation.creators": "Prof. Dr. Hans Hofmann, Universität Hamburg (Statlog project, 1994).",
        "composition.instances": "One credit applicant.",
        "distribution.licence": "CC BY 4.0 (UCI Machine Learning Repository).",
        "maintenance.owner": "UCI Machine Learning Repository; ethikit loads it from OpenML ('credit-g', version 1).",
    },
    "synthetic": {
        "url": "", "by": "ethikit (it generated this data)",
        "motivation.purpose": "Generated by ethikit to demonstrate the audit: the group attribute leaks into the "
                              "label and into a proxy feature on purpose.",
        "motivation.creators": "Generated locally by ethikit's synthetic data generator (no real source).",
        "composition.instances": "Simulated cases; no real person is represented.",
        "collection.mechanism": "Simulated, deterministic for a fixed seed.",
        "collection.consent": "Not applicable: no real people.",
        "distribution.licence": "Generated on your machine; no licence restrictions.",
        "maintenance.owner": "ethikit (ethikit/data/loaders.py).",
    },
}


def _known_key(dataset_name: str) -> str:
    """Facts follow the data actually loaded (e.g. Adult falls back to synthetic data offline)."""
    if dataset_name == "adult_income":
        return "adult"
    if dataset_name == "german_credit":
        return "german_credit"
    return "synthetic" if dataset_name.startswith("synthetic") else ""


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _prefill(profile: Dict[str, Any], config: Dict[str, Any], feature_names: Optional[List[str]]) -> Dict[str, str]:
    a: Dict[str, str] = {}
    data = config.get("data") or {}
    split = {"test_size": 0.3, "seed": 0, "scale": True, **(config.get("split") or {})}
    n, task = profile.get("n_rows"), profile.get("task")
    ts = float(split["test_size"])
    a["composition.count"] = (f"{n} instances as loaded by ethikit. Whether this is a sample of a larger "
                              f"set: see the source." if n is not None else "")
    if profile.get("positive_rate") is not None:
        a["composition.label"] = (f"Yes ({task}). {_pct(profile['positive_rate'])} of instances are in the "
                                  f"positive (favourable) class.")
    elif task:
        a["composition.label"] = f"Yes ({task})."
    nf = profile.get("n_features")
    if feature_names:
        shown = ", ".join(feature_names[:40]) + (f", ... ({len(feature_names) - 40} more)" if len(feature_names) > 40 else "")
        a["composition.features"] = f"{len(feature_names)} model features after encoding: {shown}."
    elif nf:
        a["composition.features"] = f"{nf} model features after encoding."
    if "missing" in profile:                      # only knowable from the original frame (CSV / upload)
        miss = profile["missing"]
        a["composition.missing"] = ("Share of each column missing before ethikit filled the gaps (numeric: "
                                    "median; categorical: its own level): " + "; ".join(
                                        f"{m['column']} {_pct(m['share_missing'])}" for m in miss) + "."
                                    if miss else "ethikit's profile found no missing values in the file.")
    elif data.get("source", "synthetic") in ("synthetic", "synthetic_custom", "regression"):
        a["composition.missing"] = "None: simulated data."
    groups = profile.get("groups") or []
    if profile.get("sensitive") and groups:
        rows = "; ".join(f"{g['group']}: {g['n']} ({_pct(g['share'])})" +
                         (f", base rate {_pct(g['base_rate'])}" if g.get("base_rate") is not None else "")
                         for g in groups)
        gap = profile.get("base_rate_gap")
        a["composition.subpopulations"] = (f"Yes, by '{profile['sensitive']}': {rows}." +
                                           (f" Base rates differ by {gap:.2f} between groups." if gap is not None else ""))
    elif not profile.get("sensitive"):
        a["composition.subpopulations"] = "No sensitive attribute was named for this run, so ethikit did not compare groups."
    proxy = profile.get("proxy") or {}
    if profile.get("sensitive"):
        s = f"The sensitive attribute is '{profile['sensitive']}'."
        if proxy.get("auc") is not None:
            tops = ", ".join(t["feature"] for t in proxy.get("top_features", [])[:3])
            s += (f" The other features predict it with AUC {proxy['auc']:.2f} (0.5 = no information; "
                  f"{proxy.get('method', '')})" + (f"; strongest proxies: {tops}" if tops and proxy["auc"] >= 0.6 else "") + ".")
        a["composition.sensitive"] = s + " Whether people can be re-identified: yours to answer."
    a["composition.splits"] = (f"ethikit used a random {_pct(1 - ts)} / {_pct(ts)} train / test split "
                               f"(seed {split['seed']})" + (f", and {split['cv'][0]}-fold x {split['cv'][1]} repeated "
                                                            f"cross-validation" if split.get("cv") else "") + ".")
    steps = []
    if data.get("source") in ("csv", "upload", "adult", "german_credit"):
        steps.append("categorical columns one-hot encoded (top 20 levels, rest grouped as 'other'); numeric "
                     "missing values filled with the median")
    if data.get("sensitive_threshold") not in (None, ""):
        steps.append(f"numeric sensitive column split into two groups at {data['sensitive_threshold']}")
    steps.append("features standardised (scaler fit on the training split only)" if split.get("scale", True)
                 else "features left unscaled")
    steps.append("the sensitive column was NOT given to the models" if data.get("exclude_sensitive")
                 else "the sensitive column was given to the models as a feature")
    a["preprocessing.steps"] = "By ethikit for this run: " + "; ".join(steps) + ". Earlier cleaning by the source: yours to answer."
    if profile.get("warnings"):
        a["uses.risks"] = "Found by ethikit's data profile: " + " ".join(w.rstrip(".") + "." for w in profile["warnings"])
    return a


def build_datasheet(profile: Dict[str, Any], config: Dict[str, Any],
                    feature_names: Optional[List[str]] = None, today: Optional[str] = None) -> Dict[str, Any]:
    """A datasheet dict, pre-filled where ethikit knows the answer. `profile` is profile_raw()."""
    known = KNOWN.get(_known_key(str(profile.get("dataset") or "")), {})
    auto = _prefill(profile, config, feature_names)
    items = []
    for qid, section, q in QUESTIONS:
        if qid in auto and auto[qid]:
            ans, by = auto[qid], "ethikit (computed from the data and run config)"
        elif qid in known:
            ans = known[qid]
            by = known.get("by") or f"ethikit (published facts about this dataset; verify at {known['url']})"
        else:
            ans, by = None, None
        items.append({"id": qid, "section": section, "question": q, "answer": ans, "filled_by": by})
    return {"title": f"Datasheet: {profile.get('dataset', 'dataset')}", "dataset": profile.get("dataset"),
            "after": SOURCE, "generated": today or _dt.date.today().isoformat(), "items": items}


def merge_answers(sheet: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
    """Your answers ({question id: text}) override ethikit's; empty strings are ignored."""
    unknown = set(answers) - {i["id"] for i in sheet["items"]}
    if unknown:
        raise KeyError(f"unknown datasheet question(s): {', '.join(sorted(unknown))}")
    out = json.loads(json.dumps(sheet))
    for it in out["items"]:
        if (answers.get(it["id"]) or "").strip():
            it["answer"], it["filled_by"] = answers[it["id"]].strip(), "you"
    return out


def blank_answers(sheet: Dict[str, Any]) -> Dict[str, str]:
    """The questions still open, as an answers template."""
    return {i["id"]: "" for i in sheet["items"] if not i["answer"]}


def completeness(sheet: Dict[str, Any]) -> Dict[str, int]:
    items = sheet["items"]
    return {"answered": sum(1 for i in items if i["answer"]), "total": len(items),
            "by_you": sum(1 for i in items if i["filled_by"] == "you")}


def to_markdown(sheet: Dict[str, Any]) -> str:
    c = completeness(sheet)
    lines = [f"# {sheet['title']}", "", f"_After {sheet['after']}_", "",
             f"Generated {sheet['generated']} by ethikit · {c['answered']} of {c['total']} questions answered.", ""]
    sec = None
    for it in sheet["items"]:
        if it["section"] != sec:
            sec = it["section"]
            lines += [f"## {sec}", ""]
        lines.append(f"**{it['question']}** `{it['id']}`")
        lines.append("")
        lines.append(it["answer"] if it["answer"] else "_Not answered yet -- only the people who made or chose the data can answer this._")
        if it["filled_by"]:
            lines.append(f"<sub>filled by {it['filled_by']}</sub>")
        lines.append("")
    return "\n".join(lines)


def to_html(sheet: Dict[str, Any]) -> str:
    from .report.theme import BASE, head_assets
    esc = html.escape
    c = completeness(sheet)
    body, sec = [], None
    for it in sheet["items"]:
        if it["section"] != sec:
            sec = it["section"]
            body.append(f"<h2>{esc(sec)}</h2>")
        ans = (f"<p>{esc(it['answer'])}</p>" if it["answer"] else
               "<p class='open'>Not answered yet -- only the people who made or chose the data can answer this.</p>")
        by = f"<div class='by'>filled by {esc(it['filled_by'])}</div>" if it["filled_by"] else ""
        body.append(f"<div class='q'><b>{esc(it['question'])}</b> <code>{esc(it['id'])}</code>{ans}{by}</div>")
    style = BASE + """body{max-width:860px}h1{font-size:24px}.q{margin:14px 0}.q p{margin:4px 0}
.open{color:var(--gold);font-style:italic}.by{font-size:12px;color:var(--ink-faint);font-family:var(--font-mono)}"""
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, "
            f"initial-scale=1'><title>{esc(sheet['title'])}</title>{head_assets('datasheet')}<style>{style}</style>"
            f"</head><body><main class='page'><h1>{esc(sheet['title'])}</h1><div class='sub'>after {esc(sheet['after'])}"
            f"<br>generated {esc(sheet['generated'])} · {c['answered']} of {c['total']} questions answered</div>"
            + "".join(body) + "</main></body></html>")
