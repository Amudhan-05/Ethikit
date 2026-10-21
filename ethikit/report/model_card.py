"""Per-model Model Cards -- the accountability + transparency deliverable.

Follows the Model Cards schema (Mitchell et al. 2019 / Google's Model Card
Toolkit): model details, intended use, quantitative analysis, ethical
considerations, caveats & recommendations. One card per model, so it can sit
alongside (or replace) the single aggregate `ethikit_report.html`.

Like the evaluators, this prefers the real `model-card-toolkit` package: when
it is importable we also populate its `ModelCard` dataclass and export the
library's own JSON schema as a `.mct.json` sidecar (best-effort -- wrapped so
any field mismatch across mct versions never blocks the deliverable). As of
this writing `model-card-toolkit` does not install on Python 3.12 (its
`absl-py` pin fails to build), the same kind of version constraint noted for
AIF360 in HANDOFF.md, so the hand-rolled renderer below is what actually
runs today and needs no optional dependency at all.
"""
from __future__ import annotations

import html
import os
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from .theme import BASE, head_assets
from .. import checklists as checklists_mod
from .. import coverage as coverage_mod
from ..evaluators.base import all_evaluators
from .. import principles as principles_mod
from ..benchmark.models import family_of, flat_classifiers


def _has_mct() -> bool:
    try:
        import model_card_toolkit  # noqa: F401
        return True
    except Exception:
        return False


def _model_hyperparams(model_name: str) -> Dict[str, str]:
    """Instantiate (but don't fit) the model's own factory to read its config."""
    factories = flat_classifiers()
    if model_name not in factories:
        return {}
    try:
        params = factories[model_name]().get_params()
    except Exception:
        return {}
    return {k: repr(v) for k, v in sorted(params.items())}


def _governance_only_principles() -> List[str]:
    """Principles no evaluator can measure -- process/documentation, not metrics."""
    return [p.name for p in principles_mod.PRINCIPLES if not p.tool_addressable]


def _tool_coverage_for_card(metrics: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """For every principle this model was actually scored on: which tool(s) ran
    here, and which other registered tool(s) also address that principle but
    didn't contribute a row -- the "or you could use..." second-opinion list."""
    used_by_principle: Dict[str, set] = defaultdict(set)
    for m in metrics:
        used_by_principle[m["principle"]].add(m["evaluator"])
    p2t = coverage_mod.principle_to_tools()
    rows = []
    for principle in sorted(used_by_principle):
        used = sorted(used_by_principle[principle])
        available = p2t.get(principle, [])
        other = [t for t in available if t not in used]
        rows.append({"principle": principle, "used": used, "other": other})
    return rows


def _card_data(model_name: str, dataset_name: str, sensitive_name: Optional[str],
              raw: pd.DataFrame, profile_row: "pd.Series", in_pareto: bool,
              hyperparameters: Optional[Dict[str, object]] = None,
              origin: Optional[str] = None,
              traces: Optional[Dict[str, dict]] = None,
              checklist: Optional[dict] = None) -> Dict[str, object]:
    rows = raw[(raw["model"] == model_name) & (raw["target"] != "error")]
    metrics = [
        {"evaluator": r.evaluator, "principle": r.principle, "target": r.target,
         "metric": r.metric,
         "value": None if pd.isna(r.value) else round(float(r.value), 4),
         "higher_is_better": bool(r.higher_is_better),
         "mechanism": getattr(r, "mechanism", "") or ""}
        for r in rows.itertuples()
    ]
    return {
        "model_name": model_name,
        "family": family_of(model_name),
        "hyperparameters": ({k: repr(v) for k, v in sorted(hyperparameters.items())}
                            if hyperparameters is not None else _model_hyperparams(model_name)),
        "origin": origin,
        "traces": traces or {},
        "checklist": checklist,
        "dataset_name": dataset_name,
        "sensitive_name": sensitive_name,
        "metrics": metrics,
        "principle_profile": {
            k: (None if pd.isna(v) else round(float(v), 3)) for k, v in profile_row.items()
        },
        "pareto_optimal": in_pareto,
        "governance_only_principles": _governance_only_principles(),
        "tool_coverage": _tool_coverage_for_card(metrics),
        "tools_not_run": [{"name": ev.name, "principles": list(ev.principle_ids),
                           "hint": ev.install_hint}
                          for ev in all_evaluators() if not ev.available()],
    }


def _try_export_mct_json(card: Dict[str, object], out_path: str) -> None:
    """Best-effort: also populate the real ModelCard dataclass, if installed."""
    try:
        from model_card_toolkit import ModelCard
        from model_card_toolkit.model_card import (
            Metric, ModelDetails, ModelParameters, PerformanceMetric,
            QuantitativeAnalysis,
        )

        mc = ModelCard()
        mc.model_details = ModelDetails(
            name=card["model_name"],
            overview=f"Trained on {card['dataset_name']} (family: {card['family']}).",
        )
        mc.quantitative_analysis = QuantitativeAnalysis(
            performance_metrics=[
                PerformanceMetric(type=m["metric"], value=str(m["value"]))
                for m in card["metrics"] if m["value"] is not None
            ]
        )
        with open(out_path, "w") as fh:
            fh.write(mc.to_json())
    except Exception:
        pass  # optional sidecar only -- never block the HTML deliverable on this


_STYLE = BASE + """
body{max-width:860px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:15px;margin-top:30px;border-bottom:1px solid var(--rule);padding-bottom:6px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:left}
th{font-weight:600}
.pill{padding:3px 10px;margin:2px 3px;font-size:12px}
.note{padding:10px 14px;font-size:13px;margin-top:8px}
code{padding:1px 5px;font-size:12px}
.mech{max-width:340px;white-space:normal;color:var(--ink-dim);font-size:12px}
details.tool{border:1px solid var(--rule);border-radius:6px;margin:10px 0;background:var(--bg-raised)}
details.tool summary{cursor:pointer;padding:8px 12px;font-family:var(--font-mono);font-size:13px}
details.tool summary b{color:var(--accent)}
details.tool > div{padding:0 12px 12px}
ol.steps{font-family:var(--font-mono);font-size:12px;padding-left:20px;margin:8px 0}
.small{font-size:12px}
"""


def _trace_html(tool: str, t: dict) -> str:
    esc = html.escape
    tr = t.get("trace") or {}
    parts = [f"<details class='tool'><summary><b>{esc(tool)}</b> -- formula, this run's "
             f"numbers, intermediate tables</summary><div>"]
    if t.get("formula"):
        parts.append(f"<pre>{esc(t['formula'])}</pre>")
    if t.get("limits"):
        parts.append(f"<p class='muted'><b>What this does not measure:</b> {esc(t['limits'])}</p>")
    if tr.get("steps"):
        parts.append("<ol class='steps'>" + "".join(f"<li>{esc(x)}</li>" for x in tr["steps"])
                     + "</ol>")
    for tb in tr.get("tables", []):
        rows = tb.get("rows", [])[:30]
        parts.append(f"<p class='muted'>{esc(tb.get('title', ''))}</p><table class='small'><tr>"
                     + "".join(f"<th>{esc(str(c))}</th>" for c in tb.get("columns", []))
                     + "</tr>" + "".join("<tr>" + "".join(f"<td>{esc(str(v))}</td>" for v in r)
                                         + "</tr>" for r in rows) + "</table>")
    for tx in tr.get("text", []):
        parts.append(f"<p class='muted'>{esc(tx.get('title', ''))}</p><pre>{esc(tx.get('body', ''))}</pre>")
    if t.get("params"):
        parts.append("<p class='muted'>Tool parameters used: " + ", ".join(
            f"<code>{esc(k)}={esc(repr(v))}</code>" for k, v in t["params"].items()) + "</p>")
    parts.append("</div></details>")
    return "".join(parts)


def _requirements_html(req: Optional[dict]) -> str:
    esc = html.escape
    if not req:
        return ""
    rows = "".join(
        f"<tr><td><code>{esc(r['requirement'])}</code></td><td>{'-' if r['value'] is None else r['value']}</td>"
        f"<td><span class='pill {'' if r['passes'] else 'bad'}'>{'meets' if r['passes'] else 'fails'}</span></td>"
        f"<td class='mech'>{esc(r['reason'])}</td></tr>" for r in req["results"])
    verdict = ("meets every requirement you set" if req["passes"] else
               "<b>does not</b> meet every requirement you set")
    return (f"<h2>Requirements you set</h2><p>This model {verdict}. Requirements are checked on "
            f"raw metric values, so a threshold means the same thing whatever the other candidates are.</p>"
            f"<table><tr><th>Requirement</th><th>Value</th><th></th><th>Why</th></tr>{rows}</table>")


def _profile_html(p: Optional[dict]) -> str:
    esc = html.escape
    if not p:
        return ""
    groups = "".join(
        f"<tr><td>{esc(str(g['group']))}</td><td>{g['n']}</td><td>{g['share']}</td>"
        f"<td>{g.get('base_rate', g.get('mean_target', '-'))}</td></tr>" for g in p.get("groups", []))
    px = p.get("proxy") or {}
    proxy = (f"<p class='muted'>Proxy screen: {esc(px.get('method', ''))} -> AUC <b>{px['auc']}</b> "
             f"(0.5 = the other features carry no information about the sensitive attribute).</p>"
             if px.get("auc") is not None else "")
    warns = "".join(f"<div>! {esc(w)}</div>" for w in p.get("warnings", []))
    return (f"<h2>What the data says (before any model)</h2>"
            f"<table><tr><th>Group</th><th>Rows</th><th>Share</th><th>Base rate</th></tr>{groups}</table>"
            f"{proxy}" + (f"<div class='note'>{warns}</div>" if warns else ""))


def _checklist_html(checklist: Optional[dict]) -> str:
    esc = html.escape
    if not checklist or checklist.get("checklist") not in checklists_mod.CHECKLISTS:
        return ("<p class='muted'>No governance checklist attached. Answer one in the console's "
                "<code>governance</code> tab or pass <code>--checklist answers.json</code> to "
                "<code>ethikit run</code> (templates: <code>ethikit checklist deon</code>).</p>")
    c = checklists_mod.get(checklist["checklist"])
    answers = checklist.get("answers", {})
    summary = checklists_mod.principle_summary(c.id, answers)
    srows = "".join(
        f"<tr><td>{esc(p)}</td><td>{'-' if v['score'] is None else v['score']}</td>"
        f"<td>{v['answered']} / {v['applicable']}</td></tr>"
        for p, v in sorted(summary.items()))
    irows = ""
    for i in c.items:
        a = answers.get(i.id) or {}
        ans = a.get("answer") or "unanswered"
        cls = {"yes": "", "partial": "gov", "no": "bad"}.get(ans, "other")
        ref = f"<div class='small muted'>original: {esc(i.ref)}</div>" if i.ref and i.ref != i.section else ""
        irows += (f"<tr><td><code>{esc(i.id)}</code></td><td>{esc(i.text)}{ref}</td>"
                  f"<td><span class='pill {cls}'>{esc(ans)}</span></td>"
                  f"<td class='mech'>{esc(a.get('note') or '')}</td></tr>")
    return (f"<p>Answered against <b>{esc(c.name)}</b> ({esc(c.source)}, "
            f"<a href='{esc(c.url)}'>source</a>). {esc(checklists_mod.provenance_line(c))} "
            f"This is <i>process</i> evidence about the "
            f"team and system, never mixed into the model scores above.</p>"
            f"<table><tr><th>Principle</th><th>Evidence score (yes=1, partial=0.5)</th>"
            f"<th>Answered / applicable</th></tr>{srows}</table>"
            f"<table><tr><th>Item</th><th>Question</th><th>Answer</th><th>Note</th></tr>"
            f"{irows}</table>")



def _render_card_html(card: Dict[str, object]) -> str:
    esc = html.escape
    name, family = esc(card["model_name"]), esc(card["family"])

    hparams_rows = "".join(
        f"<tr><td><code>{esc(k)}</code></td><td>{esc(v)}</td></tr>"
        for k, v in card["hyperparameters"].items()
    ) or "<tr><td colspan='2'>n/a</td></tr>"

    metric_rows = "".join(
        f"<tr><td>{esc(m['principle'])}</td><td>{esc(m['evaluator'])}</td>"
        f"<td>{esc(m['metric'])}</td><td>{esc(m['target'])}</td>"
        f"<td>{'-' if m['value'] is None else m['value']}</td>"
        f"<td>{'higher' if m['higher_is_better'] else 'lower'} is better</td>"
        f"<td class='mech'>{esc(m['mechanism']) or '-'}</td></tr>"
        for m in card["metrics"]
    ) or "<tr><td colspan='7'>no results</td></tr>"

    coverage_rows = "".join(
        f"<tr><td>{esc(c['principle'])}</td>"
        f"<td>{''.join(f'<span class=\"pill\">{esc(t)}</span>' for t in c['used'])}</td>"
        f"<td>{''.join(f'<span class=\"pill other\">{esc(t)}</span>' for t in c['other']) or '-'}</td></tr>"
        for c in card["tool_coverage"]
    ) or "<tr><td colspan='3'>no results</td></tr>"

    profile_rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{'-' if v is None else v}</td></tr>"
        for k, v in card["principle_profile"].items()
    )

    gov_pills = "".join(f'<span class="pill gov">{esc(p)}</span>'
                        for p in card["governance_only_principles"])

    pareto_line = (
        "This model is <b>Pareto-optimal</b> in this run: no other candidate beat it "
        "on every objective simultaneously."
        if card["pareto_optimal"] else
        "This model is <b>not</b> Pareto-optimal in this run: at least one other "
        "candidate matched or beat it on every objective."
    )

    sensitive = esc(card["sensitive_name"] or "n/a")
    origin = esc(card.get("origin") or "") or (
        f"A stock scikit-learn <code>{family}</code>-family model from ethikit's model zoo, "
        f"trained and evaluated on <b>{esc(card['dataset_name'])}</b>.")
    traces = "".join(_trace_html(tool, t) for tool, t in card.get("traces", {}).items()) or (
        "<p class='muted'>No traces recorded for this run.</p>")
    checklist = _checklist_html(card.get("checklist"))
    req_html = _requirements_html(card.get("requirements"))
    profile_html = _profile_html(card.get("data_profile"))
    not_run = ""
    if card.get("tools_not_run"):
        not_run = ("<h2>Tools that did not run</h2><p class='muted'>These registered tools were "
                   "unavailable on the machine that produced this card, so their principles may be "
                   "less well evidenced than a full install would give.</p><table><tr><th>Tool</th>"
                   "<th>Principles</th><th>To enable</th></tr>" + "".join(
                       f"<tr><td>{esc(t['name'])}</td><td>{esc(', '.join(t['principles']))}</td>"
                       f"<td><code>{esc(t['hint'] or 'see requirements.txt')}</code></td></tr>"
                       for t in card["tools_not_run"]) + "</table>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Card - {name}</title>{head_assets("model_cards/" + name)}<style>{_STYLE}</style></head>
<body><main class="page">
<h1>Model Card: {name}</h1>
<div class="sub">Family: <b>{family}</b> &middot; dataset: <b>{esc(card["dataset_name"])}</b>
&middot; sensitive attribute: <b>{sensitive}</b> &middot; generated by ethikit</div>

<h2>Model details</h2>
<p>{origin}</p>
<table><tr><th>Hyperparameter</th><th>Value</th></tr>{hparams_rows}</table>

<h2>Intended use</h2>
<p>Demonstration / benchmarking artifact for comparing candidate models on utility and
tool-addressable ethics principles before a human makes a deployment decision. Not
validated for production use on any real population.</p>

<h2>Quantitative analysis</h2>
<p class="muted">The Mechanism column is the specific computation behind
each number in <em>this run</em> -- not a black-box score. See "Tool coverage" below for
alternative tools you could reach for on the same principle.</p>
<table><tr><th>Principle</th><th>Evaluator</th><th>Metric</th><th>Inspects</th>
<th>Value</th><th>Direction</th><th>Mechanism (this run)</th></tr>{metric_rows}</table>

<h2>How each number was computed</h2>
<p class="muted">Every tool's formula, the same formula worked through with this run's
actual values, and the intermediate tables it was read from -- open any tool to audit it.</p>
{traces}

<h2>Per-principle profile (0-1, 1 = best among candidates in this run)</h2>
<table><tr><th>Principle</th><th>Normalized score</th></tr>{profile_rows}</table>
<p class="muted">{pareto_line}</p>

<h2>Tool coverage</h2>
<p class="muted">Which registered tool(s) actually scored this model on
each principle above (green), and which other registered tool(s) also address that principle
but weren't the source of these numbers (grey) -- a second opinion is one <code>register()</code>
call away. Full reverse map: <code>ethikit.coverage.principle_to_tools()</code>.</p>
<table><tr><th>Principle</th><th>Used here</th><th>Also available</th></tr>{coverage_rows}</table>

{req_html}
{profile_html}
{not_run}
<h2>Ethical considerations</h2>
<p>Metrics above cover only <i>tool-addressable</i> principles -- the ones a library can
measure from data, model internals, predictions, or behaviour. The following principles
are <b>governance</b> principles: they require documentation and human process, not a
metric, and are <b>not represented</b> in any score on this card.</p>
<div>{gov_pills}</div>

<h2>Governance checklist</h2>
{checklist}

<h2>Caveats and recommendations</h2>
<div class="note">
Scores are relative within this benchmark run, not absolute guarantees. Several fairness
criteria are mathematically incompatible with each other (Kleinberg / Chouldechova
impossibility results) -- satisfying one can mean violating another by construction, so
"fair" here means "fair by the specific metrics shown," not fair in every sense at once.
Treat this card as evidence for a human decision, not a verdict, and pair it with the
governance principles above before deployment.
</div>
</main></body></html>"""


def model_card_html(model_name: str, dataset_name: str, raw: pd.DataFrame,
                    profile: pd.DataFrame, pareto: List[str],
                    sensitive_name: Optional[str] = None,
                    hyperparameters: Optional[Dict[str, object]] = None,
                    origin: Optional[str] = None,
                    checklist: Optional[dict] = None,
                    requirements: Optional[dict] = None,
                    data_profile: Optional[dict] = None) -> str:
    """One model's card as an HTML string. Traces come from raw.attrs["traces"]
    (set by run_benchmark); `checklist` is a checklists answers dict; `requirements` is this
    model's entry from decision.check_requirements (plus the requirement texts); `data_profile`
    is profile.profile_raw() for the dataset."""
    traces = (raw.attrs.get("traces") or {}).get(model_name, {})
    card = _card_data(model_name, dataset_name, sensitive_name, raw, profile.loc[model_name],
                      model_name in pareto, hyperparameters, origin, traces, checklist)
    card["requirements"] = requirements
    card["data_profile"] = data_profile
    return _render_card_html(card)


def render_model_cards(dataset_name: str,
                       raw: pd.DataFrame,
                       profile: pd.DataFrame,
                       pareto: List[str],
                       out_dir: str = "model_cards",
                       sensitive_name: Optional[str] = None,
                       checklist: Optional[dict] = None,
                       hyperparameters: Optional[Dict[str, Dict[str, object]]] = None,
                       origins: Optional[Dict[str, str]] = None,
                       requirements: Optional[dict] = None,
                       data_profile: Optional[dict] = None) -> List[str]:
    """Write one Model Card HTML (+ best-effort .mct.json sidecar) per model.

    `raw` is the un-normalized run_benchmark() output (has per-metric values and
    `higher_is_better`); `profile` is the principle_profile()/objective_matrix()
    table (model x principle, normalized 0-1) used for the per-principle summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for model_name in profile.index:
        traces = (raw.attrs.get("traces") or {}).get(model_name, {})
        card = _card_data(model_name, dataset_name, sensitive_name, raw,
                          profile.loc[model_name], model_name in pareto,
                          (hyperparameters or {}).get(model_name),
                          (origins or {}).get(model_name), traces, checklist)
        card["requirements"] = ((requirements or {}).get("by_model") or {}).get(model_name)
        card["data_profile"] = data_profile
        out_path = os.path.join(out_dir, f"{model_name}.html")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(_render_card_html(card))
        paths.append(out_path)
        if _has_mct():
            _try_export_mct_json(card, os.path.join(out_dir, f"{model_name}.mct.json"))
    return paths
