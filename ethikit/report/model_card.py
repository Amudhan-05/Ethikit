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

from .. import coverage as coverage_mod
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
              raw: pd.DataFrame, profile_row: "pd.Series", in_pareto: bool) -> Dict[str, object]:
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
        "hyperparameters": _model_hyperparams(model_name),
        "dataset_name": dataset_name,
        "sensitive_name": sensitive_name,
        "metrics": metrics,
        "principle_profile": {
            k: (None if pd.isna(v) else round(float(v), 3)) for k, v in profile_row.items()
        },
        "pareto_optimal": in_pareto,
        "governance_only_principles": _governance_only_principles(),
        "tool_coverage": _tool_coverage_for_card(metrics),
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


_STYLE = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:40px;color:#1c2530;background:#fbfaf7;max-width:860px}
h1{font-size:24px;margin:0 0 4px} .sub{color:#6b7683;margin-bottom:28px}
h2{font-size:15px;text-transform:uppercase;letter-spacing:.06em;color:#3a6b5a;margin-top:30px;border-bottom:1px solid #e7e3da;padding-bottom:6px}
table{border-collapse:collapse;width:100%;font-size:13px;margin-top:8px}
th,td{padding:6px 10px;border-bottom:1px solid #e7e3da;text-align:left}
th{color:#6b7683;font-weight:600}
.pill{display:inline-block;background:#e6f0eb;color:#2f5f4e;border-radius:20px;padding:3px 10px;margin:2px 3px;font-size:12px}
.pill.gov{background:#f3ece0;color:#7a5c2a}
.pill.other{background:#eef1f5;color:#3d4f66}
.note{background:#fff6e9;border-left:3px solid #e0a54b;padding:10px 14px;font-size:13px;color:#5c4a2a;margin-top:8px}
code{background:#f2efe8;padding:1px 5px;border-radius:4px;font-size:12px}
.mech{max-width:340px;white-space:normal;color:#4a5560;font-size:12px}
"""


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

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Model Card - {name}</title><style>{_STYLE}</style></head><body>
<h1>Model Card: {name}</h1>
<div class="sub">Family: <b>{family}</b> &middot; dataset: <b>{esc(card["dataset_name"])}</b>
&middot; sensitive attribute: <b>{sensitive}</b> &middot; generated by ethikit</div>

<h2>Model details</h2>
<p>A stock scikit-learn <code>{family}</code>-family classifier from ethikit's model zoo,
trained and evaluated on <b>{esc(card["dataset_name"])}</b>.</p>
<table><tr><th>Hyperparameter</th><th>Value</th></tr>{hparams_rows}</table>

<h2>Intended use</h2>
<p>Demonstration / benchmarking artifact for comparing candidate models on utility and
tool-addressable ethics principles before a human makes a deployment decision. Not
validated for production use on any real population.</p>

<h2>Quantitative analysis</h2>
<p style="font-size:13px;color:#6b7683">The Mechanism column is the specific computation behind
each number in <em>this run</em> -- not a black-box score. See "Tool coverage" below for
alternative tools you could reach for on the same principle.</p>
<table><tr><th>Principle</th><th>Evaluator</th><th>Metric</th><th>Inspects</th>
<th>Value</th><th>Direction</th><th>Mechanism (this run)</th></tr>{metric_rows}</table>

<h2>Per-principle profile (0-1, 1 = best among candidates in this run)</h2>
<table><tr><th>Principle</th><th>Normalized score</th></tr>{profile_rows}</table>
<p style="font-size:13px;color:#6b7683">{pareto_line}</p>

<h2>Tool coverage</h2>
<p style="font-size:13px;color:#6b7683">Which registered tool(s) actually scored this model on
each principle above (green), and which other registered tool(s) also address that principle
but weren't the source of these numbers (grey) -- a second opinion is one <code>register()</code>
call away. Full reverse map: <code>ethikit.coverage.principle_to_tools()</code>.</p>
<table><tr><th>Principle</th><th>Used here</th><th>Also available</th></tr>{coverage_rows}</table>

<h2>Ethical considerations</h2>
<p>Metrics above cover only <i>tool-addressable</i> principles -- the ones a library can
measure from data, model internals, predictions, or behaviour. The following principles
are <b>governance</b> principles: they require documentation and human process, not a
metric, and are <b>not represented</b> in any score on this card.</p>
<div>{gov_pills}</div>

<h2>Caveats and recommendations</h2>
<div class="note">
Scores are relative within this benchmark run, not absolute guarantees. Several fairness
criteria are mathematically incompatible with each other (Kleinberg / Chouldechova
impossibility results) -- satisfying one can mean violating another by construction, so
"fair" here means "fair by the specific metrics shown," not fair in every sense at once.
Treat this card as evidence for a human decision, not a verdict, and pair it with the
governance principles above before deployment.
</div>
</body></html>"""


def render_model_cards(dataset_name: str,
                       raw: pd.DataFrame,
                       profile: pd.DataFrame,
                       pareto: List[str],
                       out_dir: str = "model_cards",
                       sensitive_name: Optional[str] = None) -> List[str]:
    """Write one Model Card HTML (+ best-effort .mct.json sidecar) per model.

    `raw` is the un-normalized run_benchmark() output (has per-metric values and
    `higher_is_better`); `profile` is the principle_profile()/objective_matrix()
    table (model x principle, normalized 0-1) used for the per-principle summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for model_name in profile.index:
        card = _card_data(model_name, dataset_name, sensitive_name, raw,
                          profile.loc[model_name], model_name in pareto)
        out_path = os.path.join(out_dir, f"{model_name}.html")
        with open(out_path, "w") as fh:
            fh.write(_render_card_html(card))
        paths.append(out_path)
        if _has_mct():
            _try_export_mct_json(card, os.path.join(out_dir, f"{model_name}.mct.json"))
    return paths
