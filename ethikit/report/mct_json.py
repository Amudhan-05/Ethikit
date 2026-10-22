"""Model Card JSON in Google's Model Card Toolkit schema (v0.0.2).

The Model Card Toolkit (MCT) itself does not install on Python >= 3.12, so ethikit keeps its own
HTML card and also writes the same content as JSON that validates against MCT's published schema
(vendored in schemas/, Apache-2.0). Any tool that reads MCT model cards can load it elsewhere.

What maps where:
  model_details           name, overview, documentation (origin + intended use), ethikit version
  model_parameters        architecture (family + estimator + hyperparameters), dataset, I/O format
  quantitative_analysis   every metric from every tool, typed "<principle>/<tool>/<metric>"
  considerations          users, use cases, each tool's limits, trade-offs (Pareto, profile,
                          requirements), and ethical risks (the data profile's warnings, the
                          governance-only principles, tools that did not run)

The MCT schema has no place for ethikit's worked traces; those stay in results.json and the
HTML card.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA_VERSION = "0.0.2"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / f"model_card.schema.v{SCHEMA_VERSION}.json"


def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _fmt(v) -> str:
    return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))


def card_json(card: Dict[str, object], limits: Optional[Dict[str, str]] = None,
              today: Optional[str] = None) -> dict:
    """`card` is model_card._card_data() (+ "requirements", "data_profile"); `limits` maps a tool
    name to what it does not measure."""
    from .. import __version__
    limits = limits or {}
    name = str(card["model_name"])
    hp = card.get("hyperparameters") or {}
    arch = f"{card['family']} family" + (f"; hyperparameters: " + ", ".join(f"{k}={v}" for k, v in hp.items())
                                         if hp else "")
    origin = card.get("origin") or (f"A stock {card['family']}-family model from ethikit's model zoo, "
                                    f"trained and evaluated on {card['dataset_name']}.")
    prof = card.get("data_profile") or {}
    data_desc = (f"{prof.get('n_rows', '?')} rows, {prof.get('n_features', '?')} features, task "
                 f"{prof.get('task', '?')}; held-out split used for every metric.") if prof else ""
    dataset = {"name": str(card["dataset_name"]), "description": data_desc}
    if card.get("sensitive_name"):
        dataset["sensitive"] = {"sensitive_data": [f"sensitive attribute compared across groups: "
                                                   f"{card['sensitive_name']}"]}

    metrics = [{"type": f"{m['principle']}/{m['evaluator']}/{m['metric']}", "value": _fmt(m["value"]),
                "slice": "held-out test set"}
               for m in card["metrics"] if m["value"] is not None]

    tools_used = sorted({m["evaluator"] for m in card["metrics"]})
    lims: List[dict] = [{"description": f"{t}: {limits[t]}"} for t in tools_used if limits.get(t)]
    lims.append({"description": "Scores are relative to the candidates in this run (0-1 per principle), "
                                "not absolute guarantees; several fairness criteria cannot all hold at once."})

    profile = card.get("principle_profile") or {}
    trade = [{"description": ("Pareto-optimal in this run: no other candidate was at least as good on "
                              "every principle." if card.get("pareto_optimal") else
                              "Not Pareto-optimal in this run: another candidate was at least as good "
                              "on every principle.")}]
    if profile:
        trade.append({"description": "Per-principle profile (0-1, 1 = best in this run): " +
                      ", ".join(f"{k} {_fmt(v) or 'n/a'}" for k, v in profile.items())})
    req = card.get("requirements")
    if req:
        failed = [r for r in req.get("results", []) if not r.get("passes")]
        trade.append({"description": ("Meets every hard requirement set for this run." if req.get("passes") else
                                      "Fails hard requirement(s): " + "; ".join(
                                          f"{r.get('requirement')} ({r.get('reason') or 'value ' + _fmt(r.get('value'))})"
                                          for r in failed))})

    risks = [{"name": w, "mitigation_strategy": "Investigate before deployment; see the data profile and datasheet."}
             for w in prof.get("warnings", [])]
    gov = card.get("governance_only_principles") or []
    if gov:
        risks.append({"name": "Principles no tool can measure: " + ", ".join(gov),
                      "mitigation_strategy": ("Answer a governance checklist (Deon, EU ALTAI, Datasheets, NIST AI RMF) "
                                              "and keep it with this card." if not card.get("checklist") else
                                              f"Governance checklist '{card['checklist'].get('checklist')}' attached to the HTML card.")})
    for t in card.get("tools_not_run") or []:
        risks.append({"name": f"Tool '{t['name']}' did not run ({', '.join(t['principles'])} less evidenced)",
                      "mitigation_strategy": f"Install it and re-run: {t.get('hint') or 'see requirements.txt'}"})

    return {
        "schema_version": SCHEMA_VERSION,
        "model_details": {
            "name": name,
            "overview": f"{card['family']}-family model audited by ethikit on {card['dataset_name']} "
                        f"for selection on principles of ethicality (not a ranking).",
            "documentation": origin + " Intended for comparing candidate models before a human decision; "
                                      "not validated for production use.",
            "version": {"name": f"ethikit {__version__}", "date": today or _dt.date.today().isoformat()},
            "references": [{"reference": "https://github.com/AmudhanS-05/Ethikit"}],
        },
        "model_parameters": {
            "model_architecture": arch,
            "data": [dataset],
            "input_format": "tabular feature vector (one row per case)",
            "output_format": "predicted label (and class probabilities where the model provides them)",
        },
        "quantitative_analysis": {"performance_metrics": metrics},
        "considerations": {
            "users": [{"description": "Teams choosing between candidate models, and reviewers auditing that choice."}],
            "use_cases": [{"description": "Model selection and audit on principles of ethicality."}],
            "limitations": lims,
            "tradeoffs": trade,
            "ethical_considerations": risks,
        },
    }


def validate(doc: dict) -> None:
    """Raise if `doc` does not validate against the vendored MCT schema (needs jsonschema)."""
    import jsonschema
    jsonschema.Draft7Validator(schema(), format_checker=jsonschema.FormatChecker()).validate(doc)
