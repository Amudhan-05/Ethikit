"""Checklists for the principles no tool can measure.

ethikit's principle taxonomy (principles.py) is explicit that accountability,
human oversight and well-being are *governance* principles: no library can
compute them from a model and a dataset. What a team *can* do is answer a
recognised checklist about how the system was built and will be run, and keep
those answers next to the metrics. This module ships condensed, paraphrased
versions of four widely used ones, each item tagged with the ethikit
principle(s) it gives evidence for, so the answers can sit in the Model Card
beside the numbers and close the "governance (not computed)" gap honestly.

The items are summaries for convenience, not the official text -- every
framework links to its source, and the full originals should be used for a
real audit.

  deon        -- Deon data-science ethics checklist (DrivenData, MIT licence)
  altai       -- EU Assessment List for Trustworthy AI (High-Level Expert Group, 2020)
  datasheets  -- Datasheets for Datasets (Gebru et al., 2018/2021)
  nist_ai_rmf -- NIST AI Risk Management Framework 1.0 (2023), Govern/Map/Measure/Manage

Answers are "yes" / "partial" / "no" / "n/a" plus an optional note.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ANSWERS = ("yes", "partial", "no", "n/a")


@dataclass(frozen=True)
class Item:
    id: str
    section: str
    text: str
    principles: List[str]


@dataclass(frozen=True)
class Checklist:
    id: str
    name: str
    source: str
    url: str
    scope: str
    items: List[Item] = field(default_factory=list)


def _items(rows):
    return [Item(*r) for r in rows]


CHECKLISTS: Dict[str, Checklist] = {c.id: c for c in [
    Checklist(
        id="deon", name="Deon data-science ethics checklist",
        source="DrivenData (open source, MIT licence)", url="https://deon.drivendata.org/",
        scope="The whole project lifecycle, written for the data-science team itself.",
        items=_items([
            ("A.1", "Data collection", "If people are the subject of the data, did they give informed consent to this use, and can they withdraw it?", ["accountability", "privacy"]),
            ("A.2", "Data collection", "Have sources of collection bias (who is over- or under-represented, and why) been considered?", ["fairness", "inclusiveness"]),
            ("A.3", "Data collection", "Is collection and exposure of personally identifiable information kept to what the task needs?", ["privacy"]),
            ("A.4", "Data collection", "Is there a way for affected people to find out about and query the data collected on them?", ["accountability", "human_oversight"]),
            ("B.1", "Data storage", "Is sensitive data protected (access control, encryption) and is there a plan if it leaks?", ["privacy", "safety"]),
            ("B.2", "Data storage", "Can an individual's data be removed on request?", ["privacy", "accountability"]),
            ("B.3", "Data storage", "Is there a schedule for deleting data once it is no longer needed?", ["privacy"]),
            ("C.1", "Analysis", "Have perspectives that might be missing from the team been sought (e.g. domain experts, affected communities)?", ["inclusiveness", "wellbeing"]),
            ("C.2", "Analysis", "Has the data been checked for quality issues and biases that could skew results?", ["fairness"]),
            ("C.3", "Analysis", "Are results presented honestly, including uncertainty and limitations?", ["transparency"]),
            ("C.4", "Analysis", "Is the analysis reproducible and auditable (code, data versions, decisions recorded)?", ["accountability", "transparency"]),
            ("D.1", "Modeling", "Has the model been checked for proxy variables that stand in for protected attributes?", ["fairness"]),
            ("D.2", "Modeling", "Has performance been compared across relevant groups, and are the fairness metrics chosen deliberately?", ["fairness", "inclusiveness"]),
            ("D.3", "Modeling", "Does the optimisation metric reflect the real-world costs of each kind of error?", ["wellbeing", "safety"]),
            ("D.4", "Modeling", "Can the model's decisions be explained in terms the affected people would understand?", ["explainability", "transparency"]),
            ("D.5", "Modeling", "Are the model's known shortcomings and biases communicated to decision makers?", ["transparency", "accountability"]),
            ("E.1", "Deployment", "Is there monitoring for drift, degraded performance and emerging bias after launch?", ["safety", "accountability"]),
            ("E.2", "Deployment", "Is there a process for people to contest a decision and get it corrected?", ["human_oversight", "accountability"]),
            ("E.3", "Deployment", "Can the model be rolled back or switched off quickly if it causes harm?", ["human_oversight", "safety"]),
            ("E.4", "Deployment", "Have likely misuses or unintended uses been identified and mitigated?", ["safety", "wellbeing"]),
        ])),
    Checklist(
        id="altai", name="EU Assessment List for Trustworthy AI (ALTAI)",
        source="European Commission High-Level Expert Group on AI, 2020",
        url="https://digital-strategy.ec.europa.eu/en/library/assessment-list-trustworthy-artificial-intelligence-altai-self-assessment",
        scope="Self-assessment against the EU's seven requirements for trustworthy AI.",
        items=_items([
            ("1.1", "Human agency & oversight", "Are users told they are interacting with, or being assessed by, an AI system?", ["transparency", "human_oversight"]),
            ("1.2", "Human agency & oversight", "Is the oversight model defined (human-in-the-loop, on-the-loop, or in-command) and are overseers trained?", ["human_oversight"]),
            ("1.3", "Human agency & oversight", "Can a human safely override or stop the system?", ["human_oversight", "safety"]),
            ("2.1", "Technical robustness & safety", "Have attacks (data poisoning, evasion, model extraction) and their impact been assessed?", ["safety"]),
            ("2.2", "Technical robustness & safety", "Is there a fallback plan if the system fails or behaves unexpectedly?", ["safety", "accountability"]),
            ("2.3", "Technical robustness & safety", "Is accuracy measured, monitored and communicated, including how often it may be wrong?", ["safety", "transparency"]),
            ("3.1", "Privacy & data governance", "Has a data-protection impact assessment been done, and is data minimised?", ["privacy"]),
            ("3.2", "Privacy & data governance", "Are data quality, integrity and access controls documented?", ["privacy", "accountability"]),
            ("4.1", "Transparency", "Are datasets and processes documented well enough to trace a decision back (traceability)?", ["transparency", "accountability"]),
            ("4.2", "Transparency", "Can the system's decisions be explained to the people affected?", ["explainability"]),
            ("4.3", "Transparency", "Are the system's purpose, capabilities and limits communicated to users?", ["transparency"]),
            ("5.1", "Diversity, non-discrimination & fairness", "Is there a strategy to avoid creating or reinforcing unfair bias in data and algorithm design?", ["fairness"]),
            ("5.2", "Diversity, non-discrimination & fairness", "Is the system accessible and usable by people with diverse abilities and backgrounds?", ["inclusiveness"]),
            ("5.3", "Diversity, non-discrimination & fairness", "Were affected stakeholders consulted during design?", ["inclusiveness", "wellbeing"]),
            ("6.1", "Societal & environmental well-being", "Has the environmental impact (energy, compute) of training and running the system been considered?", ["wellbeing"]),
            ("6.2", "Societal & environmental well-being", "Has the impact on work, skills and society at large been assessed?", ["wellbeing"]),
            ("7.1", "Accountability", "Can the system be independently audited (logs, documentation, access for auditors)?", ["accountability"]),
            ("7.2", "Accountability", "Is there a risk-management process, and a named owner for trade-offs and decisions?", ["accountability"]),
            ("7.3", "Accountability", "Is there an accessible mechanism for redress when the system causes harm?", ["accountability", "human_oversight"]),
        ])),
    Checklist(
        id="datasheets", name="Datasheets for Datasets",
        source="Gebru et al., Communications of the ACM 2021 (arXiv:1803.09010)",
        url="https://arxiv.org/abs/1803.09010",
        scope="Documentation of the dataset the model is trained and evaluated on.",
        items=_items([
            ("M.1", "Motivation", "Is it documented why the dataset was created, by whom, and who funded it?", ["accountability", "transparency"]),
            ("C.1", "Composition", "Is it documented what each instance represents, how many there are, and whether it is a sample of a larger set?", ["transparency"]),
            ("C.2", "Composition", "Does the dataset identify sub-populations (e.g. by age, sex), and is their distribution documented?", ["fairness", "inclusiveness"]),
            ("C.3", "Composition", "Does it contain data that could identify people or is otherwise sensitive, and is that documented?", ["privacy"]),
            ("P.1", "Collection process", "Is the collection mechanism, time frame and sampling strategy documented?", ["transparency", "fairness"]),
            ("P.2", "Collection process", "Were the people involved notified and did they consent; was an ethical review done?", ["privacy", "accountability"]),
            ("R.1", "Preprocessing", "Is all cleaning, labelling and preprocessing documented, with the raw data kept?", ["transparency", "accountability"]),
            ("U.1", "Uses", "Are tasks the dataset should NOT be used for documented?", ["safety", "wellbeing"]),
            ("U.2", "Uses", "Could the composition or collection lead to unfair treatment of individuals or groups if used naively?", ["fairness"]),
            ("D.1", "Distribution", "Is the licence / terms of use stated?", ["accountability"]),
            ("X.1", "Maintenance", "Is there a named maintainer, an erratum process, and a policy for updates and deletion requests?", ["accountability", "privacy"]),
        ])),
    Checklist(
        id="nist_ai_rmf", name="NIST AI Risk Management Framework 1.0",
        source="US National Institute of Standards and Technology, NIST AI 100-1 (2023)",
        url="https://www.nist.gov/itl/ai-risk-management-framework",
        scope="Organisational risk management across the four RMF functions.",
        items=_items([
            ("GV.1", "Govern", "Are legal and regulatory requirements for this AI system understood and documented?", ["accountability"]),
            ("GV.2", "Govern", "Are roles, responsibilities and lines of authority for AI risk clearly assigned?", ["accountability"]),
            ("GV.3", "Govern", "Is workforce diversity and domain expertise considered in AI risk decisions?", ["inclusiveness"]),
            ("GV.4", "Govern", "Are policies in place for third-party data, models and software?", ["accountability", "safety"]),
            ("MP.1", "Map", "Is the intended purpose, context of use and the people affected documented?", ["transparency", "wellbeing"]),
            ("MP.2", "Map", "Are the system's limitations and the conditions it must not be used under documented?", ["safety", "transparency"]),
            ("MP.3", "Map", "Are potential benefits and costs, including to individuals and society, characterised?", ["wellbeing"]),
            ("MS.1", "Measure", "Are appropriate metrics chosen for each identified risk, and are their limits known?", ["accountability"]),
            ("MS.2", "Measure", "Is the system evaluated for fairness, robustness, privacy and explainability (ethikit's own report counts as evidence)?", ["fairness", "safety", "privacy", "explainability"]),
            ("MS.3", "Measure", "Are risks tracked over time after deployment?", ["safety", "accountability"]),
            ("MG.1", "Manage", "Are identified risks prioritised and either mitigated, transferred or accepted with a documented rationale?", ["accountability"]),
            ("MG.2", "Manage", "Are there mechanisms to supersede, disengage or deactivate the system?", ["human_oversight", "safety"]),
            ("MG.3", "Manage", "Is there an incident-response and communication plan for when things go wrong?", ["accountability", "wellbeing"]),
        ])),
]}


def get(checklist_id: str) -> Checklist:
    return CHECKLISTS[checklist_id]


def as_dict(c: Checklist) -> dict:
    return {"id": c.id, "name": c.name, "source": c.source, "url": c.url, "scope": c.scope,
            "items": [{"id": i.id, "section": i.section, "text": i.text,
                       "principles": list(i.principles)} for i in c.items]}


def principle_summary(checklist_id: str, answers: Dict[str, dict]) -> Dict[str, dict]:
    """principle -> {"score", "answered", "items"}: share of applicable items
    answered yes (partial = 0.5). n/a items are excluded; unanswered ones count
    as not yet evidenced. A process score, never mixed into the model metrics."""
    out: Dict[str, dict] = {}
    for item in get(checklist_id).items:
        ans = (answers.get(item.id) or {}).get("answer")
        for p in item.principles:
            row = out.setdefault(p, {"points": 0.0, "applicable": 0, "answered": 0,
                                     "items": len([i for i in get(checklist_id).items
                                                   if p in i.principles])})
            if ans == "n/a":
                continue
            row["applicable"] += 1
            if ans in ANSWERS:
                row["answered"] += 1
            row["points"] += {"yes": 1.0, "partial": 0.5}.get(ans, 0.0)
    for row in out.values():
        row["score"] = round(row["points"] / row["applicable"], 3) if row["applicable"] else None
        del row["points"]
    return out


def load_answers(path: str) -> dict:
    """Answers file: {"checklist": "deon", "answers": {"A.1": {"answer": "yes", "note": "..."}}}"""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("checklist") not in CHECKLISTS:
        raise KeyError(f"unknown checklist {data.get('checklist')!r}; "
                       f"choose from {', '.join(CHECKLISTS)}")
    for k, v in data.get("answers", {}).items():
        ans = (v or {}).get("answer")
        if ans is not None and ans not in ANSWERS:  # None = not answered yet
            raise ValueError(f"item {k}: answer must be one of {ANSWERS} (or null if unanswered)")
    return data


def blank_answers(checklist_id: str) -> dict:
    return {"checklist": checklist_id,
            "answers": {i.id: {"answer": None, "note": ""} for i in get(checklist_id).items}}


def to_markdown(checklist_id: str, answers: Optional[Dict[str, dict]] = None) -> str:
    c = get(checklist_id)
    answers = answers or {}
    lines = [f"# {c.name}", "", f"Source: {c.source} -- {c.url}", "",
             "_Condensed, paraphrased items via ethikit; use the original for a formal audit._", ""]
    section = None
    for i in c.items:
        if i.section != section:
            section = i.section
            lines += ["", f"## {section}", ""]
        a = answers.get(i.id) or {}
        mark = {"yes": "[x]", "partial": "[~]", "no": "[ ]", "n/a": "[-]"}.get(a.get("answer"), "[ ]")
        note = f" -- {a['note']}" if a.get("note") else ""
        lines.append(f"- {mark} **{i.id}** {i.text} _({', '.join(i.principles)})_{note}")
    return "\n".join(lines) + "\n"
