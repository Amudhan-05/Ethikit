"""Checklists for the principles no tool can measure.

ethikit's principle taxonomy (principles.py) is explicit that accountability,
human oversight and well-being are *governance* principles: no library can
compute them from a model and a dataset. What a team *can* do is answer a
recognised checklist about how the system was built and will be run, and keep
those answers next to the metrics. This module ships condensed, paraphrased
versions of four widely used ones, each item tagged with the ethikit
principle(s) it gives evidence for, so the answers can sit in the Model Card
beside the numbers and close the "governance (not computed)" gap honestly.

Provenance is recorded per checklist (version, publication date, verbatim or
paraphrased, when and against what it was last verified, and any drift found),
and per item (`ref`: where it comes from in the original). Deon is MIT-licensed
and ships verbatim; the other three are condensed paraphrases -- use the full
originals for a formal audit.

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
    ref: str = ""        # where this item comes from in the original (its id, section or subcategory)


@dataclass(frozen=True)
class Checklist:
    id: str
    name: str
    source: str
    url: str
    scope: str
    version: str = ""            # which edition of the original
    published: str = ""          # when the original was published
    fidelity: str = "paraphrased"  # "verbatim" (licence allows) or "paraphrased" (condensed selection)
    verified_on: Optional[str] = None   # date the items were last compared with the original
    verified_against: str = ""   # what they were compared with, or why they couldn't be
    original_items: Optional[int] = None  # item count in the original, when it is a flat list
    drift_notes: List[str] = field(default_factory=list)
    items: List[Item] = field(default_factory=list)


def _items(rows):
    return [Item(*r) for r in rows]


CHECKLISTS: Dict[str, Checklist] = {c.id: c for c in [
    Checklist(
        id="deon", name="Deon data-science ethics checklist",
        source="DrivenData (open source, MIT licence)", url="https://deon.drivendata.org/",
        scope="The whole project lifecycle, written for the data-science team itself.",
        version="deon main branch, deon/assets/checklist.yml @ 8a31cf7 (after v0.3.0)",
        published="2023", fidelity="verbatim",
        verified_on="2026-10-21",
        verified_against="https://github.com/drivendataorg/deon/blob/8a31cf7/deon/assets/checklist.yml "
                         "(fetched and compared item by item; 21 items)",
        original_items=21,
        drift_notes=[
            "Before 2026-10-21 ethikit shipped a 20-item paraphrase that had drifted: its A.4 asked "
            "whether people can query the data held on them, but the original A.4 is 'downstream bias "
            "mitigation'; the original C.4 'privacy in analysis' was missing, so 'auditability' was "
            "numbered C.4 instead of C.5. Replaced by the verbatim text; answer files written before "
            "then should be re-checked for A.4, C.4 and C.5.",
            "Deon v0.3.0 (the last release) differs from main: main adds E.1 'monitoring and "
            "evaluation' and drops the separate 'concept drift' item. ethikit follows main.",
        ],
        items=_items([
            ('A.1', 'Data Collection', 'If there are human subjects, have they given informed consent, where subjects affirmatively opt-in and have a clear understanding of the data uses to which they consent?', ['accountability', 'privacy'], 'Informed consent'),
            ('A.2', 'Data Collection', 'Have we considered sources of bias that could be introduced during data collection and survey design and taken steps to mitigate those?', ['fairness', 'inclusiveness'], 'Collection bias'),
            ('A.3', 'Data Collection', "Have we considered ways to minimize exposure of personally identifiable information (PII) for example through anonymization or not collecting information that isn't relevant for analysis?", ['privacy'], 'Limit PII exposure'),
            ('A.4', 'Data Collection', 'Have we considered ways to enable testing downstream results for biased outcomes (e.g., collecting data on protected group status like race or gender)?', ['fairness', 'accountability'], 'Downstream bias mitigation'),
            ('B.1', 'Data Storage', 'Do we have a plan to protect and secure data (e.g., encryption at rest and in transit, access controls on internal users and third parties, access logs, and up-to-date software)?', ['privacy', 'safety'], 'Data security'),
            ('B.2', 'Data Storage', 'Do we have a mechanism through which an individual can request their personal information be removed?', ['privacy', 'accountability'], 'Right to be forgotten'),
            ('B.3', 'Data Storage', 'Is there a schedule or plan to delete the data after it is no longer needed?', ['privacy'], 'Data retention plan'),
            ('C.1', 'Analysis', 'Have we sought to address blindspots in the analysis through engagement with relevant stakeholders (e.g., checking assumptions and discussing implications with affected communities and subject matter experts)?', ['inclusiveness', 'wellbeing'], 'Missing perspectives'),
            ('C.2', 'Analysis', 'Have we examined the data for possible sources of bias and taken steps to mitigate or address these biases (e.g., stereotype perpetuation, confirmation bias, imbalanced classes, or omitted confounding variables)?', ['fairness'], 'Dataset bias'),
            ('C.3', 'Analysis', 'Are our visualizations, summary statistics, and reports designed to honestly represent the underlying data?', ['transparency'], 'Honest representation'),
            ('C.4', 'Analysis', 'Have we ensured that data with PII are not used or displayed unless necessary for the analysis?', ['privacy'], 'Privacy in analysis'),
            ('C.5', 'Analysis', 'Is the process of generating the analysis well documented and reproducible if we discover issues in the future?', ['accountability', 'transparency'], 'Auditability'),
            ('D.1', 'Modeling', 'Have we ensured that the model does not rely on variables or proxies for variables that are unfairly discriminatory?', ['fairness'], 'Proxy discrimination'),
            ('D.2', 'Modeling', 'Have we tested model results for fairness with respect to different affected groups (e.g., tested for disparate error rates)?', ['fairness', 'inclusiveness'], 'Fairness across groups'),
            ('D.3', 'Modeling', 'Have we considered the effects of optimizing for our defined metrics and considered additional metrics?', ['wellbeing', 'safety'], 'Metric selection'),
            ('D.4', 'Modeling', 'Can we explain in understandable terms a decision the model made in cases where a justification is needed?', ['explainability', 'transparency'], 'Explainability'),
            ('D.5', 'Modeling', 'Have we communicated the shortcomings, limitations, and biases of the model to relevant stakeholders in ways that can be generally understood?', ['transparency', 'accountability'], 'Communicate limitations'),
            ('E.1', 'Deployment', 'Do we have a clear plan to monitor the model and its impacts after it is deployed (e.g., performance monitoring, regular audit of sample predictions, human review of high-stakes decisions, reviewing downstream impacts of errors or low-confidence decisions, testing for concept drift)?', ['safety', 'accountability'], 'Monitoring and evaluation'),
            ('E.2', 'Deployment', 'Have we discussed with our organization a plan for response if users are harmed by the results (e.g., how does the data science team evaluate these cases and update analysis and models to prevent future harm)?', ['human_oversight', 'accountability'], 'Redress'),
            ('E.3', 'Deployment', 'Is there a way to turn off or roll back the model in production if necessary?', ['human_oversight', 'safety'], 'Roll back'),
            ('E.4', 'Deployment', 'Have we taken steps to identify and prevent unintended uses and abuse of the model and do we have a plan to monitor these once the model is deployed?', ['safety', 'wellbeing'], 'Unintended use'),
        ])),
    Checklist(
        id="altai", name="EU Assessment List for Trustworthy AI (ALTAI)",
        source="European Commission High-Level Expert Group on AI, 2020",
        url="https://digital-strategy.ec.europa.eu/en/library/assessment-list-trustworthy-artificial-intelligence-altai-self-assessment",
        scope="Self-assessment against the EU's seven requirements for trustworthy AI.",
        version="ALTAI final version, July 2020", published="2020-07-17", fidelity="paraphrased",
        verified_against="not machine-checked: the original could not be fetched from the build environment. "
                         "Item refs were recorded from the published document; check them by hand "
                         "(TESTING.md, M6.8).",
        drift_notes=["A 19-item condensed selection: ALTAI asks many more questions under each of its "
                     "seven requirements; each item names the ALTAI requirement and sub-section it condenses."],
        items=_items([
            ("1.1", "Human agency & oversight", "Are users told they are interacting with, or being assessed by, an AI system?", ["transparency", "human_oversight"], 'Req. 1 -- Human agency and autonomy'),
            ("1.2", "Human agency & oversight", "Is the oversight model defined (human-in-the-loop, on-the-loop, or in-command) and are overseers trained?", ["human_oversight"], 'Req. 1 -- Human oversight'),
            ("1.3", "Human agency & oversight", "Can a human safely override or stop the system?", ["human_oversight", "safety"], 'Req. 1 -- Human oversight'),
            ("2.1", "Technical robustness & safety", "Have attacks (data poisoning, evasion, model extraction) and their impact been assessed?", ["safety"], 'Req. 2 -- Resilience to attack and security'),
            ("2.2", "Technical robustness & safety", "Is there a fallback plan if the system fails or behaves unexpectedly?", ["safety", "accountability"], 'Req. 2 -- Reliability, fall-back plans and reproducibility'),
            ("2.3", "Technical robustness & safety", "Is accuracy measured, monitored and communicated, including how often it may be wrong?", ["safety", "transparency"], 'Req. 2 -- Accuracy'),
            ("3.1", "Privacy & data governance", "Has a data-protection impact assessment been done, and is data minimised?", ["privacy"], 'Req. 3 -- Privacy'),
            ("3.2", "Privacy & data governance", "Are data quality, integrity and access controls documented?", ["privacy", "accountability"], 'Req. 3 -- Data governance'),
            ("4.1", "Transparency", "Are datasets and processes documented well enough to trace a decision back (traceability)?", ["transparency", "accountability"], 'Req. 4 -- Traceability'),
            ("4.2", "Transparency", "Can the system's decisions be explained to the people affected?", ["explainability"], 'Req. 4 -- Explainability'),
            ("4.3", "Transparency", "Are the system's purpose, capabilities and limits communicated to users?", ["transparency"], 'Req. 4 -- Communication'),
            ("5.1", "Diversity, non-discrimination & fairness", "Is there a strategy to avoid creating or reinforcing unfair bias in data and algorithm design?", ["fairness"], 'Req. 5 -- Avoidance of unfair bias'),
            ("5.2", "Diversity, non-discrimination & fairness", "Is the system accessible and usable by people with diverse abilities and backgrounds?", ["inclusiveness"], 'Req. 5 -- Accessibility and universal design'),
            ("5.3", "Diversity, non-discrimination & fairness", "Were affected stakeholders consulted during design?", ["inclusiveness", "wellbeing"], 'Req. 5 -- Stakeholder participation'),
            ("6.1", "Societal & environmental well-being", "Has the environmental impact (energy, compute) of training and running the system been considered?", ["wellbeing"], 'Req. 6 -- Environmental well-being'),
            ("6.2", "Societal & environmental well-being", "Has the impact on work, skills and society at large been assessed?", ["wellbeing"], 'Req. 6 -- Impact on work and skills; impact on society at large or democracy'),
            ("7.1", "Accountability", "Can the system be independently audited (logs, documentation, access for auditors)?", ["accountability"], 'Req. 7 -- Auditability'),
            ("7.2", "Accountability", "Is there a risk-management process, and a named owner for trade-offs and decisions?", ["accountability"], 'Req. 7 -- Risk management'),
            ("7.3", "Accountability", "Is there an accessible mechanism for redress when the system causes harm?", ["accountability", "human_oversight"], 'Req. 7 -- Risk management'),
        ])),
    Checklist(
        id="datasheets", name="Datasheets for Datasets",
        source="Gebru et al., Communications of the ACM 2021 (arXiv:1803.09010)",
        url="https://arxiv.org/abs/1803.09010",
        scope="Documentation of the dataset the model is trained and evaluated on.",
        version="arXiv:1803.09010 v8 / Communications of the ACM 64(12)", published="2021-12",
        fidelity="paraphrased",
        verified_against="not machine-checked: the original could not be fetched from the build environment. "
                         "Item refs were recorded from the published document; check them by hand "
                         "(TESTING.md, M6.8).",
        drift_notes=["An 11-item condensed selection covering all seven sections of the original's "
                     "question list; the full datasheet (ethikit datasheet) uses the original section "
                     "structure."],
        items=_items([
            ("M.1", "Motivation", "Is it documented why the dataset was created, by whom, and who funded it?", ["accountability", "transparency"], 'Motivation'),
            ("C.1", "Composition", "Is it documented what each instance represents, how many there are, and whether it is a sample of a larger set?", ["transparency"], 'Composition'),
            ("C.2", "Composition", "Does the dataset identify sub-populations (e.g. by age, sex), and is their distribution documented?", ["fairness", "inclusiveness"], 'Composition'),
            ("C.3", "Composition", "Does it contain data that could identify people or is otherwise sensitive, and is that documented?", ["privacy"], 'Composition'),
            ("P.1", "Collection process", "Is the collection mechanism, time frame and sampling strategy documented?", ["transparency", "fairness"], 'Collection process'),
            ("P.2", "Collection process", "Were the people involved notified and did they consent; was an ethical review done?", ["privacy", "accountability"], 'Collection process'),
            ("R.1", "Preprocessing", "Is all cleaning, labelling and preprocessing documented, with the raw data kept?", ["transparency", "accountability"], 'Preprocessing/cleaning/labeling'),
            ("U.1", "Uses", "Are tasks the dataset should NOT be used for documented?", ["safety", "wellbeing"], 'Uses'),
            ("U.2", "Uses", "Could the composition or collection lead to unfair treatment of individuals or groups if used naively?", ["fairness"], 'Uses'),
            ("D.1", "Distribution", "Is the licence / terms of use stated?", ["accountability"], 'Distribution'),
            ("X.1", "Maintenance", "Is there a named maintainer, an erratum process, and a policy for updates and deletion requests?", ["accountability", "privacy"], 'Maintenance'),
        ])),
    Checklist(
        id="nist_ai_rmf", name="NIST AI Risk Management Framework 1.0",
        source="US National Institute of Standards and Technology, NIST AI 100-1 (2023)",
        url="https://www.nist.gov/itl/ai-risk-management-framework",
        scope="Organisational risk management across the four RMF functions.",
        version="AI RMF 1.0 (NIST AI 100-1)", published="2023-01-26", fidelity="paraphrased",
        verified_against="not machine-checked: the original could not be fetched from the build environment. "
                         "Item refs were recorded from the published document; check them by hand "
                         "(TESTING.md, M6.8).",
        drift_notes=["A 13-item condensed selection of the RMF's subcategories; each item names the "
                     "subcategory (e.g. GOVERN 1.1) it condenses."],
        items=_items([
            ("GV.1", "Govern", "Are legal and regulatory requirements for this AI system understood and documented?", ["accountability"], 'GOVERN 1.1'),
            ("GV.2", "Govern", "Are roles, responsibilities and lines of authority for AI risk clearly assigned?", ["accountability"], 'GOVERN 2.1'),
            ("GV.3", "Govern", "Is workforce diversity and domain expertise considered in AI risk decisions?", ["inclusiveness"], 'GOVERN 3.1'),
            ("GV.4", "Govern", "Are policies in place for third-party data, models and software?", ["accountability", "safety"], 'GOVERN 6.1'),
            ("MP.1", "Map", "Is the intended purpose, context of use and the people affected documented?", ["transparency", "wellbeing"], 'MAP 1.1'),
            ("MP.2", "Map", "Are the system's limitations and the conditions it must not be used under documented?", ["safety", "transparency"], 'MAP 2.2'),
            ("MP.3", "Map", "Are potential benefits and costs, including to individuals and society, characterised?", ["wellbeing"], 'MAP 3.1, MAP 3.2'),
            ("MS.1", "Measure", "Are appropriate metrics chosen for each identified risk, and are their limits known?", ["accountability"], 'MEASURE 1.1'),
            ("MS.2", "Measure", "Is the system evaluated for fairness, robustness, privacy and explainability (ethikit's own report counts as evidence)?", ["fairness", "safety", "privacy", "explainability"], 'MEASURE 2.6, 2.7, 2.9, 2.10, 2.11'),
            ("MS.3", "Measure", "Are risks tracked over time after deployment?", ["safety", "accountability"], 'MEASURE 3.1'),
            ("MG.1", "Manage", "Are identified risks prioritised and either mitigated, transferred or accepted with a documented rationale?", ["accountability"], 'MANAGE 1.2, 1.3'),
            ("MG.2", "Manage", "Are there mechanisms to supersede, disengage or deactivate the system?", ["human_oversight", "safety"], 'MANAGE 2.4'),
            ("MG.3", "Manage", "Is there an incident-response and communication plan for when things go wrong?", ["accountability", "wellbeing"], 'MANAGE 4.3'),
        ])),
]}


def get(checklist_id: str) -> Checklist:
    return CHECKLISTS[checklist_id]


def as_dict(c: Checklist) -> dict:
    return {"id": c.id, "name": c.name, "source": c.source, "url": c.url, "scope": c.scope,
            "provenance": provenance(c),
            "items": [{"id": i.id, "section": i.section, "text": i.text, "ref": i.ref,
                       "principles": list(i.principles)} for i in c.items]}


def provenance_line(c: Checklist) -> str:
    """One sentence on what this checklist is relative to its original."""
    how = ("the original's text, verbatim" if c.fidelity == "verbatim"
           else "ethikit's condensed paraphrase -- use the original for a formal audit")
    checked = (f"verified against the original on {c.verified_on}" if c.verified_on
               else "not yet verified item by item against the original")
    return f"{c.version} (published {c.published}); items are {how}; {checked}."


def provenance(c: Checklist) -> dict:
    return {"version": c.version, "published": c.published, "fidelity": c.fidelity,
            "verified_on": c.verified_on, "verified_against": c.verified_against,
            "original_items": c.original_items, "drift_notes": list(c.drift_notes),
            "line": provenance_line(c)}


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
    verified = (f"verified against the original on {c.verified_on}" if c.verified_on
                else "not yet verified against the original")
    lines = [f"# {c.name}", "", f"Source: {c.source} -- {c.url}", "",
             f"Version: {c.version} (published {c.published}); {c.fidelity}; {verified}.", "",
             ("_Verbatim text of the original._" if c.fidelity == "verbatim" else
              "_Condensed, paraphrased items via ethikit; use the original for a formal audit._"), ""]
    section = None
    for i in c.items:
        if i.section != section:
            section = i.section
            lines += ["", f"## {section}", ""]
        a = answers.get(i.id) or {}
        mark = {"yes": "[x]", "partial": "[~]", "no": "[ ]", "n/a": "[-]"}.get(a.get("answer"), "[ ]")
        note = f" -- {a['note']}" if a.get("note") else ""
        ref = f" [orig: {i.ref}]" if i.ref and i.ref != i.section else ""
        lines.append(f"- {mark} **{i.id}** {i.text} _({', '.join(i.principles)})_{ref}{note}")
    return "\n".join(lines) + "\n"
