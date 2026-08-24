"""Canonical ethical-AI principle taxonomy.

This module harmonises the principle sets published by four influential
bodies into ONE canonical vocabulary the whole framework uses. Every
evaluator tags its output with one of these principles, and every
principle records which of the four sources names it (its "provenance").

Sources
-------
- UNESCO  : Recommendation on the Ethics of AI (2021) - 4 core values +
            10 principles, adopted by 194 member states.
- MICROSOFT: Responsible AI Standard v2 - 6 principles.
- IBM     : "Pillars of Trust" - 5 pillars, each with an open-source tool.
- GOOGLE  : AI Principles (2018 seven-principle set, restructured in 2025
            into three pillars; we map the operative, testable ones).

Only some principles are *tool-addressable* (a library can measure them on
a model + data). The rest are *governance* principles - handled by process,
documentation, and human review, not by a metric. We are explicit about
which is which so the framework never pretends a number covers everything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List


class Source(str, Enum):
    UNESCO = "UNESCO"
    MICROSOFT = "Microsoft"
    IBM = "IBM"
    GOOGLE = "Google"


class InspectionTarget(str, Enum):
    """Which part of the AI system an evaluator actually inspects."""
    DATA = "data"                      # the dataset itself (base rates, representation)
    MODEL_INTERNALS = "model_internals"  # weights / structure / global attributions
    PREDICTIONS = "predictions"        # per-instance & per-group outputs
    BEHAVIOR = "behavior"              # response to perturbation / shift / attack
    PROCESS = "process"               # governance / documentation (not computed)


@dataclass(frozen=True)
class Principle:
    id: str
    name: str
    plain: str                        # one-line explanation in simple words
    tool_addressable: bool
    targets: List[InspectionTarget]
    sources: List[Source]             # which bodies name this principle


# ---------------------------------------------------------------------------
# The canonical set. Each entry is traceable to the four source frameworks.
# ---------------------------------------------------------------------------
PRINCIPLES: List[Principle] = [
    Principle(
        id="fairness",
        name="Fairness & Non-discrimination",
        plain="The model treats different groups of people equitably and does not encode bias.",
        tool_addressable=True,
        targets=[InspectionTarget.DATA, InspectionTarget.PREDICTIONS],
        sources=[Source.UNESCO, Source.MICROSOFT, Source.IBM, Source.GOOGLE],
    ),
    Principle(
        id="transparency",
        name="Transparency",
        plain="How the model was built and how it decides can be inspected and documented.",
        tool_addressable=True,
        targets=[InspectionTarget.MODEL_INTERNALS, InspectionTarget.PROCESS],
        sources=[Source.UNESCO, Source.MICROSOFT, Source.IBM, Source.GOOGLE],
    ),
    Principle(
        id="explainability",
        name="Explainability",
        plain="A human can understand why the model made a particular prediction.",
        tool_addressable=True,
        targets=[InspectionTarget.MODEL_INTERNALS, InspectionTarget.PREDICTIONS],
        sources=[Source.UNESCO, Source.IBM, Source.GOOGLE],
    ),
    Principle(
        id="privacy",
        name="Privacy & Data Protection",
        plain="The model does not leak or over-expose the personal data it was trained on.",
        tool_addressable=True,
        targets=[InspectionTarget.DATA, InspectionTarget.MODEL_INTERNALS, InspectionTarget.BEHAVIOR],
        sources=[Source.UNESCO, Source.MICROSOFT, Source.IBM, Source.GOOGLE],
    ),
    Principle(
        id="safety",
        name="Safety, Security & Robustness",
        plain="The model behaves reliably under noisy, shifted, or adversarial inputs.",
        tool_addressable=True,
        targets=[InspectionTarget.BEHAVIOR],
        sources=[Source.UNESCO, Source.MICROSOFT, Source.IBM, Source.GOOGLE],
    ),
    Principle(
        id="accountability",
        name="Accountability",
        plain="There is a clear, documented owner and audit trail for the model's decisions.",
        tool_addressable=False,  # supported by documentation, not a metric
        targets=[InspectionTarget.PROCESS],
        sources=[Source.UNESCO, Source.MICROSOFT, Source.GOOGLE],
    ),
    Principle(
        id="human_oversight",
        name="Human Oversight & Determination",
        plain="Humans stay in control and can review, override, or appeal decisions.",
        tool_addressable=False,
        targets=[InspectionTarget.PROCESS],
        sources=[Source.UNESCO],
    ),
    Principle(
        id="inclusiveness",
        name="Inclusiveness & Diversity",
        plain="The system works for people of all backgrounds and abilities, and small groups are not ignored.",
        tool_addressable=True,  # partially: subgroup / slice analysis
        targets=[InspectionTarget.DATA, InspectionTarget.PREDICTIONS],
        sources=[Source.UNESCO, Source.MICROSOFT, Source.GOOGLE],
    ),
    Principle(
        id="wellbeing",
        name="Sustainability & Well-being",
        plain="The system benefits society and the environment and avoids broad harm.",
        tool_addressable=False,
        targets=[InspectionTarget.PROCESS],
        sources=[Source.UNESCO, Source.GOOGLE],
    ),
]

BY_ID = {p.id: p for p in PRINCIPLES}


def get(principle_id: str) -> Principle:
    return BY_ID[principle_id]


def tool_addressable() -> List[Principle]:
    return [p for p in PRINCIPLES if p.tool_addressable]


def coverage_table() -> str:
    """Return a plain-text principle x source coverage table."""
    header = f"{'Principle':32} | UNESCO | MSFT | IBM | Google | Tool?"
    lines = [header, "-" * len(header)]
    for p in PRINCIPLES:
        def mark(s: Source) -> str:
            return "  x   " if s in p.sources else "      "
        lines.append(
            f"{p.name:32} |{mark(Source.UNESCO)}|{mark(Source.MICROSOFT)[:6]}|"
            f"{mark(Source.IBM)[:5]}|{mark(Source.GOOGLE)}| {'yes' if p.tool_addressable else 'no'}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(coverage_table())
