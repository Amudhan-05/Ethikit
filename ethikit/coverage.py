"""Coverage: which tool addresses which principle, and vice versa.

The forward direction -- given an evaluator, which principles does it speak
to -- already lives on every Evaluator as `principle_ids`
(ethikit/evaluators/base.py). This module is the reverse direction: given a
principle, which registered tool(s) can you reach for. It also carries each
tool's `mechanism` (the general, run-independent explanation of *how* it
decides what it decides) and the InspectionTarget(s) it touches, so choosing
a tool also tells you what part of the model/data it actually looks at.

Two distinct questions this answers:
  1. "I care about principle X -- which tool(s) should I run?"
     -> principle_to_tools() / coverage_table()
  2. "I'm using tool Y -- which principles does it speak to, and how does it
      actually decide?"
     -> tool_to_principles() / tool_info()

This is the *static*, run-independent counterpart to EvalResult.mechanism
(ethikit/core/types.py), which carries the same idea filled in with one
particular run's actual numbers (sample sizes, backend used, groups
compared, ...). Use this module to pick a tool before running anything;
read a result's `mechanism` field afterwards to see what it actually did.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from . import principles as principles_mod
from .evaluators.base import all_evaluators


def principle_to_tools() -> Dict[str, List[str]]:
    """principle_id -> sorted list of registered evaluator names that address it."""
    out: Dict[str, List[str]] = defaultdict(list)
    for ev in all_evaluators():
        for pid in ev.principle_ids:
            out[pid].append(ev.name)
    return {pid: sorted(names) for pid, names in out.items()}


def tool_to_principles() -> Dict[str, List[str]]:
    """evaluator name -> its principle_ids (a name-keyed view of the registry)."""
    return {ev.name: list(ev.principle_ids) for ev in all_evaluators()}


def tool_info(name: str) -> Dict[str, object]:
    """One evaluator's full self-description: principles, mechanism, targets, availability."""
    for ev in all_evaluators():
        if ev.name == name:
            return {
                "name": ev.name,
                "principle_ids": list(ev.principle_ids),
                "mechanism": ev.mechanism,
                "supports": [t.value for t in ev.supports],
                "available": ev.available(),
            }
    raise KeyError(f"no registered evaluator named {name!r}")


def coverage_table() -> str:
    """Human-readable principle -> tools (+ mechanism) table, mirroring the
    style of principles.coverage_table() (principle -> source bodies)."""
    p2t = principle_to_tools()
    lines = []
    for pid in sorted(p2t):
        principle = principles_mod.BY_ID.get(pid)
        title = f"{principle.name} ({pid})" if principle else pid
        lines.append(title)
        for name in p2t[pid]:
            info = tool_info(name)
            flag = "" if info["available"] else "  [currently unavailable]"
            lines.append(f"  - {name}{flag}: {info['mechanism']}")
    return "\n".join(lines)


if __name__ == "__main__":
    # importing this package registers all built-in evaluators
    from . import evaluators as _evaluators  # noqa: F401
    print(coverage_table())
