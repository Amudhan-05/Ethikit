"""Evaluator base class + a simple registry.

The whole extensibility story lives here. To add a new ethics tool you
subclass Evaluator, implement `evaluate`, and register it. Nothing else in
the framework has to change -- the runner discovers evaluators from the
registry, and the aggregator/report consume the standardised EvalResults.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type

from ..core.types import EvalContext, EvalResult, TaskType


class Evaluator(ABC):
    #: human-readable name, used in reports
    name: str = "evaluator"
    #: which canonical principle ids this evaluator speaks to
    principle_ids: List[str] = []
    #: which task types it supports
    supports: List[TaskType] = [TaskType.CLASSIFICATION, TaskType.REGRESSION]
    #: general, run-independent explanation of how this tool decides what it
    #: decides (which algorithm, what part of the model/data it looks at).
    #: This is the static counterpart to EvalResult.mechanism, which carries
    #: the same idea but filled in with this particular run's actual numbers.
    #: Powers ethikit.coverage's reverse principle -> tool lookup.
    mechanism: str = "(no mechanism description registered)"
    #: the maths behind the headline metric(s), as plain-text/Unicode lines so it
    #: renders the same in a terminal, a model card, and the web console. Each
    #: evaluate() call additionally puts the *same* formula filled in with this
    #: run's numbers into details["trace"] (see make_trace()).
    formula: str = ""
    #: how to install the optional dependency this tool needs ("" = nothing to install);
    #: shown in the console, `ethikit list tools` and every model card when the tool is skipped.
    install_hint: str = ""
    #: what this tool does NOT measure -- shown next to its numbers so nobody over-reads them.
    limits: str = ""

    def params(self) -> Dict[str, Any]:
        """Tunable knobs: every public scalar attribute set in __init__ (sample
        sizes, noise levels, attack budgets, seeds...). These are what
        `ethikit run --tool-param <tool>.<key>=<value>` and the console's
        advanced panel let a user override."""
        return {k: v for k, v in vars(self).items()
                if not k.startswith("_") and isinstance(v, (bool, int, float, str))}

    def configured(self, **overrides: Any) -> "Evaluator":
        """A copy of this evaluator with some params() overridden (cast to the
        default's type, so "0.2" from a CLI flag becomes a float)."""
        clone = copy.copy(self)
        current = self.params()
        for key, value in overrides.items():
            if key not in current:
                raise KeyError(f"{self.name} has no tunable param {key!r} "
                               f"(available: {', '.join(current) or 'none'})")
            kind = type(current[key])
            if kind is bool and isinstance(value, str):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            setattr(clone, key, kind(value))
        return clone

    def available(self) -> bool:
        """Override to return False when optional dependencies are missing."""
        return True

    def applies_to(self, ctx: EvalContext) -> bool:
        return ctx.dataset.task in self.supports

    @abstractmethod
    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        ...


_REGISTRY: Dict[str, Evaluator] = {}


def register(evaluator: Evaluator) -> Evaluator:
    _REGISTRY[evaluator.name] = evaluator
    return evaluator


def all_evaluators() -> List[Evaluator]:
    return list(_REGISTRY.values())


def available_evaluators() -> List[Evaluator]:
    return [e for e in _REGISTRY.values() if e.available()]


def make_trace(steps: List[str] = None, tables: List[dict] = None,
               charts: List[dict] = None, text: List[dict] = None) -> Dict[str, Any]:
    """Standard shape for details["trace"]: the intermediate artefacts behind a
    number, so every UI can show *how* it was computed without per-tool code.

    steps  -- the formula worked through with this run's actual values
    tables -- {"title", "columns": [...], "rows": [[...], ...]}
    charts -- {"title", "type": "bars"|"hist", "labels": [...],
               "series": [{"name", "values": [...]}], "threshold": optional float,
               "x_label"/"y_label": optional}
    text   -- {"title", "body"} for preformatted output (e.g. a surrogate tree's rules)
    """
    return {"steps": steps or [], "tables": tables or [],
            "charts": charts or [], "text": text or []}
