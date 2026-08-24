"""Evaluator base class + a simple registry.

The whole extensibility story lives here. To add a new ethics tool you
subclass Evaluator, implement `evaluate`, and register it. Nothing else in
the framework has to change -- the runner discovers evaluators from the
registry, and the aggregator/report consume the standardised EvalResults.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Type

from ..core.types import EvalContext, EvalResult, TaskType


class Evaluator(ABC):
    #: human-readable name, used in reports
    name: str = "evaluator"
    #: which canonical principle ids this evaluator speaks to
    principle_ids: List[str] = []
    #: which task types it supports
    supports: List[TaskType] = [TaskType.CLASSIFICATION, TaskType.REGRESSION]

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
