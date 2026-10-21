"""Importing this package registers all built-in evaluators."""
from . import (  # noqa: F401
    performance, fairness, explainability, robustness, privacy, giskard_scan,
    art_robustness, drift, aif360_eval,
)
from .base import Evaluator, register, all_evaluators, available_evaluators  # noqa: F401
