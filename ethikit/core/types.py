"""Shared data types that flow through the whole pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from ..principles import InspectionTarget


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class Dataset:
    """A tabular dataset plus the sensitive attribute used for fairness checks."""
    name: str
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    task: TaskType
    sensitive_train: Optional[np.ndarray] = None   # protected group per train row
    sensitive_test: Optional[np.ndarray] = None    # protected group per test row
    sensitive_name: Optional[str] = None
    feature_names: Optional[List[str]] = None


@dataclass
class EvalContext:
    """Everything an evaluator needs to inspect one fitted model."""
    model: Any
    dataset: Dataset
    y_pred: Any = None          # cached test predictions
    y_pred_train: Any = None    # cached train predictions (for leakage checks)
    y_proba: Any = None         # cached test probabilities if available


@dataclass
class EvalResult:
    """One standardised measurement, tagged so it can be aggregated and scored."""
    evaluator: str
    principle_id: str
    target: InspectionTarget
    metric: str
    value: float
    higher_is_better: bool
    details: Dict[str, Any] = field(default_factory=dict)

    def as_record(self, model_name: str) -> Dict[str, Any]:
        return {
            "model": model_name,
            "evaluator": self.evaluator,
            "principle": self.principle_id,
            "target": self.target.value,
            "metric": self.metric,
            "value": self.value,
            "higher_is_better": self.higher_is_better,
        }
