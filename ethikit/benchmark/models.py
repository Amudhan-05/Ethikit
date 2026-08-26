"""Readily-available model factories, grouped by family.

These are all stock scikit-learn estimators so the demo needs no extra
installs. `pip install xgboost lightgbm` in Claude Code lets you drop
gradient-boosted trees into the tree_based family the same way.
"""
from __future__ import annotations

from typing import Callable, Dict

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# family -> {model_name: factory}
CLASSIFIERS: Dict[str, Dict[str, Callable]] = {
    "linear": {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
    },
    "tree_based": {
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, random_state=0),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
        "gradient_boosting": lambda: GradientBoostingClassifier(random_state=0),
    },
    "instance_based": {
        "knn": lambda: KNeighborsClassifier(),
    },
    "probabilistic": {
        "naive_bayes": lambda: GaussianNB(),
    },
    "kernel": {
        "svc": lambda: SVC(probability=True, random_state=0),
    },
}

# family -> {model_name: factory}, mirroring CLASSIFIERS' families for regression
# (Phase 2: exercises FairnessEvaluator's group-MAE/MSE branch).
REGRESSORS: Dict[str, Dict[str, Callable]] = {
    "linear": {
        "ridge": lambda: Ridge(),
    },
    "tree_based": {
        "decision_tree_reg": lambda: DecisionTreeRegressor(max_depth=6, random_state=0),
        "random_forest_reg": lambda: RandomForestRegressor(n_estimators=200, random_state=0),
        "gradient_boosting_reg": lambda: GradientBoostingRegressor(random_state=0),
    },
    "instance_based": {
        "knn_reg": lambda: KNeighborsRegressor(),
    },
}


def flat_classifiers() -> Dict[str, Callable]:
    """{model_name: factory} across all classifier families."""
    return _flatten(CLASSIFIERS)


def flat_regressors() -> Dict[str, Callable]:
    """{model_name: factory} across all regressor families."""
    return _flatten(REGRESSORS)


def _flatten(registry: Dict[str, Dict[str, Callable]]) -> Dict[str, Callable]:
    out = {}
    for family in registry.values():
        out.update(family)
    return out


def family_of(model_name: str) -> str:
    for registry in (CLASSIFIERS, REGRESSORS):
        for fam, models in registry.items():
            if model_name in models:
                return fam
    return "unknown"
