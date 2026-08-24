"""Readily-available model factories, grouped by family.

These are all stock scikit-learn estimators so the demo needs no extra
installs. `pip install xgboost lightgbm` in Claude Code lets you drop
gradient-boosted trees into the tree_based family the same way.
"""
from __future__ import annotations

from typing import Callable, Dict

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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


def flat_classifiers() -> Dict[str, Callable]:
    """{model_name: factory} across all families."""
    out = {}
    for family in CLASSIFIERS.values():
        out.update(family)
    return out


def family_of(model_name: str) -> str:
    for fam, models in CLASSIFIERS.items():
        if model_name in models:
            return fam
    return "unknown"
