"""Readily-available model factories, grouped by family.

These are all stock scikit-learn estimators so the demo needs no extra
installs. `pip install xgboost lightgbm` in Claude Code lets you drop
gradient-boosted trees into the tree_based family the same way.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
    # deep learning on tabular data (M5). scikit-learn's MLP is always available; the PyTorch MLP
    # and its differentially-private twin are added below when torch / opacus are installed.
    "deep": {
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                                     random_state=0),
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
    "deep": {
        "mlp_reg": lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                                        random_state=0),
    },
}


def _register_torch_models() -> None:
    """PyTorch models join the 'deep' family only if torch is importable (and DP only with opacus)."""
    from ..deep import TorchMLPClassifier, has_opacus, has_torch
    if has_torch():
        CLASSIFIERS["deep"]["torch_mlp"] = lambda: TorchMLPClassifier()
        if has_opacus():
            CLASSIFIERS["deep"]["torch_mlp_dp"] = lambda: TorchMLPClassifier(dp=True)


_register_torch_models()

#: what to install to unlock models that aren't registered on this machine
OPTIONAL_MODELS = {"torch_mlp": "pip install torch", "torch_mlp_dp": "pip install torch opacus"}


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


# ---------------------------------------------------------------------------
# Power-user hooks: every hyperparameter, any estimator, your own fitted model
# ---------------------------------------------------------------------------
def default_params(model_name: str) -> Dict[str, object]:
    """Every constructor parameter of a zoo model, as sklearn's get_params()
    reports it (shallow), with the zoo's own overrides already applied."""
    factory = {**flat_classifiers(), **flat_regressors()}[model_name]
    return factory().get_params(deep=False)


def _coerce(value, default):
    """CLI/console strings -> the type the estimator expects: Python literals
    ("200", "0.1", "None", "'balanced'", "(64, 32)") or a bare string ("rbf")."""
    import ast
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def with_params(factory: Callable, overrides: Dict[str, object]) -> Callable:
    """Wrap a factory so the estimator it builds gets `overrides` via set_params
    (unknown keys fail loudly when the model is built, naming the valid ones)."""
    if not overrides:
        return factory

    def build():
        est = factory()
        current = est.get_params(deep=False)
        bad = [k for k in overrides if k not in current]
        if bad:
            raise KeyError(f"{type(est).__name__} has no parameter(s) {', '.join(bad)}; "
                           f"valid: {', '.join(sorted(current))}")
        est.set_params(**{k: _coerce(v, current[k]) for k, v in overrides.items()})
        return est
    return build


def estimator_from_path(path: str, params: Optional[Dict[str, object]] = None) -> Callable:
    """Factory for ANY importable estimator class, e.g.
    'sklearn.ensemble.ExtraTreesClassifier' or 'xgboost.XGBClassifier' -- so
    the zoo is a default, not a limit."""
    import importlib
    module, _, cls = path.rpartition(".")
    klass = getattr(importlib.import_module(module), cls)
    return with_params(klass, params or {})


def load_fitted_model(path: str):
    """Load an already-trained estimator saved with joblib/pickle, to evaluate
    as-is. Only load files you trust: unpickling runs arbitrary code. The model
    must accept the same feature columns ethikit builds from your data (see
    prepare_frame), and -- if you trained it on unscaled features -- run with
    scaling disabled (`--no-scale`)."""
    from ..deep import adapt
    low = str(path).lower()
    if low.endswith((".pt", ".pth")):
        import torch
        return adapt(torch.load(path, weights_only=False))
    if low.endswith((".keras", ".h5")):
        import keras                                   # only when you actually load a Keras file
        return adapt(keras.models.load_model(path))
    try:
        import joblib
        return adapt(joblib.load(path))
    except ImportError:
        import pickle
        with open(path, "rb") as fh:
            return adapt(pickle.load(fh))
