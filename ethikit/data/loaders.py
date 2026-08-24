"""Readily-available datasets for the demo.

`load_adult()` tries the classic UCI Adult income dataset via OpenML (needs
network -- works in Claude Code). If that fails, it transparently falls back
to a synthetic dataset with a KNOWN injected bias, so the pipeline always
runs and the fairness metrics always have something to detect.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..core.types import Dataset, TaskType


def make_synthetic_fairness_dataset(n=4000, seed=0) -> Dataset:
    """Synthetic binary-classification data with a deliberately biased label.

    The sensitive attribute `group` influences the label through a proxy
    feature, so a naive model will show demographic-parity gaps -- perfect
    for demonstrating the fairness evaluator.
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=n)                       # protected attribute
    x1 = rng.normal(0, 1, n) + group * 0.8                   # proxy correlated w/ group
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    logits = 1.2 * x1 + 0.6 * x2 - 0.3 * x3 + 0.9 * group    # group leaks into label
    p = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(int)

    X = np.column_stack([x1, x2, x3, group.astype(float)])
    feature_names = ["proxy_income", "feature_2", "feature_3", "group"]

    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X[tr])
    return Dataset(
        name="synthetic_biased",
        X_train=scaler.transform(X[tr]), X_test=scaler.transform(X[te]),
        y_train=y[tr], y_test=y[te], task=TaskType.CLASSIFICATION,
        sensitive_train=group[tr], sensitive_test=group[te],
        sensitive_name="group", feature_names=feature_names,
    )


def load_adult(seed=0) -> Dataset:
    """UCI Adult income; sensitive attribute = sex. Falls back to synthetic."""
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("adult", version=2, as_frame=True)
        df = data.frame.dropna()
        y = (df["class"].astype(str).str.contains(">50K")).astype(int).to_numpy()
        sensitive = (df["sex"].astype(str).values)
        feats = df.drop(columns=["class"])
        feats = feats.select_dtypes(include="number")  # keep it simple for the demo
        feature_names = list(feats.columns)
        X = feats.to_numpy(dtype=float)

        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=y)
        scaler = StandardScaler().fit(X[tr])
        return Dataset(
            name="adult_income",
            X_train=scaler.transform(X[tr]), X_test=scaler.transform(X[te]),
            y_train=y[tr], y_test=y[te], task=TaskType.CLASSIFICATION,
            sensitive_train=sensitive[tr], sensitive_test=sensitive[te],
            sensitive_name="sex", feature_names=feature_names,
        )
    except Exception as exc:  # no network / offline demo
        print(f"[loaders] Adult unavailable ({type(exc).__name__}); using synthetic data.")
        return make_synthetic_fairness_dataset(seed=seed)
