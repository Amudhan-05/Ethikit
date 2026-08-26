"""Readily-available datasets for the demo.

`load_adult()` tries the classic UCI Adult income dataset via OpenML (needs
network -- works in Claude Code). If that fails, it transparently falls back
to a synthetic dataset with a KNOWN injected bias, so the pipeline always
runs and the fairness metrics always have something to detect.

Every loader is now built on two layers so both a single train/test split
and repeated cross-validation (ethikit.benchmark.cv) scale/split the SAME
underlying data consistently, with no leakage either way:
  - a `RawData` (unsplit X/y + optional sensitive attribute) -- fetched/built
    once and cached in-process, since CV needs many splits of the same data
    and refetching per split would be wasteful (and, for OpenML, slow).
  - `split_dataset(raw, ...)` -- one train/test split, scaled with a
    StandardScaler fit on train only. `make_synthetic_fairness_dataset()`
    and `load_adult()` are thin wrappers around this for backward
    compatibility with the single-split demo flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..core.types import Dataset, TaskType


@dataclass
class RawData:
    """Unsplit X/y (+ optional sensitive attribute): the shared input to both
    a single split and ethikit.benchmark.cv's repeated K-fold splitting."""
    X: np.ndarray
    y: np.ndarray
    sensitive: Optional[np.ndarray]
    sensitive_name: Optional[str]
    feature_names: List[str]
    name: str
    task: TaskType = TaskType.CLASSIFICATION


def split_dataset(raw: RawData, seed: int = 0, test_size: float = 0.3,
                  train_idx: Optional[np.ndarray] = None,
                  test_idx: Optional[np.ndarray] = None) -> Dataset:
    """One train/test split of a RawData, scaled with a scaler fit on train
    only (no leakage). Pass explicit train_idx/test_idx (e.g. from a
    StratifiedKFold) to reuse this for one CV fold instead of a fresh random
    split -- ethikit.benchmark.cv does exactly that.
    """
    if train_idx is None or test_idx is None:
        idx = np.arange(len(raw.y))
        stratify = raw.y if raw.task == TaskType.CLASSIFICATION else None
        train_idx, test_idx = train_test_split(idx, test_size=test_size,
                                               random_state=seed, stratify=stratify)
    scaler = StandardScaler().fit(raw.X[train_idx])
    return Dataset(
        name=raw.name,
        X_train=scaler.transform(raw.X[train_idx]), X_test=scaler.transform(raw.X[test_idx]),
        y_train=raw.y[train_idx], y_test=raw.y[test_idx], task=raw.task,
        sensitive_train=None if raw.sensitive is None else raw.sensitive[train_idx],
        sensitive_test=None if raw.sensitive is None else raw.sensitive[test_idx],
        sensitive_name=raw.sensitive_name, feature_names=raw.feature_names,
    )


def _synthetic_raw(n: int = 4000, seed: int = 0) -> RawData:
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
    return RawData(
        X=X, y=y, sensitive=group, sensitive_name="group",
        feature_names=["proxy_income", "feature_2", "feature_3", "group"],
        name="synthetic_biased", task=TaskType.CLASSIFICATION,
    )


def make_synthetic_fairness_raw(n: int = 4000, seed: int = 0) -> RawData:
    """Unsplit version of make_synthetic_fairness_dataset(), for repeated CV
    (ethikit.benchmark.cv.run_benchmark_cv needs the data before any split)."""
    return _synthetic_raw(n=n, seed=seed)


def make_synthetic_fairness_dataset(n=4000, seed=0) -> Dataset:
    """Synthetic binary-classification data with a deliberately biased label."""
    return split_dataset(make_synthetic_fairness_raw(n=n, seed=seed), seed=seed)


def make_synthetic_fairness_regression_dataset(n: int = 4000, seed: int = 0) -> Dataset:
    """Synthetic regression data with a deliberately biased continuous target.

    Same shape as make_synthetic_fairness_dataset but for TaskType.REGRESSION,
    so FairnessEvaluator's regression branch (group MAE/MSE gap) has
    something real to detect: `group` leaks directly into the target, so a
    naive regressor systematically over/under-predicts for one group.
    """
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 2, size=n)
    x1 = rng.normal(0, 1, n) + group * 0.8
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    noise = rng.normal(0, 1.0, n)
    y = 3.0 * x1 + 1.5 * x2 - 0.8 * x3 + 4.0 * group + noise
    X = np.column_stack([x1, x2, x3, group.astype(float)])
    raw = RawData(
        X=X, y=y, sensitive=group, sensitive_name="group",
        feature_names=["proxy_income", "feature_2", "feature_3", "group"],
        name="synthetic_biased_regression", task=TaskType.REGRESSION,
    )
    return split_dataset(raw, seed=seed)


_ADULT_RAW_CACHE: Optional[RawData] = None


def load_adult_raw(seed: int = 0) -> RawData:
    """Fetch + clean UCI Adult (unsplit), cached in-process so repeated CV
    folds don't refetch/reparse. `seed` only affects the synthetic fallback
    used if the network fetch fails -- the real fetch is seed-independent.
    """
    global _ADULT_RAW_CACHE
    if _ADULT_RAW_CACHE is not None:
        return _ADULT_RAW_CACHE
    try:
        from sklearn.datasets import fetch_openml
        data = fetch_openml("adult", version=2, as_frame=True)
        df = data.frame.dropna()
        y = (df["class"].astype(str).str.contains(">50K")).astype(int).to_numpy()
        sensitive = df["sex"].astype(str).values
        feats = df.drop(columns=["class"]).select_dtypes(include="number")  # keep it simple
        _ADULT_RAW_CACHE = RawData(
            X=feats.to_numpy(dtype=float), y=y, sensitive=sensitive, sensitive_name="sex",
            feature_names=list(feats.columns), name="adult_income", task=TaskType.CLASSIFICATION,
        )
    except Exception as exc:  # no network / offline demo
        print(f"[loaders] Adult unavailable ({type(exc).__name__}); using synthetic data.")
        _ADULT_RAW_CACHE = _synthetic_raw(seed=seed)
    return _ADULT_RAW_CACHE


def load_adult(seed=0) -> Dataset:
    """UCI Adult income; sensitive attribute = sex. Falls back to synthetic."""
    return split_dataset(load_adult_raw(seed=seed), seed=seed)
