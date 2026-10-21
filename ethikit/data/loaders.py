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
                  test_idx: Optional[np.ndarray] = None, scale: bool = True) -> Dataset:
    """One train/test split of a RawData, scaled with a scaler fit on train
    only (no leakage). Pass explicit train_idx/test_idx (e.g. from a
    StratifiedKFold) to reuse this for one CV fold instead of a fresh random
    split -- ethikit.benchmark.cv does exactly that. `scale=False` keeps raw
    feature values (needed when evaluating a model you trained yourself on
    unscaled data).
    """
    if train_idx is None or test_idx is None:
        idx = np.arange(len(raw.y))
        stratify = raw.y if raw.task == TaskType.CLASSIFICATION else None
        train_idx, test_idx = train_test_split(idx, test_size=test_size,
                                               random_state=seed, stratify=stratify)
    if scale:
        scaler = StandardScaler().fit(raw.X[train_idx])
        X_train, X_test = scaler.transform(raw.X[train_idx]), scaler.transform(raw.X[test_idx])
    else:
        X_train, X_test = raw.X[train_idx], raw.X[test_idx]
    return Dataset(
        name=raw.name,
        X_train=X_train, X_test=X_test,
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
    """Split version of make_synthetic_fairness_regression_raw()."""
    return split_dataset(make_synthetic_fairness_regression_raw(n=n, seed=seed), seed=seed)


def make_synthetic_fairness_regression_raw(n: int = 4000, seed: int = 0) -> RawData:
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
    return raw


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
        _ADULT_RAW_CACHE = adult_from_frame(data.frame)
    except Exception as exc:  # no network / offline demo
        print(f"[loaders] Adult unavailable ({type(exc).__name__}); using synthetic data.")
        _ADULT_RAW_CACHE = _synthetic_raw(seed=seed)
    return _ADULT_RAW_CACHE


def adult_from_frame(df) -> RawData:
    """UCI Adult as a RawData, categoricals included (workclass, education, marital-status,
    occupation, relationship, race, sex, native-country are one-hot encoded; levels beyond
    the top 20 collapse to 'other'). Target: income >50K = 1. Sensitive attribute: sex.
    `sex` stays in the feature set, like the synthetic data's `group` column, so the audit
    reflects a model that can see it; use `--exclude-sensitive` on your own CSVs to test
    fairness-through-unawareness."""
    df = df.dropna()
    df = df.assign(**{"class": df["class"].astype(str).str.strip().str.rstrip(".")})
    return prepare_frame(df, target="class", sensitive="sex", positive_label=">50K",
                         name="adult_income")


def load_adult(seed=0) -> Dataset:
    """UCI Adult income; sensitive attribute = sex. Falls back to synthetic."""
    return split_dataset(load_adult_raw(seed=seed), seed=seed)


# ---------------------------------------------------------------------------
# Bring-your-own data
# ---------------------------------------------------------------------------
def prepare_frame(df, target: str, sensitive: Optional[str] = None, name: str = "custom",
                  task: Optional[str] = None, drop: Optional[List[str]] = None,
                  exclude_sensitive: bool = False, sensitive_threshold: Optional[float] = None,
                  positive_label: Optional[str] = None, max_categories: int = 20) -> RawData:
    """Turn any pandas DataFrame into a RawData the whole pipeline can run on.

    - target: column to predict. Classification vs regression is inferred
      (non-numeric, or <= 20 distinct integer values -> classification) unless
      `task` is given. Class labels are encoded 0..k-1 in sorted order, so for
      a binary target the *later* label ("yes" > "no", ">50K" > "<=50K",
      "good" > "bad") is the positive / favourable class=1 unless
      `positive_label` says otherwise.
    - sensitive: column whose groups the fairness tools compare. A numeric
      column with many values (e.g. age) is split into two groups at
      `sensitive_threshold` (default: the median) -- e.g. "age<35" / "age>=35".
      `exclude_sensitive=True` drops it from the model's features
      ("fairness through unawareness" -- proxies can still leak it).
    - features: everything else minus `drop`; numeric NaNs -> median,
      categoricals one-hot encoded (top `max_categories` levels, rest -> "other").
    """
    import pandas as pd

    df = df.copy()
    drop = [c for c in (drop or []) if c in df.columns]
    missing = [c for c in [target, sensitive] if c and c not in df.columns]
    if missing:
        raise KeyError(f"column(s) not in data: {', '.join(missing)} "
                       f"(available: {', '.join(map(str, df.columns))})")
    df = df.dropna(subset=[c for c in [target, sensitive] if c])
    if len(df) < 50:
        raise ValueError(f"only {len(df)} usable rows after dropping missing target/sensitive "
                         f"values; need at least 50")

    y_raw = df[target]
    if task is None:
        numeric = pd.api.types.is_numeric_dtype(y_raw)
        few_ints = numeric and y_raw.nunique() <= 20 and np.allclose(y_raw, y_raw.round())
        task = "classification" if (not numeric or few_ints) else "regression"
    task_t = TaskType(task)
    if task_t == TaskType.CLASSIFICATION:
        labels = sorted(y_raw.astype(str).unique())
        if len(labels) < 2:
            raise ValueError(f"target '{target}' has a single class: {labels}")
        if positive_label is not None and len(labels) == 2:
            if str(positive_label) not in labels:
                raise ValueError(f"positive_label {positive_label!r} not in {labels}")
            labels = [l for l in labels if l != str(positive_label)] + [str(positive_label)]
        y = y_raw.astype(str).map({l: i for i, l in enumerate(labels)}).to_numpy()
    else:
        y = y_raw.astype(float).to_numpy()

    sens = None
    sens_name = sensitive
    if sensitive:
        s = df[sensitive]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            thr = float(s.median()) if sensitive_threshold is None else float(sensitive_threshold)
            sens = np.where(s >= thr, f"{sensitive}>={thr:g}", f"{sensitive}<{thr:g}")
        else:
            sens = s.astype(str).to_numpy()

    feats = df.drop(columns=[target] + drop + ([sensitive] if sensitive and exclude_sensitive else []))
    num = feats.select_dtypes(include="number")
    num = num.fillna(num.median())
    cat = feats.drop(columns=num.columns)
    if len(cat.columns):
        cat = cat.astype(str).fillna("missing")
        for c in cat.columns:
            top = cat[c].value_counts().index[:max_categories]
            cat[c] = cat[c].where(cat[c].isin(top), "other")
        cat = pd.get_dummies(cat, prefix_sep="=", dtype=float)
    X = pd.concat([num.astype(float), cat], axis=1)
    if X.shape[1] == 0:
        raise ValueError("no feature columns left after removing target/sensitive/drop")
    return RawData(X=X.to_numpy(dtype=float), y=y, sensitive=sens, sensitive_name=sens_name,
                   feature_names=[str(c) for c in X.columns], name=name, task=task_t)


def load_csv(path: str, target: str, sensitive: Optional[str] = None, **kw) -> RawData:
    """prepare_frame() over a CSV file on disk; dataset name = file stem."""
    import os
    import pandas as pd
    kw.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return prepare_frame(pd.read_csv(path), target, sensitive, **kw)


# ---------------------------------------------------------------------------
# Synthetic data with knobs -- "different kinds of data" without a network
# ---------------------------------------------------------------------------
def make_synthetic_raw(n: int = 4000, n_features: int = 4, bias: float = 0.9,
                       proxy_strength: float = 0.8, minority_share: float = 0.5,
                       positive_rate: float = 0.5, noise: float = 1.0,
                       n_categorical: int = 0, seed: int = 0,
                       task: str = "classification") -> RawData:
    """A tunable version of the synthetic biased dataset.

    bias            how strongly group membership drives the label directly
    proxy_strength  how strongly group leaks into the first feature (a proxy,
                    so dropping the group column doesn't remove the bias)
    minority_share  fraction of rows in group 1 (small = an under-represented group)
    positive_rate   approximate overall share of y=1 (class imbalance)
    noise           label noise; larger = harder problem, flatter models
    n_features      total numeric features (>= 2), most of them weak/irrelevant
    n_categorical   extra categorical (one-hot) features, to exercise mixed data
    """
    rng = np.random.default_rng(seed)
    n_features = max(2, int(n_features))
    group = (rng.uniform(size=n) < minority_share).astype(int)
    X = rng.normal(0, 1, size=(n, n_features))
    X[:, 0] += group * proxy_strength
    coefs = np.concatenate([[1.2, 0.6], rng.normal(0, 0.25, n_features - 2)])
    signal = X @ coefs + bias * group + rng.normal(0, noise, n)
    names = ["proxy_income"] + [f"feature_{i}" for i in range(2, n_features + 1)]
    cats = []
    for k in range(int(n_categorical)):
        levels = rng.integers(0, 4, size=n)
        signal += 0.3 * (levels == 0)
        onehot = np.eye(4)[levels]
        cats.append(onehot)
        names += [f"cat_{k + 1}={lvl}" for lvl in "abcd"]
    if cats:
        X = np.column_stack([X] + cats)
    X = np.column_stack([X, group.astype(float)])
    names.append("group")
    if task == "regression":
        y = signal * 2.0
    else:
        y = (signal > np.quantile(signal, 1 - positive_rate)).astype(int)
    return RawData(X=X, y=y, sensitive=group, sensitive_name="group", feature_names=names,
                   name=f"synthetic(n={n},bias={bias:g},minority={minority_share:g})",
                   task=TaskType(task))


_GERMAN_RAW_CACHE: Optional[RawData] = None


def load_german_credit_raw() -> RawData:
    """Statlog German Credit (OpenML 'credit-g', 1000 rows): good vs bad credit
    risk, sensitive attribute = age split at 25 (the conventional split in the
    fairness literature). Needs network; raises if unavailable (no silent
    fallback -- the console reports it instead)."""
    global _GERMAN_RAW_CACHE
    if _GERMAN_RAW_CACHE is None:
        from sklearn.datasets import fetch_openml
        df = fetch_openml("credit-g", version=1, as_frame=True).frame
        _GERMAN_RAW_CACHE = prepare_frame(df, target="class", sensitive="age",
                                          sensitive_threshold=25, positive_label="good",
                                          name="german_credit")
    return _GERMAN_RAW_CACHE
