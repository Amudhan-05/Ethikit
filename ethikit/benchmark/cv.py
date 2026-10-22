"""Repeated cross-validation: replaces a single train/test split with a
DISTRIBUTION of scores per model, across both folds and seeds.

Why this matters: everything in `run_demo.py` up to Phase 1 was a single
random split. A model that looks best can just be the one that got lucky on
that particular 30% test slice -- there's no way to tell from one number.
`run_benchmark_cv()` reuses `run_benchmark()` unchanged (the architecture
invariant holds: every row is still an EvalResult -> as_record()) but calls
it once per (repeat, fold) pair and tags each batch of rows with `repeat` /
`fold` / `seed` columns, so the result is a long DataFrame with N samples per
model per metric instead of 1. That's also exactly the shape
`ethikit.scoring.significance` needs for a Friedman test: one score per
model per independent sample.

"Nested" here means folds nested inside repeats (reshuffled each repeat),
not nested CV in the hyperparameter-tuning sense -- the model zoo has no
tunable hyperparameters yet to nest a search over. If/when one gets added,
the inner loop belongs inside this same per-fold loop, wrapping the model
factory in a search before it's fit.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ..core.types import TaskType
from ..data.loaders import RawData, split_dataset
from .runner import run_benchmark


def run_benchmark_cv(raw: RawData,
                     model_factories: Dict[str, Callable],
                     n_splits: int = 5,
                     n_repeats: int = 3,
                     seed: int = 0,
                     test_size: Optional[float] = None,
                     evaluators=None, on_event=None) -> pd.DataFrame:
    """Run the full evaluator battery over `n_repeats` x `n_splits` folds of
    `raw`. Each repeat reshuffles the folds with a different seed (drawn from
    `seed`), so this covers both cross-validation (multiple folds) and
    multi-seed (multiple shufflings) in one call.

    `test_size` is accepted for API symmetry with split_dataset but unused --
    K-fold's test fraction is 1/n_splits by construction; pass a different
    `n_splits` instead (e.g. n_splits=5 -> 20% test folds).

    Returns the same schema as run_benchmark() plus `repeat`, `fold`, `seed`.
    """
    splitter_cls = StratifiedKFold if raw.task == TaskType.CLASSIFICATION else KFold
    rng = np.random.default_rng(seed)
    batches = []
    traces = {}

    for repeat in range(n_repeats):
        fold_seed = int(rng.integers(0, 2**31 - 1))
        splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=fold_seed)
        split_args = (raw.X, raw.y) if raw.task == TaskType.CLASSIFICATION else (raw.X,)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(*split_args)):
            ds = split_dataset(raw, train_idx=train_idx, test_idx=test_idx)
            batch = run_benchmark(ds, model_factories, evaluators=evaluators, on_event=on_event)
            batch["repeat"] = repeat
            batch["fold"] = fold
            batch["seed"] = fold_seed
            batches.append(batch)
            traces.update({k: v for k, v in batch.attrs.get("traces", {}).items()})

    out = pd.concat(batches, ignore_index=True)
    out.attrs["traces"] = traces          # the last fold's traces: enough to show how a number is made
    return out


def score_distribution(cv_raw: pd.DataFrame, metric: str,
                       evaluator: Optional[str] = None,
                       principle: Optional[str] = None) -> pd.DataFrame:
    """model x (repeat, fold) matrix of one metric's raw value -- the shape
    ethikit.scoring.significance.friedman_test() and rank_matrix() expect.
    Pass `evaluator` and/or `principle` to disambiguate when a metric name
    isn't unique on its own -- e.g. giskard's "giskard_issue_score" is
    reported once per principle it buckets issues into (fairness, safety);
    without `principle` those two series get silently averaged together by
    the groupby below, which is wrong (they measure different things).
    """
    rows = cv_raw[cv_raw["metric"] == metric]
    if evaluator is not None:
        rows = rows[rows["evaluator"] == evaluator]
    if principle is not None:
        rows = rows[rows["principle"] == principle]
    dupes = rows.duplicated(subset=["model", "repeat", "fold"]).sum()
    if dupes:
        raise ValueError(
            f"metric={metric!r} evaluator={evaluator!r} principle={principle!r} still has "
            f"{dupes} duplicate (model, repeat, fold) rows -- narrow with `evaluator=`/"
            f"`principle=` (e.g. giskard emits this metric once per principle)."
        )
    rows = rows.assign(sample=list(zip(rows["repeat"], rows["fold"])))
    return rows.pivot_table(index="sample", columns="model", values="value")


def summarize_cv(cv_raw: pd.DataFrame) -> pd.DataFrame:
    """model x principle x metric mean +/- std across all (repeat, fold)
    samples -- the quick "is this stable or was it luck" sanity check before
    reaching for a significance test. Grouped by principle as well as metric
    because a metric name isn't always unique on its own -- see
    score_distribution()'s docstring (giskard's "giskard_issue_score")."""
    valid = cv_raw[cv_raw["target"] != "error"]
    agg = (valid.groupby(["model", "evaluator", "principle", "metric"])["value"]
          .agg(["mean", "std", "count"]))
    return agg.round(4).reset_index()
