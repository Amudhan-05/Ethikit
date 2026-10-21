"""Meta-benchmark: run the model zoo + ethics battery across MANY datasets
(via PMLB -- https://github.com/EpistasisLab/pmlb) so a family-level tendency
is an aggregate over many datasets, not a one-dataset anecdote (HANDOFF.md
Phase 2).

LIMITS: the default suite is a generic sample with no sensitive attribute (so the
fairness tools do not run in it) and is not organised by domain. For per-domain
evidence with fairness columns use ethikit.domains (reference suites).

PMLB over OpenML-CC18 here: no API key, direct GitHub-hosted CSVs, and a
stable, well-known dataset list -- fewer moving parts for a demo that needs
to "just run" in Claude Code.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import pandas as pd

from ..core.types import TaskType
from ..data.loaders import RawData, split_dataset
from .runner import run_benchmark

# A small, fast, reliably-fetchable slice of PMLB's binary-classification
# datasets -- confirmed by hand to fetch successfully and be genuinely
# binary (several names in pmlb.classification_dataset_names 404 against the
# actual hosted files, and plenty are multi-class; load_pmlb_dataset() skips
# both cases defensively either way). Deliberately excludes PMLB's "adult"
# (48842 rows, same dataset ethikit.data.load_adult() already fetches via
# OpenML) to avoid the same SVC(probability=True)-on-tens-of-thousands-of-
# rows slowdown documented in HANDOFF.md. Override with `dataset_names=` for
# a bigger sweep, e.g. a filtered slice of pmlb.classification_dataset_names.
DEFAULT_SUITE: List[str] = [
    "mushroom", "spambase", "ionosphere", "sonar", "tic_tac_toe",
    "breast_cancer", "biomed", "kr_vs_kp", "phoneme", "monk1",
]


def load_pmlb_dataset(name: str) -> Optional[RawData]:
    """Fetch one PMLB dataset and adapt it to RawData. Returns None (with a
    printed note) instead of raising if it's not fetchable, not binary
    classification, or has no numeric features -- mirrors load_adult()'s
    "keep the sweep running" philosophy rather than aborting the whole meta
    run over one bad dataset name.

    PMLB doesn't label a sensitive attribute, so FairnessEvaluator naturally
    sits out these rows (its applies_to() requires sensitive_test) while
    every other evaluator still runs; Giskard's PerformanceBiasDetector is
    the auto-discovered analogue that still catches subgroup disparities
    without one declared.
    """
    try:
        from pmlb import fetch_data
    except ImportError as exc:
        raise RuntimeError("pip install pmlb to run the meta-benchmark") from exc

    try:
        df = fetch_data(name)
    except Exception as exc:
        print(f"[meta] {name}: unfetchable ({type(exc).__name__}), skipping")
        return None
    if "target" not in df.columns:
        print(f"[meta] {name}: no 'target' column, skipping")
        return None
    if df["target"].nunique() != 2:
        print(f"[meta] {name}: not binary ({df['target'].nunique()} classes), skipping")
        return None
    feats = df.drop(columns=["target"]).select_dtypes(include="number")
    if feats.shape[1] == 0:
        print(f"[meta] {name}: no numeric features, skipping")
        return None

    y = (df["target"] == df["target"].max()).astype(int).to_numpy()
    return RawData(
        X=feats.to_numpy(dtype=float), y=y, sensitive=None, sensitive_name=None,
        feature_names=list(feats.columns), name=name, task=TaskType.CLASSIFICATION,
    )


def run_meta_benchmark(model_factories: Dict[str, Callable],
                       dataset_names: Optional[List[str]] = None,
                       seed: int = 0, test_size: float = 0.3) -> pd.DataFrame:
    """Run the evaluator battery once per dataset in `dataset_names` (default
    DEFAULT_SUITE), tagging every row with a `dataset` column. Datasets that
    fail to load (see load_pmlb_dataset) are skipped, not fatal.
    """
    names = dataset_names or DEFAULT_SUITE
    batches = []
    for name in names:
        raw = load_pmlb_dataset(name)
        if raw is None:
            continue
        ds = split_dataset(raw, seed=seed, test_size=test_size)
        batch = run_benchmark(ds, model_factories)
        batch["dataset"] = name
        batches.append(batch)
    if not batches:
        raise RuntimeError(
            "No datasets loaded successfully -- check network access or the "
            "dataset_names list (pmlb.classification_dataset_names for valid names)."
        )
    return pd.concat(batches, ignore_index=True)


def aggregate_across_datasets(meta_raw: pd.DataFrame) -> pd.DataFrame:
    """model x principle mean normalized score, averaged ACROSS datasets.

    Normalizes within each dataset first (so differing scales/units across
    datasets never distort the average), then means per model per principle
    over every dataset that model actually ran on -- a model missing from
    some datasets (e.g. one that errored) is simply averaged over fewer.
    """
    from ..scoring import normalize, principle_profile
    frames = [principle_profile(normalize(grp)) for _, grp in meta_raw.groupby("dataset")]
    combined = pd.concat(frames)
    return combined.groupby(combined.index).mean(numeric_only=True).round(3)


def weighted_score_across_datasets(meta_raw: pd.DataFrame,
                                   weights: Dict[str, float]) -> pd.DataFrame:
    """model x {mean_weighted_score, family, n_datasets}, averaged across
    datasets -- the generalised version of run_demo.py's single-dataset
    top-scoring-family block. Sort and groupby('family').head(1) on the
    result for the leading model of each family across this suite.
    """
    from ..scoring import normalize, objective_matrix, weighted_score
    from .models import family_of

    per_dataset = []
    for name, grp in meta_raw.groupby("dataset"):
        obj = objective_matrix(normalize(grp))
        obj["weighted_score"] = weighted_score(obj, weights)
        per_dataset.append(obj[["weighted_score"]].assign(dataset=name))
    stacked = pd.concat(per_dataset)

    board = stacked.groupby(stacked.index)["weighted_score"].mean().round(3)
    board = board.rename("mean_weighted_score").to_frame()
    board["family"] = [family_of(m) for m in board.index]
    board["n_datasets"] = stacked.groupby(stacked.index).size()
    return board.sort_values("mean_weighted_score", ascending=False)
