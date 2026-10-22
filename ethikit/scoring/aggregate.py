"""Aggregation & scoring.

Key design choice: we DO NOT collapse everything into one "ethics score" by
default. Several fairness criteria are mathematically incompatible (Kleinberg /
Chouldechova impossibility results), so a single ranking hides real trade-offs (ethikit is a selection-and-audit aid, not a ranker).
Instead we produce:

  1. normalize()      -> every metric mapped to [0,1] where 1 = better
  2. principle_profile() -> model x principle table (the radar data)
  3. pareto_front()   -> the set of non-dominated models over chosen objectives
  4. weighted_score() -> OPTIONAL single ranking, only once a human supplies
                          explicit weights (making the value judgement visible)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `norm` column: each metric min-max scaled so 1.0 is always best."""
    df = df[df["target"] != "error"].copy()
    df["norm"] = np.nan
    for metric, grp in df.groupby("metric"):
        vals = grp["value"].astype(float)
        lo, hi = vals.min(), vals.max()
        if np.isclose(hi, lo):
            scaled = pd.Series(1.0, index=vals.index)
        else:
            scaled = (vals - lo) / (hi - lo)
        if not bool(grp["higher_is_better"].iloc[0]):
            scaled = 1.0 - scaled
        df.loc[grp.index, "norm"] = scaled
    return df


def principle_profile(df_norm: pd.DataFrame) -> pd.DataFrame:
    """model x principle mean normalized score (the radar-chart matrix).

    Excludes the performance evaluator: utility is a separate axis, not a
    principle, and must not be double-counted inside the principle columns.
    """
    principles_only = df_norm[df_norm["evaluator"] != "performance"]
    return (principles_only.pivot_table(index="model", columns="principle",
                                        values="norm", aggfunc="mean")
            .round(3))


def _utility_series(df_norm: pd.DataFrame) -> pd.Series:
    util = df_norm[df_norm["evaluator"] == "performance"]
    return util.groupby("model")["norm"].mean()


def objective_matrix(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Principle profile + a `utility` column: the full trade-off space that
    the Pareto front is computed over (so fairness-vs-accuracy shows up)."""
    prof = principle_profile(df_norm)
    prof.insert(0, "utility", _utility_series(df_norm).round(3))
    prof.attrs = {}          # don't drag the run's traces along: pandas deep-copies attrs on every op
    return prof


def pareto_front(profile: pd.DataFrame,
                 objectives: Optional[List[str]] = None) -> List[str]:
    """Return models not dominated on ALL chosen objective columns."""
    cols = objectives or [c for c in profile.columns]
    mat = profile[cols].fillna(0.0)
    models = list(mat.index)
    front = []
    for m in models:
        dominated = False
        for other in models:
            if other == m:
                continue
            if (mat.loc[other] >= mat.loc[m]).all() and (mat.loc[other] > mat.loc[m]).any():
                dominated = True
                break
        if not dominated:
            front.append(m)
    return front


def weighted_score(profile: pd.DataFrame,
                   weights: Dict[str, float]) -> pd.Series:
    """Single ranking from explicit, caller-supplied per-principle weights."""
    cols = [c for c in weights if c in profile.columns]
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    score = (profile[cols].fillna(0.0).to_numpy() @ w)
    return pd.Series(score, index=profile.index, name="weighted_score").sort_values(ascending=False)


def leaderboard(df: pd.DataFrame,
                weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Convenience: utility + per-principle profile + optional weighted score.

    `weights` keys are principle ids; utility is never auto-weighted (add a
    "utility" key yourself if you want to trade it off explicitly).
    """
    dn = normalize(df)
    prof = objective_matrix(dn)
    if weights:
        prof["weighted_score"] = weighted_score(prof, weights).round(3)
        prof = prof.sort_values("weighted_score", ascending=False)
    return prof
