"""Friedman + Nemenyi significance testing across multiple independent
samples (CV folds, or datasets in a meta-benchmark) -- so a "model A beats
model B" claim is backed by a hypothesis test, not just a higher mean that
could be one lucky split away from reversing.

Expects a `scores` DataFrame shaped exactly like
ethikit.benchmark.cv.score_distribution()'s output: index = independent
sample (a (repeat, fold) pair, or a dataset name), columns = model, values =
one comparable score per (sample, model) pair -- e.g. one metric's raw
value, or a normalized principle score. Same shape works whether the
samples come from CV folds or from ethikit.benchmark.meta's per-dataset runs.

Rule of thumb (Demsar, 2006, "Statistical Comparisons of Classifiers over
Multiple Data Sets"): the Friedman test needs >= 3 models and >= ~5
independent samples to have real statistical power; `friedman_test()` flags
this via `enough_data` rather than silently producing a p-value that looks
more authoritative than the sample size supports.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from scipy import stats


def rank_matrix(scores: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Rank-transform each row (sample) across models, 1 = best.

    `ascending=False` (default) is for "higher is better" scores -- matches
    every ethikit.scoring.normalize() output, where 1.0 is always best. Pass
    `ascending=True` if you hand this a raw lower-is-better metric instead
    (e.g. an unnormalized error rate straight from score_distribution()).
    """
    return scores.rank(axis=1, ascending=ascending, method="average")


def friedman_test(scores: pd.DataFrame, ascending: bool = False) -> Dict[str, object]:
    """Friedman chi-square test: null hypothesis = every model has the same
    average rank across samples. Returns the statistic, p-value, each
    model's average rank (best first), and whether there's enough data
    (Demsar's rule of thumb) for the result to mean much.
    """
    if scores.shape[1] < 3:
        raise ValueError(
            f"Friedman needs >= 3 models to compare; got {scores.shape[1]}: "
            f"{list(scores.columns)}"
        )
    clean = scores.dropna()
    if len(clean) < 2:
        raise ValueError(
            f"Only {len(clean)} complete (no-NaN) samples across {scores.shape[1]} "
            "models -- need at least a couple of independent samples to test anything."
        )
    stat, p = stats.friedmanchisquare(*[clean[c].to_numpy() for c in clean.columns])
    ranks = rank_matrix(clean, ascending=ascending).mean(axis=0).sort_values()
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "n_samples": int(len(clean)),
        "n_models": int(scores.shape[1]),
        "average_ranks": ranks,
        "significant_at_0.05": bool(p < 0.05),
        "enough_data": len(clean) >= 5 and scores.shape[1] >= 3,
    }


def nemenyi_posthoc(scores: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Nemenyi post-hoc p-values (model x model), via
    scikit-posthocs. Only meaningful to interpret once friedman_test() has
    already rejected the null (p < 0.05) -- Nemenyi tells you WHICH pairs
    differ, not WHETHER any of them do.
    """
    import scikit_posthocs as sp
    return sp.posthoc_nemenyi_friedman(scores.dropna())


def compare_models(scores: pd.DataFrame, ascending: bool = False) -> Dict[str, object]:
    """One call: Friedman, then Nemenyi if (and only if) Friedman is
    significant. Always returns average_ranks; `nemenyi` is None when the
    Friedman test didn't reject the null (or scikit-posthocs isn't
    installed), so the caller doesn't accidentally read pairwise
    significance into noise-level differences.
    """
    result = friedman_test(scores, ascending=ascending)
    result["nemenyi"] = None
    if result["significant_at_0.05"]:
        try:
            result["nemenyi"] = nemenyi_posthoc(scores)
        except ImportError:
            pass
    return result


def critical_difference_diagram(scores: pd.DataFrame, out_path: str,
                                ascending: bool = False,
                                title: Optional[str] = None) -> str:
    """Render the classic Demsar critical-difference diagram: each model's
    average rank on a line, with a horizontal bar connecting every group of
    models that are NOT significantly different from each other (Nemenyi).
    Needs scikit-posthocs (for the plotting helper) and matplotlib.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless -- this may run outside a GUI/notebook context
    import matplotlib.pyplot as plt
    import scikit_posthocs as sp

    clean = scores.dropna()
    ranks = rank_matrix(clean, ascending=ascending).mean(axis=0).sort_values()
    sig_matrix = nemenyi_posthoc(clean)

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.4 * len(ranks)))
    sp.critical_difference_diagram(ranks, sig_matrix, ax=ax)
    ax.set_title(title or f"Critical difference (Nemenyi, n={len(clean)} samples)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
