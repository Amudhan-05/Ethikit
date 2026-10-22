"""Dataset profile: what the data itself says before any model is trained.

Most of ethikit inspects predictions or behaviour. Several of the most common fairness problems
are visible in the *data* already:

  * representation -- how many rows each sensitive group has (a 3% group gets 3% of the say),
  * base rates      -- how often each group has the positive label (a model that learns the
                       data faithfully reproduces any gap here),
  * proxies         -- features that let a model recover the sensitive attribute even when that
                       attribute is removed ("fairness through unawareness" fails when proxies exist),
  * missingness     -- columns that were mostly empty before imputation.

The proxy screen fits a small decision tree to predict the sensitive attribute from the *other*
features (cross-validated AUC: 0.5 = no information, 1.0 = perfectly recoverable) and ranks single
features by mutual information with it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .core.types import TaskType
from .data.loaders import RawData

SMALL_GROUP_SHARE = 0.10
PROXY_AUC_WARN = 0.65


def _sensitive_columns(raw: RawData) -> List[int]:
    """Indices of feature columns that ARE the sensitive attribute (or its one-hot parts)."""
    name = (raw.sensitive_name or "").lower()
    if not name:
        return []
    return [i for i, f in enumerate(raw.feature_names)
            if f.lower() == name or f.lower().startswith(name + "=")]


def profile_raw(raw: RawData, source_frame: Optional[pd.DataFrame] = None,
                seed: int = 0, max_rows: int = 20000) -> Dict[str, Any]:
    n = len(raw.y)
    out: Dict[str, Any] = {
        "dataset": raw.name, "task": raw.task.value, "n_rows": int(n),
        "n_features": int(raw.X.shape[1]), "sensitive": raw.sensitive_name, "warnings": [],
    }
    if raw.task == TaskType.CLASSIFICATION:
        out["positive_rate"] = round(float(np.mean(raw.y == 1)), 4)

    # ---- groups: size, share, base rate
    if raw.sensitive is not None:
        g = pd.Series(np.asarray(raw.sensitive).astype(str))
        rows = []
        for grp, idx in g.groupby(g).groups.items():
            idx = np.asarray(list(idx))
            row = {"group": grp, "n": int(len(idx)), "share": round(len(idx) / n, 4)}
            if raw.task == TaskType.CLASSIFICATION:
                row["base_rate"] = round(float(np.mean(raw.y[idx] == 1)), 4)
            else:
                row["mean_target"] = round(float(np.mean(raw.y[idx])), 4)
            rows.append(row)
        rows.sort(key=lambda r: -r["n"])
        out["groups"] = rows
        small = [r["group"] for r in rows if r["share"] < SMALL_GROUP_SHARE]
        if small:
            out["warnings"].append(f"under-represented group(s) below {SMALL_GROUP_SHARE:.0%} of rows: "
                                   f"{', '.join(small)} -- their metrics are noisier and easier to ignore")
        if raw.task == TaskType.CLASSIFICATION and len(rows) > 1:
            rates = [r["base_rate"] for r in rows]
            gap = max(rates) - min(rates)
            out["base_rate_gap"] = round(gap, 4)
            if gap >= 0.1:
                out["warnings"].append(f"base rates differ by {gap:.2f} between groups in the data itself: "
                                       f"a model that fits the data well will tend to reproduce that gap")
        out["proxy"] = _proxy_screen(raw, seed=seed, max_rows=max_rows)
        if out["proxy"].get("auc") is not None and out["proxy"]["auc"] >= PROXY_AUC_WARN:
            top = ", ".join(f["feature"] for f in out["proxy"]["top_features"][:3])
            out["warnings"].append(f"the other features predict '{raw.sensitive_name}' with AUC "
                                   f"{out['proxy']['auc']:.2f}: dropping it would not remove it (main proxies: {top})")

    # ---- missingness (only knowable from the original frame, before imputation)
    if source_frame is not None:
        miss = source_frame.isna().mean().sort_values(ascending=False)
        out["missing"] = [{"column": str(c), "share_missing": round(float(v), 4)}
                          for c, v in miss.items() if v > 0][:20]
        high = [m["column"] for m in out["missing"] if m["share_missing"] > 0.2]
        if high:
            out["warnings"].append(f"columns more than 20% empty before imputation: {', '.join(high)}")
    return out


def _proxy_screen(raw: RawData, seed: int, max_rows: int) -> Dict[str, Any]:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    s = np.asarray(raw.sensitive).astype(str)
    groups, counts = np.unique(s, return_counts=True)
    if len(groups) < 2:
        return {"auc": None, "reason": "only one group present"}
    drop = set(_sensitive_columns(raw))
    keep = [i for i in range(raw.X.shape[1]) if i not in drop]
    if not keep:
        return {"auc": None, "reason": "no features other than the sensitive attribute"}
    X = raw.X[:, keep]
    names = [raw.feature_names[i] for i in keep]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(s), size=min(len(s), max_rows), replace=False)
    X, s = X[idx], s[idx]
    # one-vs-rest on the largest group keeps the AUC readable for >2 groups
    target = (s == groups[np.argmax(counts)]).astype(int)
    if target.min() == target.max():
        return {"auc": None, "reason": "sample contains a single group"}
    folds = max(2, min(5, int(np.bincount(target).min())))
    if folds < 2:
        return {"auc": None, "reason": "too few rows in the smaller group"}
    tree = DecisionTreeClassifier(max_depth=4, random_state=seed)
    proba = cross_val_predict(tree, X, target, cv=StratifiedKFold(folds, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1]
    auc = float(roc_auc_score(target, proba))
    mi = mutual_info_classif(X, target, random_state=seed)
    order = np.argsort(mi)[::-1][:10]
    return {"auc": round(auc, 4), "reference_group": str(groups[np.argmax(counts)]),
            "method": f"depth-4 decision tree, {folds}-fold out-of-fold AUC, predicting "
                      f"'{groups[np.argmax(counts)]}' vs the rest from {len(names)} non-sensitive features",
            "top_features": [{"feature": names[i], "mutual_information": round(float(mi[i]), 4)}
                             for i in order if mi[i] > 0]}
