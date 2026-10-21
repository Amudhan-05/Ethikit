"""Run a reference suite and turn the per-dataset results into family-level evidence.

Pipeline, per dataset:  load -> (split | repeated k-fold) -> run the tool battery for every
model -> normalise each metric across the models *on that dataset* -> a model x principle
profile.  Then across the suite:

  model_table   mean +/- std over datasets of each model's score per principle
  family_table  same, after first averaging the models inside a family (so a family with more
                members isn't favoured)
  agreement     on how many datasets the family that leads overall also leads *there*
  significance  Friedman test over datasets, per principle, only when there is enough data

Everything here is a *tendency within this suite*. No function in this module names a "best
model"; shortlist() returns a starting set to re-test on your own data.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..benchmark.cv import run_benchmark_cv
from ..benchmark.models import CLASSIFIERS, family_of, flat_classifiers
from ..benchmark.runner import jsonable, run_benchmark, select_evaluators
from ..data.loaders import split_dataset
from ..scoring import normalize, objective_matrix
from ..scoring.significance import friedman_test
from .spec import DatasetUnavailable, Domain, STRENGTH_TEXT, load_spec

#: lighter than "everything": the slow adversarial attack and the optional heavy scanners are off
#: by default so a 15-dataset suite finishes in minutes; pass tools=[...] to change it.
DEFAULT_SUITE_TOOLS = ["performance", "fairness", "aif360", "explainability", "robustness",
                       "membership_inference"]
DEFAULT_SUITE_MODELS = ["logistic_regression", "decision_tree", "random_forest",
                        "gradient_boosting", "naive_bayes", "knn"]

Event = Dict[str, Any]

#: raw (un-normalised) metrics kept so the relative 0-1 scores can be read against real values.
#: (metric name -> short label). Relative scores say "better than the others here"; these say
#: "how good, actually".
KEY_METRICS = {
    "accuracy": "accuracy", "roc_auc": "AUC",
    "demographic_parity_diff": "parity gap", "equalized_odds_diff": "odds gap",
    "disparate_impact_gap": "|1-DI|", "membership_inference_advantage": "MI advantage",
    "perturbation_stability": "stability", "surrogate_fidelity": "fidelity",
}

#: shortlist default weights. Utility is included on purpose: a shortlist that ignores accuracy
#: would happily recommend models that are bad at the job. These are a value judgement, so the
#: weights used are always returned and shown.
SHORTLIST_WEIGHTS = {"utility": 0.25, "fairness": 0.30, "explainability": 0.15, "safety": 0.15,
                     "privacy": 0.10, "inclusiveness": 0.05}


@dataclass
class DomainResult:
    domain: str
    title: str
    strength: str
    models: List[str]
    tools: List[str]
    datasets: List[dict]                       # those that ran: id, title, n_rows, sensitive, seconds
    skipped: List[dict]                        # those that did not, with the reason
    per_dataset: pd.DataFrame                  # dataset, model, family, principle, score
    model_table: pd.DataFrame
    family_table: pd.DataFrame
    absolute_table: pd.DataFrame               # family x key metric, raw values (mean over datasets)
    top_family: Dict[str, Optional[str]]
    agreement: Dict[str, Optional[float]]
    significance: Dict[str, dict]
    caveats: List[str]
    cv: Optional[dict] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def frame(df):
            return {"columns": [str(c) for c in df.columns],
                    "rows": df.reset_index().to_dict(orient="records")}
        return jsonable({
            "domain": self.domain, "title": self.title, "strength": self.strength,
            "strength_text": STRENGTH_TEXT[self.strength], "models": self.models,
            "tools": self.tools, "datasets": self.datasets, "skipped": self.skipped,
            "per_dataset": self.per_dataset.to_dict(orient="records"),
            "model_table": self.model_table.reset_index().to_dict(orient="records"),
            "family_table": self.family_table.reset_index().to_dict(orient="records"),
            "absolute_table": self.absolute_table.reset_index().to_dict(orient="records"),
            "key_metric_labels": KEY_METRICS,
            "top_family": self.top_family, "agreement": self.agreement,
            "significance": self.significance, "caveats": self.caveats, "cv": self.cv,
            "notes": self.notes,
        })


class DomainError(RuntimeError):
    pass


def _profile(raw: pd.DataFrame) -> pd.DataFrame:
    """model x ({utility} + principles) normalised 0-1 for one batch of records."""
    return objective_matrix(normalize(raw))


def _long(profile: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = profile.reset_index().melt(id_vars="model", var_name="principle", value_name="score")
    out.insert(0, "dataset", dataset)
    return out


def run_domain(domain: Domain,
               models: Optional[Sequence[str]] = None,
               tools: Optional[Sequence[str]] = None,
               *, max_datasets: Optional[int] = None, max_rows: Optional[int] = 5000,
               seed: int = 0, test_size: float = 0.3,
               cv: Optional[Sequence[int]] = None,
               on_event: Optional[Callable[[Event], None]] = None,
               use_cache: bool = True) -> DomainResult:
    emit = on_event or (lambda e: None)
    if not domain.datasets:
        raise DomainError(f"{domain.id}: {STRENGTH_TEXT['gap']}. {' '.join(domain.caveats)}")
    zoo = flat_classifiers()
    names = list(models or DEFAULT_SUITE_MODELS)
    bad = [m for m in names if m not in zoo]
    if bad:
        raise DomainError(f"unknown model(s) {bad}; choose from {sorted(zoo)}")
    factories = {m: zoo[m] for m in names}
    tool_names = list(tools or DEFAULT_SUITE_TOOLS)
    evaluators = select_evaluators([t for t in tool_names if t])
    specs = domain.datasets[:max_datasets] if max_datasets else domain.datasets

    emit({"type": "domain_start", "domain": domain.id, "title": domain.title,
          "datasets": [{"id": s.id, "title": s.title} for s in specs], "models": names,
          "tools": [e.name for e in evaluators], "strength": domain.strength,
          "cv": list(cv) if cv else None})

    frames, raws, ran, skipped = [], [], [], []
    for i, spec in enumerate(specs):
        emit({"type": "dataset_start", "id": spec.id, "title": spec.title, "i": i + 1, "n": len(specs)})
        t0 = time.perf_counter()
        try:
            raw = load_spec(spec, max_rows=max_rows, seed=seed, use_cache=use_cache)
            if cv:
                folds, repeats = int(cv[0]), int(cv[1])
                cvdf = run_benchmark_cv(raw, factories, n_splits=folds, n_repeats=repeats, seed=seed,
                                        evaluators=evaluators)
                per = [_profile(g.drop(columns=["repeat", "fold", "seed"]))
                       for _, g in cvdf.groupby(["repeat", "fold"])]
                profile = pd.concat(per).groupby(level=0).mean()
                raw_records = cvdf
            else:
                ds = split_dataset(raw, seed=seed, test_size=test_size)
                raw_records = run_benchmark(ds, factories, evaluators=evaluators)
                profile = _profile(raw_records)
        except DatasetUnavailable as exc:
            skipped.append({"id": spec.id, "title": spec.title, "reason": str(exc)})
            emit({"type": "dataset_skip", "id": spec.id, "reason": str(exc)})
            continue
        frames.append(_long(profile, spec.id))
        keep = raw_records[raw_records["metric"].isin(KEY_METRICS)]
        raws.append(keep.groupby(["model", "metric"])["value"].mean().reset_index().assign(dataset=spec.id))
        info = {"id": spec.id, "title": spec.title, "n_rows": int(len(raw.y)),
                "positive_rate": round(float(np.mean(raw.y)), 3),
                "sensitive": spec.sensitive, "sensitive_groups": sorted(map(str, set(raw.sensitive))) if raw.sensitive is not None else [],
                "source": spec.source, "verified": spec.verified, "note": spec.note,
                "seconds": round(time.perf_counter() - t0, 1)}
        ran.append(info)
        emit({"type": "dataset_done", **info})

    if not frames:
        raise DomainError("no dataset in this suite could be loaded:\n  - " +
                          "\n  - ".join(f"{s['id']}: {s['reason']}" for s in skipped))

    per_dataset = pd.concat(frames, ignore_index=True)
    per_dataset["family"] = per_dataset["model"].map(family_of)
    result = _aggregate(domain, names, [e.name for e in evaluators], ran, skipped, per_dataset,
                        list(cv) if cv else None, pd.concat(raws, ignore_index=True))
    emit({"type": "domain_done", "domain": domain.id, "n_datasets": len(ran), "n_skipped": len(skipped)})
    return result


def _aggregate(domain, names, tool_names, ran, skipped, per, cv, raw_metrics) -> DomainResult:
    per = per[per["principle"].notna()]
    model_table = (per.groupby(["model", "principle"])["score"].agg(["mean", "std", "count"])
                   .rename(columns={"count": "n_datasets"}).round(3))
    fam_ds = per.groupby(["dataset", "family", "principle"])["score"].mean().reset_index()
    family_table = (fam_ds.groupby(["family", "principle"])["score"].agg(["mean", "std", "count"])
                    .rename(columns={"count": "n_datasets"}).round(3))

    raw_metrics = raw_metrics.assign(family=raw_metrics["model"].map(family_of))
    absolute = (raw_metrics.groupby(["dataset", "family", "metric"])["value"].mean().reset_index()
                .groupby(["family", "metric"])["value"].mean().unstack("metric").round(3))
    absolute = absolute[[m for m in KEY_METRICS if m in absolute.columns]]

    top_family: Dict[str, Optional[str]] = {}
    agreement: Dict[str, Optional[float]] = {}
    significance: Dict[str, dict] = {}
    notes: List[str] = []
    for principle, g in fam_ds.groupby("principle"):
        overall = g.groupby("family")["score"].mean()
        lead = overall.idxmax() if len(overall) else None
        top_family[principle] = lead
        wins = [d["family"].iloc[int(np.argmax(d["score"].to_numpy()))] for _, d in g.groupby("dataset")]
        agreement[principle] = round(float(np.mean([w == lead for w in wins])), 3) if wins else None

        mat = per[per["principle"] == principle].pivot_table(index="dataset", columns="model",
                                                               values="score")
        if mat.shape[0] < 3 or mat.shape[1] < 3:
            significance[principle] = {"skipped": f"needs >= 3 datasets and >= 3 models "
                                       f"(have {mat.shape[0]} and {mat.shape[1]})"}
            continue
        try:
            f = friedman_test(mat)
            significance[principle] = {"p_value": round(f["p_value"], 4),
                                       "significant_at_0.05": f["significant_at_0.05"],
                                       "n_datasets": f["n_samples"], "enough_data": f["enough_data"],
                                       "average_ranks": {k: round(float(v), 2) for k, v in f["average_ranks"].items()}}
        except Exception as exc:                       # constant columns, scipy missing, ...
            significance[principle] = {"skipped": f"{type(exc).__name__}: {exc}"}
    if len(ran) < 5:
        notes.append(f"Only {len(ran)} dataset(s) ran: treat every number here as indicative, not conclusive.")
    if skipped:
        notes.append(f"{len(skipped)} dataset(s) were skipped (see 'skipped'); results cover only those that ran.")
    return DomainResult(domain.id, domain.title, domain.strength if not skipped or len(ran) >= 5 else
                        ("small" if len(ran) > 1 else "single"),
                        names, tool_names, ran, skipped, per, model_table, family_table, absolute, top_family,
                        agreement, significance, list(domain.caveats), cv, notes)


def shortlist(result: DomainResult, weights: Optional[Dict[str, float]] = None,
              k: int = 2) -> Dict[str, Any]:
    """Which families to try first on your own data, given what you care about. Weighted mean
    of each family's suite-average score per principle (utility included only if weighted).
    A starting set to re-test, never a recommendation to deploy."""
    ft = result.family_table["mean"].unstack("principle")
    given = dict(weights or SHORTLIST_WEIGHTS)
    given.setdefault("utility", SHORTLIST_WEIGHTS["utility"])      # never silently ignore accuracy
    w = {p: v for p, v in given.items() if p in ft.columns and v > 0}
    if not w:
        raise DomainError("no weighted principle is present in this suite's results")
    total = sum(w.values())
    score = sum(ft[p].fillna(0) * v / total for p, v in w.items()).sort_values(ascending=False)
    fams = list(score.index[:k])
    models = [m for m in result.models if family_of(m) in fams]
    return {"families": fams, "models": models, "weights_used": {p: round(v / total, 3) for p, v in w.items()},
            "scores": {f: round(float(s), 3) for f, s in score.items()},
            "caveat": f"Evidence from {len(result.datasets)} dataset(s) in the '{result.domain}' suite. "
                      "Re-test on your own data before relying on it."}


def compare_domains(results: Dict[str, DomainResult]) -> pd.DataFrame:
    """Does the family ordering change from one domain to another? One row per
    (principle, domain pair): Spearman rho between the two orderings and whether the leading
    family is the same. Deliberately returns *no* single cross-domain winner."""
    rows = []
    ids = list(results)
    principles = sorted(set.intersection(*[set(r.top_family) for r in results.values()])) if results else []
    for p in principles:
        series = {d: results[d].family_table["mean"].xs(p, level="principle") for d in ids}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = series[ids[i]], series[ids[j]]
                common = a.index.intersection(b.index)
                rho = float(a[common].rank().corr(b[common].rank())) if len(common) >= 3 else np.nan
                rows.append({"principle": p, "domain_a": ids[i], "domain_b": ids[j],
                             "families_compared": len(common),
                             "spearman_rho": None if np.isnan(rho) else round(rho, 3),
                             "same_leading_family": bool(a[common].idxmax() == b[common].idxmax()) if len(common) else None,
                             "leading_a": a[common].idxmax() if len(common) else None,
                             "leading_b": b[common].idxmax() if len(common) else None})
    return pd.DataFrame(rows)
