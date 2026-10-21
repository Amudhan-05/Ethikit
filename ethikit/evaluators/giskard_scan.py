"""Giskard evaluator.

Giskard (https://github.com/Giskard-AI/giskard) is a single unified scanner
that probes a fitted sklearn/XGBoost/LightGBM/CatBoost model for several
vulnerability classes in one pass: subgroup performance bias, perturbation
robustness, spurious correlations, overconfidence, etc. It's the long-term
backbone HANDOFF.md calls out -- one dependency that later extends to LLMs
too (`giskard.scan` also runs against text/chat models).

Unlike the other Phase-1 additions (SHAP, AIF360), Giskard installs cleanly
on Python 3.12 -- confirmed by hand during this session -- so this evaluator
has no fallback path; `available()` just gates on the import succeeding, the
same as every optional-dependency evaluator in this file's peers.

Mapping Giskard's own issue taxonomy onto ethikit's canonical principles
(see ethikit/principles.py):
  - "Performance" issues (a data slice performs worse than the global
    average) -> fairness: an auto-discovered, unlabelled analogue of the
    FairnessEvaluator's demographic-parity check, catching disparities on
    slices that aren't the one declared sensitive attribute.
  - "Robustness" / "Ethical" (protected-attribute perturbation) / spurious
    correlation / overconfidence / calibration issues -> safety: they are
    all about whether the model's behaviour holds up under perturbation.
  - We deliberately do NOT scan for "Data Leakage" here (that's Giskard's
    privacy-adjacent category) -- it's slower and Phase 1's privacy story is
    already an explicit proxy pending the Phase 3 membership-inference
    upgrade; claiming a `privacy` principle_id from an unrun detector would
    be dishonest. Revisit once Phase 3 lands.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from ..core.types import EvalContext, EvalResult, TaskType
from ..principles import InspectionTarget
from .base import Evaluator, make_trace, register

_LEVEL_WEIGHT = {"major": 3, "medium": 2, "minor": 1}

# Substring match against Giskard's IssueGroup.name (lowercased) -> our principle id.
_GROUP_TO_PRINCIPLE = [
    ("performance", "fairness"),
    ("ethical", "fairness"),
    ("spurious", "fairness"),
    ("robustness", "safety"),
    ("overconfidence", "safety"),
    ("underconfidence", "safety"),
    ("calibration", "safety"),
    ("stochasticity", "safety"),
]
_DEFAULT_PRINCIPLE = "safety"


def _has_giskard() -> bool:
    try:
        import giskard  # noqa: F401
        return True
    except Exception:
        return False


def _principle_for(group_name: str) -> str:
    name = (group_name or "").lower()
    for needle, principle in _GROUP_TO_PRINCIPLE:
        if needle in name:
            return principle
    return _DEFAULT_PRINCIPLE


def _quiet_giskard() -> None:
    """Disable telemetry and drop Giskard's own logger to WARNING.

    Two independent reasons: (1) an ethics-scanning tool phoning home by
    default is exactly the kind of thing this project should be explicit
    about, so we opt out; (2) Giskard's INFO logs contain a literal 'Δ'
    character, and Python's default stdout encoding on Windows (cp1252)
    can't encode it -- non-fatal (the logging module swallows the error)
    but it floods stderr with a "Logging error" traceback per line scanned.
    """
    try:
        import giskard
        giskard.settings.disable_analytics = True
        giskard.settings.loglevel = "WARNING"
    except Exception:
        pass
    # giskard.scanner.logger.logger sets its OWN explicit INFO level at import time
    # (giskard/scanner/logger.py: `logger.setLevel(logging.INFO)`), which overrides
    # inheritance from the "giskard" ancestor logger below -- has to be silenced by
    # its exact dotted name, not just the package root.
    logging.getLogger("giskard").setLevel(logging.WARNING)
    logging.getLogger("giskard.scanner.logger").setLevel(logging.WARNING)


class GiskardEvaluator(Evaluator):
    name = "giskard"
    principle_ids = ["fairness", "safety"]
    supports = [TaskType.CLASSIFICATION]  # giskard.Model wiring below is classification-only
    mechanism = (
        "Runs Giskard's automated scanner over a random sample of test rows, "
        "treating the model as a black-box predict_proba function: "
        "PerformanceBiasDetector auto-discovers data slices via loss-guided "
        "slicing and flags ones whose performance metric is significantly worse "
        "than the global average (-> fairness); NumericalPerturbationDetector and "
        "EthicalBiasDetector perturb feature values (noise, rescaling, "
        "protected-attribute swaps) and flag high failure rates (-> safety). "
        "Issues are weighted by severity (major=3/medium=2/minor=1) and summed "
        "per principle bucket. Deliberately skips the data-leakage detector -- "
        "see this module's docstring."
    )
    formula = (
        "giskard.scan(model, sample, only=[performance, robustness]) → issues\n"
        "weight(issue) = 3 (major) | 2 (medium) | 1 (minor)\n"
        "giskard_issue_score(principle) = Σ weight(issue) over issues mapped to that principle"
    )

    def __init__(self, sample_size: int = 300, seed: int = 0):
        self.sample_size = sample_size
        self.seed = seed

    def available(self) -> bool:
        return _has_giskard()

    def applies_to(self, ctx: EvalContext) -> bool:
        return ctx.dataset.task in self.supports and _has_giskard()

    def evaluate(self, ctx: EvalContext) -> List[EvalResult]:
        _quiet_giskard()
        import giskard

        ds = ctx.dataset
        names = ds.feature_names or [f"f{i}" for i in range(np.asarray(ds.X_test).shape[1])]
        Xt = np.asarray(ds.X_test, dtype=float)
        y = np.asarray(ds.y_test)

        rng = np.random.default_rng(self.seed)
        n = min(self.sample_size, len(Xt))
        idx = rng.choice(len(Xt), size=n, replace=False)

        df = pd.DataFrame(Xt[idx], columns=names)
        df["__target__"] = y[idx].astype(str)

        classes = [str(c) for c in getattr(ctx.model, "classes_", sorted(set(df["__target__"])))]
        predict_fn = ctx.model.predict_proba if hasattr(ctx.model, "predict_proba") else \
            (lambda X: np.eye(len(classes))[ctx.model.predict(X).astype(int)])

        gmodel = giskard.Model(
            model=predict_fn, model_type="classification",
            classification_labels=classes, feature_names=names,
        )
        gdataset = giskard.Dataset(
            df=df, target="__target__",
            column_types={f: "numeric" for f in names},
        )

        report = giskard.scan(gmodel, gdataset, only=["performance", "robustness"], verbose=False)

        buckets: Dict[str, Dict[str, object]] = {
            p: {"weight": 0.0, "issues": []} for p in set(pr for _, pr in _GROUP_TO_PRINCIPLE)
        }
        for issue in report.issues:
            principle = _principle_for(issue.group.name)
            level = getattr(issue.level, "value", issue.level)
            weight = _LEVEL_WEIGHT.get(str(level).lower(), 1)
            bucket = buckets.setdefault(principle, {"weight": 0.0, "issues": []})
            bucket["weight"] += weight
            bucket["issues"].append({
                "level": str(level), "detector": getattr(issue, "detector_name", "?"),
                "description": (issue.description or "")[:200],
            })

        results = []
        for principle in self.principle_ids:
            data = buckets.get(principle, {"weight": 0.0, "issues": []})
            n_issues = len(data["issues"])
            detectors = sorted(set(i["detector"] for i in data["issues"])) or ["none triggered"]
            mechanism = (
                f"Giskard scan (performance + robustness categories) over {n} sampled "
                f"test rows: {n_issues} issue(s) found for '{principle}' via "
                f"{', '.join(detectors)}; severity-weighted score = {data['weight']:.1f}."
            )
            results.append(EvalResult(
                self.name, principle, InspectionTarget.BEHAVIOR, "giskard_issue_score",
                float(data["weight"]), False, mechanism,
                {"n_issues": n_issues, "issues": data["issues"][:5], "sample_size": n,
                 "trace": make_trace(
                     steps=[f"score({principle}) = Σ severity weights = {data['weight']:.1f}"],
                     tables=[{"title": f"Issues Giskard raised that map to '{principle}'",
                              "columns": ["level", "detector", "description"],
                              "rows": [[i["level"], i["detector"], i["description"]]
                                       for i in data["issues"]]}])},
            ))
        return results


register(GiskardEvaluator())
