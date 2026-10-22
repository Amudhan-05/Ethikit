"""Decision support: the "select" half of selection-and-audit.

Two things a person needs before choosing a model, beyond a profile:

1. **Requirements** (hard constraints). "Only models whose demographic_parity_diff <= 0.1 and
   whose membership-inference advantage <= 0.2." Checked on the raw, un-normalised metric values,
   because a threshold must mean the same thing whatever the other candidates are. Every model
   that fails is listed with *which* requirement it failed and by how much -- never silently
   dropped. A metric that a model wasn't measured on counts as a failure ("not measured"), because
   an unmeasured property is not a satisfied one.

2. **Choice stability**. A weighted order depends on weights that are a value judgement. Before
   trusting "model A comes first", it helps to know how far the weights would have to move for
   the first place to change. We report, per weighted principle, the range of that weight (others
   kept in proportion) over which the same model stays first and who takes over outside it; plus
   the share of nearby random weight vectors that keep the same first place.

Neither function crowns anything: requirements give a feasible set, stability says how fragile
an ordering is. The person still decides.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

_OPS = {"<=": np.less_equal, "<": np.less, ">=": np.greater_equal, ">": np.greater}
_PATTERN = re.compile(r"^\s*(?:(?P<tool>[\w\-]+)\.)?(?P<metric>[\w\-]+)\s*(?P<op><=|>=|<|>)\s*"
                      r"(?P<value>-?\d+(?:\.\d*)?(?:[eE]-?\d+)?)\s*$")


@dataclass(frozen=True)
class Requirement:
    metric: str
    op: str
    value: float
    tool: Optional[str] = None

    @property
    def text(self) -> str:
        return f"{self.tool + '.' if self.tool else ''}{self.metric} {self.op} {self.value:g}"

    def passes(self, x: float) -> bool:
        return bool(_OPS[self.op](x, self.value))


def parse_requirement(s: str) -> Requirement:
    """'demographic_parity_diff<=0.1' or 'aif360.disparate_impact_gap<0.2' -> Requirement."""
    m = _PATTERN.match(s or "")
    if not m:
        raise ValueError(f"can't read requirement {s!r}; write it like "
                         f"'demographic_parity_diff<=0.1' or 'aif360.theil_index<0.2' "
                         f"(operators: <=, <, >=, >)")
    return Requirement(m["metric"], m["op"], float(m["value"]), m["tool"])


def parse_requirements(items: Optional[Iterable]) -> List[Requirement]:
    out = []
    for it in items or []:
        if isinstance(it, Requirement):
            out.append(it)
        elif isinstance(it, dict):
            out.append(Requirement(it["metric"], it["op"], float(it["value"]), it.get("tool")))
        else:
            out.append(parse_requirement(str(it)))
    return out


def check_requirements(raw: pd.DataFrame, requirements: List[Requirement],
                       models: Optional[List[str]] = None) -> Dict[str, dict]:
    """model -> {"passes": bool, "results": [{requirement, value, passes, margin, reason}]}.

    `margin` is how far inside (+) or outside (-) the threshold the value sits, in the metric's
    own units, so "failed by 0.002" and "failed by 0.3" read differently.
    """
    rows = raw[raw["target"] != "error"]
    models = models or sorted(rows["model"].unique())
    known = set(rows["metric"])
    out: Dict[str, dict] = {}
    for m in models:
        mine = rows[rows["model"] == m]
        results = []
        for r in requirements:
            if r.metric not in known:
                raise ValueError(f"requirement {r.text!r}: no tool in this run reports "
                                 f"'{r.metric}'. Metrics available: {', '.join(sorted(known))}")
            hits = mine[mine["metric"] == r.metric]
            if r.tool:
                hits = hits[hits["evaluator"] == r.tool]
            if len(hits["evaluator"].unique()) > 1:
                raise ValueError(f"requirement {r.text!r} is ambiguous: '{r.metric}' comes from "
                                 f"{sorted(hits['evaluator'].unique())}; prefix the tool, e.g. "
                                 f"'{hits['evaluator'].iloc[0]}.{r.metric}{r.op}{r.value:g}'")
            vals = hits["value"].astype(float).dropna()
            if vals.empty:
                results.append({"requirement": r.text, "value": None, "passes": False,
                                "margin": None, "reason": "not measured for this model"})
                continue
            v = float(vals.mean())
            ok = r.passes(v)
            margin = (r.value - v) if r.op in ("<=", "<") else (v - r.value)
            results.append({"requirement": r.text, "value": round(v, 4), "passes": ok,
                            "margin": round(margin, 4),
                            "reason": "meets it" if ok else
                            f"{v:.4f} is {abs(margin):.4f} {'above' if r.op in ('<=', '<') else 'below'} the limit"})
        out[m] = {"passes": all(x["passes"] for x in results), "results": results}
    return out


# ---------------------------------------------------------------------------------- stability
class _Matrix:
    """The objective table as a plain array: the stability sweep evaluates thousands of weight
    vectors, and pandas ops would deep-copy the frame's .attrs (a run's traces) on every one."""

    def __init__(self, obj: pd.DataFrame):
        self.models = [str(m) for m in obj.index]
        self.cols = {str(c): i for i, c in enumerate(obj.columns)}
        self.values = np.nan_to_num(obj.to_numpy(dtype=float), nan=0.0)

    def leader(self, w: Dict[str, float]) -> str:
        vec = np.zeros(self.values.shape[1])
        tot = sum(v for c, v in w.items() if c in self.cols and v > 0) or 1.0
        for c, v in w.items():
            if c in self.cols and v > 0:
                vec[self.cols[c]] = v / tot
        return self.models[int(np.argmax(self.values @ vec))]


def choice_stability(objective: pd.DataFrame, weights: Dict[str, float],
                     n_samples: int = 2000, concentration: float = 50.0, seed: int = 0,
                     candidates: Optional[List[str]] = None) -> dict:
    """How fragile is "first place under these weights"?

    For each weighted principle p, vary p's share of the total weight from 0 to 1 (the other
    weights keep their relative proportions) and find the interval around today's share in which
    the same model stays first, plus which model takes over just outside it.
    Then sample weight vectors from a Dirichlet centred on today's weights (higher
    `concentration` = closer to today's) and report the share that keeps the same first place.
    `candidates` restricts the comparison (e.g. to the models that meet the requirements).
    """
    obj = objective.loc[candidates] if candidates else objective
    w = {p: float(v) for p, v in weights.items() if p in obj.columns and v > 0}
    if obj.empty or not w:
        return {"available": False, "reason": "no candidates or no weighted principle in this run"}
    if len(obj) == 1:
        return {"available": False, "reason": "only one candidate: nothing to compare against"}
    total = sum(w.values())
    share = {p: v / total for p, v in w.items()}
    mat = _Matrix(obj)
    first = mat.leader(share)

    grid = np.linspace(0, 1, 201)
    per_principle = {}
    for p in share:
        rest = {q: s for q, s in share.items() if q != p}
        rest_tot = sum(rest.values())
        leaders = []
        for g in grid:
            ww = {q: (s / rest_tot) * (1 - g) for q, s in rest.items()} if rest_tot else {}
            ww[p] = g
            leaders.append(mat.leader(ww) if sum(ww.values()) > 0 else first)
        i0 = int(np.argmin(np.abs(grid - share[p])))
        lo = i0
        while lo > 0 and leaders[lo - 1] == first:
            lo -= 1
        hi = i0
        while hi < len(grid) - 1 and leaders[hi + 1] == first:
            hi += 1
        below = leaders[lo - 1] if lo > 0 else None
        above = leaders[hi + 1] if hi < len(grid) - 1 else None
        per_principle[p] = {
            "current_share": round(share[p], 3), "stable_from": round(float(grid[lo]), 3),
            "stable_to": round(float(grid[hi]), 3), "taken_over_below_by": below,
            "taken_over_above_by": above,
            "sentence": _sentence(p, share[p], grid[lo], grid[hi], below, above, first),
        }

    rng = np.random.default_rng(seed)
    alpha = np.array([share[p] for p in share]) * concentration + 1e-3
    keys = list(share)
    same = 0
    rivals: Dict[str, int] = {}
    for _ in range(n_samples):
        v = rng.dirichlet(alpha)
        lead = mat.leader(dict(zip(keys, v)))
        if lead == first:
            same += 1
        else:
            rivals[lead] = rivals.get(lead, 0) + 1
    return {
        "available": True, "first": first, "weights_share": {p: round(s, 3) for p, s in share.items()},
        "per_principle": per_principle,
        "nearby_weights_keep_first": round(same / n_samples, 3),
        "main_rivals": dict(sorted(rivals.items(), key=lambda kv: -kv[1])[:3]),
        "n_samples": n_samples, "concentration": concentration,
    }


def _sentence(p, cur, lo, hi, below, above, first) -> str:
    if lo <= 0 and hi >= 1:
        return f"{first} stays first for any {p} weight."
    parts = []
    if lo > 0:
        parts.append(f"below {lo:.2f} {below} takes over")
    if hi < 1:
        parts.append(f"above {hi:.2f} {above} takes over")
    return f"{first} stays first while {p}'s share of the weight is {lo:.2f}-{hi:.2f} (now {cur:.2f}); " + \
           "; ".join(parts) + "."
