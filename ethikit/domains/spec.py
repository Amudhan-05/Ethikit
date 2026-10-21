"""Types for reference suites: a Domain is a named set of public datasets from one field
(lending, census/income, health, criminal justice, ...), each with a declared target and
sensitive attribute. Running the same battery over a suite gives *evidence about which kinds
of model tend to do well in that field* -- a prior for an organisation that has no data of its
own yet. It is never a verdict about any particular model on any particular population.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from ..core.types import TaskType
from ..data.loaders import RawData


class DatasetUnavailable(RuntimeError):
    """A dataset couldn't be fetched/read. The message says why and how to fix it; the suite
    runner records it as skipped instead of silently substituting other data."""


@dataclass
class DatasetSpec:
    id: str
    title: str
    loader: Callable[[], RawData]
    source: str                     # where the data comes from (URL / citation)
    sensitive: str                  # which attribute the fairness tools compare, in words
    needs: str = "network"          # "network" | "manual" | "none"
    note: str = ""                  # modelling choices a reader should know about
    verified: bool = True           # False: loader written against documented schema but not yet run end to end


@dataclass
class Domain:
    id: str
    title: str
    blurb: str
    datasets: List[DatasetSpec]
    caveats: List[str] = field(default_factory=list)
    for_whom: str = ""

    @property
    def strength(self) -> str:
        """How much cross-dataset evidence this suite can honestly offer."""
        n = len(self.datasets)
        return "gap" if n == 0 else "single" if n == 1 else "small" if n < 5 else "suite"


STRENGTH_TEXT = {
    "gap": "no public benchmark found -- use your own data (ethikit run --data your.csv)",
    "single": "one dataset: a starting point, NOT cross-dataset evidence",
    "small": "2-4 datasets: indicative only; significance tests have little power",
    "suite": "5+ datasets: enough for family-level tendencies within this domain",
}


# ---------------------------------------------------------------------------
# cache (fetched data is stored once, under the user's home, never in the repo)
# ---------------------------------------------------------------------------
def cache_dir() -> Path:
    p = Path(os.environ.get("ETHIKIT_CACHE") or Path.home() / ".cache" / "ethikit")
    p.mkdir(parents=True, exist_ok=True)
    return p


def cap_rows(raw: RawData, max_rows: Optional[int], seed: int = 0) -> RawData:
    """Deterministic row subsample (keeps the sensitive attribute aligned) so a big census
    state doesn't make a model sweep take an hour. None = keep everything."""
    n = len(raw.y)
    if not max_rows or n <= max_rows:
        return raw
    idx = np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False))
    return RawData(X=raw.X[idx], y=raw.y[idx],
                   sensitive=None if raw.sensitive is None else raw.sensitive[idx],
                   sensitive_name=raw.sensitive_name, feature_names=raw.feature_names,
                   name=raw.name, task=raw.task)


def load_spec(spec: DatasetSpec, max_rows: Optional[int] = 5000, seed: int = 0,
              use_cache: bool = True) -> RawData:
    """spec.loader() with an on-disk cache of the *full* cleaned RawData, then a deterministic
    row cap. Any failure becomes DatasetUnavailable naming the dataset."""
    path = cache_dir() / f"{spec.id}.pkl"
    raw: Optional[RawData] = None
    if use_cache and path.exists():
        try:
            with open(path, "rb") as fh:
                raw = pickle.load(fh)
        except Exception:
            raw = None
    if raw is None:
        try:
            raw = spec.loader()
        except DatasetUnavailable:
            raise
        except Exception as exc:
            raise DatasetUnavailable(f"{spec.id}: {type(exc).__name__}: {exc}") from exc
        if use_cache:
            try:
                with open(path, "wb") as fh:
                    pickle.dump(raw, fh)
            except Exception:
                pass
    if raw.task != TaskType.CLASSIFICATION:
        raise DatasetUnavailable(f"{spec.id}: reference suites are classification-only for now")
    return cap_rows(raw, max_rows, seed)
