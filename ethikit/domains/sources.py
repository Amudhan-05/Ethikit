"""Loaders for the public datasets behind the reference suites.

Rules every loader follows:
  * Fail loudly with a message that says what to do (never silently substitute synthetic data).
  * Return a RawData with the sensitive attribute declared, via prepare_frame().
  * Say what modelling choices were made (which column is "sensitive", how it was binarised,
    whether it stays in the feature set) in the DatasetSpec note.
Which of these have been run end-to-end against the real source is recorded in `verified`.
"""
from __future__ import annotations

import io
import urllib.request
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..data.loaders import RawData, adult_from_frame, prepare_frame
from .spec import DatasetUnavailable, cache_dir

COMPAS_URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/master/"
              "compas-scores-two-years.csv")


# --------------------------------------------------------------------------- justice
def compas_from_frame(df: pd.DataFrame) -> RawData:
    """ProPublica's COMPAS two-year recidivism data, cleaned the way ProPublica's own analysis
    did (screened within 30 days of arrest, known recidivism outcome, ordinary charge degree,
    a score on file). Target: re-offended within two years. Sensitive: race, with the three tiny
    groups (Asian, Native American, Other) merged into 'Other'. The COMPAS risk score itself
    (decile_score, score_text) is NOT a feature -- it is the existing tool's *output*, and using
    it would leak. Race is excluded from the features (the real tool did not take it as input);
    proxies for it can still be present, which is exactly what the audit looks for."""
    d = df[(df["days_b_screening_arrest"].abs() <= 30) & (df["is_recid"] != -1)
           & (df["c_charge_degree"] != "O") & (df["score_text"] != "N/A")].copy()
    d["race"] = d["race"].where(d["race"].isin(["African-American", "Caucasian", "Hispanic"]), "Other")
    keep = ["age", "sex", "juv_fel_count", "juv_misd_count", "juv_other_count", "priors_count",
            "c_charge_degree", "race", "two_year_recid"]
    return prepare_frame(d[keep], target="two_year_recid", sensitive="race",
                         name="compas_recidivism", exclude_sensitive=True)


def load_compas() -> RawData:
    cached = cache_dir() / "compas-scores-two-years.csv"
    try:
        if cached.exists():
            df = pd.read_csv(cached)
        else:
            with urllib.request.urlopen(COMPAS_URL, timeout=60) as r:
                data = r.read()
            cached.write_bytes(data)
            df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise DatasetUnavailable(
            f"compas: could not fetch {COMPAS_URL} ({type(exc).__name__}). Download that CSV "
            f"manually and save it as {cached}") from exc
    return compas_from_frame(df)


# --------------------------------------------------------------------------- lending / income
def load_german_credit() -> RawData:
    from ..data.loaders import load_german_credit_raw
    return load_german_credit_raw()


def load_adult_strict() -> RawData:
    """UCI Adult from OpenML. Unlike load_adult(), no synthetic fallback: a reference suite
    must not quietly report results for fake data."""
    from sklearn.datasets import fetch_openml
    return adult_from_frame(fetch_openml("adult", version=2, as_frame=True).frame)


def _fairlearn_dataset(fetch_name: str, name: str, sensitive_candidates: Sequence[str],
                       **prep) -> RawData:
    """Adapt a `fairlearn.datasets.fetch_*` loader. Column names are matched
    case-insensitively against `sensitive_candidates`; if none match, the error lists the
    real columns so the spec can be corrected in one edit."""
    try:
        from fairlearn import datasets as fd
        bunch = getattr(fd, fetch_name)(as_frame=True)
    except Exception as exc:
        raise DatasetUnavailable(
            f"{name}: fairlearn.datasets.{fetch_name}() failed ({type(exc).__name__}: {exc}); "
            f"it downloads from OpenML on first use, so it needs network access once") from exc
    df = bunch.data.copy()
    real_columns = list(df.columns)                    # before our own target column is added
    df["__target__"] = bunch.target
    lowered = {str(c).lower(): c for c in real_columns}
    col = next((lowered[c.lower()] for c in sensitive_candidates if c.lower() in lowered), None)
    if col is None:
        raise DatasetUnavailable(
            f"{name}: none of {list(sensitive_candidates)} is a column of fairlearn's {fetch_name} "
            f"data; columns are {real_columns}. Update sensitive_candidates in "
            f"ethikit/domains/sources.py")
    return prepare_frame(df, target="__target__", sensitive=col, name=name, **prep)


def load_credit_card_default() -> RawData:
    return _fairlearn_dataset("fetch_credit_card", "credit_card_default", ["SEX", "x2", "X2"])


def load_bank_marketing() -> RawData:
    return _fairlearn_dataset("fetch_bank_marketing", "bank_marketing", ["age", "V1", "x1"])


# --------------------------------------------------------------------------- health
def load_diabetes_hospital() -> RawData:
    return _fairlearn_dataset("fetch_diabetes_hospital", "diabetes_hospital_readmission",
                              ["race", "gender"])


# --------------------------------------------------------------------------- census (folktables ACS)
ACS_TASKS = ["ACSIncome", "ACSEmployment", "ACSPublicCoverage", "ACSMobility", "ACSTravelTime"]


def acs_from_pums(df: pd.DataFrame, task_name: str, name: str) -> RawData:
    """Turn one state's ACS PUMS table into a RawData for a folktables task. The label comes
    from folktables' own task definition (its population filter and target transform); the
    sensitive attribute is SEX (relabelled Male/Female). Features are the task's integer-coded
    ACS columns used as numbers, as in folktables' own baselines (occupation/place codes are
    therefore crude -- see the domain caveats)."""
    import folktables
    task = getattr(folktables, task_name)
    X, y, _ = task.df_to_pandas(df)
    X = X.copy()
    if "SEX" in X.columns:
        X["SEX"] = X["SEX"].map({1: "Male", 2: "Female"}).fillna("Other")
        sensitive = "SEX"
    else:
        sensitive = "RAC1P"
    X["__target__"] = np.asarray(y).astype(int).ravel()
    return prepare_frame(X, target="__target__", sensitive=sensitive, name=name)


def acs_loader(task_name: str, state: str, year: str = "2018") -> Callable[[], RawData]:
    def load() -> RawData:
        try:
            from folktables import ACSDataSource
        except ImportError as exc:
            raise DatasetUnavailable("census suite needs `pip install folktables`") from exc
        try:
            src = ACSDataSource(survey_year=year, horizon="1-Year", survey="person",
                                root_dir=str(cache_dir() / "acs"))
            df = src.get_data(states=[state], download=True)
        except Exception as exc:
            raise DatasetUnavailable(
                f"{task_name}/{state}: could not download ACS PUMS {year} ({type(exc).__name__}: "
                f"{exc}). It is fetched once from census.gov (tens of MB per state) and cached "
                f"under {cache_dir() / 'acs'}") from exc
        return acs_from_pums(df, task_name, f"{task_name}_{state}_{year}")
    return load
