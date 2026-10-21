"""Reference-suite registry, loaders (offline fixtures) and the cache."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.fake_pums import fake_pums  # noqa: E402

from ethikit.core.types import TaskType  # noqa: E402
from ethikit.data import make_synthetic_raw  # noqa: E402
from ethikit.domains import DatasetUnavailable, get_domains, load_spec  # noqa: E402
from ethikit.domains import sources as S  # noqa: E402
from ethikit.domains.spec import DatasetSpec, STRENGTH_TEXT, cap_rows  # noqa: E402


def test_every_domain_is_described_and_honest_about_its_size():
    for d in get_domains().values():
        assert d.title and d.blurb and d.caveats, d.id
        assert d.strength in STRENGTH_TEXT
        assert (d.strength == "gap") == (len(d.datasets) == 0)
        if d.strength in ("single", "small"):
            assert any("dataset" in c.lower() for c in d.caveats), f"{d.id} must say it is thin"


def test_every_dataset_declares_source_sensitive_attribute_and_unique_id():
    seen = set()
    for d in get_domains().values():
        for s in d.datasets:
            assert s.source and s.sensitive and s.title and callable(s.loader), s.id
            assert s.id not in seen, f"duplicate dataset id {s.id}"
            seen.add(s.id)


def test_census_suite_is_tasks_by_states_plus_adult():
    c = get_domains(["CA", "WA"])["census"]
    ids = [s.id for s in c.datasets]
    assert len(ids) == len(S.ACS_TASKS) * 2 + 1 and "ACSIncome_WA_2018" in ids and "adult_income" in ids
    assert c.strength == "suite"


def test_hiring_is_an_explicit_gap_not_a_synthetic_stand_in():
    h = get_domains()["hiring"]
    assert h.datasets == [] and h.strength == "gap" and "your own" in " ".join(h.caveats).lower()


@pytest.mark.parametrize("task", S.ACS_TASKS)
def test_acs_adapter_builds_a_raw_dataset_from_pums_shaped_data(task):
    raw = S.acs_from_pums(fake_pums(task), task, f"{task}_fake")
    assert raw.task == TaskType.CLASSIFICATION and raw.X.shape[0] > 500
    assert set(raw.sensitive) == {"Male", "Female"} and raw.sensitive_name == "SEX"
    assert 0.05 < raw.y.mean() < 0.95            # both classes present after folktables' own filter


def test_compas_adapter_applies_propublica_cleaning_and_excludes_leaky_columns():
    r = np.random.default_rng(0); n = 400
    df = pd.DataFrame({
        "age": r.integers(18, 70, n), "sex": r.choice(["Male", "Female"], n),
        "juv_fel_count": 0, "juv_misd_count": 0, "juv_other_count": 0,
        "priors_count": r.integers(0, 10, n), "c_charge_degree": r.choice(["F", "M", "O"], n),
        "race": r.choice(["African-American", "Caucasian", "Hispanic", "Asian", "Other"], n),
        "days_b_screening_arrest": r.integers(-60, 60, n), "is_recid": r.choice([-1, 0, 1], n),
        "score_text": r.choice(["Low", "High", "N/A"], n), "decile_score": r.integers(1, 10, n),
        "two_year_recid": r.integers(0, 2, n)})
    raw = S.compas_from_frame(df)
    kept = df[(df.days_b_screening_arrest.abs() <= 30) & (df.is_recid != -1)
              & (df.c_charge_degree != "O") & (df.score_text != "N/A")]
    assert len(raw.y) == len(kept)
    assert set(raw.sensitive) <= {"African-American", "Caucasian", "Hispanic", "Other"}
    assert not any(f.startswith("race") or "decile" in f or "score" in f for f in raw.feature_names)


def test_fairlearn_adapter_names_the_real_columns_when_the_schema_differs(monkeypatch):
    pytest.importorskip("fairlearn")
    import types
    import fairlearn.datasets as fd
    fake = types.SimpleNamespace(data=pd.DataFrame({"a": range(100), "b": range(100)}),
                                 target=pd.Series([0, 1] * 50))
    monkeypatch.setattr(fd, "fetch_credit_card", lambda as_frame=True: fake)
    with pytest.raises(DatasetUnavailable, match=r"columns are \['a', 'b'\]"):
        S.load_credit_card_default()


def test_fairlearn_adapter_reports_network_failure_clearly(monkeypatch):
    pytest.importorskip("fairlearn")
    import fairlearn.datasets as fd

    def boom(as_frame=True):
        raise OSError("no network")
    monkeypatch.setattr(fd, "fetch_bank_marketing", boom)
    with pytest.raises(DatasetUnavailable, match="needs network access once"):
        S.load_bank_marketing()


def test_cap_rows_is_deterministic_and_keeps_alignment(tiny_raw):
    a, b = cap_rows(tiny_raw, 200, seed=3), cap_rows(tiny_raw, 200, seed=3)
    assert len(a.y) == 200 and (a.X == b.X).all() and (a.sensitive == b.sensitive).all()
    assert len(cap_rows(tiny_raw, None).y) == len(tiny_raw.y)
    # alignment: the sensitive group column is also a feature in the synthetic data
    assert (a.X[:, -1] == a.sensitive).all()


def test_cache_roundtrip_and_failures_become_dataset_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("ETHIKIT_CACHE", str(tmp_path))
    calls = []

    def loader():
        calls.append(1)
        return make_synthetic_raw(n=300, seed=1)
    spec = DatasetSpec("toy", "Toy", loader, "synthetic", "group", needs="none")
    load_spec(spec); load_spec(spec)
    assert len(calls) == 1 and (tmp_path / "toy.pkl").exists()       # second call came from cache
    load_spec(spec, use_cache=False)
    assert len(calls) == 2

    def broken():
        raise ValueError("schema changed")
    with pytest.raises(DatasetUnavailable, match="bad: ValueError: schema changed"):
        load_spec(DatasetSpec("bad", "Bad", broken, "x", "y"), use_cache=False)


def test_regression_datasets_are_refused_in_suites(tmp_path, monkeypatch):
    monkeypatch.setenv("ETHIKIT_CACHE", str(tmp_path))
    spec = DatasetSpec("reg", "Reg", lambda: make_synthetic_raw(n=300, task="regression"), "s", "g")
    with pytest.raises(DatasetUnavailable, match="classification-only"):
        load_spec(spec, use_cache=False)
