import numpy as np
import pandas as pd
import pytest

from ethikit.core.types import TaskType
from ethikit.data import make_synthetic_raw, prepare_frame, split_dataset
from ethikit.data.loaders import adult_from_frame


def _frame(n=300, seed=0):
    r = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": r.integers(18, 70, n), "income": r.normal(50, 10, n),
        "job": r.choice(["a", "b", "c"], n), "sex": r.choice(["F", "M"], n),
        "y": r.choice(["no", "yes"], n)})


def test_basic_classification_and_positive_class_is_later_label():
    raw = prepare_frame(_frame(), "y", "sex")
    assert raw.task == TaskType.CLASSIFICATION
    assert sorted(set(raw.sensitive)) == ["F", "M"]
    df = _frame()
    assert (raw.y == (df["y"] == "yes").astype(int).to_numpy()).all()   # "yes" > "no" -> 1


def test_positive_label_override():
    raw = prepare_frame(_frame(), "y", "sex", positive_label="no")
    df = _frame()
    assert (raw.y == (df["y"] == "no").astype(int).to_numpy()).all()


def test_nans_in_features_are_imputed_but_rows_with_missing_target_dropped():
    df = _frame()
    df.loc[:9, "income"] = np.nan
    df.loc[10:14, "y"] = np.nan
    raw = prepare_frame(df, "y", "sex")
    assert not np.isnan(raw.X).any() and len(raw.y) == 295


def test_many_level_categorical_collapses_to_other():
    df = _frame(500)
    df["zip"] = np.random.default_rng(1).choice([f"z{i}" for i in range(60)], 500)
    raw = prepare_frame(df, "y", "sex", max_categories=10)
    zip_cols = [f for f in raw.feature_names if f.startswith("zip=")]
    assert len(zip_cols) <= 11 and "zip=other" in zip_cols


def test_numeric_sensitive_column_splits_at_median_by_default_and_at_threshold():
    raw = prepare_frame(_frame(), "y", "age")
    assert len(set(raw.sensitive)) == 2
    raw25 = prepare_frame(_frame(), "y", "age", sensitive_threshold=25)
    assert set(raw25.sensitive) == {"age<25", "age>=25"}


def test_exclude_sensitive_removes_it_from_features():
    with_s = prepare_frame(_frame(), "y", "sex")
    without = prepare_frame(_frame(), "y", "sex", exclude_sensitive=True)
    assert any(f.startswith("sex") for f in with_s.feature_names)
    assert not any(f.startswith("sex") for f in without.feature_names)


def test_regression_is_inferred_from_a_continuous_target():
    df = _frame()
    assert prepare_frame(df, "income", "sex").task == TaskType.REGRESSION


def test_single_class_target_raises_a_clear_error():
    df = _frame(); df["y"] = "yes"
    with pytest.raises(ValueError, match="single class"):
        prepare_frame(df, "y", "sex")


def test_unknown_column_error_lists_available_columns():
    with pytest.raises(KeyError, match="available"):
        prepare_frame(_frame(), "nope", "sex")


def test_too_few_rows_is_rejected():
    with pytest.raises(ValueError, match="at least 50"):
        prepare_frame(_frame(20), "y", "sex")


def test_scaling_uses_train_statistics_only(tiny_raw):
    ds = split_dataset(tiny_raw, seed=3)
    assert abs(ds.X_train.mean()) < 1e-6          # train is standardised...
    unscaled = split_dataset(tiny_raw, seed=3, scale=False)
    assert not np.allclose(unscaled.X_train, ds.X_train)


def test_synthetic_knobs_change_the_data():
    small = make_synthetic_raw(n=1000, minority_share=0.1, seed=0)
    assert abs(small.sensitive.mean() - 0.1) < 0.03
    skew = make_synthetic_raw(n=2000, positive_rate=0.2, seed=0)
    assert abs(skew.y.mean() - 0.2) < 0.03
    cat = make_synthetic_raw(n=500, n_categorical=2, seed=0)
    assert any(f.startswith("cat_1=") for f in cat.feature_names)


def test_adult_now_includes_categorical_columns():
    r = np.random.default_rng(0); n = 400
    df = pd.DataFrame({
        "age": r.integers(18, 80, n), "workclass": r.choice(["Private", "Gov", "Self"], n),
        "native-country": r.choice([f"c{i}" for i in range(30)], n),
        "sex": r.choice(["Male", "Female"], n), "class": r.choice(["<=50K", ">50K"], n)})
    raw = adult_from_frame(df)
    assert "workclass=Private" in raw.feature_names
    assert raw.sensitive_name == "sex" and set(raw.sensitive) == {"Male", "Female"}
    assert (raw.y == (df["class"] == ">50K").astype(int).to_numpy()).all()
    assert len([f for f in raw.feature_names if f.startswith("native-country=")]) <= 21
