"""M5: deep learning on tabular data. scikit-learn MLP tests always run; PyTorch / Captum / Opacus
tests skip cleanly when those aren't installed."""
import numpy as np
import pytest

from ethikit.benchmark import CLASSIFIERS, run_benchmark, select_evaluators
from ethikit.benchmark.models import family_of, load_fitted_model
from ethikit.benchmark.runner import _predict_all
from ethikit.data import make_synthetic_raw, split_dataset
from ethikit.deep import ProbaModelAdapter, adapt


@pytest.fixture(scope="module")
def ds():
    return split_dataset(make_synthetic_raw(n=1200, seed=5), seed=5)


def test_sklearn_mlp_is_in_the_deep_family_and_runs_the_battery(ds):
    assert family_of("mlp") == "deep"
    df = run_benchmark(ds, {"mlp": CLASSIFIERS["deep"]["mlp"]},
                       evaluators=select_evaluators(["performance", "fairness", "membership_inference"]))
    assert not (df["target"] == "error").any() and {"accuracy", "demographic_parity_diff"} <= set(df["metric"])


class FakeKeras:
    """Anything with .predict(X) -> probabilities, like a Keras model."""
    def __init__(self, w):
        self.w = w

    def predict(self, X):
        return 1 / (1 + np.exp(-(X @ self.w)))          # one column: P(class 1)


def test_probability_models_are_adapted_and_evaluated_as_is(ds):
    m = adapt(FakeKeras(np.ones(ds.X_test.shape[1])))
    assert isinstance(m, ProbaModelAdapter)
    p = m.predict_proba(ds.X_test)
    assert p.shape == (len(ds.X_test), 2) and np.allclose(p.sum(1), 1)
    df = run_benchmark(ds, {}, prefit={"keras_like": m}, evaluators=select_evaluators(["performance", "fairness"]))
    assert set(df["model"]) == {"keras_like"} and not (df["target"] == "error").any()


def test_adapt_rejects_things_that_cannot_predict():
    with pytest.raises(TypeError, match="no predict"):
        adapt(object())


