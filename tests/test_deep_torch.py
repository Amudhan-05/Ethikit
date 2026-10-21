"""M5: PyTorch / Captum / Opacus. The whole file skips cleanly when torch isn't installed."""
import numpy as np
import pytest

from ethikit.benchmark import CLASSIFIERS
from ethikit.benchmark.models import family_of, load_fitted_model
from ethikit.benchmark.runner import _predict_all
from ethikit.data import make_synthetic_raw, split_dataset
from ethikit.deep import adapt


@pytest.fixture(scope="module")
def ds():
    return split_dataset(make_synthetic_raw(n=1200, seed=5), seed=5)


# ------------------------------------------------------------------ PyTorch
torch = pytest.importorskip("torch")
from ethikit.deep import TorchMLPClassifier, TorchModuleAdapter  # noqa: E402


def test_torch_mlp_speaks_scikit_learn_and_is_deterministic(ds):
    from sklearn.base import clone
    m = TorchMLPClassifier(epochs=5, seed=1)
    assert clone(m).get_params()["epochs"] == 5
    a = m.fit(ds.X_train, ds.y_train).predict_proba(ds.X_test)
    b = clone(m).fit(ds.X_train, ds.y_train).predict_proba(ds.X_test)
    assert np.allclose(a.sum(1), 1) and np.allclose(a, b)
    assert set(m.predict(ds.X_test)) <= set(np.unique(ds.y_train))
    assert family_of("torch_mlp") == "deep"


def test_a_raw_torch_module_and_a_saved_pt_file_are_evaluated(ds, tmp_path):
    net = TorchMLPClassifier(epochs=3).fit(ds.X_train, ds.y_train).module_
    adapted = adapt(net)
    assert isinstance(adapted, TorchModuleAdapter) and adapted.predict_proba(ds.X_test).shape[1] == 2
    path = tmp_path / "model.pt"
    torch.save(net, path)
    loaded = load_fitted_model(str(path))
    assert np.allclose(loaded.predict_proba(ds.X_test), adapted.predict_proba(ds.X_test))


def test_captum_attributions_satisfy_completeness(ds):
    pytest.importorskip("captum")
    from ethikit.evaluators.captum_ig import CaptumIGEvaluator
    m = TorchMLPClassifier(epochs=10).fit(ds.X_train, ds.y_train)
    res = {r.metric: r for r in CaptumIGEvaluator(sample_size=40, n_steps=128).evaluate(_predict_all(m, ds))}
    assert res["ig_completeness_error"].value < 1e-2
    assert 0 < res["ig_concentration"].value <= 1
    assert res["ig_concentration"].details["trace"]["steps"]


def test_captum_does_not_apply_to_non_torch_models(ds):
    pytest.importorskip("captum")
    from ethikit.evaluators.captum_ig import CaptumIGEvaluator
    m = CLASSIFIERS["linear"]["logistic_regression"]().fit(ds.X_train, ds.y_train)
    assert not CaptumIGEvaluator().applies_to(_predict_all(m, ds))


def test_dp_training_reports_its_budget_and_leaks_no_more_than_the_plain_model():
    pytest.importorskip("opacus")
    from ethikit.evaluators.privacy import MembershipInferenceEvaluator
    # small, noisy data + a big network + many epochs: the plain model overfits on purpose
    d = split_dataset(make_synthetic_raw(n=500, noise=2.5, n_features=12, seed=7), seed=7)
    plain = TorchMLPClassifier(hidden_layers=(256, 256), epochs=150, learning_rate=3e-3, seed=0)
    dp = TorchMLPClassifier(hidden_layers=(256, 256), epochs=150, learning_rate=3e-3, seed=0, dp=True, target_epsilon=1.0)
    plain.fit(d.X_train, d.y_train); dp.fit(d.X_train, d.y_train)
    assert dp.epsilon_ is not None and dp.epsilon_ <= 1.0 * 1.05 and plain.epsilon_ is None
    mi = MembershipInferenceEvaluator()
    adv_plain = {r.metric: r.value for r in mi.evaluate(_predict_all(plain, d))}["membership_inference_advantage"]
    res_dp = mi.evaluate(_predict_all(dp, d))
    adv_dp = {r.metric: r.value for r in res_dp}["membership_inference_advantage"]
    assert adv_dp < adv_plain - 0.1, (adv_dp, adv_plain)        # measured here: 0.05 vs 0.41
    assert "differential privacy" in res_dp[0].details["trace"]["steps"][0]


def test_captum_works_on_a_dp_trained_model(ds):
    """Regression: Opacus left gradient hooks behind and Captum crashed with 'pop from empty list'."""
    pytest.importorskip("captum"); pytest.importorskip("opacus")
    from ethikit.evaluators.captum_ig import CaptumIGEvaluator
    m = TorchMLPClassifier(epochs=3, dp=True).fit(ds.X_train, ds.y_train)
    r = {x.metric: x.value for x in CaptumIGEvaluator(sample_size=20).evaluate(_predict_all(m, ds))}
    assert r["ig_completeness_error"] < 1e-2
