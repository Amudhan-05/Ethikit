"""Shared fixtures. Everything here is small and offline so the suite runs in seconds."""
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def tiny_raw():
    from ethikit.data import make_synthetic_raw
    return make_synthetic_raw(n=600, seed=1)


@pytest.fixture(scope="session")
def tiny_ds(tiny_raw):
    from ethikit.data import split_dataset
    return split_dataset(tiny_raw, seed=1)


@pytest.fixture(scope="session")
def fitted_ctx(tiny_ds):
    """A logistic-regression EvalContext on the tiny dataset (cheap to share)."""
    from sklearn.linear_model import LogisticRegression
    from ethikit.benchmark.runner import _predict_all
    model = LogisticRegression(max_iter=500).fit(tiny_ds.X_train, tiny_ds.y_train)
    return _predict_all(model, tiny_ds)


@pytest.fixture(scope="session")
def tiny_reg_ds():
    from ethikit.data import make_synthetic_raw, split_dataset
    return split_dataset(make_synthetic_raw(n=500, seed=2, task="regression"), seed=2)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def server():
    """The real console server on a free local port, in a background thread."""
    import threading
    from http.server import ThreadingHTTPServer
    from ethikit.server import Handler
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown(); httpd.server_close()
