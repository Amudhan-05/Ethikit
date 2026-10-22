"""Deep learning on tabular data.

Everything in ethikit talks to a model through scikit-learn's surface: fit / predict /
predict_proba (+ get_params for the console's hyperparameter panel). This module makes neural
networks fit that surface, so every tool, the console, the model card and the CLI treat them
exactly like any other model:

  TorchMLPClassifier   a small PyTorch multi-layer perceptron, trained by this module.
                       dp=True trains it with differential privacy (DP-SGD via Opacus):
                       per-example gradients are clipped and noised, and the privacy budget
                       actually spent (epsilon, at the given delta) is recorded as `epsilon_`.
  TorchModuleAdapter   wraps an already-trained torch.nn.Module (load one with --model-file x.pt).
  ProbaModelAdapter    wraps anything whose .predict(X) returns class probabilities -- e.g. a
                       Keras model -- duck-typed, so TensorFlow is never imported here.

torch / opacus are optional: without them the "deep" family still has scikit-learn's MLP, and
the torch models simply don't appear (they say how to enable them).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def has_opacus() -> bool:
    try:
        import opacus  # noqa: F401
        return True
    except Exception:
        return False


def _build_mlp(n_in: int, n_out: int, hidden: Sequence[int], dropout: float):
    import torch.nn as nn
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, int(h)), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = int(h)
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """sklearn-compatible PyTorch MLP. Deterministic for a fixed `seed` on CPU."""

    def __init__(self, hidden_layers: Sequence[int] = (64, 32), learning_rate: float = 1e-3,
                 epochs: int = 30, batch_size: int = 128, weight_decay: float = 0.0,
                 dropout: float = 0.0, seed: int = 0, dp: bool = False,
                 target_epsilon: float = 3.0, target_delta: Optional[float] = None,
                 max_grad_norm: float = 1.0):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed
        self.dp = dp
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm

    def fit(self, X, y):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        X = np.asarray(X, dtype=np.float32)
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        self.n_features_in_ = X.shape[1]
        model = _build_mlp(X.shape[1], len(self.classes_), self.hidden_layers, self.dropout)
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        g = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y_idx.astype(np.int64))),
                            batch_size=int(self.batch_size), shuffle=True, generator=g)
        self.epsilon_, self.delta_ = None, None
        if self.dp:
            if not has_opacus():
                raise ImportError("dp=True needs Opacus: pip install opacus")
            from opacus import PrivacyEngine
            delta = self.target_delta or 1.0 / (10 * len(X))
            engine = PrivacyEngine(secure_mode=False)
            model, opt, loader = engine.make_private_with_epsilon(
                module=model, optimizer=opt, data_loader=loader, epochs=int(self.epochs),
                target_epsilon=float(self.target_epsilon), target_delta=delta,
                max_grad_norm=float(self.max_grad_norm))
        loss_fn = torch.nn.CrossEntropyLoss()
        model.train()
        for _ in range(int(self.epochs)):
            for xb, yb in loader:
                if len(xb) == 0:            # Poisson sampling (DP) can draw an empty batch
                    continue
                opt.zero_grad()
                loss_fn(model(xb), yb).backward()
                opt.step()
        if self.dp:
            self.delta_ = delta
            self.epsilon_ = float(engine.get_epsilon(delta))
            # Opacus leaves per-sample-gradient hooks on the network; any later backward pass
            # (e.g. Captum's integrated gradients) would trip over them. Training is done: remove them.
            if hasattr(model, "remove_hooks"):
                model.remove_hooks()
            model = model._module if hasattr(model, "_module") else model
            for prm in model.parameters():
                if hasattr(prm, "grad_sample"):
                    del prm.grad_sample
        model.eval()
        self.module_ = model
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            logits = self.module_(torch.from_numpy(np.asarray(X, dtype=np.float32)))
            return torch.softmax(logits, dim=1).numpy().astype(float)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class TorchModuleAdapter(ClassifierMixin, BaseEstimator):
    """An already-trained torch.nn.Module (outputs logits per class) as a sklearn classifier."""

    def __init__(self, module=None, classes: Sequence = (0, 1)):
        self.module = module
        self.classes = classes

    def fit(self, X, y=None):                 # pre-trained: evaluated as-is
        self.classes_ = np.asarray(self.classes)
        self.module_ = self.module
        return self

    def predict_proba(self, X):
        import torch
        if not hasattr(self, "module_"):
            self.fit(X)
        self.module_.eval()
        with torch.no_grad():
            out = self.module_(torch.as_tensor(np.asarray(X, dtype=np.float32)))
            return torch.softmax(out, dim=1).numpy().astype(float)

    def predict(self, X):
        if not hasattr(self, "classes_"):
            self.fit(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ProbaModelAdapter(ClassifierMixin, BaseEstimator):
    """Anything whose .predict(X) returns class probabilities (a Keras model, say) as a sklearn
    classifier. A single probability column is read as P(class 1)."""

    def __init__(self, model=None, classes: Sequence = (0, 1)):
        self.model = model
        self.classes = classes

    def fit(self, X, y=None):
        self.classes_ = np.asarray(self.classes)
        return self

    def predict_proba(self, X):
        p = np.asarray(self.model.predict(np.asarray(X, dtype=np.float32)), dtype=float)
        if p.ndim == 1 or p.shape[1] == 1:
            p = p.reshape(-1)
            p = np.column_stack([1 - p, p])
        return p

    def predict(self, X):
        if not hasattr(self, "classes_"):
            self.fit(X)
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def adapt(obj):
    """Wrap a loaded object so ethikit can evaluate it, or return it unchanged if it already
    speaks scikit-learn."""
    if hasattr(obj, "predict") and hasattr(obj, "get_params"):
        return obj
    try:
        import torch
        if isinstance(obj, torch.nn.Module):
            return TorchModuleAdapter(obj).fit(None)
    except ImportError:
        pass
    if hasattr(obj, "predict"):
        return ProbaModelAdapter(obj).fit(None)
    raise TypeError(f"{type(obj).__name__} has no predict(); ethikit can't evaluate it")
