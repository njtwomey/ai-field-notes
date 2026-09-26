"""Binary logistic regression trained by full-batch gradient descent."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]


@dataclass(frozen=True)
class Fit:
    weights: Array
    bias: float
    losses: list[float]

    def predict_proba(self, x: Array) -> Array:
        return sigmoid(x @ self.weights + self.bias)


def sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-z))


def make_blobs(n: int, *, separation: float, seed: int) -> tuple[Array, Array]:
    """Two isotropic Gaussian classes whose means are ``separation`` apart along the diagonal."""
    rng = np.random.default_rng(seed)
    half = separation / (2.0 * np.sqrt(2.0))
    y = rng.integers(0, 2, size=n).astype(float)
    centres = np.where(y[:, None] == 1.0, half, -half)
    return centres + rng.normal(size=(n, 2)), y


def cross_entropy(p: Array, y: Array) -> float:
    eps = 1e-12
    return float(-np.mean(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))


def fit(x: Array, y: Array, *, lr: float, epochs: int, l2: float = 0.0) -> Fit:
    n, d = x.shape
    w = np.zeros(d)
    b = 0.0
    losses: list[float] = []
    for _ in range(epochs):
        p = sigmoid(x @ w + b)
        losses.append(cross_entropy(p, y) + 0.5 * l2 * float(w @ w))
        # The gradient of the mean cross-entropy is Xᵀ(p - y)/n.
        w -= lr * (x.T @ (p - y) / n + l2 * w)
        b -= lr * float(np.mean(p - y))
    return Fit(weights=w, bias=b, losses=losses)
