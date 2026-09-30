"""Binary logistic regression fitted three ways: gradient descent, Newton's method (IRLS) and mini-batch SGD.

Every solver works on the design matrix with a leading column of ones, so ``theta = (b, w₁, …, w_d)``. The L2
penalty applies to the weights only, never to the bias.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]


@dataclass(frozen=True)
class Fit:
    theta: Array
    losses: list[float]
    """Penalised mean cross-entropy before the first update and after each pass over the data."""

    @property
    def bias(self) -> float:
        return float(self.theta[0])

    @property
    def weights(self) -> Array:
        return self.theta[1:]

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


def with_intercept(x: Array) -> Array:
    return np.column_stack([np.ones(len(x)), x])


def penalty_mask(d: int) -> Array:
    """1 for each penalised coefficient: every weight, but not the bias in position 0."""
    mask = np.ones(d)
    mask[0] = 0.0
    return mask


def loss(theta: Array, x1: Array, y: Array, l2: float) -> float:
    z = x1 @ theta
    # Each term is log(1 + e^z) - y z, the cross-entropy written in the score. logaddexp avoids overflow.
    data = float(np.mean(np.logaddexp(0.0, z) - y * z))
    return data + 0.5 * l2 * float(theta[1:] @ theta[1:])


def gradient(theta: Array, x1: Array, y: Array, l2: float) -> Array:
    """∇L = Xᵀ(p - y)/n + λθ, with the bias unpenalised."""
    p = sigmoid(x1 @ theta)
    return x1.T @ (p - y) / len(y) + l2 * penalty_mask(len(theta)) * theta


def gradient_descent(x: Array, y: Array, *, lr: float, epochs: int, l2: float = 0.0) -> Fit:
    x1 = with_intercept(x)
    theta = np.zeros(x1.shape[1])
    losses = [loss(theta, x1, y, l2)]
    for _ in range(epochs):
        theta = theta - lr * gradient(theta, x1, y, l2)
        losses.append(loss(theta, x1, y, l2))
    return Fit(theta, losses)


def newton(x: Array, y: Array, *, iterations: int, l2: float = 0.0, halving: bool = True) -> Fit:
    """Newton's method written as iteratively reweighted least squares (IRLS).

    Each iteration solves the weighted least-squares problem (XᵀSX + nλD) θ = XᵀS z, where S = diag(p(1 - p)) holds
    the weights and z = Xθ + (y - p)/s is the working response. The solution is exactly the Newton step. With
    ``halving``, the step is halved until the loss decreases, which makes the method converge from any start.
    """
    x1 = with_intercept(x)
    n, d = x1.shape
    theta = np.zeros(d)
    losses = [loss(theta, x1, y, l2)]
    for _ in range(iterations):
        eta = x1 @ theta
        p = sigmoid(eta)
        s = np.maximum(p * (1.0 - p), 1e-12)
        z = eta + (y - p) / s
        a = x1.T @ (s[:, None] * x1) + n * l2 * np.diag(penalty_mask(d))
        proposal = np.linalg.solve(a, x1.T @ (s * z))
        step = proposal - theta
        t = 1.0
        while halving and loss(theta + t * step, x1, y, l2) > losses[-1] and t > 1e-8:
            t /= 2.0
        theta = theta + t * step
        losses.append(loss(theta, x1, y, l2))
    return Fit(theta, losses)


def sgd(x: Array, y: Array, *, lr: float, epochs: int, batch: int, l2: float = 0.0, seed: int = 0) -> Fit:
    """Mini-batch SGD. The step size decays as lr / (1 + epoch), so the iterates settle instead of hovering."""
    x1 = with_intercept(x)
    n = len(y)
    rng = np.random.default_rng(seed)
    theta = np.zeros(x1.shape[1])
    losses = [loss(theta, x1, y, l2)]
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            rows = order[start : start + batch]
            # The mini-batch gradient is an unbiased estimate of the full gradient.
            theta = theta - lr / (1.0 + epoch) * gradient(theta, x1[rows], y[rows], l2)
        losses.append(loss(theta, x1, y, l2))
    return Fit(theta, losses)
