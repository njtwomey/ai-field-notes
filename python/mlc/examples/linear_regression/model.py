"""Ordinary least squares, solved exactly and by gradient descent."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]


@dataclass(frozen=True)
class Fit:
    weights: Array
    """Coefficients, one per feature."""
    bias: float
    losses: list[float]
    """Mean squared error per epoch. Empty for the closed-form solution."""

    def predict(self, x: Array) -> Array:
        return x @ self.weights + self.bias


def make_data(n: int, *, slope: float, intercept: float, noise: float, seed: int) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3.0, 3.0, size=(n, 1))
    y = slope * x[:, 0] + intercept + rng.normal(0.0, noise, size=n)
    return x, y


def fit_normal_equation(x: Array, y: Array) -> Fit:
    design = np.column_stack([np.ones(len(x)), x])
    # lstsq is the numerically stable form of (XᵀX)⁻¹Xᵀy.
    theta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return Fit(weights=theta[1:], bias=float(theta[0]), losses=[])


def fit_gradient_descent(x: Array, y: Array, *, lr: float, epochs: int) -> Fit:
    n, d = x.shape
    w = np.zeros(d)
    b = 0.0
    losses: list[float] = []
    for _ in range(epochs):
        residual = x @ w + b - y
        losses.append(float(np.mean(residual**2)))
        w -= lr * (2.0 / n) * (x.T @ residual)
        b -= lr * (2.0 / n) * float(residual.sum())
    return Fit(weights=w, bias=b, losses=losses)


def r_squared(y: Array, y_hat: Array) -> float:
    return 1.0 - float(np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2))
