"""Gradient descent, with optional heavy-ball momentum, on two small problems.

- Linear regression with an uncentred feature. The MSE is a quadratic in (w, b) whose valley is long and diagonal.
- 1-D logistic regression with an uncentred feature. The cross-entropy valley is longer still, and not quadratic.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]
type LossAndGrad = Callable[[Array], tuple[float, Array]]

DIVERGED = 1e8
"""Loss above which a run is stopped and reported as diverged."""


def make_regression(n: int = 50, *, seed: int = 0) -> tuple[Array, Array]:
    """y = 1.5x - 0.5 + noise with x in [0, 4]. The feature is not centred, which couples w and b."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 4.0, size=n)
    return x, 1.5 * x - 0.5 + rng.normal(0.0, 0.5, size=n)


def mse(x: Array, y: Array) -> LossAndGrad:
    """MSE of y ≈ w·x + b as a function of theta = (w, b), with its gradient."""

    def f(theta: Array) -> tuple[float, Array]:
        residual = theta[0] * x + theta[1] - y
        return float(np.mean(residual**2)), 2.0 * np.array([np.mean(residual * x), np.mean(residual)])

    return f


def mse_optimum(x: Array, y: Array) -> Array:
    """Exact least-squares (w, b)."""
    theta, *_ = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)
    return theta


def mse_hessian(x: Array) -> Array:
    """The MSE Hessian is constant: (2/n) XᵀX with a column of ones for the bias."""
    return 2.0 * np.array([[np.mean(x * x), np.mean(x)], [np.mean(x), 1.0]])


def make_valley(n: int = 100, *, seed: int = 0) -> tuple[Array, Array]:
    """Two unit-variance classes centred at x = 3 and x = 5. The optimum is near w = 2, b = -8."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(float)
    return rng.normal(3.0 + 2.0 * y, 1.0), y


def cross_entropy(x: Array, y: Array) -> LossAndGrad:
    """Mean cross-entropy of P(y = 1) = σ(w·x + b) as a function of theta = (w, b), with its gradient."""

    def f(theta: Array) -> tuple[float, Array]:
        z = theta[0] * x + theta[1]
        # log(1 + e^z) - y·z is the stable form of the per-sample cross-entropy.
        loss = float(np.mean(np.logaddexp(0.0, z) - y * z))
        residual = 1.0 / (1.0 + np.exp(-z)) - y
        return loss, np.array([np.mean(residual * x), np.mean(residual)])

    return f


def cross_entropy_optimum(x: Array, y: Array, *, iterations: int = 50) -> Array:
    """Exact (w, b) by Newton's method. Used only as the reference optimum, never as the method being shown."""
    design = np.column_stack([x, np.ones_like(x)])
    theta = np.zeros(2)
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(design @ theta)))
        hessian = design.T @ (design * (p * (1.0 - p))[:, None]) / len(x)
        theta = theta - np.linalg.solve(hessian, design.T @ (p - y) / len(x))
    return theta


@dataclass(frozen=True)
class Trajectory:
    thetas: Array
    """Shape (steps + 1, 2): the start and every iterate."""
    losses: list[float]
    diverged: bool


def descend(f: LossAndGrad, theta0: Array, *, lr: float, steps: int, momentum: float = 0.0) -> Trajectory:
    """Heavy-ball update: v ← βv − η∇L(θ), θ ← θ + v. With β = 0 this is plain gradient descent."""
    theta = np.asarray(theta0, dtype=float).copy()
    velocity = np.zeros_like(theta)
    thetas = [theta.copy()]
    losses: list[float] = []
    for _ in range(steps):
        loss, grad = f(theta)
        losses.append(loss)
        if not np.isfinite(loss) or loss > DIVERGED:
            return Trajectory(np.array(thetas), losses, diverged=True)
        velocity = momentum * velocity - lr * grad
        theta = theta + velocity
        thetas.append(theta.copy())
    losses.append(f(theta)[0])
    return Trajectory(np.array(thetas), losses, diverged=False)


def excess(losses: list[float], optimum: float, floor: float = 1e-12) -> Array:
    """L - L*, floored so that a log scale stays finite once a run has converged."""
    return np.maximum(np.asarray(losses) - optimum, floor)


def loss_grid(f: LossAndGrad, w: Array, b: Array) -> Array:
    """Loss at every (w[j], b[i]), row-major by b."""
    return np.array([[f(np.array([wj, bi]))[0] for wj in w] for bi in b])
