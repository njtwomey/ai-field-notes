"""Gaussian mixture model with diagonal covariances, fitted by expectation-maximisation (EM)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mlc.examples.kmeans.model import Init, initialise

type Array = NDArray[np.float64]
type Labels = NDArray[np.int64]

VARIANCE_FLOOR = 1e-3
"""Smallest allowed variance. Without it a component can collapse onto one point and the likelihood diverges."""


@dataclass(frozen=True)
class Mixture:
    weights: Array
    """Shape (k,). Mixing proportions π, summing to 1."""
    means: Array
    """Shape (k, d)."""
    variances: Array
    """Shape (k, d). The diagonal of each component's covariance."""

    @property
    def k(self) -> int:
        return len(self.weights)


@dataclass(frozen=True)
class Step:
    mixture: Mixture
    responsibilities: Array
    """Shape (n, k). Posterior probability that point i came from component j."""
    log_likelihood: float
    """Mean log-likelihood per point under `mixture`."""


def make_data(n: int = 400, *, seed: int = 0) -> tuple[Array, Labels]:
    """Two long, thin, parallel clusters close together and one small round one. Returns points and true labels.

    k-means cuts the long clusters across their length, because it favours round clusters of equal size. Diagonal
    Gaussians fit them.
    """
    rng = np.random.default_rng(seed)
    means = np.array([[0.0, 1.2], [0.0, -1.2], [4.0, 3.5]])
    stds = np.array([[2.2, 0.35], [2.2, 0.35], [0.4, 0.4]])
    labels = rng.choice(3, size=n, p=[0.4, 0.4, 0.2])
    return means[labels] + stds[labels] * rng.normal(size=(n, 2)), labels


def log_densities(x: Array, mixture: Mixture) -> Array:
    """Shape (n, k): log π_j + log N(x_i | μ_j, diag σ²_j)."""
    diff = x[:, None, :] - mixture.means[None]
    log_normal = -0.5 * np.sum(np.log(2 * np.pi * mixture.variances)[None] + diff**2 / mixture.variances[None], axis=2)
    return np.log(mixture.weights)[None] + log_normal


def e_step(x: Array, mixture: Mixture) -> Step:
    log_joint = log_densities(x, mixture)
    # log-sum-exp over components, shifted by the row maximum for numerical stability.
    top = log_joint.max(axis=1, keepdims=True)
    log_marginal = top[:, 0] + np.log(np.exp(log_joint - top).sum(axis=1))
    return Step(mixture, np.exp(log_joint - log_marginal[:, None]), float(log_marginal.mean()))


def m_step(x: Array, responsibilities: Array) -> Mixture:
    """Weighted maximum-likelihood estimates, with each point weighted by its responsibility."""
    counts = responsibilities.sum(axis=0)
    means = responsibilities.T @ x / counts[:, None]
    variances = np.stack(
        [(responsibilities[:, j, None] * (x - means[j]) ** 2).sum(axis=0) / counts[j] for j in range(len(counts))]
    )
    return Mixture(counts / len(x), means, np.maximum(variances, VARIANCE_FLOOR))


def initial_mixture(x: Array, k: int, init: Init, seed: int) -> Mixture:
    """Means from k-means seeding; every component starts with the overall variance and equal weight."""
    means = initialise(x, k, init, np.random.default_rng(seed))
    return Mixture(np.full(k, 1.0 / k), means, np.tile(x.var(axis=0), (k, 1)))


def fit(x: Array, k: int, *, init: Init, seed: int, max_iter: int = 200, tol: float = 1e-6) -> list[Step]:
    """Every EM iteration, so callers can inspect convergence. The last step is the result."""
    steps = [e_step(x, initial_mixture(x, k, init, seed))]
    for _ in range(max_iter):
        steps.append(e_step(x, m_step(x, steps[-1].responsibilities)))
        if steps[-1].log_likelihood - steps[-2].log_likelihood < tol:
            break
    return steps


def parameter_count(k: int, d: int) -> int:
    """k means and k diagonal variances in d dimensions, plus k − 1 free weights."""
    return 2 * k * d + (k - 1)


def aic(mean_log_likelihood: float, n: int, k: int, d: int) -> float:
    """Akaike information criterion, −2 log L + 2p. Lower is better."""
    return float(-2.0 * n * mean_log_likelihood + 2.0 * parameter_count(k, d))


def bic(mean_log_likelihood: float, n: int, k: int, d: int) -> float:
    """Bayesian information criterion, −2 log L + p log n. Lower is better."""
    return float(-2.0 * n * mean_log_likelihood + parameter_count(k, d) * np.log(n))
