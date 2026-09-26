"""Lloyd's algorithm for k-means with random or k-means++ initialisation."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]
type Labels = NDArray[np.int64]


class Init(StrEnum):
    random = "random"
    plusplus = "kmeans++"


@dataclass(frozen=True)
class Step:
    centroids: Array
    labels: Labels
    inertia: float


def make_blobs(n: int, *, k: int, spread: float, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
    centres = 3.0 * np.column_stack([np.cos(angles), np.sin(angles)])
    return centres[rng.integers(0, k, size=n)] + rng.normal(0.0, spread, size=(n, 2))


def initialise(x: Array, k: int, method: Init, rng: np.random.Generator) -> Array:
    if method is Init.random:
        return x[rng.choice(len(x), size=k, replace=False)].copy()
    centroids = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        # Sample the next centre with probability proportional to squared distance from the nearest chosen centre.
        d2 = np.min(((x[:, None, :] - np.array(centroids)[None]) ** 2).sum(-1), axis=1)
        centroids.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    return np.array(centroids)


def assign(x: Array, centroids: Array) -> tuple[Labels, float]:
    d2 = ((x[:, None, :] - centroids[None]) ** 2).sum(-1)
    labels = d2.argmin(axis=1)
    return labels, float(d2[np.arange(len(x)), labels].sum())


def lloyd(x: Array, k: int, *, init: Init, max_iter: int, seed: int) -> list[Step]:
    """Return every step, so callers can inspect convergence. The last step is the result."""
    rng = np.random.default_rng(seed)
    centroids = initialise(x, k, init, rng)
    steps: list[Step] = []
    for _ in range(max_iter):
        labels, inertia = assign(x, centroids)
        steps.append(Step(centroids, labels, inertia))
        updated = np.array([x[labels == j].mean(0) if np.any(labels == j) else centroids[j] for j in range(k)])
        if np.allclose(updated, centroids):
            break
        centroids = updated
    return steps
