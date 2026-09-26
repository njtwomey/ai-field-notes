"""Singular value decomposition and low-rank approximation (Eckart–Young)."""

import numpy as np
from numpy.typing import NDArray

type Array = NDArray[np.float64]


def make_image(size: int = 64) -> Array:
    """A test image in [0, 1]: a smooth gradient, a disc, a bar, a diagonal band and a checkerboard patch.

    Smooth and axis-aligned parts are low rank; the diagonal band and the checkerboard need many components.
    """
    y, x = np.mgrid[0:size, 0:size] / (size - 1)
    image = 0.25 + 0.2 * x
    image = np.where((x - 0.3) ** 2 + (y - 0.35) ** 2 < 0.04, 0.95, image)
    image = np.where((x > 0.6) & (x < 0.9) & (y > 0.15) & (y < 0.3), 0.85, image)
    image = np.where(np.abs(x - y - 0.25) < 0.04, 0.05, image)
    checker = ((np.floor(x * 16) + np.floor(y * 16)) % 2).astype(float)
    image = np.where((x > 0.6) & (y > 0.6) & (x < 0.9) & (y < 0.9), checker, image)
    return image


def low_rank(u: Array, s: Array, vt: Array, k: int) -> Array:
    """Truncated SVD: the sum of the first k rank-one terms σᵢ uᵢ vᵢᵀ."""
    return (u[:, :k] * s[:k]) @ vt[:k]
