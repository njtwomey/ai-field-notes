import numpy as np

from mlc.core.contracts import Grid2d, Strict
from mlc.core.figures import figure
from mlc.examples.welch.model import type_i_error


class TypeIErrorMaps(Strict):
    """Type I error rates of the pooled and Welch tests over the variance ratio and the second group's size."""

    pooled: Grid2d
    welch: Grid2d
    n1: int
    alpha: float


@figure("welch-t-test/type-i-error", title="Type I error of pooled and Welch t-tests, n₁ = 20")
def type_i_error_maps() -> TypeIErrorMaps:
    n1, alpha = 20, 0.05
    log_ratios = np.linspace(-2.0, 2.0, 17)
    n2s = np.arange(5, 61, 5)
    pooled = np.zeros((len(n2s), len(log_ratios)))
    welch = np.zeros_like(pooled)
    for i, n2 in enumerate(n2s):
        for j, r in enumerate(log_ratios):
            pooled[i, j], welch[i, j] = type_i_error(n1, int(n2), 2.0**r, alpha=alpha, reps=10_000, seed=i * 100 + j)

    def grid(z: np.ndarray) -> Grid2d:
        return Grid2d(
            x=log_ratios.round(3).tolist(),
            y=n2s.astype(float).tolist(),
            z=z.round(4).tolist(),
            x_label="log₂(sd₂ / sd₁)",
            y_label="n₂",
            z_label="Type I error rate",
        )

    return TypeIErrorMaps(pooled=grid(pooled), welch=grid(welch), n1=n1, alpha=alpha)
