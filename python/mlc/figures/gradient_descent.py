import numpy as np

from mlc.core.contracts import Grid2d, Point2d, PointCloud2d, Samples1d, Strict
from mlc.core.figures import figure
from mlc.examples.gradient_descent.model import (
    cross_entropy,
    cross_entropy_optimum,
    loss_grid,
    make_regression,
    make_valley,
    mse,
    mse_optimum,
)


class RegressionSurface(Strict):
    """Log-MSE of y ≈ w·x + b over (w, b), with the data it was computed on."""

    surface: Grid2d
    data: PointCloud2d
    optimum: Point2d
    """Exact least-squares (w, b)."""
    optimum_loss: float


class LogisticValley(Strict):
    """Log-cross-entropy of a 1-D logistic regression over (w, b), with the data it was computed on."""

    surface: Grid2d
    data: Samples1d
    optimum: Point2d
    """Exact (w, b), found by Newton's method."""
    optimum_loss: float


@figure("gradient-descent/regression-surface", title="log₁₀ MSE over (w, b) for an uncentred feature")
def regression_surface() -> RegressionSurface:
    x, y = make_regression()
    w = np.linspace(-1.5, 4.5, 61).round(4)
    b = np.linspace(-5.0, 4.0, 61).round(4)
    f = mse(x, y)
    theta = mse_optimum(x, y)
    z = np.log10(loss_grid(f, w, b))
    return RegressionSurface(
        optimum=Point2d(x=float(theta[0]), y=float(theta[1])),
        optimum_loss=f(theta)[0],
        surface=Grid2d(
            x=w.tolist(), y=b.tolist(), z=z.round(4).tolist(), x_label="w", y_label="b", z_label="log₁₀ MSE"
        ),
        data=PointCloud2d(x=x.round(4).tolist(), y=y.round(4).tolist()),
    )


@figure("gradient-descent/logistic-valley", title="log₁₀ cross-entropy over (w, b) for a 1-D logistic regression")
def logistic_valley() -> LogisticValley:
    x, y = make_valley()
    w = np.linspace(-1.0, 5.0, 81).round(4)
    b = np.linspace(-20.0, 4.0, 81).round(4)
    f = cross_entropy(x, y)
    theta = cross_entropy_optimum(x, y)
    z = np.log10(loss_grid(f, w, b))
    return LogisticValley(
        optimum=Point2d(x=float(theta[0]), y=float(theta[1])),
        optimum_loss=f(theta)[0],
        surface=Grid2d(
            x=w.tolist(), y=b.tolist(), z=z.round(4).tolist(), x_label="w", y_label="b", z_label="log₁₀ cross-entropy"
        ),
        data=Samples1d(x=x.round(4).tolist(), group=y.astype(int).tolist(), group_names=["y = 0", "y = 1"]),
    )
