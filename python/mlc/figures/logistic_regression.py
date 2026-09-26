import numpy as np

from mlc.core.contracts import Grid2d, PointCloud2d, Strict
from mlc.core.figures import figure
from mlc.examples.logistic_regression.model import cross_entropy, make_blobs, sigmoid


class LossSurface(Strict):
    """Cross-entropy over the two weights, with the dataset it was computed on."""

    surface: Grid2d
    data: PointCloud2d


@figure("logistic-regression/loss-surface", title="Cross-entropy over (w₁, w₂) with bias fixed at 0")
def loss_surface() -> LossSurface:
    x, y = make_blobs(200, separation=3.0, seed=0)
    axis = np.linspace(-4.0, 4.0, 81).round(4)
    loss = [[cross_entropy(sigmoid(x @ np.array([a, b])), y) for a in axis] for b in axis]
    return LossSurface(
        surface=Grid2d(x=axis.tolist(), y=axis.tolist(), z=np.round(loss, 5).tolist(), x_label="w₁", y_label="w₂"),
        data=PointCloud2d(
            x=x[:, 0].round(4).tolist(),
            y=x[:, 1].round(4).tolist(),
            group=y.astype(int).tolist(),
            group_names=["y = 0", "y = 1"],
        ),
    )
