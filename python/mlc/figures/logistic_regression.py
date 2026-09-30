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


class SolverDataset(Strict):
    """A two-feature dataset for the solver comparison, with its unpenalised loss surface (bias fixed at 0)."""

    name: str
    description: str
    separable: bool
    """True when a line through the origin separates the classes, so the unpenalised loss has no minimum."""
    surface: Grid2d
    data: PointCloud2d


class SolverDatasets(Strict):
    datasets: list[SolverDataset]


def _correlated(n: int, *, correlation: float, gap: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Features with the given correlation; the class means differ by ``gap`` along the low-variance axis (1, -1)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n).astype(float)
    x = rng.multivariate_normal([0.0, 0.0], [[1.0, correlation], [correlation, 1.0]], size=n)
    axis = np.array([1.0, -1.0]) / np.sqrt(2.0)
    return x + np.where(y[:, None] == 1.0, 0.5, -0.5) * gap * axis, y


def _noisy(n: int, *, separation: float, flip: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = make_blobs(n, separation=separation, seed=seed)
    flipped = np.random.default_rng(seed + 1).random(n) < flip
    return x, np.where(flipped, 1.0 - y, y)


@figure("logistic-regression/solver-datasets", title="Datasets for the solver comparison, with their loss surfaces")
def solver_datasets() -> SolverDatasets:
    cases = [
        (
            "overlapping",
            "Two overlapping Gaussian classes. A well-conditioned bowl.",
            make_blobs(200, separation=3.0, seed=0),
        ),
        (
            "correlated",
            "Features with correlation 0.95; the classes differ along the narrow axis. An ill-conditioned valley.",
            _correlated(200, correlation=0.95, gap=0.25, seed=1),
        ),
        (
            "noisy labels",
            "Overlapping classes with 15% of labels flipped. A shallow bowl.",
            _noisy(200, separation=2.5, flip=0.15, seed=2),
        ),
        (
            "separable",
            "A line through the origin separates the classes. The loss has no minimum.",
            make_blobs(200, separation=6.0, seed=0),
        ),
    ]
    axis = np.linspace(-6.0, 6.0, 81).round(4)
    datasets = []
    for name, description, (x, y) in cases:
        loss = [[cross_entropy(sigmoid(x @ np.array([a, b])), y) for a in axis] for b in axis]
        signs = 2.0 * y - 1.0
        datasets.append(
            SolverDataset(
                name=name,
                description=description,
                separable=bool(np.min(signs * (x @ np.array([1.0, 1.0]))) > 0),
                surface=Grid2d(
                    x=axis.tolist(), y=axis.tolist(), z=np.round(loss, 5).tolist(), x_label="w₁", y_label="w₂"
                ),
                data=PointCloud2d(
                    x=x[:, 0].round(4).tolist(),
                    y=x[:, 1].round(4).tolist(),
                    group=y.astype(int).tolist(),
                    group_names=["y = 0", "y = 1"],
                ),
            )
        )
    return SolverDatasets(datasets=datasets)
