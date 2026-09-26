from mlc.core.contracts import PointCloud2d
from mlc.core.figures import figure
from mlc.examples.kmeans.model import make_blobs


@figure("k-means/blobs", title="Three Gaussian blobs, 300 points")
def blobs() -> PointCloud2d:
    x = make_blobs(300, k=3, spread=0.8, seed=0).round(4)
    return PointCloud2d(x=x[:, 0].tolist(), y=x[:, 1].tolist())
