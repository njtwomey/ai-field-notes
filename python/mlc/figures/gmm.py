import numpy as np
from pydantic import Field

from mlc.core.contracts import PointCloud2d, Strict
from mlc.core.figures import figure
from mlc.examples.gmm.model import fit, make_data
from mlc.examples.kmeans.model import Init


@figure("gaussian-mixture-model/blobs", title="Three axis-aligned clusters of different shapes, 400 points")
def blobs() -> PointCloud2d:
    x, _ = make_data()
    x = x.round(4)
    return PointCloud2d(x=x[:, 0].tolist(), y=x[:, 1].tolist())


class ModelSelectionTable(Strict):
    """Deviance −2 log L of the best EM fit for every sample size, restart budget and k. AIC and BIC follow from it."""

    ks: list[int]
    ns: list[int]
    """Sample sizes. Each uses the first n points of `blobs`, which are a random subsample."""
    max_restarts: int
    deviance: list[list[list[float]]] = Field(description="Indexed [restarts − 1][n index][k index].")


@figure("gaussian-mixture-model/model-selection", title="Best-fit deviance by n, restarts and k")
def model_selection() -> ModelSelectionTable:
    x, _ = make_data()
    ks = list(range(1, 9))
    ns = list(range(30, 401, 10))
    max_restarts = 5
    # best[r][i][j]: best mean log-likelihood over k-means++ seeds 0..r for ns[i] points and ks[j] components.
    best = np.full((max_restarts, len(ns), len(ks)), -np.inf)
    for i, n in enumerate(ns):
        for j, k in enumerate(ks):
            running = -np.inf
            for r in range(max_restarts):
                running = max(running, fit(x[:n], k, init=Init.plusplus, seed=r, max_iter=200)[-1].log_likelihood)
                best[r, i, j] = running
    deviance = -2.0 * best * np.array(ns)[None, :, None]
    return ModelSelectionTable(ks=ks, ns=ns, max_restarts=max_restarts, deviance=deviance.round(3).tolist())
