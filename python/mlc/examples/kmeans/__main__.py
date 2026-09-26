"""Cluster synthetic 2-D blobs with k-means."""

import numpy as np
from pydantic import Field
from pydantic_settings import CliApp, CliSubCommand

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.kmeans.model import Init, lloyd, make_blobs


class Data(Command):
    """Options for the synthetic data: three Gaussian blobs on a circle."""

    n: int = Field(300, ge=3, description="Number of points.")
    spread: float = Field(0.8, ge=0.0, description="Standard deviation of each blob.")
    seed: int = Field(0, description="Seed for the data generator and the initialisation.")


class Fit(Data):
    """Run Lloyd's algorithm once and report every iteration."""

    k: int = Field(3, ge=1, description="Number of clusters.")
    init: Init = Field(Init.plusplus, description="Centroid initialisation.")
    max_iter: int = Field(50, ge=1, description="Iteration cap. Lloyd's algorithm usually stops earlier.")

    def cli_cmd(self) -> None:
        x = make_blobs(self.n, k=3, spread=self.spread, seed=self.seed)
        steps = lloyd(x, self.k, init=self.init, max_iter=self.max_iter, seed=self.seed)
        emit.table(["iteration", "inertia"], [[i, s.inertia] for i, s in enumerate(steps)], title="Convergence")
        final = steps[-1]
        emit.chart(
            emit.scatter("points", x[:, 0], x[:, 1], group=final.labels),
            emit.scatter("centroids", final.centroids[:, 0], final.centroids[:, 1]),
            title=f"Clusters after {len(steps)} iterations",
            x_label="x₁",
            y_label="x₂",
        )


class Elbow(Data):
    """Report the final inertia for k = 1..k_max. The bend in the curve suggests a choice of k."""

    k_max: int = Field(8, ge=2, description="Largest k to try.")

    def cli_cmd(self) -> None:
        x = make_blobs(self.n, k=3, spread=self.spread, seed=self.seed)
        ks = np.arange(1, self.k_max + 1)
        inertias = [lloyd(x, int(k), init=Init.plusplus, max_iter=100, seed=self.seed)[-1].inertia for k in ks]
        emit.chart(emit.line("inertia", ks, inertias), title="Elbow curve", x_label="k", y_label="inertia")


class KMeans(Command):
    """k-means clustering on synthetic 2-D data."""

    fit: CliSubCommand[Fit] = Field(description="Cluster once and show the result.")
    elbow: CliSubCommand[Elbow] = Field(description="Sweep k and plot inertia.")

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


if __name__ == "__main__":
    KMeans.main()
