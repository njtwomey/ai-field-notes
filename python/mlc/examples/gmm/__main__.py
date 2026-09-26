"""Fit a diagonal-covariance Gaussian mixture by EM, compare it with k-means, and choose k by BIC."""

from itertools import permutations

import numpy as np
from pydantic import Field
from pydantic_settings import CliApp, CliSubCommand

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.gmm.model import aic, bic, fit, log_densities, make_data
from mlc.examples.kmeans.model import Init, lloyd


class Data(Command):
    """Options for the synthetic data: three axis-aligned clusters of different shapes."""

    n: int = Field(400, ge=10, description="Number of points.")
    seed: int = Field(0, description="Seed for the data generator and the initialisation.")


class Fit(Data):
    """Run EM once and report every iteration, the clusters and the fitted density."""

    k: int = Field(3, ge=1, description="Number of mixture components.")
    init: Init = Field(Init.plusplus, description="How to choose the initial means.")

    def cli_cmd(self) -> None:
        x, _ = make_data(self.n, seed=self.seed)
        steps = fit(x, self.k, init=self.init, seed=self.seed)
        emit.table(
            ["iteration", "mean log-likelihood"],
            [[i, s.log_likelihood] for i, s in enumerate(steps)],
            title="Convergence. EM never decreases the likelihood.",
        )
        final = steps[-1]
        emit.table(
            ["component", "weight", "mean x₁", "mean x₂", "σ x₁", "σ x₂"],
            [
                [j + 1, final.mixture.weights[j], *final.mixture.means[j], *np.sqrt(final.mixture.variances[j])]
                for j in range(final.mixture.k)
            ],
            title="Fitted components",
        )
        grid = np.linspace(-6.0, 6.0, 61)
        gx, gy = np.meshgrid(grid, grid)
        log_joint = log_densities(np.column_stack([gx.ravel(), gy.ravel()]), final.mixture)
        log_density = np.logaddexp.reduce(log_joint, axis=1).reshape(gx.shape)
        emit.heatmap(
            grid,
            grid,
            log_density,
            title="Fitted log density with points coloured by most likely component",
            x_label="x₁",
            y_label="x₂",
            overlay=[emit.scatter("points", x[:, 0], x[:, 1], group=final.responsibilities.argmax(axis=1))],
        )


def best_accuracy(labels: np.ndarray, truth: np.ndarray, k: int) -> float:
    """Accuracy under the best matching of cluster ids to true labels. Cluster ids are arbitrary."""
    return max(float(np.mean(np.array(p)[labels] == truth)) for p in permutations(range(k)))


class Compare(Data):
    """Fit k-means and the Gaussian mixture to the same data and compare their clusters with the true labels."""

    def cli_cmd(self) -> None:
        x, truth = make_data(self.n, seed=self.seed)
        km = lloyd(x, 3, init=Init.plusplus, max_iter=100, seed=self.seed)[-1].labels
        gm = fit(x, 3, init=Init.plusplus, seed=self.seed)[-1].responsibilities.argmax(axis=1)
        emit.metrics(
            {"k-means accuracy": best_accuracy(km, truth, 3), "GMM accuracy": best_accuracy(gm, truth, 3)},
            title="Agreement with the true clusters",
        )
        emit.chart(emit.scatter("k-means", x[:, 0], x[:, 1], group=km), title="k-means", x_label="x₁", y_label="x₂")
        emit.chart(
            emit.scatter("GMM", x[:, 0], x[:, 1], group=gm), title="Gaussian mixture", x_label="x₁", y_label="x₂"
        )


class Select(Data):
    """Fit k = 1..k_max and report −2 log L, AIC and BIC. Only the likelihood always improves with k."""

    k_max: int = Field(8, ge=2, description="Largest k to try.")
    restarts: int = Field(3, ge=1, description="k-means++ starts per k. The best likelihood is kept.")

    def cli_cmd(self) -> None:
        x, _ = make_data(self.n, seed=self.seed)
        n, d = x.shape
        ks = np.arange(1, self.k_max + 1)
        lls = [
            max(fit(x, int(k), init=Init.plusplus, seed=self.seed + r)[-1].log_likelihood for r in range(self.restarts))
            for k in ks
        ]
        aics = [aic(ll, n, int(k), d) for k, ll in zip(ks, lls, strict=True)]
        bics = [bic(ll, n, int(k), d) for k, ll in zip(ks, lls, strict=True)]
        emit.table(
            ["k", "−2 log L", "AIC", "BIC"],
            [[int(k), -2 * n * ll, a, b] for k, ll, a, b in zip(ks, lls, aics, bics, strict=True)],
        )
        emit.metrics(
            {"AIC picks k": int(ks[int(np.argmin(aics))]), "BIC picks k": int(ks[int(np.argmin(bics))])},
            title="Selected number of components",
        )
        emit.chart(
            emit.line("−2 log L", ks, [-2 * n * ll for ll in lls]),
            emit.line("AIC", ks, aics),
            emit.line("BIC", ks, bics),
            title="Criteria by number of components. Lower is better.",
            x_label="k",
            y_label="criterion",
        )


class GaussianMixture(Command):
    """Gaussian mixture models with diagonal covariance."""

    fit: CliSubCommand[Fit] = Field(description=Fit.__doc__)
    compare: CliSubCommand[Compare] = Field(description=Compare.__doc__)
    select: CliSubCommand[Select] = Field(description=Select.__doc__)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


if __name__ == "__main__":
    GaussianMixture.main()
