"""Fit ordinary least squares and its regularised variants to the diabetes data with scikit-learn."""

from enum import StrEnum

import numpy as np
from pydantic import Field
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge, lars_path

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.linear_models.model import Array, Data, Penalty, diabetes, exact, objective, sgd_trajectory


class Model(StrEnum):
    ols = "ols"
    ridge = "ridge"
    lasso = "lasso"
    elastic_net = "elastic-net"
    lars = "lars"
    sgd = "sgd"


class LinearModels(Command):
    """Fit one linear model to the standardised diabetes data and report its coefficients."""

    model: Model = Field(Model.ols, description="ols, ridge, lasso, elastic-net, lars (the whole path) or sgd.")
    alpha: float = Field(
        1.0,
        ge=0.0,
        description="Penalty strength, in scikit-learn's scaling for the chosen model (Ridge, Lasso, ElasticNet, SGD).",
    )
    l1_ratio: float = Field(0.5, ge=0.0, le=1.0, description="Share of the L1 term for elastic-net and SGD.")
    penalty: Penalty = Field(Penalty.l1, description="SGD only: l2, l1 or elasticnet.")
    epochs: int = Field(50, ge=1, description="SGD only: passes over the data.")
    seed: int = Field(0, description="SGD only: seed for the sample order.")

    def cli_cmd(self) -> None:
        data = diabetes()
        if self.model is Model.lars:
            self.lars(data.x, data.y, data.features)
            return
        if self.model is Model.sgd:
            w, b = self.sgd(data)
        else:
            estimator = {
                Model.ols: LinearRegression(),
                Model.ridge: Ridge(alpha=self.alpha),
                Model.lasso: Lasso(alpha=self.alpha),
                Model.elastic_net: ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio),
            }[self.model]
            estimator.fit(data.x, data.y)
            w, b = np.asarray(estimator.coef_, dtype=float), float(estimator.intercept_)

        residual = data.y - data.x @ w - b
        r2 = 1 - residual @ residual / (data.y @ data.y)
        emit.metrics(
            {
                "R² (training)": float(r2),
                "nonzero coefficients": int(np.count_nonzero(w)),
                "‖w‖₁": float(np.abs(w).sum()),
            },
            title="Fit",
        )
        emit.table(
            ["feature", "coefficient"],
            [[name, round(float(c), 3)] for name, c in zip(data.features, w, strict=True)],
            title="Coefficients (per standard deviation of each feature)",
        )

    def sgd(self, data: Data) -> tuple[Array, float]:
        weights, intercepts = sgd_trajectory(
            data, self.penalty, self.alpha, self.l1_ratio, epochs=self.epochs, seed=self.seed
        )
        w_star, b_star = exact(data, self.penalty, self.alpha, self.l1_ratio)
        best = objective(data, w_star, b_star, self.penalty, self.alpha, self.l1_ratio)
        gaps = [
            objective(data, w, b, self.penalty, self.alpha, self.l1_ratio) - best
            for w, b in zip(weights, intercepts, strict=True)
        ]
        emit.chart(
            emit.line("objective − optimum", np.arange(len(gaps)), np.maximum(gaps, 1e-12)),
            title="SGD against the exact optimum",
            x_label="epoch",
            y_label="objective gap",
        )
        emit.table(
            ["feature", "SGD", "exact"],
            [
                [name, round(float(a), 3), round(float(e), 3)]
                for name, a, e in zip(data.features, weights[-1], w_star, strict=True)
            ],
            title="Final SGD weights against the exact solution",
        )
        return weights[-1], float(intercepts[-1])

    @staticmethod
    def lars(x: Array, y: Array, features: list[str]) -> None:
        _, active, coefs = lars_path(x, y, method="lar")
        norms = np.abs(coefs).sum(axis=0)
        emit.chart(
            *(emit.line(name, norms, coefs[j]) for j, name in enumerate(features)),
            title="LARS coefficient paths",
            x_label="‖w‖₁",
            y_label="coefficient",
        )
        emit.table(
            ["step", "feature entering"], [[i + 1, features[j]] for i, j in enumerate(active)], title="Order of entry"
        )


if __name__ == "__main__":
    LinearModels.main()
