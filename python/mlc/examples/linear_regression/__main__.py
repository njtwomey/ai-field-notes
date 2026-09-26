"""Fit a one-feature linear regression to synthetic data."""

from enum import StrEnum

import numpy as np
from pydantic import Field

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.linear_regression.model import fit_gradient_descent, fit_normal_equation, make_data, r_squared


class Method(StrEnum):
    normal = "normal"
    gd = "gd"


class LinearRegression(Command):
    """Fit y = wx + b to noisy samples of y = 1.5x - 0.5 and report the fit."""

    method: Method = Field(Method.normal, description="normal: closed-form least squares. gd: gradient descent.")
    n: int = Field(100, ge=2, description="Number of samples.")
    noise: float = Field(0.5, ge=0.0, description="Standard deviation of the Gaussian noise added to y.")
    lr: float = Field(0.05, gt=0.0, description="Gradient-descent step size. Ignored by the normal method.")
    epochs: int = Field(200, ge=1, description="Gradient-descent iterations. Ignored by the normal method.")
    seed: int = Field(0, description="Seed for the data generator.")

    def cli_cmd(self) -> None:
        x, y = make_data(self.n, slope=1.5, intercept=-0.5, noise=self.noise, seed=self.seed)
        if self.method is Method.normal:
            fit = fit_normal_equation(x, y)
        else:
            fit = fit_gradient_descent(x, y, lr=self.lr, epochs=self.epochs)
        y_hat = fit.predict(x)

        emit.metrics(
            {
                "slope": fit.weights[0],
                "intercept": fit.bias,
                "mse": float(np.mean((y - y_hat) ** 2)),
                "r2": r_squared(y, y_hat),
            },
            title="Fitted parameters",
        )
        grid = np.linspace(-3.0, 3.0, 50)
        emit.chart(
            emit.scatter("data", x[:, 0], y),
            emit.line("fit", grid, fit.predict(grid[:, None])),
            title="Data and fitted line",
        )
        if fit.losses:
            emit.chart(
                emit.line("mse", np.arange(len(fit.losses)), fit.losses),
                title="Training loss",
                x_label="epoch",
                y_label="MSE",
            )


if __name__ == "__main__":
    LinearRegression.main()
