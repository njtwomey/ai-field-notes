"""Train logistic regression on two Gaussian classes, compare three solvers and plot the decision surface."""

from typing import Literal

import numpy as np
from pydantic import Field

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.logistic_regression.model import Fit, gradient_descent, make_blobs, newton, sgd

type Solver = Literal["gradient-descent", "newton", "sgd"]


class LogisticRegression(Command):
    """Fit binary logistic regression by gradient descent, Newton's method (IRLS) and SGD; report the chosen fit."""

    n: int = Field(200, ge=2, description="Number of samples, split randomly between the two classes.")
    separation: float = Field(3.0, ge=0.0, description="Distance between the two class means.")
    l2: float = Field(0.0, ge=0.0, description="L2 penalty strength on the weights (not the bias). 0 disables it.")
    solver: Solver = Field("newton", description="Solver whose fit is reported and plotted.")
    epochs: int = Field(30, ge=1, description="Passes over the data for every solver.")
    lr: float = Field(1.0, gt=0.0, description="Gradient-descent step size, and SGD's initial step size.")
    batch: int = Field(10, ge=1, description="SGD mini-batch size.")
    seed: int = Field(0, description="Seed for the data generator and the SGD shuffles.")

    def cli_cmd(self) -> None:
        x, y = make_blobs(self.n, separation=self.separation, seed=self.seed)
        fits: dict[Solver, Fit] = {
            "gradient-descent": gradient_descent(x, y, lr=self.lr, epochs=self.epochs, l2=self.l2),
            "newton": newton(x, y, iterations=self.epochs, l2=self.l2),
            "sgd": sgd(x, y, lr=self.lr, epochs=self.epochs, batch=self.batch, l2=self.l2, seed=self.seed),
        }
        model = fits[self.solver]
        accuracy = float(np.mean((model.predict_proba(x) >= 0.5) == (y == 1.0)))

        emit.metrics(
            {
                "solver": self.solver,
                "w1": model.weights[0],
                "w2": model.weights[1],
                "bias": model.bias,
                "loss": model.losses[-1],
                "accuracy": accuracy,
            },
            title="Fitted parameters",
        )

        # Each solver's gap to the best loss any of them reached, on a log scale. Every solver makes one pass over
        # the data per point: one gradient step, one Newton step, or n / batch SGD steps.
        best = min(min(f.losses) for f in fits.values())
        passes = np.arange(self.epochs + 1)
        gaps = {name: np.log10(np.maximum(np.array(f.losses) - best, 1e-16)) for name, f in fits.items()}
        emit.chart(
            *(emit.line(name, passes, gap) for name, gap in gaps.items()),
            title="Convergence: log₁₀(loss − best loss)",
            x_label="passes over the data",
            y_label="log₁₀ gap",
        )
        checkpoints = [k for k in (1, 2, 5, 10) if k < self.epochs] + [self.epochs]
        emit.table(
            ["solver", *(f"after {k}" for k in checkpoints)],
            [[name, *(f"{f.losses[k] - best:.2e}" for k in checkpoints)] for name, f in fits.items()],
            title="Loss above the best, after k passes",
        )

        grid = np.linspace(-4.0, 4.0, 41)
        gx, gy = np.meshgrid(grid, grid)
        proba = model.predict_proba(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        emit.heatmap(
            grid,
            grid,
            proba,
            title="P(y = 1 | x) with training data",
            x_label="x₁",
            y_label="x₂",
            overlay=[emit.scatter("data", x[:, 0], x[:, 1], group=y.astype(int))],
        )


if __name__ == "__main__":
    LogisticRegression.main()
