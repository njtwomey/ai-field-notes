"""Train logistic regression on two Gaussian classes and plot the decision surface."""

import numpy as np
from pydantic import Field

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.logistic_regression.model import fit, make_blobs


class LogisticRegression(Command):
    """Train binary logistic regression by gradient descent and report P(y = 1 | x) over the plane."""

    n: int = Field(200, ge=2, description="Number of samples, split randomly between the two classes.")
    separation: float = Field(3.0, ge=0.0, description="Distance between the two class means.")
    l2: float = Field(0.0, ge=0.0, description="L2 penalty strength on the weights. 0 disables it.")
    lr: float = Field(0.5, gt=0.0, description="Gradient-descent step size.")
    epochs: int = Field(300, ge=1, description="Gradient-descent iterations.")
    seed: int = Field(0, description="Seed for the data generator.")

    def cli_cmd(self) -> None:
        x, y = make_blobs(self.n, separation=self.separation, seed=self.seed)
        model = fit(x, y, lr=self.lr, epochs=self.epochs, l2=self.l2)
        accuracy = float(np.mean((model.predict_proba(x) >= 0.5) == (y == 1.0)))

        emit.metrics(
            {
                "w1": model.weights[0],
                "w2": model.weights[1],
                "bias": model.bias,
                "loss": model.losses[-1],
                "accuracy": accuracy,
            },
            title="Fitted parameters",
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
        emit.chart(
            emit.line("loss", np.arange(self.epochs), model.losses),
            title="Training loss",
            x_label="epoch",
            y_label="loss",
        )


if __name__ == "__main__":
    LogisticRegression.main()
