"""Gradient descent on linear and logistic regression: step size and momentum."""

import numpy as np
from pydantic import Field
from pydantic_settings import CliApp, CliSubCommand

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.gradient_descent.model import (
    cross_entropy,
    cross_entropy_optimum,
    descend,
    excess,
    loss_grid,
    make_regression,
    make_valley,
    mse,
    mse_hessian,
    mse_optimum,
)


class StepSizes(Command):
    """Run gradient descent on linear regression with step sizes set as fractions of the stability limit 2/λ_max."""

    fractions: list[float] = Field(
        default_factory=lambda: [0.1, 0.45, 0.95, 1.02],
        description="Step sizes as fractions of 2/λ_max. Above 1 the iteration diverges.",
    )
    steps: int = Field(40, ge=1, description="Iterations per run.")
    seed: int = Field(0, description="Seed for the data generator.")

    def cli_cmd(self) -> None:
        x, y = make_regression(seed=self.seed)
        eigs = np.linalg.eigvalsh(mse_hessian(x))
        limit = 2.0 / eigs.max()
        emit.metrics(
            {"λ_min": eigs.min(), "λ_max": eigs.max(), "condition number": eigs.max() / eigs.min(), "2/λ_max": limit},
            title="Curvature of the MSE",
        )
        f = mse(x, y)
        best = f(mse_optimum(x, y))[0]
        start = np.array([-1.0, 3.0])
        runs = [(frac, descend(f, start, lr=frac * limit, steps=self.steps)) for frac in self.fractions]
        emit.table(
            ["fraction of 2/λ_max", "η", "final loss", "status"],
            [
                [
                    frac,
                    frac * limit,
                    t.losses[-1],
                    "diverged" if t.diverged or t.losses[-1] > t.losses[0] else "converged",
                ]
                for frac, t in runs
            ],
            title="Final loss by step size",
        )
        emit.chart(
            *(
                emit.line(f"η = {frac:g}·2/λ_max", np.arange(len(t.losses)), np.log10(excess(t.losses, best)))
                for frac, t in runs
            ),
            title="Excess loss per iteration. A straight line is linear convergence; its slope is the rate.",
            x_label="iteration",
            y_label="log₁₀ (MSE − MSE*)",
        )
        w = np.linspace(-1.5, 4.5, 61)
        b = np.linspace(-5.0, 4.0, 61)
        emit.heatmap(
            w,
            b,
            np.log10(loss_grid(f, w, b)),
            title="Trajectories on the loss surface (log₁₀ MSE)",
            x_label="w",
            y_label="b",
            overlay=[emit.line(f"{frac:g}·2/λ_max", t.thetas[:, 0], t.thetas[:, 1]) for frac, t in runs[:3]],
        )


class Momentum(Command):
    """Compare plain gradient descent with heavy-ball momentum on an ill-conditioned logistic regression."""

    lr: float = Field(0.1, gt=0.0, description="Step size η, shared by both optimisers.")
    beta: float = Field(0.9, ge=0.0, lt=1.0, description="Momentum coefficient β.")
    steps: int = Field(300, ge=1, description="Iterations per run.")
    seed: int = Field(0, description="Seed for the data generator.")

    def cli_cmd(self) -> None:
        x, y = make_valley(seed=self.seed)
        f = cross_entropy(x, y)
        best = f(cross_entropy_optimum(x, y))[0]
        start = np.array([4.0, 0.0])
        plain = descend(f, start, lr=self.lr, steps=self.steps)
        heavy = descend(f, start, lr=self.lr, steps=self.steps, momentum=self.beta)
        emit.metrics(
            {
                "optimal loss": best,
                "plain excess loss": plain.losses[-1] - best,
                "momentum excess loss": heavy.losses[-1] - best,
            },
            title=f"After {self.steps} steps",
        )
        emit.chart(
            emit.line("plain", np.arange(len(plain.losses)), np.log10(excess(plain.losses, best))),
            emit.line(
                f"momentum β = {self.beta:g}", np.arange(len(heavy.losses)), np.log10(excess(heavy.losses, best))
            ),
            title="Excess loss per iteration",
            x_label="iteration",
            y_label="log₁₀ (L − L*)",
        )
        w = np.linspace(-1.0, 5.0, 61)
        b = np.linspace(-20.0, 4.0, 61)
        emit.heatmap(
            w,
            b,
            np.log10(loss_grid(f, w, b)),
            title="Trajectories on the loss surface (log₁₀ cross-entropy)",
            x_label="w",
            y_label="b",
            overlay=[
                emit.line("plain", plain.thetas[:, 0], plain.thetas[:, 1]),
                emit.line("momentum", heavy.thetas[:, 0], heavy.thetas[:, 1]),
            ],
        )


class GradientDescent(Command):
    """Gradient descent experiments."""

    step_sizes: CliSubCommand[StepSizes] = Field(description=StepSizes.__doc__)
    momentum: CliSubCommand[Momentum] = Field(description=Momentum.__doc__)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


if __name__ == "__main__":
    GradientDescent.main()
