"""Compare Student's pooled t-test with Welch's t-test."""

import numpy as np
from pydantic import Field
from pydantic_settings import CliApp, CliSubCommand

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.welch.model import pooled_t, type_i_error, welch_t


class Compare(Command):
    """Run both tests on one pair of samples with unequal variances and sizes."""

    n1: int = Field(40, ge=2, description="Size of group 1.")
    n2: int = Field(10, ge=2, description="Size of group 2.")
    sd1: float = Field(1.0, gt=0.0, description="Standard deviation of group 1.")
    sd2: float = Field(3.0, gt=0.0, description="Standard deviation of group 2.")
    shift: float = Field(0.0, description="True difference in means, group 2 minus group 1.")
    seed: int = Field(0, description="Seed for the data generator.")

    def cli_cmd(self) -> None:
        rng = np.random.default_rng(self.seed)
        x = rng.normal(0.0, self.sd1, self.n1)
        y = rng.normal(self.shift, self.sd2, self.n2)
        pooled, welch = pooled_t(x, y), welch_t(x, y)
        emit.table(
            ["test", "t", "degrees of freedom", "p (two-sided)"],
            [["pooled (Student)", pooled.t, pooled.df, pooled.p], ["Welch", welch.t, welch.df, welch.p]],
            title="Same data, two tests",
        )


class Sweep(Command):
    """Estimate each test's Type I error rate across variance ratios, with a small or large second group."""

    n1: int = Field(20, ge=2, description="Size of group 1.")
    alpha: float = Field(0.05, gt=0.0, lt=1.0, description="Significance level.")
    reps: int = Field(20_000, ge=100, description="Simulated experiments per setting.")
    seed: int = Field(0, description="Seed for the simulations.")

    def cli_cmd(self) -> None:
        log_ratios = np.linspace(-2.0, 2.0, 9)
        rows = []
        curves: dict[str, list[float]] = {}
        for n2 in (10, 40):
            pooled, welch = zip(
                *(
                    type_i_error(self.n1, n2, 2.0**r, alpha=self.alpha, reps=self.reps, seed=self.seed)
                    for r in log_ratios
                ),
                strict=True,
            )
            curves[f"pooled, n₂ = {n2}"] = list(pooled)
            curves[f"Welch, n₂ = {n2}"] = list(welch)
            rows += [[n2, float(2.0**r), p, w] for r, p, w in zip(log_ratios, pooled, welch, strict=True)]
        emit.table(["n₂", "sd₂ / sd₁", "pooled", "Welch"], rows, title=f"Type I error rate, n₁ = {self.n1}")
        emit.chart(
            *(emit.line(name, log_ratios, rates) for name, rates in curves.items()),
            emit.line("nominal α", [-2.0, 2.0], [self.alpha, self.alpha]),
            title="Type I error against log₂(sd₂ / sd₁)",
            x_label="log₂(sd₂ / sd₁)",
            y_label="rejection rate under H₀",
        )


class Welch(Command):
    """Student's pooled t-test against Welch's t-test."""

    compare: CliSubCommand[Compare] = Field(description=Compare.__doc__)
    sweep: CliSubCommand[Sweep] = Field(description=Sweep.__doc__)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


if __name__ == "__main__":
    Welch.main()
