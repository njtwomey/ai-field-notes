"""Fit ordinal thresholds with and without the ordering constraint θ_1 ≤ … ≤ θ_{K−1}, and report the multipliers."""

from enum import StrEnum

import torch
from pydantic import Field

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.ordinal_thresholds.data import synthetic, worked_example
from mlc.examples.ordinal_thresholds.fit import Method, Problem, Result, fit
from mlc.examples.ordinal_thresholds.model import class_probabilities, contradictory, predict
from mlc.examples.ordinal_thresholds.ordering import constraints

torch.set_default_dtype(torch.float64)


class Data(StrEnum):
    worked_example = "worked-example"
    synthetic = "synthetic"


class OrdinalThresholds(Command):
    """Fit K − 1 thresholds on one score by four methods and compare order, class probabilities and multipliers."""

    data: Data = Field(Data.worked_example, description="The note's worked example (w fixed) or a 2-D synthetic set.")
    class2: int = Field(1, ge=0, description="Training examples in class 2, the class that shrinks.")
    others: int = Field(30, ge=1, description="Training examples in each other class (synthetic data only).")
    noise: float = Field(0.6, gt=0, description="Noise standard deviation (synthetic data only).")
    seed: int = Field(0, description="Seed for the synthetic data.")
    steps: int = Field(4000, ge=1, description="Gradient steps per method.")
    lr: float = Field(0.2, gt=0, description="Step size.")
    rho: float = Field(1.0, gt=0, description="Penalty weight of the augmented Lagrangian.")
    rounds: int = Field(40, ge=1, description="Multiplier updates of the augmented Lagrangian, spread over the steps.")
    l2: float = Field(1.0, ge=0, description="L2 penalty on w when w is learnt (synthetic data only).")

    def cli_cmd(self) -> None:
        if self.data is Data.worked_example:
            x, y = worked_example(self.class2)
            problem = Problem(x, y, learn_w=False, l2=0.0)
            w0 = torch.ones(1)
            test_x, test_y = x, y
        else:
            counts = [self.others, self.class2, self.others, self.others]
            x, y = synthetic(counts, self.noise, self.seed)
            test_x, test_y = synthetic([200] * 4, self.noise, self.seed + 1)
            problem = Problem(x, y, learn_w=True, l2=self.l2)
            w0 = torch.tensor([1.0, 0.0])
        k = 4
        s0 = x @ w0
        theta0 = torch.linspace(float(s0.min()), float(s0.max()), k + 1)[1:-1]

        results = [
            fit(m, problem, theta0, w0, steps=self.steps, lr=self.lr, rho=self.rho, rounds=self.rounds) for m in Method
        ]
        report(results, problem, test_x, test_y, k)


def report(results: list[Result], problem: Problem, test_x: torch.Tensor, test_y: torch.Tensor, k: int) -> None:
    names = [f"θ{i + 1}" for i in range(k - 1)]
    rows: list[list[str | float | None]] = []
    for r in results:
        s = test_x @ r.w
        # The worst class probability over the test scores and a grid around them: negative means an invalid model.
        grid = torch.linspace(float(s.min()) - 1, float(s.max()) + 1, 401)
        worst = float(class_probabilities(torch.cat([s, grid]), r.theta).min())
        has_mu = r.method in (Method.projected, Method.lagrangian)
        rows.append(
            [
                r.method.value,
                *[float(t) for t in r.theta],
                "yes" if bool((constraints(r.theta) <= 1e-9).all()) else "no",
                worst,
                float(problem.loss(r.w, r.theta)),
                *([float(m) for m in r.mu] if has_mu else [None] * (k - 2)),
                float(contradictory(s, r.theta).double().mean()),
                float((predict(s, r.theta) - test_y).abs().double().mean()),
            ]
        )
    emit.table(
        [
            "method",
            *names,
            "ordered",
            "min P(y = k)",
            "loss",
            *[f"μ{i + 1}" for i in range(k - 2)],
            "contradictory",
            "MAE",
        ],
        rows,
        title="Thresholds, class probabilities and multipliers by method",
    )

    lagrangian = next(r for r in results if r.method is Method.lagrangian)
    slack = constraints(lagrangian.theta)
    emit.metrics(
        {
            **{f"μ{i + 1}": float(m) for i, m in enumerate(lagrangian.mu)},
            "active set": ", ".join(f"θ{i + 1} = θ{i + 2}" for i, g in enumerate(slack) if g > -1e-4) or "none",
            **{
                f"μ{i + 1}(θ{i + 1} − θ{i + 2})": float(m * g)
                for i, (m, g) in enumerate(zip(lagrangian.mu, slack, strict=True))
            },
        },
        title="Augmented Lagrangian: multipliers and complementary slackness",
    )

    for r in results:
        emit.chart(
            *[emit.line(n, r.path_steps, [p[i] for p in r.path]) for i, n in enumerate(names)],
            title=f"Threshold paths: {r.method.value}",
            x_label="step",
            y_label="θ",
        )
    rounds = list(range(1, len(lagrangian.mu_path) + 1))
    emit.chart(
        *[emit.line(f"μ{i + 1}", rounds, [m[i] for m in lagrangian.mu_path]) for i in range(k - 2)],
        title="Multipliers over the augmented Lagrangian rounds",
        x_label="round",
        y_label="μ",
    )


if __name__ == "__main__":
    OrdinalThresholds.main()
