"""Solve a six-point soft-margin SVM exactly, as a quadratic programme and by SMO, and report every slack."""

from fractions import Fraction

from pydantic import Field
from pydantic_settings import CliApp, CliSubCommand

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.svm_solvers.active_set import Status, enumerate_active_sets, try_guess
from mlc.examples.svm_solvers.data import Point, X, Y, dot, slacks, weights
from mlc.examples.svm_solvers.smo import smo


def position(margin: Fraction) -> str:
    if margin > 1:
        return "beyond the margin"
    if margin == 1:
        return "on the margin"
    return "inside the margin" if margin > 0 else "misclassified"


def report(x: list[Point], y: list[int], alpha: list[Fraction], b: Fraction) -> None:
    w = weights(x, y, alpha)
    xi = slacks(x, y, w, b)
    emit.metrics({"w": f"({w[0]}, {w[1]})", "b": str(b), "‖w‖²": str(dot(w, w))}, title="Solution")
    emit.table(
        ["point", "x", "y", "α", "f(x)", "y f(x)", "ξ", "position"],
        [
            [
                f"x{i + 1}",
                f"({p[0]}, {p[1]})",
                t,
                str(a),
                str(dot(w, p) + b),
                str(t * (dot(w, p) + b)),
                str(s),
                position(t * (dot(w, p) + b)),
            ]
            for i, (p, t, a, s) in enumerate(zip(x, y, alpha, xi, strict=True))
        ],
        title="Multipliers and slacks",
    )


class Options(Command):
    """Options shared by both solvers."""

    c: float = Field(0.5, gt=0, description="Price C of one unit of slack. Read as an exact fraction.")

    @property
    def cost(self) -> Fraction:
        return Fraction(self.c).limit_denominator(1000)


class Qp(Options):
    """Solve the dual QP by trying active sets, then recover w, b and every slack."""

    def cli_cmd(self) -> None:
        wrong = try_guess(
            X, Y, self.cost, (Status.zero, Status.free, Status.free, Status.bound, Status.zero, Status.free)
        )
        if wrong is not None:
            emit.text("\n".join(wrong.violations) or "no violations", title="A wrong guess: x2 free instead of at C")
        guess, tried = enumerate_active_sets(X, Y, self.cost)
        emit.metrics(
            {"active set": " ".join(f"x{i + 1}:{s}" for i, s in enumerate(guess.status)), "guesses tried": tried},
            title="Active set",
        )
        report(X, Y, guess.alpha, guess.b)


class Smo(Options):
    """Run SMO from α = 0 in exact arithmetic and report every step."""

    def cli_cmd(self) -> None:
        alpha, b, steps = smo(X, Y, self.cost)
        emit.table(
            ["step", "pair (i, j)", "E before", "E_j − E_i", "η", "α_j unclipped", "[L, H]", "α after", "b", "dual"],
            [
                [
                    n,
                    f"(x{s.i + 1}, x{s.j + 1})",
                    ", ".join(map(str, s.errors)),
                    str(s.gap),
                    str(s.eta),
                    str(s.aj_unclipped),
                    f"[{s.low}, {s.high}]",
                    ", ".join(map(str, s.alpha)),
                    str(s.b),
                    str(s.objective),
                ]
                for n, s in enumerate(steps, start=1)
            ],
            title=f"SMO steps ({len(steps)} to an exact optimum)",
        )
        report(X, Y, alpha, b)


class SvmSolvers(Command):
    """The worked example of solving a soft-margin SVM two ways."""

    qp: CliSubCommand[Qp] = Field(description="Quadratic programming by active-set guessing.")
    smo: CliSubCommand[Smo] = Field(description="Sequential minimal optimisation, step by step.")

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


if __name__ == "__main__":
    SvmSolvers.main()
