"""Solve the SVM dual as a quadratic programme by guessing its active set.

Each point is guessed to be at α = 0, free (0 < α < C), or at α = C. With the guess fixed, the KKT conditions are
linear: y_i f(x_i) = 1 for every free point and Σ α_i y_i = 0. Solving them gives α and b; the guess is right when the
solution also satisfies the inequalities it did not impose.
"""

import itertools
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

from mlc.examples.svm_solvers.data import Point, dot, gram, weights


class Status(StrEnum):
    zero = "0"
    free = "free"
    bound = "C"


@dataclass(frozen=True)
class Guess:
    status: tuple[Status, ...]
    alpha: list[Fraction]
    b: Fraction
    margins: list[Fraction]
    violations: list[str]


def solve_linear(a: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction] | None:
    """Gauss–Jordan elimination in exact arithmetic; None if the matrix is singular."""
    n = len(a)
    m = [[*row, r] for row, r in zip(a, rhs, strict=True)]
    for c in range(n):
        pivot = next((r for r in range(c, n) if m[r][c] != 0), None)
        if pivot is None:
            return None
        m[c], m[pivot] = m[pivot], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [u - f * v for u, v in zip(m[r], m[c], strict=True)]
    return [m[i][n] / m[i][i] for i in range(n)]


def try_guess(x: list[Point], y: list[int], c: Fraction, status: tuple[Status, ...]) -> Guess | None:
    """Solve the KKT equations for one guess and list the inequalities it breaks. None if b is not determined."""
    n = len(x)
    k = gram(x)
    free = [i for i in range(n) if status[i] is Status.free]
    if not free:
        return None
    fixed = [c if s is Status.bound else Fraction(0) for s in status]
    # Unknowns: α_free, then b. Rows: y_i f(x_i) = 1 for each free i, then Σ α_i y_i = 0.
    rows: list[list[Fraction]] = []
    rhs: list[Fraction] = []
    for i in free:
        rows.append([Fraction(y[i] * y[j]) * k[i][j] for j in free] + [Fraction(y[i])])
        rhs.append(1 - sum((fixed[j] * y[i] * y[j] * k[i][j] for j in range(n) if j not in free), Fraction(0)))
    rows.append([Fraction(y[j]) for j in free] + [Fraction(0)])
    rhs.append(-sum((fixed[j] * y[j] for j in range(n) if j not in free), Fraction(0)))
    sol = solve_linear(rows, rhs)
    if sol is None:
        return None
    alpha = fixed[:]
    for i, v in zip(free, sol, strict=False):
        alpha[i] = v
    b = sol[-1]
    w = weights(x, y, alpha)
    margins = [t * (dot(w, p) + b) for p, t in zip(x, y, strict=True)]
    violations: list[str] = []
    for i, s in enumerate(status):
        if s is Status.zero and margins[i] < 1:
            violations.append(f"x{i + 1}: α = 0 needs y f ≥ 1, got {margins[i]}")
        if s is Status.bound and margins[i] > 1:
            violations.append(f"x{i + 1}: α = C needs y f ≤ 1, got {margins[i]}")
        if s is Status.free and not 0 < alpha[i] < c:
            violations.append(f"x{i + 1}: free α must lie in (0, C), got {alpha[i]}")
    return Guess(status, alpha, b, margins, violations)


def enumerate_active_sets(x: list[Point], y: list[int], c: Fraction) -> tuple[Guess, int]:
    """Try all 3ⁿ guesses in order and return the first that satisfies every KKT condition, with the count tried."""
    for count, status in enumerate(itertools.product(list(Status), repeat=len(x)), start=1):
        guess = try_guess(x, y, c, status)
        if guess is not None and not guess.violations:
            return guess, count
    raise ValueError("no active set satisfies the KKT conditions; the solution has no free support vector")
