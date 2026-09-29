"""Sequential minimal optimisation (Platt 1998) for the linear soft-margin SVM, recording every step.

The working pair is the maximal violating pair: i minimises the error E over points whose α can move to raise y f,
j maximises E over points whose α can move the other way. Ties go to the lower index.
"""

from dataclasses import dataclass
from fractions import Fraction

from mlc.examples.svm_solvers.data import Point, dual_objective, gram


@dataclass(frozen=True)
class Step:
    i: int
    j: int
    errors: list[Fraction]
    gap: Fraction
    eta: Fraction
    aj_unclipped: Fraction
    low: Fraction
    high: Fraction
    alpha: list[Fraction]
    b: Fraction
    objective: Fraction


def smo(x: list[Point], y: list[int], c: Fraction, max_steps: int = 100) -> tuple[list[Fraction], Fraction, list[Step]]:
    n = len(x)
    k = gram(x)
    alpha = [Fraction(0)] * n
    b = Fraction(0)
    steps: list[Step] = []
    for _ in range(max_steps):
        # E_t = f(x_t) − y_t with f(x) = Σ_s α_s y_s K(x_s, x) + b.
        errors = [sum((alpha[s] * y[s] * k[s][t] for s in range(n)), b) - y[t] for t in range(n)]
        up = [t for t in range(n) if (alpha[t] < c if y[t] > 0 else alpha[t] > 0)]
        down = [t for t in range(n) if (alpha[t] > 0 if y[t] > 0 else alpha[t] < c)]
        i = min(up, key=lambda t: (errors[t], t))
        j = max(down, key=lambda t: (errors[t], -t))
        gap = errors[j] - errors[i]
        if gap <= 0:  # KKT holds exactly: no pair can raise the dual objective.
            break
        eta = k[i][i] + k[j][j] - 2 * k[i][j]
        aj_unclipped = alpha[j] + y[j] * (errors[i] - errors[j]) / eta
        if y[i] != y[j]:
            low, high = max(Fraction(0), alpha[j] - alpha[i]), min(c, c + alpha[j] - alpha[i])
        else:
            low, high = max(Fraction(0), alpha[i] + alpha[j] - c), min(c, alpha[i] + alpha[j])
        aj = min(max(aj_unclipped, low), high)
        ai = alpha[i] + y[i] * y[j] * (alpha[j] - aj)
        di, dj = ai - alpha[i], aj - alpha[j]
        # Platt's bias update: the value that makes f exact on a free multiplier of the pair.
        b1 = b - errors[i] - y[i] * di * k[i][i] - y[j] * dj * k[i][j]
        b2 = b - errors[j] - y[i] * di * k[i][j] - y[j] * dj * k[j][j]
        b = b1 if 0 < ai < c else b2 if 0 < aj < c else (b1 + b2) / 2
        alpha[i], alpha[j] = ai, aj
        steps.append(Step(i, j, errors, gap, eta, aj_unclipped, low, high, alpha[:], b, dual_objective(k, y, alpha)))
    return alpha, b, steps
