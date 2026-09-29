"""The six-point dataset of the worked example, in exact rational arithmetic."""

from fractions import Fraction

type Point = tuple[Fraction, Fraction]

# Three points of class −1, then three of class +1.
X: list[Point] = [(Fraction(a), Fraction(b)) for a, b in [(1, 1), (2, 1), (1, 2), (2, 2), (4, 3), (4, 1)]]
Y: list[int] = [-1, -1, -1, 1, 1, 1]


def dot(a: Point, b: Point) -> Fraction:
    return a[0] * b[0] + a[1] * b[1]


def gram(x: list[Point]) -> list[list[Fraction]]:
    """Linear kernel K_ij = x_iᵀx_j."""
    return [[dot(a, b) for b in x] for a in x]


def weights(x: list[Point], y: list[int], alpha: list[Fraction]) -> Point:
    """w = Σ α_i y_i x_i."""
    return (
        sum((a * t * p[0] for a, t, p in zip(alpha, y, x, strict=True)), Fraction(0)),
        sum((a * t * p[1] for a, t, p in zip(alpha, y, x, strict=True)), Fraction(0)),
    )


def dual_objective(k: list[list[Fraction]], y: list[int], alpha: list[Fraction]) -> Fraction:
    """Σ α_i − ½ Σ_ij α_i α_j y_i y_j K_ij, to be maximised."""
    n = len(alpha)
    quad = sum((alpha[i] * alpha[j] * y[i] * y[j] * k[i][j] for i in range(n) for j in range(n)), Fraction(0))
    return sum(alpha, Fraction(0)) - quad / 2


def slacks(x: list[Point], y: list[int], w: Point, b: Fraction) -> list[Fraction]:
    """The smallest slack each point needs: ξ_i = max(0, 1 − y_i f(x_i))."""
    return [max(Fraction(0), 1 - t * (dot(w, p) + b)) for p, t in zip(x, y, strict=True)]
