"""Golden values for aifn/gam: a penalised additive fit solved directly in numpy (scipy B-spline designs with the
sum-to-zero constraint imposed through the KKT system), its EDF, and the GCV and REML curves over λ for one smooth."""

import numpy as np
from scipy import interpolate


def design(x, lo, hi, k, degree=3):
    seg = k - degree
    h = (hi - lo) / seg
    knots = lo + np.arange(-degree, seg + degree + 1) * h
    return interpolate.BSpline.design_matrix(np.clip(x, lo, hi), knots, degree).toarray()


def fit(blocks, y, lambdas):
    """min ‖y − α − Σ Bⱼβⱼ‖² + Σ λⱼ‖D βⱼ‖² subject to 1ᵀBⱼβⱼ = 0, by the KKT system."""
    n = len(y)
    X = np.column_stack([np.ones(n), *blocks])
    p = X.shape[1]
    S = np.zeros((p, p))
    C = []
    o = 1
    for B, lam in zip(blocks, lambdas, strict=True):
        q = B.shape[1]
        D = np.diff(np.eye(q), 2, axis=0)
        S[o : o + q, o : o + q] = lam * D.T @ D
        c = np.zeros(p)
        c[o : o + q] = B.sum(axis=0)
        C.append(c)
        o += q
    C = np.array(C)
    m = len(C)
    K = np.block([[X.T @ X + S, C.T], [C, np.zeros((m, m))]])
    sol = np.linalg.solve(K, np.r_[X.T @ y, np.zeros(m)])
    beta = sol[:p]
    fitted = X @ beta
    # EDF = tr of the constrained hat matrix: columns of the KKT inverse's top-left block.
    Kinv = np.linalg.inv(K)[:p, :p]
    edf = np.trace(Kinv @ X.T @ X)
    return fitted, edf, S, X, C


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    n = 120
    x = rng.uniform(0, 1, size=(n, 2))
    y = np.sin(2 * np.pi * x[:, 0]) + (x[:, 1] - 0.5) ** 2 * 4 + 0.3 * rng.normal(size=n)
    k = 10
    blocks = [design(x[:, j], x[:, j].min(), x[:, j].max(), k) for j in range(2)]
    fitted, edf, *_ = fit(blocks, y, [0.5, 3.0])

    # GCV over λ for a single smooth of x0.
    grid = np.linspace(-4, 4, 161)
    gcv = []
    for log_lam in grid:
        f, e, *_ = fit(blocks[:1], y, [np.exp(log_lam)])
        rss = np.sum((y - f) ** 2)
        gcv.append(n * rss / (n - e) ** 2)
    return {
        "x": x,
        "y": y,
        "k": k,
        "lambdas": [0.5, 3.0],
        "fitted": fitted,
        "edf": edf,
        "gcv_grid": grid,
        "gcv": gcv,
        "gcv_best": grid[int(np.argmin(gcv))],
    }
