"""Reference fits for aifn/learning/calibration: isotonic regression from scikit-learn's IsotonicRegression (ties in x
pooled to their weighted mean, increasing and decreasing, with weights) and scipy.optimize.isotonic_regression (on an
ordered sequence)."""

import numpy as np
from scipy.optimize import isotonic_regression
from sklearn.isotonic import IsotonicRegression


def cases() -> dict:
    rng = np.random.default_rng(7)
    out: dict = {"isotonicRegression": [], "poolAdjacentViolatorsSteps": []}
    for n, increasing, weighted, ties in [
        (12, True, False, False),
        (30, True, True, False),
        (25, False, True, False),
        (40, True, True, True),
    ]:
        x = np.sort(rng.uniform(0, 10, n)) if not ties else np.round(rng.uniform(0, 10, n))
        sign = 1 if increasing else -1
        y = np.sin(x / 3) * sign + 0.4 * rng.standard_normal(n) + x / 10 * sign
        w = rng.uniform(0.2, 2.0, n) if weighted else np.ones(n)
        fit = IsotonicRegression(increasing=increasing).fit(x, y, sample_weight=w).predict(x)
        out["isotonicRegression"].append({"x": x, "y": y, "weights": w, "increasing": increasing, "fit": fit})
    for n in (8, 50):
        y = rng.standard_normal(n).cumsum() * 0.3 + rng.standard_normal(n)
        w = rng.uniform(0.5, 1.5, n)
        res = isotonic_regression(y, weights=w, increasing=True)
        out["poolAdjacentViolatorsSteps"].append({"y": y, "weights": w, "fit": res.x, "blocks": res.blocks})
    return out
