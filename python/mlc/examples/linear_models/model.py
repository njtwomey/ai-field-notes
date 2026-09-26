"""Regularised linear regression on the diabetes data with scikit-learn: exact solvers and stochastic gradient descent.

Objectives, with n samples, standardised features and a centred target:

- ridge: ‖y − Xw‖² + λ‖w‖² (scikit-learn ``Ridge(alpha=λ)``)
- lasso: (1/2n)‖y − Xw‖² + α‖w‖₁ (``Lasso(alpha=α)``)
- elastic net: (1/2n)‖y − Xw‖² + αρ‖w‖₁ + ½α(1 − ρ)‖w‖² (``ElasticNet(alpha=α, l1_ratio=ρ)``)

``SGDRegressor`` minimises (1/n)Σ ½(yᵢ − ŷᵢ)² + α R(w), so its l1 and elastic-net penalties match the lasso and elastic
net above with the same α, and its l2 penalty ½α‖w‖² matches ridge with λ = nα.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.preprocessing import StandardScaler

type Array = NDArray[np.float64]


# Age, sex, body-mass index, mean blood pressure and six blood serum measurements, in the dataset's column order.
FEATURES = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]


class Penalty(StrEnum):
    l2 = "l2"
    l1 = "l1"
    elasticnet = "elasticnet"


@dataclass(frozen=True)
class Data:
    x: Array
    """Standardised features, shape (n, 10)."""
    y: Array
    """Target minus its mean: disease progression one year after baseline."""
    features: list[str]


def diabetes() -> Data:
    """The diabetes data of Efron et al. (2004): 442 patients, 10 baseline measurements."""
    x, y = (np.asarray(a, dtype=float) for a in load_diabetes(return_X_y=True))
    return Data(x=StandardScaler().fit_transform(x), y=y - y.mean(), features=FEATURES)


def penalty_value(w: Array, penalty: Penalty, alpha: float, l1_ratio: float) -> float:
    """α R(w) in SGDRegressor's scaling."""
    match penalty:
        case Penalty.l2:
            return float(0.5 * alpha * w @ w)
        case Penalty.l1:
            return float(alpha * np.abs(w).sum())
        case Penalty.elasticnet:
            return float(alpha * (l1_ratio * np.abs(w).sum() + 0.5 * (1 - l1_ratio) * w @ w))


def objective(data: Data, w: Array, b: float, penalty: Penalty, alpha: float, l1_ratio: float) -> float:
    """(1/n)Σ ½(yᵢ − xᵢᵀw − b)² + α R(w), the objective SGDRegressor minimises."""
    residual = data.y - data.x @ w - b
    return float(0.5 * np.mean(residual**2) + penalty_value(w, penalty, alpha, l1_ratio))


def exact(data: Data, penalty: Penalty, alpha: float, l1_ratio: float) -> tuple[Array, float]:
    """The exact minimiser of the same objective: ridge by a direct solve, the others by coordinate descent."""
    match penalty:
        case Penalty.l2:
            model = Ridge(alpha=len(data.y) * alpha)
        case Penalty.l1:
            model = Lasso(alpha=alpha, tol=1e-10, max_iter=100_000)
        case Penalty.elasticnet:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, tol=1e-10, max_iter=100_000)
    model.fit(data.x, data.y)
    return np.asarray(model.coef_, dtype=float), float(model.intercept_)


def sgd_trajectory(
    data: Data,
    penalty: Penalty,
    alpha: float,
    l1_ratio: float,
    *,
    epochs: int,
    learning_rate: str = "invscaling",
    eta0: float = 0.01,
    seed: int = 0,
) -> tuple[Array, Array]:
    """Weights and intercept after each epoch of SGDRegressor, starting from zero. Row 0 is the starting point."""
    sgd = SGDRegressor(
        penalty=penalty.value,
        alpha=alpha,
        l1_ratio=l1_ratio,
        learning_rate=learning_rate,
        eta0=eta0,
        random_state=seed,
    )
    weights = [np.zeros(data.x.shape[1])]
    intercepts = [0.0]
    for _ in range(epochs):
        sgd.partial_fit(data.x, data.y)
        weights.append(np.asarray(sgd.coef_, dtype=float).copy())
        intercepts.append(float(np.asarray(sgd.intercept_).ravel()[0]))
    return np.array(weights), np.array(intercepts)


def subgradient_lasso_trajectory(
    data: Data, alpha: float, *, epochs: int, eta0: float = 0.01, power_t: float = 0.25, seed: int = 0
) -> tuple[Array, Array]:
    """Plain stochastic subgradient descent on the lasso objective, with SGDRegressor's invscaling schedule.

    Each step moves w by η(rᵢxᵢ − α sign(w)). Unlike SGDRegressor's truncated cumulative penalty, nothing ever sets a
    weight to exactly zero, so weights that should vanish jitter around it instead.
    """
    rng = np.random.default_rng(seed)
    n, d = data.x.shape
    w, b, t = np.zeros(d), 0.0, 1
    weights, intercepts = [w.copy()], [b]
    for _ in range(epochs):
        for i in rng.permutation(n):
            eta = eta0 / t**power_t
            r = data.y[i] - data.x[i] @ w - b
            w += eta * (r * data.x[i] - alpha * np.sign(w))
            b += eta * r
            t += 1
        weights.append(w.copy())
        intercepts.append(b)
    return np.array(weights), np.array(intercepts)
