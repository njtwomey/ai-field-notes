from typing import Literal

import numpy as np
from pydantic import Field
from sklearn.linear_model import enet_path, lars_path, lasso_path

from mlc.core.contracts import Strict
from mlc.core.figures import figure
from mlc.examples.linear_models.model import (
    Penalty,
    diabetes,
    exact,
    objective,
    sgd_trajectory,
    subgradient_lasso_trajectory,
)


def _r(values: np.ndarray, digits: int = 4) -> list[float]:
    return np.asarray(values, dtype=float).round(digits).tolist()


def _rows(values: np.ndarray, digits: int = 4) -> list[list[float]]:
    return np.asarray(values, dtype=float).round(digits).tolist()


class CoefficientPath(Strict):
    """Coefficients of a regularised fit along a grid of penalty values, one row per feature."""

    features: list[str]
    penalty: list[float] = Field(description="Penalty values, decreasing: from all-zero (or near) to least penalised.")
    coef: list[list[float]] = Field(description="Indexed [feature][penalty index].")


class RidgePath(Strict):
    """Ridge coefficients against λ on the diabetes data, with effective degrees of freedom and the OLS solution."""

    path: CoefficientPath
    dof: list[float] = Field(description="Effective degrees of freedom Σ dᵢ²/(dᵢ² + λ) at each λ.")
    ols: list[float]


class LassoPath(Strict):
    """Lasso coefficients against α on the diabetes data, with the OLS solution."""

    path: CoefficientPath
    nonzero: list[int] = Field(description="Number of nonzero coefficients at each α.")
    ols: list[float]


class ElasticNetPaths(Strict):
    """Elastic-net paths for several mixing ratios; ratio 1 is the lasso."""

    l1_ratios: list[float]
    paths: list[CoefficientPath]


class PathEvent(Strict):
    knot: int = Field(description="Index of the knot at which the event happens.")
    alpha: float = Field(description="Penalty at that knot, in scikit-learn's lasso scaling.")
    feature: int
    kind: Literal["enter", "drop"]


class KnotPath(Strict):
    """A piecewise-linear path given by its knots; coefficients are linear in ‖w‖₁ between consecutive knots."""

    l1_norm: list[float]
    alphas: list[float]
    coef: list[list[float]] = Field(description="Indexed [feature][knot].")
    events: list[PathEvent] = Field(description="Features entering or leaving the active set, in order.")


class LarsPaths(Strict):
    """LARS and the LARS–lasso modification on the diabetes data (Efron et al. 2004)."""

    features: list[str]
    lar: KnotPath
    lasso: KnotPath


class SgdRun(Strict):
    penalty: Penalty
    label: str
    coef: list[list[float]] = Field(description="Indexed [feature][epoch]; epoch 0 is the zero start.")
    exact: list[float] = Field(description="The exact minimiser of the same objective.")
    gap: list[float] = Field(description="Objective minus the exact optimum at each epoch.")
    zeros: list[int] = Field(description="Number of coefficients exactly zero at each epoch.")
    exact_zeros: int


class SgdTrajectories(Strict):
    """SGDRegressor weights per epoch for several penalties, against the exact solutions."""

    features: list[str]
    alpha: float
    l1_ratio: float
    runs: list[SgdRun]


def _events(coef: np.ndarray, alphas: np.ndarray) -> list[PathEvent]:
    """A feature joins the active set at the knot where its coefficient starts to move away from zero, and leaves it
    at the knot where its coefficient returns to zero."""
    events: list[PathEvent] = []
    for k in range(1, coef.shape[1]):
        before, now = np.abs(coef[:, k - 1]) > 1e-10, np.abs(coef[:, k]) > 1e-10
        joined = np.flatnonzero(now & ~before)
        left = np.flatnonzero(before & ~now)
        events += [PathEvent(knot=k - 1, alpha=float(alphas[k - 1]), feature=int(j), kind="enter") for j in joined]
        events += [PathEvent(knot=k, alpha=float(alphas[k]), feature=int(j), kind="drop") for j in left]
    return sorted(events, key=lambda e: (e.knot, e.kind == "enter"))


@figure("ridge-regression/path", title="Ridge coefficients against λ on the diabetes data")
def ridge_path() -> RidgePath:
    data = diabetes()
    lambdas = np.logspace(5, -2, 71)
    # Closed form through the SVD: w(λ) = V diag(dᵢ / (dᵢ² + λ)) Uᵀy.
    u, d, vt = np.linalg.svd(data.x, full_matrices=False)
    uty = u.T @ data.y
    coef = np.stack([vt.T @ (d / (d**2 + lam) * uty) for lam in lambdas], axis=1)
    dof = [(d**2 / (d**2 + lam)).sum() for lam in lambdas]
    ols = vt.T @ (uty / d)
    return RidgePath(
        path=CoefficientPath(features=data.features, penalty=_r(lambdas, 6), coef=_rows(coef)),
        dof=_r(np.array(dof)),
        ols=_r(ols),
    )


@figure("lasso/path", title="Lasso coefficients against α on the diabetes data")
def lasso_coefficient_path() -> LassoPath:
    data = diabetes()
    alpha_max = np.abs(data.x.T @ data.y).max() / len(data.y)
    alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max) - 4, 81)
    _, coef, _ = lasso_path(data.x, data.y, alphas=alphas, tol=1e-10, max_iter=100_000)
    ols = np.linalg.lstsq(data.x, data.y, rcond=None)[0]
    coef = np.where(np.abs(coef) < 1e-10, 0.0, coef)
    return LassoPath(
        path=CoefficientPath(features=data.features, penalty=_r(alphas, 6), coef=_rows(coef)),
        nonzero=[int(np.count_nonzero(coef[:, k])) for k in range(coef.shape[1])],
        ols=_r(ols),
    )


@figure("elastic-net/paths", title="Elastic-net paths on the diabetes data for four mixing ratios")
def elastic_net_paths() -> ElasticNetPaths:
    data = diabetes()
    ratios = [0.1, 0.5, 0.9, 1.0]
    paths: list[CoefficientPath] = []
    for ratio in ratios:
        alpha_max = np.abs(data.x.T @ data.y).max() / (len(data.y) * ratio)
        alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_max) - 4, 81)
        _, coef, _ = enet_path(data.x, data.y, l1_ratio=ratio, alphas=alphas, tol=1e-10, max_iter=100_000)
        coef = np.where(np.abs(coef) < 1e-10, 0.0, coef)
        paths.append(CoefficientPath(features=data.features, penalty=_r(alphas, 6), coef=_rows(coef)))
    return ElasticNetPaths(l1_ratios=ratios, paths=paths)


def _knot_path(method: Literal["lar", "lasso"]) -> KnotPath:
    data = diabetes()
    alphas, _, coef = lars_path(data.x, data.y, method=method)
    coef = np.where(np.abs(coef) < 1e-10, 0.0, coef)
    return KnotPath(
        l1_norm=_r(np.abs(coef).sum(axis=0)),
        alphas=_r(alphas, 6),
        coef=_rows(coef),
        events=_events(coef, alphas),
    )


@figure("least-angle-regression/paths", title="LARS and LARS–lasso paths on the diabetes data")
def lars_paths() -> LarsPaths:
    return LarsPaths(features=diabetes().features, lar=_knot_path("lar"), lasso=_knot_path("lasso"))


@figure("sgd-for-linear-models/trajectories", title="SGDRegressor weights per epoch against the exact solutions")
def sgd_trajectories() -> SgdTrajectories:
    data = diabetes()
    alpha, l1_ratio, epochs = 1.0, 0.5, 60
    runs: list[SgdRun] = []
    settings = [
        (Penalty.l2, "ridge (l2)", "invscaling"),
        (Penalty.l1, "lasso (l1)", "invscaling"),
        (Penalty.elasticnet, "elastic net", "invscaling"),
        (Penalty.l1, "lasso, constant step", "constant"),
        (Penalty.l1, "lasso, plain subgradient", "subgradient"),
    ]
    for penalty, label, schedule in settings:
        if schedule == "subgradient":
            weights, intercepts = subgradient_lasso_trajectory(data, alpha, epochs=epochs)
        else:
            weights, intercepts = sgd_trajectory(data, penalty, alpha, l1_ratio, epochs=epochs, learning_rate=schedule)
        w_star, b_star = exact(data, penalty, alpha, l1_ratio)
        best = objective(data, w_star, b_star, penalty, alpha, l1_ratio)
        gap = [objective(data, w, b, penalty, alpha, l1_ratio) - best for w, b in zip(weights, intercepts, strict=True)]
        runs.append(
            SgdRun(
                penalty=penalty,
                label=label,
                coef=_rows(weights.T),
                exact=_r(w_star),
                gap=np.maximum(np.array(gap), 1e-12).round(10).tolist(),
                zeros=[int((w == 0).sum()) for w in weights],
                exact_zeros=int((w_star == 0).sum()),
            )
        )
    return SgdTrajectories(features=data.features, alpha=alpha, l1_ratio=l1_ratio, runs=runs)
