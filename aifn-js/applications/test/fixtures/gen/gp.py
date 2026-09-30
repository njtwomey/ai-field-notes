"""Golden values for aifn/kernels and aifn/gp, from scikit-learn's kernels, GaussianProcessRegressor and
GaussianProcessClassifier (Laplace)."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    x = rng.uniform(-2, 2, size=(7, 2))
    y2 = rng.uniform(-2, 2, size=(4, 2))
    ard = np.array([0.7, 1.9])
    kernels = {
        "rbf": (ConstantKernel(1.7) * RBF(0.8), {"lengthscale": 0.8, "variance": 1.7}),
        "rbf_ard": (ConstantKernel(1.3) * RBF(ard), {"lengthscale": ard, "variance": 1.3}),
        "matern12": (ConstantKernel(0.9) * Matern(0.6, nu=0.5), {"lengthscale": 0.6, "variance": 0.9}),
        "matern32": (ConstantKernel(1.1) * Matern(1.2, nu=1.5), {"lengthscale": 1.2, "variance": 1.1}),
        "matern52": (ConstantKernel(2.0) * Matern(ard, nu=2.5), {"lengthscale": ard, "variance": 2.0}),
        "rq": (
            ConstantKernel(1.4) * RationalQuadratic(0.9, alpha=0.7),
            {"lengthscale": 0.9, "alpha": 0.7, "variance": 1.4},
        ),
        "periodic": (
            ConstantKernel(0.8) * ExpSineSquared(1.1, periodicity=2.3),
            {"lengthscale": 1.1, "period": 2.3, "variance": 0.8},
        ),
        "linear": (ConstantKernel(0.5) * DotProduct(sigma_0=0.0) + ConstantKernel(0.3), {"variance": 0.5, "bias": 0.3}),
        "polynomial": (
            (ConstantKernel(0.3) + ConstantKernel(0.5) * DotProduct(sigma_0=0.0)) ** 3,
            {"variance": 0.5, "bias": 0.3},
        ),
    }
    grams = {name: {"params": p, "xy": k(x, y2), "xx": k(x)} for name, (k, p) in kernels.items()}

    # Regression: 1-D inputs, fixed hyperparameters (optimizer=None), noise via alpha.
    n = 12
    xr = np.sort(rng.uniform(0, 5, size=n))[:, None]
    yr = np.sin(1.3 * xr[:, 0]) + 0.2 * rng.normal(size=n)
    xs = np.linspace(-0.5, 5.5, 9)[:, None]
    noise = 0.05
    kr = ConstantKernel(1.5) * RBF(0.7)
    gpr = GaussianProcessRegressor(kernel=kr, alpha=noise, optimizer=None).fit(xr, yr)
    mean, cov = gpr.predict(xs, return_cov=True)
    lml, _grad = gpr.log_marginal_likelihood(gpr.kernel_.theta, eval_gradient=True)
    # Gradient w.r.t. log noise too: use a WhiteKernel for the noise.
    kw = ConstantKernel(1.5) * RBF(0.7) + WhiteKernel(noise)
    gw = GaussianProcessRegressor(kernel=kw, alpha=0.0, optimizer=None).fit(xr, yr)
    lml_w, grad_w = gw.log_marginal_likelihood(gw.kernel_.theta, eval_gradient=True)
    # Fitted hyperparameters (L-BFGS-B from the same start, several restarts).
    fitted = GaussianProcessRegressor(kernel=kw, alpha=0.0, n_restarts_optimizer=5, random_state=0).fit(xr, yr)
    fp = fitted.kernel_.get_params()

    # Classification by Laplace, fixed hyperparameters.
    xc = rng.uniform(-3, 3, size=(30, 1))
    yc = (np.sin(xc[:, 0]) + 0.3 * rng.normal(size=30) > 0).astype(int)
    kc = ConstantKernel(2.0) * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kc, optimizer=None).fit(xc, yc)
    base = gpc.base_estimator_
    xcs = np.linspace(-3, 3, 7)[:, None]

    return {
        "x": x,
        "y": y2,
        "grams": grams,
        "regression": {
            "x": xr[:, 0],
            "y": yr,
            "xs": xs[:, 0],
            "noise": noise,
            "variance": 1.5,
            "lengthscale": 0.7,
            "mean": mean,
            "cov": cov,
            "lml": lml,
            # scikit-learn's theta order: log constant, log lengthscale, log noise.
            "log_gradient": grad_w,
            "lml_white": lml_w,
            "fitted": {
                "variance": fp["k1__k1__constant_value"],
                "lengthscale": fp["k1__k2__length_scale"],
                "noise": fp["k2__noise_level"],
                "lml": fitted.log_marginal_likelihood_value_,
            },
        },
        "classification": {
            "x": xc[:, 0],
            "y": yc,
            "xs": xcs[:, 0],
            "variance": 2.0,
            "lengthscale": 1.0,
            "mode": base.f_cached,
            "lml": base.log_marginal_likelihood_value_,
            "proba": gpc.predict_proba(xcs)[:, 1],
        },
    }
