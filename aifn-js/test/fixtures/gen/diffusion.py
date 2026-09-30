"""Golden noise schedules for aifn/diffusion, transcribed from the papers' reference code in numpy.

- Linear schedule and posterior coefficients: Ho et al. (2020), diffusion_utils_2.py (betas linspace 1e-4 → 0.02,
  alphas_cumprod, posterior_variance, posterior_mean_coef1/2).
- Cosine schedule: Nichol & Dhariwal (2021), improved-diffusion's betas_for_alpha_bar with max_beta 0.999.
- VP SDE marginal: m(t) = exp(-1/2 ∫ β) by quadrature (Song et al., 2021).
"""

import math

import numpy as np
from scipy import integrate


def cases() -> dict[str, object]:
    T = 1000
    betas = np.linspace(1e-4, 0.02, T, dtype=np.float64)
    alphas_cumprod = np.cumprod(1.0 - betas)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(1.0 - betas) / (1.0 - alphas_cumprod)

    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    cos_betas = np.array([min(1 - alpha_bar((i + 1) / T) / alpha_bar(i / T), 0.999) for i in range(T)])

    beta_min, beta_max = 0.1, 20.0
    times = [0.0, 0.1, 0.37, 0.8, 1.0]
    vp_mean = [math.exp(-0.5 * integrate.quad(lambda s: beta_min + s * (beta_max - beta_min), 0, t)[0]) for t in times]
    pick = [0, 1, 9, 99, 499, 998, 999]
    return {
        "linear": {
            "betas": betas[pick],
            "alphaBars": alphas_cumprod[pick],
            "posteriorVariance": posterior_variance[pick],
            "coef1": coef1[pick],
            "coef2": coef2[pick],
            "steps": [p + 1 for p in pick],
        },
        "cosine": {
            "betas": cos_betas[pick],
            "alphaBars": np.cumprod(1 - cos_betas)[pick],
            "steps": [p + 1 for p in pick],
        },
        "vp": {"times": times, "meanScale": vp_mean},
    }
