"""Student's pooled two-sample t-test and Welch's unequal-variance t-test, and their Type I error by simulation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

type Array = NDArray[np.float64]


@dataclass(frozen=True)
class TestResult:
    t: float
    df: float
    p: float
    """Two-sided p-value."""


def pooled_t(x: Array, y: Array) -> TestResult:
    """Student's test: one variance estimate pooled across groups, df = n₁ + n₂ − 2."""
    n1, n2 = len(x), len(y)
    pooled = ((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2)
    t = (x.mean() - y.mean()) / np.sqrt(pooled * (1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    return TestResult(float(t), float(df), float(2 * stats.t.sf(abs(t), df)))


def welch_t(x: Array, y: Array) -> TestResult:
    """Welch's test: each group keeps its own variance; df from the Welch–Satterthwaite approximation."""
    n1, n2 = len(x), len(y)
    a, b = x.var(ddof=1) / n1, y.var(ddof=1) / n2
    t = (x.mean() - y.mean()) / np.sqrt(a + b)
    df = (a + b) ** 2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))
    return TestResult(float(t), float(df), float(2 * stats.t.sf(abs(t), df)))


def type_i_error(n1: int, n2: int, sd_ratio: float, *, alpha: float, reps: int, seed: int) -> tuple[float, float]:
    """Rejection rates of (pooled, Welch) when the means are equal and sd₂/sd₁ = sd_ratio. Vectorised over reps."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(reps, n1))
    y = rng.normal(0.0, sd_ratio, size=(reps, n2))
    v1, v2 = x.var(axis=1, ddof=1), y.var(axis=1, ddof=1)
    diff = x.mean(axis=1) - y.mean(axis=1)

    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    t_pooled = diff / np.sqrt(pooled * (1 / n1 + 1 / n2))
    reject_pooled = np.abs(t_pooled) > stats.t.ppf(1 - alpha / 2, n1 + n2 - 2)

    a, b = v1 / n1, v2 / n2
    t_welch = diff / np.sqrt(a + b)
    df = (a + b) ** 2 / (a**2 / (n1 - 1) + b**2 / (n2 - 1))
    reject_welch = np.abs(t_welch) > stats.t.ppf(1 - alpha / 2, df)
    return float(reject_pooled.mean()), float(reject_welch.mean())
