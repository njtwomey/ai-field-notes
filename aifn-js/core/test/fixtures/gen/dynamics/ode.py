"""Golden values for aifn/ode from scipy: solve_ivp's RK45 step sequence, expm and general eigenvalues."""

import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp


def lotka_volterra(_t: float, x: np.ndarray) -> np.ndarray:
    return np.array([1.5 * x[0] - x[0] * x[1], -3.0 * x[1] + x[0] * x[1]])


def robertson(_t: float, y: np.ndarray) -> np.ndarray:
    return np.array(
        [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ]
    )


def robertson_jac(_t: float, y: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [-0.04, 1e4 * y[2], 1e4 * y[1]],
            [0.04, -1e4 * y[2] - 6e7 * y[1], -1e4 * y[1]],
            [0.0, 6e7 * y[1], 0.0],
        ]
    )


MU = 1000.0


def van_der_pol(_t: float, y: np.ndarray) -> np.ndarray:
    return np.array([y[1], MU * (1 - y[0] ** 2) * y[1] - y[0]])


def van_der_pol_jac(_t: float, y: np.ndarray) -> np.ndarray:
    return np.array([[0.0, 1.0], [-2 * MU * y[0] * y[1] - 1, MU * (1 - y[0] ** 2)]])


def bdf_cases() -> dict[str, object]:
    """scipy BDF (analytic Jacobian) step sequences, and tight Radau references at the end time."""
    problems = {
        "robertson": (robertson, robertson_jac, (0.0, 1e5), [1.0, 0.0, 0.0], 1e-6, [1e-8, 1e-14, 1e-8]),
        "van_der_pol": (van_der_pol, van_der_pol_jac, (0.0, 3000.0), [2.0, 0.0], 1e-3, [1e-6, 1e-6]),
        "van_der_pol_tight": (van_der_pol, van_der_pol_jac, (0.0, 3000.0), [2.0, 0.0], 1e-7, [1e-9, 1e-9]),
    }
    out = {}
    for key, (f, jac, span, y0, rtol, atol) in problems.items():
        sol = solve_ivp(f, span, y0, method="BDF", jac=jac, rtol=rtol, atol=atol)
        ref = solve_ivp(f, span, y0, method="Radau", jac=jac, rtol=1e-11, atol=[a * 1e-4 for a in atol])
        out[key] = {
            "span": span,
            "y0": y0,
            "rtol": rtol,
            "atol": atol,
            "t": sol.t,
            "x": sol.y.T,
            "nfev": sol.nfev,
            "njev": sol.njev,
            "nlu": sol.nlu,
            "reference": ref.y[:, -1],
        }
    return out


def cases() -> dict[str, object]:
    rng = np.random.default_rng(3)
    rk45 = {}
    for rtol, atol in [(1e-3, 1e-6), (1e-8, 1e-10)]:
        sol = solve_ivp(lotka_volterra, (0.0, 10.0), [10.0, 5.0], method="RK45", rtol=rtol, atol=atol)
        rk45[f"{rtol:g}"] = {
            "rtol": rtol,
            "atol": atol,
            "t": sol.t,
            "x": sol.y.T,
            "nfev": sol.nfev,
        }
    reference = solve_ivp(lotka_volterra, (0.0, 10.0), [10.0, 5.0], method="DOP853", rtol=1e-13, atol=1e-13)

    matrices = {
        "rotation": np.array([[0.0, 1.0], [-1.0, 0.0]]),
        "random": rng.normal(size=(4, 4)),
        "large": 8.0 * rng.normal(size=(5, 5)),
        "nilpotent": np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]]),
        "stiff": np.array([[-1000.0, 1.0], [0.0, -0.5]]),
    }
    expm = {k: {"a": a, "expm": sla.expm(a)} for k, a in matrices.items()}

    eig = {}
    for k, a in {
        "random": rng.normal(size=(6, 6)),
        "companion": np.array([[0, 1, 0], [0, 0, 1], [6, -11, 6.0]]),
    }.items():
        w = np.linalg.eigvals(a)
        order = np.lexsort((-w.imag, -w.real))
        eig[k] = {"a": a, "real": w.real[order], "imag": w.imag[order]}

    return {
        "rk45": rk45,
        "lotka_volterra_x10": reference.y[:, -1],
        "expm": expm,
        "eig": eig,
        "bdf": bdf_cases(),
    }
