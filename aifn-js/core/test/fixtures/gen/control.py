"""Golden values for aifn/control from scipy: Riccati and Lyapunov equations, discretisation, tf/ss conversion, Bode."""

import numpy as np
import scipy.linalg as sla
import scipy.signal as sig
from scipy.optimize import brentq


def margins_reference(num: list[float], den: list[float]) -> dict[str, float]:
    """Gain and phase margins by root finding on |L(jw)| = 1 and Im L(jw) = 0 with Re L(jw) < 0."""

    def response(w: float) -> complex:
        s = 1j * w
        return complex(np.polyval(num, s) / np.polyval(den, s))

    ws = np.logspace(-3, 3, 20001)
    mags = np.array([abs(response(w)) for w in ws])
    ims = np.array([response(w).imag for w in ws])
    gc = [
        brentq(lambda w: abs(response(w)) - 1, ws[i], ws[i + 1])
        for i in range(len(ws) - 1)
        if (mags[i] - 1) * (mags[i + 1] - 1) < 0
    ]
    pc = [
        brentq(lambda w: response(w).imag, ws[i], ws[i + 1])
        for i in range(len(ws) - 1)
        if ims[i] * ims[i + 1] < 0 and response(ws[i]).real < 0
    ]
    w_gc = gc[0]
    w_pc = pc[0]
    return {
        "gainCrossover": w_gc,
        "phaseMargin": 180 + float(np.degrees(np.angle(response(w_gc)))),
        "phaseCrossover": w_pc,
        "gainMargin": 1 / abs(response(w_pc)),
    }


def cases() -> dict[str, object]:
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 3))
    b = rng.normal(size=(3, 2))
    m = rng.normal(size=(3, 3))
    q = m @ m.T + np.eye(3)
    r = np.array([[2.0, 0.3], [0.3, 1.0]])

    # Linearised cart-pole (states x, x', theta, theta'; open-loop unstable).
    cp_a = np.array([[0, 1, 0, 0], [0, 0, -0.98, 0], [0, 0, 0, 1], [0, 0, 21.56, 0]], dtype=float)
    cp_b = np.array([[0], [1], [0], [-2]], dtype=float)
    cp_q = np.diag([1.0, 0.1, 10.0, 0.1])
    cp_r = np.array([[0.5]])

    ad = a / (1.2 * max(abs(np.linalg.eigvals(a))))  # a stable discrete A

    stable = a - (max(np.linalg.eigvals(a).real) + 0.5) * np.eye(3)
    c = rng.normal(size=(2, 3))
    d = rng.normal(size=(2, 2))
    zoh = sig.cont2discrete((a, b, c, d), 0.1, method="zoh")
    tustin = sig.cont2discrete((a, b, c, d), 0.1, method="bilinear")

    num = [1.0, 3.0]
    den = [1.0, 2.0, 3.0, 4.0]
    A_tf, B_tf, C_tf, D_tf = sig.tf2ss(num, den)
    ss_num, ss_den = sig.ss2tf(cp_a, cp_b, np.array([[1.0, 0, 0, 0]]), np.array([[0.0]]))

    w = np.logspace(-2, 2, 50)
    _, mag, phase = sig.bode(sig.lti([2.0, 1.0], [1.0, 0.5, 4.0, 0.0]), w)

    return {
        "care": {"A": a, "B": b, "Q": q, "R": r, "P": sla.solve_continuous_are(a, b, q, r)},
        "cartpole": {"A": cp_a, "B": cp_b, "Q": cp_q, "R": cp_r, "P": sla.solve_continuous_are(cp_a, cp_b, cp_q, cp_r)},
        "dare": {"A": a, "B": b, "Q": q, "R": r, "P": sla.solve_discrete_are(a, b, q, r)},
        "dare_stable": {"A": ad, "B": b, "Q": q, "R": r, "P": sla.solve_discrete_are(ad, b, q, r)},
        "lyapunov": {
            "A": stable,
            "Q": q,
            "X": sla.solve_continuous_lyapunov(stable, -q),
            "Ad": ad,
            "Xd": sla.solve_discrete_lyapunov(ad, q),
        },
        "discretise": {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "dt": 0.1,
            "zoh": {"A": zoh[0], "B": zoh[1], "C": zoh[2], "D": zoh[3]},
            "tustin": {"A": tustin[0], "B": tustin[1], "C": tustin[2], "D": tustin[3]},
        },
        "tf2ss": {"num": num, "den": den, "A": A_tf, "B": B_tf, "C": C_tf, "D": D_tf},
        "ss2tf": {"num": ss_num[0], "den": ss_den},
        "bode": {"num": [2.0, 1.0], "den": [1.0, 0.5, 4.0, 0.0], "w": w, "mag": mag, "phase": phase},
        "margins": {"num": [2.0], "den": [1.0, 3.0, 2.0, 0.0], **margins_reference([2.0], [1.0, 3.0, 2.0, 0.0])},
    }
