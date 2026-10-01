"""Golden values for aifn/geometry: convex hulls from scipy.spatial and meshgrids from numpy."""

import numpy as np
from scipy.spatial import ConvexHull


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20260930)
    pts = rng.normal(size=(40, 2))
    hull = ConvexHull(pts)
    x = np.linspace(-1, 1, 4)
    y = np.linspace(0, 2, 3)
    xx, yy = np.meshgrid(x, y)
    xi, yi = np.meshgrid(x, y, indexing="ij")
    return {
        "hull": {"points": pts, "vertices": sorted(hull.vertices.tolist()), "area": hull.volume},
        "meshgrid": {"x": x, "y": y, "xx": xx, "yy": yy, "xi": xi, "yi": yi},
        "logspace": np.logspace(-2, 1, 7),
    }
