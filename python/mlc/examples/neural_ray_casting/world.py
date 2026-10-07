"""Grid worlds and a vectorised ray caster.

The maps are read from the site's ``world.ts``, so the Python training data and the browser figure use the same worlds.
A world is a grid of square cells of side 1, empty (0) or wall (1 to 4); x grows to the right and y downwards, and a ray
angle is measured from the +x axis towards +y. Rays are traced with the digital differential analyser of Amanatides and
Woo (1987), vectorised over rays.
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
WORLD_TS = REPO / "content/notes/part-7-application-domains/games/_shared/world.ts"
MAX_DISTANCE = 40.0
FOV = math.radians(66)


@dataclass(frozen=True)
class World:
    id: str
    cells: np.ndarray  # (height, width) uint8, 0 empty

    @property
    def height(self) -> int:
        return self.cells.shape[0]

    @property
    def width(self) -> int:
        return self.cells.shape[1]


def load_worlds() -> dict[str, World]:
    """Every map in world.ts, by id."""
    src = WORLD_TS.read_text()
    worlds: dict[str, World] = {}
    for m in re.finditer(r"id: '([^']+)',.*?rows: \[(.*?)\]", src, re.S):
        rows = re.findall(r"'([^']*)'", m.group(2))
        grid = np.array([[int(c) if c in "1234" else 0 for c in r] for r in rows], dtype=np.uint8)
        worlds[m.group(1)] = World(m.group(1), grid)
    return worlds


def occupied(w: World, cx: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """True for wall cells; everything outside the map counts as wall."""
    inside = (cx >= 0) & (cy >= 0) & (cx < w.width) & (cy < w.height)
    v = np.ones(cx.shape, dtype=bool)
    v[inside] = w.cells[cy[inside], cx[inside]] != 0
    return v


def cast(w: World, x: np.ndarray, y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Distance to the first wall along every ray (x, y, angle)."""
    dx, dy = np.cos(a), np.sin(a)
    cx, cy = np.floor(x).astype(int), np.floor(y).astype(int)
    sx = np.where(dx < 0, -1, 1)
    sy = np.where(dy < 0, -1, 1)
    # Distance along the ray between successive vertical (x) and horizontal (y) grid lines, and to the first of each.
    with np.errstate(divide="ignore"):
        ddx = np.where(dx == 0, np.inf, np.abs(1 / dx))
        ddy = np.where(dy == 0, np.inf, np.abs(1 / dy))
    nx = np.where(dx < 0, (x - cx) * ddx, (cx + 1 - x) * ddx)
    ny = np.where(dy < 0, (y - cy) * ddy, (cy + 1 - y) * ddy)
    d = np.full(x.shape, MAX_DISTANCE)
    live = np.ones(x.shape, dtype=bool)
    for _ in range(4 * (w.width + w.height)):
        if not live.any():
            break
        step_x = live & (nx < ny)
        step_y = live & ~(nx < ny)
        t = np.where(step_x, nx, ny)
        nx = np.where(step_x, nx + ddx, nx)
        ny = np.where(step_y, ny + ddy, ny)
        cx = np.where(step_x, cx + sx, cx)
        cy = np.where(step_y, cy + sy, cy)
        hit = live & occupied(w, cx, cy)
        d[hit] = t[hit]
        live &= ~hit
    return d


def wall_line(w: World, x: np.ndarray, y: np.ndarray, a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    The grid line each ray stops on, as a class: vertical lines x = c are classes c = 0..W, horizontal lines y = c are
    classes W + 1 + c. Every wall face lies on one of these lines, so the line fixes the hit distance exactly.
    """
    hx, hy = x + d * np.cos(a), y + d * np.sin(a)
    vertical = np.abs(hx - np.round(hx)) <= np.abs(hy - np.round(hy))
    return np.where(vertical, np.round(hx), w.width + 1 + np.round(hy)).astype(int)


def sample_free(w: World, n: int, rng: np.random.Generator, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """n points uniform over the free floor, at least `margin` from every wall cell."""
    xs, ys = [], []
    got = 0
    while got < n:
        x = rng.uniform(0, w.width, 2 * n)
        y = rng.uniform(0, w.height, 2 * n)
        ok = np.ones(x.shape, dtype=bool)
        for ox in (-margin, margin):
            for oy in (-margin, margin):
                ok &= ~occupied(w, np.floor(x + ox).astype(int), np.floor(y + oy).astype(int))
        xs.append(x[ok])
        ys.append(y[ok])
        got += int(ok.sum())
    return np.concatenate(xs)[:n], np.concatenate(ys)[:n]


def ray_offsets(r: int, fov: float = FOV) -> np.ndarray:
    """Angles of r screen columns from the heading, through equally spaced points of a flat projection plane."""
    return np.arctan((2 * (np.arange(r) + 0.5) / r - 1) * np.tan(fov / 2))


def traverse(w: World, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Every cell each ray enters, in order, until it leaves the map, whatever the cells contain: flat cell indices
    (y · W + x; −1 once outside, the solid boundary) and the distance at which the ray enters each, shape
    (N, W + H + 2).
    This is the part of ray casting that does not depend on the map, so a model of the map can reuse it.
    """
    x, y, a = rays[:, 0], rays[:, 1], rays[:, 2]
    dx, dy = np.cos(a), np.sin(a)
    cx, cy = np.floor(x).astype(int), np.floor(y).astype(int)
    sx, sy = np.where(dx < 0, -1, 1), np.where(dy < 0, -1, 1)
    with np.errstate(divide="ignore"):
        ddx = np.where(dx == 0, np.inf, np.abs(1 / dx))
        ddy = np.where(dy == 0, np.inf, np.abs(1 / dy))
    nx = np.where(dx < 0, (x - cx) * ddx, (cx + 1 - x) * ddx)
    ny = np.where(dy < 0, (y - cy) * ddy, (cy + 1 - y) * ddy)
    k_max = w.width + w.height + 2
    cells = np.full((len(x), k_max), -1, dtype=np.int64)
    ts = np.zeros((len(x), k_max))
    outside = np.zeros(len(x), dtype=bool)
    for k in range(k_max):
        step_x = nx < ny
        t = np.where(step_x, nx, ny)
        nx = np.where(step_x, nx + ddx, nx)
        ny = np.where(~step_x, ny + ddy, ny)
        cx = np.where(step_x, cx + sx, cx)
        cy = np.where(~step_x, cy + sy, cy)
        out_now = (cx < 0) | (cy < 0) | (cx >= w.width) | (cy >= w.height)
        # Once outside, a ray stays at its exit distance; the boundary is solid, so it stops there.
        ts[:, k] = np.where(outside, ts[:, k - 1] if k else 0.0, t)
        cells[:, k] = np.where(out_now | outside, -1, cy * w.width + cx)
        outside |= out_now
    return cells, ts
