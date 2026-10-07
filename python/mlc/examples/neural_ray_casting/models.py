"""Models of the depth function d(x, y, φ): each maps rays (N, 3) = (x, y, φ) in cells and radians to distances (N,).

Three classes:

- **Distance regression**: a network outputs log d directly (Fourier features + MLP, SIREN, or a mixture of candidate
  distances). It smooths every occlusion edge, where d jumps.
- **Wall-line classifier**: a network picks the grid line the ray stops on; the distance to that line is computed
  exactly, so walls are straight and edges sharp.
- **Density field + volume rendering**: a network learns a 2D density σ(x, y), the map itself, and the distance is the
  expected termination distance along the ray, as in NeRF. Occlusion is the renderer's job.
"""

import math

import torch
from torch import nn

from mlc.examples.neural_ray_casting.world import World


class FourierMLP(nn.Module):
    """Positional encoding of (x/W, y/H, φ) at octaves 2^0..2^(L−1), sines and cosines, then a ReLU MLP to log d."""

    def __init__(self, w: World, octaves: int = 6, width: int = 64, depth: int = 2):
        super().__init__()
        self.wh = torch.tensor([w.width, w.height], dtype=torch.float32)
        self.f = 2.0 ** torch.arange(octaves)
        layers, n = [], 6 * octaves
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.ReLU()]
            n = width
        self.net = nn.Sequential(*layers, nn.Linear(n, 1))

    def features(self, r: torch.Tensor) -> torch.Tensor:
        uv = r[:, :2] / self.wh
        z = torch.cat([uv[:, :1] * self.f * math.pi, uv[:, 1:] * self.f * math.pi, r[:, 2:3] * self.f], 1)
        return torch.cat([z.sin(), z.cos()], 1)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.net(self.features(r)).squeeze(1))


class Sine(nn.Module):
    """A SIREN layer sin(ω₀ (W x + b)) with the initialisation of Sitzmann et al. (2020)."""

    def __init__(self, n_in: int, n_out: int, w0: float, first: bool):
        super().__init__()
        self.lin, self.w0 = nn.Linear(n_in, n_out), w0
        bound = 1 / n_in if first else math.sqrt(6 / n_in) / w0
        nn.init.uniform_(self.lin.weight, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.lin(x))


class Siren(nn.Module):
    """Sine activations on (2u − 1, 2v − 1, cos φ, sin φ), output log d."""

    def __init__(self, w: World, width: int = 128, depth: int = 3, w0: float = 30.0):
        super().__init__()
        self.wh = torch.tensor([w.width, w.height], dtype=torch.float32)
        layers = [Sine(4, width, w0, True)] + [Sine(width, width, w0, False) for _ in range(depth - 1)]
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        uv = 2 * r[:, :2] / self.wh - 1
        return torch.exp(self.net(torch.cat([uv, r[:, 2:3].cos(), r[:, 2:3].sin()], 1)).squeeze(1))


class Mixture(nn.Module):
    """
    K candidate log distances μ_k with Laplace scales b_k and weights π_k; trained by the mixture likelihood, predicting
    the μ of the most probable component, so near an edge it picks a side instead of averaging.
    """

    def __init__(self, w: World, k: int = 4, octaves: int = 6, width: int = 128, depth: int = 3):
        super().__init__()
        self.body = FourierMLP(w, octaves, width, depth)
        self.body.net[-1] = nn.Linear(width, 3 * k)
        self.K = k

    def params(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_b, logit = self.body.net(self.body.features(r)).split(self.K, 1)
        return mu, log_b.clamp(-6, 2), logit

    def nll(self, r: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        mu, log_b, logit = self.params(r)
        comp = -(torch.log(d)[:, None] - mu).abs() / log_b.exp() - log_b - math.log(2)
        return -(torch.logsumexp(comp + torch.log_softmax(logit, 1), 1)).mean()

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        mu, _, logit = self.params(r)
        return torch.exp(mu.gather(1, logit.argmax(1, keepdim=True)).squeeze(1))


def line_distance(w: World, r: torch.Tensor) -> torch.Tensor:
    """
    Distance along each ray to every grid line, (N, W + H + 2): (c − x)/cos φ to the vertical line x = c and
    (c − y)/sin φ to the horizontal line y = c. Non-positive values are lines behind the camera.
    """
    c = torch.arange(w.width + w.height + 2, dtype=torch.float32)
    vertical = c <= w.width
    position = torch.where(vertical, c, c - (w.width + 1))
    num = position[None] - torch.where(vertical[None], r[:, 0:1], r[:, 1:2])
    den = torch.where(vertical[None], torch.cos(r[:, 2:3]), torch.sin(r[:, 2:3]))
    den = torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
    return num / den


class Walls(nn.Module):
    """
    Classify the grid line a ray stops on (W + H + 2 classes) and return the exact distance to it. Lines behind the
    camera are masked before the softmax and the argmax.
    """

    def __init__(self, w: World, octaves: int = 6, width: int = 64, depth: int = 2):
        super().__init__()
        self.w = w
        self.body = FourierMLP(w, octaves, width, depth)
        self.body.net[-1] = nn.Linear(width, w.width + w.height + 2)

    def logits(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = line_distance(self.w, r)
        return self.body.net(self.body.features(r)).masked_fill(dist <= 1e-4, -1e4), dist

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        logit, dist = self.logits(r)
        return dist.gather(1, logit.argmax(1, keepdim=True)).squeeze(1)


def grid_lookup(table: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Bilinear interpolation of a grid of features `table` (H + 1, W + 1, F) at points p (N, 2) in grid units."""
    res = table.shape[:2]
    base = torch.floor(p)
    frac = p - base
    base = base.long()
    out = torch.zeros(p.shape[0], table.shape[2], device=p.device)
    for bx in (0, 1):
        for by in (0, 1):
            i = (base[:, 0] + bx).clamp(0, res[0] - 1)
            j = (base[:, 1] + by).clamp(0, res[1] - 1)
            wgt = (frac[:, 0] if bx else 1 - frac[:, 0]) * (frac[:, 1] if by else 1 - frac[:, 1])
            out = out + wgt[:, None] * table[i, j]
    return out


class Density(nn.Module):
    """
    A 2D density field σ(x, y), multi-resolution feature grids (to `levels[-1]` cells across the map) and a small MLP,
    rendered along each ray: with K samples t_k at spacing δ on [0, T], α_k = 1 − exp(−σ_k δ) and transmittance
    T_k = Π_{j<k}(1 − α_j), the distance is d̂ = Σ_k T_k α_k t_k + T_K T. Outside the map is solid.
    """

    wh: torch.Tensor

    def __init__(self, w: World, levels: tuple[int, ...] = (4, 8, 16, 32, 64), feats: int = 2, width: int = 32):
        super().__init__()
        self.register_buffer("wh", torch.tensor([w.width, w.height], dtype=torch.float32))
        self.far = math.hypot(w.width, w.height)
        self.K = int(8 * self.far)  # eight samples per cell of ray length
        self.levels = levels
        self.tables = nn.ParameterList([nn.Parameter(1e-2 * torch.randn(s + 1, s + 1, feats)) for s in levels])
        self.net = nn.Sequential(nn.Linear(feats * len(levels), width), nn.ReLU(), nn.Linear(width, 1))

    def sigma(self, xy: torch.Tensor) -> torch.Tensor:
        uv = xy / self.wh
        fs = [grid_lookup(t, uv * s) for t, s in zip(self.tables, self.levels, strict=True)]
        inside = ((uv >= 0) & (uv <= 1)).all(1)
        s = nn.functional.softplus(self.net(torch.cat(fs, 1)).squeeze(1) + 2)
        return torch.where(inside, s, torch.full_like(s, 1e3))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        n, dev = r.shape[0], r.device
        delta = self.far / self.K
        t = (torch.arange(self.K, dtype=torch.float32, device=dev) + 0.5) * delta
        # Training jitters the samples within their intervals (stratified sampling), so no grid of t is privileged.
        t = t + (torch.rand(n, self.K, device=dev) - 0.5) * delta if self.training else t.expand(n, self.K)
        dirs = torch.stack([r[:, 2].cos(), r[:, 2].sin()], 1)
        sig = self.sigma((r[:, None, :2] + t[..., None] * dirs[:, None, :]).reshape(-1, 2)).reshape(n, self.K)
        alpha = 1 - torch.exp(-sig * delta)
        # Transmittance as exp(cumsum(log)) rather than cumprod, which ONNX lacks; the two are equal.
        trans = torch.exp(torch.cumsum(torch.log(torch.cat([torch.ones(n, 1, device=dev), 1 - alpha + 1e-10], 1)), 1))
        return (trans[:, :-1] * alpha * t).sum(1) + trans[:, -1] * self.far


class OccupancyMap(nn.Module):
    """
    The map itself: one logit per cell, P(wall) = σ(θ_c). Training sees each ray's traversal (the cells it enters and
    the exact distance at which it enters each, from `world.traverse`); the ray stops in the k-th cell with probability
    p_k = o_k Π_{j<k}(1 − o_j), so the expected distance is Σ_k p_k t_k, differentiable in the occupancies. Prediction
    thresholds the map at ½ and ray-casts it exactly, with the grid traversal written in tensor operations so that it
    exports to ONNX.
    """

    def __init__(self, w: World):
        super().__init__()
        self.W, self.H = w.width, w.height
        self.logit = nn.Parameter(torch.full((w.width * w.height,), -2.0))

    def cell_logits(self) -> torch.Tensor:
        # Index −1 (outside the map) is always solid.
        return torch.cat([self.logit, torch.tensor([20.0], device=self.logit.device)])

    def expected(self, cells: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        lg = self.cell_logits()[cells]
        log_free = -nn.functional.softplus(lg)  # log(1 − o)
        before = torch.cumsum(log_free, 1) - log_free  # log Π_{j<k}(1 − o_j)
        p = torch.exp(-nn.functional.softplus(-lg) + before)  # o_k Π_{j<k}(1 − o_j)
        return (p * ts).sum(1) + torch.exp(before[:, -1] + log_free[:, -1]) * ts[:, -1]

    @staticmethod
    def first_hit(occupied: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Entry distance of the first occupied cell along each traversal (the last entry is the solid boundary)."""
        return ts.gather(1, occupied.float().argmax(1, keepdim=True)).squeeze(1)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        occ = (self.logit > 0).float()
        x, y, a = r[:, 0], r[:, 1], r[:, 2]
        dx, dy = torch.cos(a), torch.sin(a)
        cx, cy = torch.floor(x), torch.floor(y)
        sx, sy = torch.where(dx < 0, -1.0, 1.0), torch.where(dy < 0, -1.0, 1.0)
        ddx = 1 / dx.abs().clamp_min(1e-9)
        ddy = 1 / dy.abs().clamp_min(1e-9)
        nx = torch.where(dx < 0, (x - cx) * ddx, (cx + 1 - x) * ddx)
        ny = torch.where(dy < 0, (y - cy) * ddy, (cy + 1 - y) * ddy)
        d = torch.full_like(x, 1e3)
        # Masks are kept as 0/1 floats: ONNX Runtime has no Where for boolean values.
        done = torch.zeros_like(x)
        for _ in range(self.W + self.H + 2):
            step_x = nx < ny
            t = torch.where(step_x, nx, ny)
            nx = torch.where(step_x, nx + ddx, nx)
            ny = torch.where(step_x, ny, ny + ddy)
            cx = torch.where(step_x, cx + sx, cx)
            cy = torch.where(step_x, cy, cy + sy)
            inside = (cx >= 0) & (cy >= 0) & (cx < self.W) & (cy < self.H)
            index = (cy.clamp(0, self.H - 1) * self.W + cx.clamp(0, self.W - 1)).long()
            solid = torch.where(inside, occ[index], torch.ones_like(x))
            hit = solid * (1 - done)
            d = torch.where(hit > 0.5, t, d)
            done = done + hit
        return d


class NormalPolicy(nn.Module):
    """
    A ray caster as a policy that names the wall's facing and estimates its distance, with no grid lines given.

    Walls face along x (east or west) or along y (north or south); for a ray with direction u = (cos φ, sin φ) the side
    it can see follows from the sign of u, so the facing is a choice of two. For each facing the network gives the mean
    and log spread of log p, where p is the wall's distance from the camera measured along the wall's normal. The
    distance along the ray is the projection d = p / |n · u|: p / |cos φ| for a wall across x, p / |sin φ| across y.
    Prediction takes the more probable facing and p = exp(mean).
    """

    def __init__(self, w: World, octaves: int = 6, width: int = 64, depth: int = 2):
        super().__init__()
        self.body = FourierMLP(w, octaves, width, depth)
        self.body.net[-1] = nn.Linear(width, 6)  # 2 facing logits, 2 means and 2 log spreads of log p

    def heads(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.body.net(self.body.features(r))
        logit, mean, log_sd = out[:, 0:2], out[:, 2:4], out[:, 4:6].clamp(-5, 1)
        # |n · u| for a wall across x and across y; floored so a ray along a wall cannot divide by zero.
        cosine = torch.stack([torch.cos(r[:, 2]).abs(), torch.sin(r[:, 2]).abs()], 1).clamp_min(1e-3)
        return logit, mean, log_sd, cosine

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        logit, mean, _, cosine = self.heads(r)
        facing = logit.argmax(1, keepdim=True)
        return (torch.exp(mean.gather(1, facing)) / cosine.gather(1, facing)).squeeze(1)


class WallSegments(nn.Module):
    """
    A ray caster made of learned straight walls. The parameters are K segments, each two endpoints a_k, b_k. A ray
    o + t u meets segment k where o + t u = a_k + s (b_k − a_k); with e = b_k − a_k and the 2D cross product
    p × q = p_x q_y − p_y q_x,

        t_k = ((a_k − o) × e) / (u × e),    s_k = ((a_k − o) × u) / (u × e),

    and the ray crosses the segment when t_k > 0 and 0 ≤ s_k ≤ 1. The distance is the nearest crossing, the minimum
    of t_k over the segments crossed: occlusion is exact by construction. Training uses the expected first crossing
    (`soft`), with the crossing tests softened to sigmoids of width `sharpness`, which is annealed towards the hard
    test. Walls may face any direction.
    """

    def __init__(self, w: World, k: int = 128):
        super().__init__()
        self.far = math.hypot(w.width, w.height)
        g = torch.Generator().manual_seed(0)
        centre = torch.rand(k, 2, generator=g) * torch.tensor([w.width, w.height])
        angle = torch.rand(k, generator=g) * math.pi
        half = 0.5 * torch.stack([torch.cos(angle), torch.sin(angle)], 1)
        self.a = nn.Parameter(centre - half)
        self.b = nn.Parameter(centre + half)
        self.sharpness = 0.05  # width of the soft crossing tests (s in segment lengths, t in cells)

    def place(self, hits: torch.Tensor) -> None:
        """Centre the segments on observed hit points (o + d u of training rays), keeping their random angles."""
        with torch.no_grad():
            half = (self.b - self.a) / 2
            pick = hits[torch.randperm(len(hits), generator=torch.Generator().manual_seed(1))[: len(half)]]
            self.a.copy_(pick - half)
            self.b.copy_(pick + half)

    def crossings(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        o, u = r[:, :2], torch.stack([torch.cos(r[:, 2]), torch.sin(r[:, 2])], 1)
        e = self.b - self.a  # (K, 2)
        ao = self.a[None] - o[:, None]  # (N, K, 2)
        den = u[:, None, 0] * e[None, :, 1] - u[:, None, 1] * e[None, :, 0]
        den = torch.where(den.abs() < 1e-6, torch.full_like(den, 1e-6), den)
        t = (ao[..., 0] * e[None, :, 1] - ao[..., 1] * e[None, :, 0]) / den
        s = (ao[..., 0] * u[:, None, 1] - ao[..., 1] * u[:, None, 0]) / den
        return t, s

    def soft(self, r: torch.Tensor) -> torch.Tensor:
        """
        Expected first crossing. Sort the segments along the ray; segment k is crossed with probability c_k (sigmoids
        of s, 1 − s and t at `sharpness`), and the ray stops there with probability p_k = c_k Π_{j nearer} (1 − c_j).
        A segment the ray misses contributes nothing however near it is, and the rule is exact once every c_k is 0 or 1.
        """
        t, s = self.crossings(r)
        k = self.sharpness
        log_c = (
            nn.functional.logsigmoid(s / k) + nn.functional.logsigmoid((1 - s) / k) + nn.functional.logsigmoid(t / k)
        )
        order = t.clamp_min(-1.0).argsort(1)
        t_sorted = t.gather(1, order).clamp(1e-3, self.far)
        log_c = log_c.gather(1, order)
        log_miss = torch.log1p(-log_c.exp().clamp(max=1 - 1e-6))
        before = torch.cumsum(log_miss, 1) - log_miss  # log Π over nearer segments of (1 − c_j)
        p = torch.exp(log_c + before)
        return (p * t_sorted).sum(1) + torch.exp(before[:, -1] + log_miss[:, -1]) * self.far

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        t, s = self.crossings(r)
        hit = (s >= 0) & (s <= 1) & (t > 1e-6)
        return torch.where(hit, t, torch.full_like(t, self.far)).min(1).values
