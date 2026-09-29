"""Three ways to keep thresholds in order: reparameterise, project, or attach Lagrange multipliers."""

import torch
import torch.nn.functional as F
from torch import Tensor


def ordered(theta1: Tensor, delta: Tensor) -> Tensor:
    """θ_1 followed by softplus increments: every value of (θ_1, δ) gives strictly ordered thresholds."""
    return torch.cat([theta1.reshape(1), theta1 + torch.cumsum(F.softplus(delta), dim=0)])


def inverse_softplus(gap: Tensor) -> Tensor:
    """The δ whose softplus is a given positive gap."""
    return gap + torch.log(-torch.expm1(-gap))


def pool_adjacent_violators(v: Tensor) -> Tensor:
    """Euclidean projection onto θ_1 ≤ … ≤ θ_{K−1}: replace each out-of-order run by its mean."""
    blocks: list[tuple[float, int]] = []  # (sum, count) of each pooled run
    for value in v.tolist():
        blocks.append((value, 1))
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1] = (blocks[-1][0] + total, blocks[-1][1] + count)
    return torch.tensor([total / count for total, count in blocks for _ in range(count)], dtype=v.dtype)


def constraints(theta: Tensor) -> Tensor:
    """g_k(θ) = θ_k − θ_{k+1}; the order holds when every g_k ≤ 0."""
    return theta[:-1] - theta[1:]


def augmented_penalty(theta: Tensor, mu: Tensor, rho: float) -> Tensor:
    """The augmented Lagrangian term for g ≤ 0 with multipliers μ ≥ 0 and penalty weight ρ."""
    return (torch.clamp(mu + rho * constraints(theta), min=0) ** 2 - mu**2).sum() / (2 * rho)


def update_multipliers(theta: Tensor, mu: Tensor, rho: float) -> Tensor:
    """First-order multiplier update: μ grows while a constraint is violated and falls to 0 when it is slack."""
    return torch.clamp(mu + rho * constraints(theta), min=0)


def stationarity_multipliers(grad: Tensor) -> Tensor:
    """μ_k = −(∂_1 L + … + ∂_k L), the multipliers that make ∇L + Σ μ_k ∇g_k = 0 at a constrained optimum."""
    return -torch.cumsum(grad, dim=0)[:-1]
