"""A threshold model on one score s = wᵀx: cumulative logit probabilities and the immediate-threshold loss.

Classes are 0-based in code: class k lies between θ[k − 1] and θ[k], where θ has K − 1 entries.
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def class_probabilities(s: Tensor, theta: Tensor) -> Tensor:
    """P(y = k | s) = σ(θ_k − s) − σ(θ_{k−1} − s), shape (N, K). A class between crossed thresholds goes negative."""
    cdf = torch.sigmoid(theta[None, :] - s[:, None])  # P(y ≤ k | s), shape (N, K − 1)
    zeros, ones = torch.zeros_like(cdf[:, :1]), torch.ones_like(cdf[:, :1])
    return torch.diff(torch.cat([zeros, cdf, ones], dim=1), dim=1)


def immediate_threshold_loss(s: Tensor, y: Tensor, theta: Tensor) -> Tensor:
    """Logistic loss of each example against the two thresholds beside its class, summed over examples."""
    has_upper = y < len(theta)  # the score should lie below θ[y]
    has_lower = y > 0  # the score should lie above θ[y − 1]
    loss = F.softplus(s[has_upper] - theta[y[has_upper]]).sum()
    return loss + F.softplus(theta[y[has_lower] - 1] - s[has_lower]).sum()


def predict(s: Tensor, theta: Tensor) -> Tensor:
    """The threshold rule: the class is the number of thresholds below the score."""
    return (s[:, None] > theta[None, :]).sum(dim=1)


def contradictory(s: Tensor, theta: Tensor) -> Tensor:
    """Scores at which the binary answers "is y > k?" are not monotone in k: yes at some k but no at a smaller k."""
    above = s[:, None] > theta[None, :]
    return (above[:, 1:] & ~above[:, :-1]).any(dim=1)
