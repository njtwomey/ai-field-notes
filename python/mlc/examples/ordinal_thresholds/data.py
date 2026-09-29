"""The note's worked example and a small synthetic set with a rare class."""

import math

import torch
from torch import Tensor


def worked_example(n2: int) -> tuple[Tensor, Tensor]:
    """Scores 0, 1, 2, 3 for classes 1 to 4 (0-based labels 0 to 3), with 3, n2, 3 and 3 examples."""
    counts = [3, n2, 3, 3]
    s = torch.tensor([float(k) for k, n in enumerate(counts) for _ in range(n)], dtype=torch.float64)
    y = torch.tensor([k for k, n in enumerate(counts) for _ in range(n)])
    return s[:, None], y


def synthetic(counts: list[int], noise: float, seed: int) -> tuple[Tensor, Tensor]:
    """2-D points whose class k is centred k units along a line at 30°, with isotropic Gaussian noise."""
    g = torch.Generator().manual_seed(seed)
    direction = torch.tensor([math.cos(math.pi / 6), math.sin(math.pi / 6)], dtype=torch.float64)
    y = torch.tensor([k for k, n in enumerate(counts) for _ in range(n)])
    centres = y.to(torch.float64)[:, None] * direction[None, :]
    return centres + noise * torch.randn(len(y), 2, generator=g, dtype=torch.float64), y
