"""Fit the thresholds (and optionally w) by gradient descent in four ways, recording the path of every run."""

from dataclasses import dataclass, field
from enum import StrEnum

import torch
from torch import Tensor

from mlc.examples.ordinal_thresholds.model import immediate_threshold_loss
from mlc.examples.ordinal_thresholds.ordering import (
    augmented_penalty,
    inverse_softplus,
    ordered,
    pool_adjacent_violators,
    stationarity_multipliers,
    update_multipliers,
)


class Method(StrEnum):
    unconstrained = "unconstrained"
    reparameterised = "reparameterised"
    projected = "projected"
    lagrangian = "lagrangian"


@dataclass
class Problem:
    x: Tensor
    y: Tensor
    learn_w: bool
    l2: float

    def loss(self, w: Tensor, theta: Tensor) -> Tensor:
        """The immediate-threshold loss of the scores s = xw, plus an L2 penalty on w when w is learnt."""
        penalty = 0.5 * self.l2 * (w**2).sum() if self.learn_w else 0.0
        return immediate_threshold_loss(self.x @ w, self.y, theta) + penalty


@dataclass
class Result:
    method: Method
    w: Tensor
    theta: Tensor
    mu: Tensor
    path_steps: list[int] = field(default_factory=list[int])
    path: list[list[float]] = field(default_factory=list[list[float]])
    mu_path: list[list[float]] = field(default_factory=list[list[float]])


def fit(
    method: Method,
    problem: Problem,
    theta0: Tensor,
    w0: Tensor,
    *,
    steps: int,
    lr: float,
    rho: float,
    rounds: int,
) -> Result:
    """Run `steps` gradient steps; the Lagrangian method updates its multipliers `rounds` times along the way."""
    w = w0.clone().requires_grad_(problem.learn_w)
    theta1 = theta0[:1].clone().requires_grad_()
    delta = inverse_softplus(theta0[1:] - theta0[:-1]).requires_grad_()
    theta = theta0.clone().requires_grad_()
    mu = torch.zeros(len(theta0) - 1, dtype=theta0.dtype)
    result = Result(method, w, theta0, mu)

    def current() -> Tensor:
        return ordered(theta1, delta) if method is Method.reparameterised else theta

    params = [theta1, delta] if method is Method.reparameterised else [theta]
    if problem.learn_w:
        params.append(w)

    per_round = max(1, steps // rounds)
    every = max(1, steps // 200)  # keep about 200 points of each path
    for step in range(steps):
        objective = problem.loss(w, current())
        if method is Method.lagrangian:
            objective = objective + augmented_penalty(current(), mu, rho)
        grads = torch.autograd.grad(objective, params)
        with torch.no_grad():
            for p, g in zip(params, grads, strict=True):
                p -= lr * g
            if method is Method.projected:
                theta.copy_(pool_adjacent_violators(theta))
            if method is Method.lagrangian and (step + 1) % per_round == 0:
                mu = update_multipliers(theta, mu, rho)
                result.mu_path.append(mu.tolist())
        if step % every == 0 or step == steps - 1:
            result.path_steps.append(step + 1)
            result.path.append(current().detach().tolist())

    result.theta = current().detach()
    result.w = w.detach()
    if method is Method.lagrangian:
        result.mu = mu
    else:
        # The multipliers that stationarity asks of this point: meaningful for the projected method, whose point is
        # the constrained optimum. Rounding leaves tiny values on inactive constraints.
        theta_at = result.theta.clone().requires_grad_()
        (grad,) = torch.autograd.grad(problem.loss(result.w, theta_at), [theta_at])
        result.mu = stationarity_multipliers(grad).clamp(min=0)
    return result
