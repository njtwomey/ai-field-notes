"""Data, training and scoring for the models of d(x, y, φ).

The training set is the ray caster itself: rays drawn uniformly over the free floor and all angles, labelled with their
distance (and, for the wall-line classifier, the line they stop on). The test set is whole views: held-out poses, each
with the 48 rays of a 66° field of view, scored by the share of rays within 10 % of the true distance.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from mlc.examples.neural_ray_casting.models import (
    Density,
    FourierMLP,
    Mixture,
    NormalPolicy,
    OccupancyMap,
    Siren,
    Walls,
    WallSegments,
)
from mlc.examples.neural_ray_casting.world import World, cast, ray_offsets, sample_free, traverse, wall_line

RAYS_PER_VIEW = 48


@dataclass(frozen=True)
class Variant:
    """A trained model within a class: how to build it, its learning rate and a description."""

    family: str
    label: str
    build: Callable[[World], nn.Module]
    lr: float


CLASSES = {
    "segments": "learned wall segments + nearest crossing",
    "map": "learned map + exact ray casting",
    "rl": "reinforcement learning",
    "density": "density field + volume rendering",
    "walls": "wall-line classifier",
    "regression": "distance regression",
}

VARIANTS: dict[str, Variant] = {
    "segments-64": Variant("segments", "64 learned segments", lambda w: WallSegments(w, 64), 1e-2),
    "segments-128": Variant("segments", "128 learned segments", lambda w: WallSegments(w, 128), 1e-2),
    "segments-256": Variant("segments", "256 learned segments", lambda w: WallSegments(w, 256), 1e-2),
    "map": Variant("map", "one value per cell", lambda w: OccupancyMap(w), 0.1),
    "map-rl": Variant("rl", "map by REINFORCE", lambda w: OccupancyMap(w), 0.05),
    "walls-rl": Variant("rl", "wall-line policy, REINFORCE, MLP 64 × 2", lambda w: Walls(w), 3e-3),
    "normal-rl": Variant("rl", "facing + distance policy, REINFORCE, MLP 64 × 2", lambda w: NormalPolicy(w), 3e-3),
    "normal-rl-big": Variant(
        "rl",
        "facing + distance policy, REINFORCE, MLP 256 × 4",
        lambda w: NormalPolicy(w, octaves=8, width=256, depth=4),
        1e-3,
    ),
    "walls-grpo": Variant("rl", "wall-line classifier + GRPO (verifiable reward)", lambda w: Walls(w), 1e-4),
    "walls-rl-big": Variant(
        "rl", "wall-line policy, REINFORCE, MLP 256 × 4", lambda w: Walls(w, octaves=8, width=256, depth=4), 1e-3
    ),
    "density-32": Variant("density", "grid to 32 × 32", lambda w: Density(w, levels=(4, 8, 16, 32)), 1e-2),
    "density": Variant("density", "grid to 64 × 64", lambda w: Density(w), 1e-2),
    "density-128": Variant("density", "grid to 128 × 128", lambda w: Density(w, levels=(4, 8, 16, 32, 64, 128)), 1e-2),
    "walls": Variant("walls", "MLP 64 × 2", lambda w: Walls(w), 3e-3),
    "walls-big": Variant("walls", "MLP 256 × 4", lambda w: Walls(w, octaves=8, width=256, depth=4), 1e-3),
    "pe": Variant("regression", "Fourier features + MLP 64 × 2", lambda w: FourierMLP(w), 3e-3),
    "pe-big": Variant(
        "regression", "Fourier features + MLP 256 × 4", lambda w: FourierMLP(w, octaves=8, width=256, depth=4), 1e-3
    ),
    "siren": Variant("regression", "SIREN 128 × 3", lambda w: Siren(w), 1e-4),
    "mixture": Variant("regression", "mixture of 4 distances", lambda w: Mixture(w), 1e-3),
}


@dataclass(frozen=True)
class Data:
    rays: torch.Tensor  # (n, 3) training rays (x, y, φ)
    distance: torch.Tensor  # (n,)
    line: torch.Tensor  # (n,) the grid line each training ray stops on
    test_rays: torch.Tensor  # (views × 48, 3), whole views
    test_distance: torch.Tensor


def make_data(w: World, n_train: int, n_views: int, seed: int = 0) -> Data:
    rng = np.random.default_rng(seed)
    x, y = sample_free(w, n_train, rng)
    a = rng.uniform(-math.pi, math.pi, n_train)
    d = cast(w, x, y, a)
    tx, ty = sample_free(w, n_views, rng)
    heading = rng.uniform(-math.pi, math.pi, n_views)
    vx, vy = np.repeat(tx, RAYS_PER_VIEW), np.repeat(ty, RAYS_PER_VIEW)
    va = (heading[:, None] + ray_offsets(RAYS_PER_VIEW)[None, :]).ravel()

    def stack(*v: np.ndarray) -> torch.Tensor:
        return torch.tensor(np.stack(v, 1), dtype=torch.float32)

    return Data(
        rays=stack(x, y, a),
        distance=torch.tensor(d, dtype=torch.float32),
        line=torch.tensor(wall_line(w, x, y, a, d)),
        test_rays=stack(vx, vy, va),
        test_distance=torch.tensor(cast(w, vx, vy, va), dtype=torch.float32),
    )


def test_views(w: World) -> Data:
    """The held-out views every command scores on: the same draw as training with the default 200 000 rays."""
    return make_data(w, 200_000, 300)


def scores(pred: torch.Tensor | np.ndarray, truth: torch.Tensor | np.ndarray) -> dict[str, float]:
    """Share of rays within 10 % of the truth, median relative error and RMS error in cells."""
    p, t = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    rel = np.abs(p - t) / t
    return {
        "within10": float((rel < 0.1).mean()),
        "median_rel": float(np.median(rel)),
        "rmse": float(np.sqrt(((p - t) ** 2).mean())),
    }


@torch.no_grad()
def predict(model: nn.Module, rays: torch.Tensor, chunk: int = 170 * RAYS_PER_VIEW) -> torch.Tensor:
    model.eval()
    dev = next(model.parameters()).device
    out = torch.cat([model(rays[i : i + chunk].to(dev)).cpu() for i in range(0, len(rays), chunk)])
    model.train()
    return out


def loss(model: nn.Module, rays: torch.Tensor, distance: torch.Tensor, line: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on the line for the wall-line classifier, the mixture likelihood, else L1 on log distance."""
    if isinstance(model, Walls):
        return nn.functional.cross_entropy(model.logits(rays)[0], line)
    if isinstance(model, Mixture):
        return model.nll(rays, distance)
    return (torch.log(model(rays).clamp_min(1e-3)) - torch.log(distance)).abs().mean()


SAMPLES = 32  # maps sampled per REINFORCE step
SHAPING = 0.1  # weight of the straight-wall bonus in the map's reward


def wall_shape(maps: torch.Tensor, w: World) -> torch.Tensor:
    """
    Shaping reward per sampled map: wall cells joined side by side (runs of wall, so straight lines and tight corners)
    count for, wall cells touching only at a corner (a diagonal leak no grid wall has) count against, per cell.
    """
    g = maps.view(-1, w.height, w.width)
    side = (g[:, :, 1:] * g[:, :, :-1]).sum((1, 2)) + (g[:, 1:, :] * g[:, :-1, :]).sum((1, 2))
    a, b, c, d = g[:, :-1, :-1], g[:, :-1, 1:], g[:, 1:, :-1], g[:, 1:, 1:]
    diagonal = (a * d * (1 - b) * (1 - c) + b * c * (1 - a) * (1 - d)).sum((1, 2))
    return (side - diagonal) / (w.width * w.height)


def reinforce_map(
    model: OccupancyMap, w: World, cells: torch.Tensor, ts: torch.Tensor, distance: torch.Tensor
) -> torch.Tensor:
    """
    REINFORCE on the map: sample whole maps cell by cell from P(wall) = σ(θ), ray-cast each exactly (no gradient flows
    through the renderer), reward R = −mean |log d̂ − log d| plus the straight-wall shaping, and step along
    Σ_m (R_m − b_m) ∇ log p(map_m), with b_m the mean reward of the other samples.
    """
    p = torch.sigmoid(model.logit)
    maps = torch.bernoulli(p.detach().expand(SAMPLES, -1))
    solid = torch.cat([maps, torch.ones(SAMPLES, 1)], 1)
    occupied = solid[:, cells] > 0.5  # (samples, rays, steps)
    hit = ts.expand(SAMPLES, -1, -1).gather(2, occupied.float().argmax(2, keepdim=True)).squeeze(2)
    reward = -(torch.log(hit.clamp_min(1e-3)) - torch.log(distance)).abs().mean(1) + SHAPING * wall_shape(maps, w)
    baseline = (reward.sum() - reward) / (SAMPLES - 1)
    log_p = -(nn.functional.softplus(-model.logit) * maps + nn.functional.softplus(model.logit) * (1 - maps)).sum(1)
    return -((reward - baseline).detach() * log_p).mean()


GUESSES = 8  # wall choices sampled per ray in one REINFORCE step
ENTROPY = 3e-2  # weight of the entropy bonus that keeps the wall policy exploring


def reinforce_walls(model: Walls, rays: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """
    The wall-line classifier trained by REINFORCE, as a replacement for the ray caster that never sees the correct wall.

    For each ray the policy π(line | ray) is the classifier's softmax over the lines in front of the camera. Sample
    GUESSES lines per ray; each guess earns r = −|log d_line − log d|, where d_line is the exact distance to the guessed
    line and d the measured distance. The baseline of a guess is the mean reward of the ray's other guesses
    (leave-one-out), so only guesses better than their siblings are reinforced. The step follows
    mean (r − b) ∇ log π(line), plus a small entropy bonus.
    """
    logit, dist = model.logits(rays)
    policy = torch.distributions.Categorical(logits=logit)
    lines = policy.sample((GUESSES,))  # (guesses, rays)
    d_line = dist.T.gather(0, lines)  # distance to each guessed line
    reward = -(torch.log(d_line.clamp_min(1e-3)) - torch.log(distance)[None]).abs()
    baseline = (reward.sum(0, keepdim=True) - reward) / (GUESSES - 1)
    gain = ((reward - baseline).detach() * policy.log_prob(lines)).mean()
    return -gain - ENTROPY * policy.entropy().mean()


def reinforce_normal(model: NormalPolicy, rays: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """
    REINFORCE for the facing + distance policy. Each guess picks a facing from the softmax and a log p from that
    facing's Gaussian; the distance along the ray is p / |n · u|; the reward is −|log d − log d_true|. As for the wall
    policy, each ray gets GUESSES guesses with a leave-one-out baseline, plus an entropy bonus on the facing.
    """
    logit, mean, log_sd, cosine = model.heads(rays)
    facing_policy = torch.distributions.Categorical(logits=logit)
    facing = facing_policy.sample((GUESSES,))  # (guesses, rays)
    m = mean.T.gather(0, facing)
    sd = log_sd.T.gather(0, facing).exp()
    log_p = (m + sd * torch.randn_like(m)).detach()
    log_d = log_p - torch.log(cosine.T.gather(0, facing))
    reward = -(log_d - torch.log(distance)[None]).abs()
    baseline = (reward.sum(0, keepdim=True) - reward) / (GUESSES - 1)
    log_prob = facing_policy.log_prob(facing) + torch.distributions.Normal(m, sd).log_prob(log_p)
    gain = ((reward - baseline).detach() * log_prob).mean()
    return -gain - ENTROPY * facing_policy.entropy().mean()


GRPO_STEPS = 1500  # RL steps after the supervised warm start
GRPO_KL = 0.05  # weight of the KL penalty to the warm-start policy
GRPO_LR = 1e-4  # learning rate of the GRPO phase


def grpo_walls(
    model: Walls, reference: Walls, rays: torch.Tensor, distance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One GRPO step with a verifiable reward. For each ray (the prompt) sample a group of GUESSES walls (the answers);
    the verifier is the geometry: an answer passes (reward 1) when the distance to its line is within 10 % of the true
    distance, else 0. Advantages are standardised within each ray's group, and a KL penalty keeps the policy near the
    supervised model it started from. Returns the loss and the mean reward.
    """
    logit, dist = model.logits(rays)
    policy = torch.distributions.Categorical(logits=logit)
    lines = policy.sample((GUESSES,))
    d_line = dist.T.gather(0, lines)
    reward = ((d_line - distance[None]).abs() / distance[None] < 0.1).float()
    advantage = (reward - reward.mean(0, keepdim=True)) / (reward.std(0, keepdim=True) + 1e-4)
    with torch.no_grad():
        ref_logit, _ = reference.logits(rays)
    kl = torch.distributions.kl_divergence(policy, torch.distributions.Categorical(logits=ref_logit)).mean()
    loss = -(advantage.detach() * policy.log_prob(lines)).mean() + GRPO_KL * kl
    return loss, reward.mean()


def train_grpo(
    w: World, data: "Data", *, steps: int, batch: int, log_every: int
) -> tuple[nn.Module, list[dict[str, float]]]:
    """The supervised wall-line classifier (`steps` steps), then GRPO_STEPS steps of GRPO from it."""
    model, history = train("walls", w, data, steps=steps, batch=batch, log_every=log_every)
    assert isinstance(model, Walls)
    reference = Walls(w)
    reference.load_state_dict(model.state_dict())
    reference.eval()
    opt = torch.optim.Adam(model.parameters(), lr=GRPO_LR)
    start = time.time() - history[-1]["seconds"]
    for step in range(1, GRPO_STEPS + 1):
        i = torch.randint(0, len(data.rays), (batch // GUESSES,))
        value, _ = grpo_walls(model, reference, data.rays[i], data.distance[i])
        opt.zero_grad()
        value.backward()
        opt.step()
        if step % log_every == 0 or step == GRPO_STEPS:
            s = scores(predict(model, data.test_rays), data.test_distance)
            history.append({"step": steps + step, "seconds": time.time() - start, **s})
    return model, history


def train(
    name: str, w: World, data: Data, *, steps: int, batch: int, device: str = "cpu", log_every: int = 500
) -> tuple[nn.Module, list[dict[str, float]]]:
    """Adam with a cosine schedule; density models take an eighth of the batch, since each ray costs K samples."""
    if name == "walls-grpo":
        return train_grpo(w, data, steps=steps, batch=batch, log_every=log_every)
    torch.manual_seed(0)
    variant = VARIANTS[name]
    model = variant.build(w)
    dev = torch.device(device if isinstance(model, Density) else "cpu")
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=variant.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=variant.lr / 20)
    b = batch // 8 if isinstance(model, Density) else batch
    # Map models train on each ray's traversal, which does not depend on the map, so it is computed once.
    path = (torch.empty(0, dtype=torch.long), torch.empty(0))
    if isinstance(model, OccupancyMap):
        cells, ts = traverse(w, data.rays.numpy().astype(float))
        path = (torch.tensor(cells), torch.tensor(ts, dtype=torch.float32))
    if isinstance(model, WallSegments):
        # Start the segments on observed hit points: where rays stopped is where walls are.
        u = torch.stack([torch.cos(data.rays[:, 2]), torch.sin(data.rays[:, 2])], 1)
        model.place(data.rays[:, :2] + data.distance[:, None] * u)
    history = []
    start = time.time()
    for step in range(1, steps + 1):
        i = torch.randint(0, len(data.rays), (b,))
        if isinstance(model, WallSegments):
            # Anneal the crossing tests from soft (every nearby segment gets a gradient) to nearly hard.
            model.sharpness = 0.05 * (0.002 / 0.05) ** (step / steps)
            pred = model.soft(data.rays[i]).clamp_min(1e-3)
            value = (torch.log(pred) - torch.log(data.distance[i])).abs().mean()
        elif name == "map-rl" and isinstance(model, OccupancyMap):
            value = reinforce_map(model, w, path[0][i], path[1][i], data.distance[i])
        elif isinstance(model, NormalPolicy):
            value = reinforce_normal(model, data.rays[i], data.distance[i])
        elif name in ("walls-rl", "walls-rl-big") and isinstance(model, Walls):
            value = reinforce_walls(model, data.rays[i], data.distance[i])
        elif isinstance(model, OccupancyMap):
            pred = model.expected(path[0][i], path[1][i]).clamp_min(1e-3)
            value = (torch.log(pred) - torch.log(data.distance[i])).abs().mean()
        else:
            value = loss(model, data.rays[i].to(dev), data.distance[i].to(dev), data.line[i].to(dev))
        opt.zero_grad()
        value.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            s = scores(predict(model, data.test_rays), data.test_distance)
            history.append({"step": step, "seconds": time.time() - start, **s})
    model.cpu()
    return model, history
