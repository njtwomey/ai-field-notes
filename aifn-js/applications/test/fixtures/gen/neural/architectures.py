"""Golden values for aifn-applied/neural/architectures: a transformer block evaluated in torch with
torch.nn.functional's layer_norm, rms_norm, gelu (exact) and scaled_dot_product_attention, from fixed random
parameters in aifn's layout (Linear weights [in, out], applied as x W + b). Cases cover pre- and post-norm, layer and
RMS norm, causal and full attention, with the output and the gradient of Σ y ⊙ r in x."""

import numpy as np
import torch
import torch.nn.functional as F

D, HEADS, HIDDEN, BATCH, T = 8, 2, 16, 2, 5


def params(rng: np.random.Generator) -> dict[str, object]:
    def lin(i: int, o: int) -> dict[str, np.ndarray]:
        return {"weight": rng.normal(size=(i, o)) / np.sqrt(i), "bias": 0.1 * rng.normal(size=o)}

    def norm() -> dict[str, np.ndarray]:
        return {"gamma": 1 + 0.2 * rng.normal(size=D), "beta": 0.1 * rng.normal(size=D)}

    return {
        "attentionNorm": norm(),
        "attention": {"query": lin(D, D), "key": lin(D, D), "value": lin(D, D), "output": lin(D, D)},
        "feedForwardNorm": norm(),
        "feedForward": {"up": lin(D, HIDDEN), "down": lin(HIDDEN, D)},
    }


def block(p, x: torch.Tensor, placement: str, norm: str, causal: bool) -> torch.Tensor:
    t = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731

    def normalise(np_, h):
        if norm == "rms":
            return F.rms_norm(h, (D,), weight=t(np_["gamma"]), eps=1e-6)
        return F.layer_norm(h, (D,), weight=t(np_["gamma"]), bias=t(np_["beta"]), eps=1e-5)

    def linear(h, lp):
        return h @ t(lp["weight"]) + t(lp["bias"])

    def attention(h):
        a = p["attention"]
        split = lambda u: u.reshape(*u.shape[:-1], HEADS, D // HEADS).transpose(-3, -2)  # noqa: E731
        q, k, v = (split(linear(h, a[name])) for name in ("query", "key", "value"))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return linear(o.transpose(-3, -2).reshape(*h.shape), a["output"])

    def feed_forward(h):
        ff = p["feedForward"]
        return linear(F.gelu(linear(h, ff["up"])), ff["down"])

    def branch(f, np_, h):
        return h + f(normalise(np_, h)) if placement == "pre" else normalise(np_, h + f(h))

    h = branch(attention, p["attentionNorm"], x)
    return branch(feed_forward, p["feedForwardNorm"], h)


def cases() -> dict[str, object]:
    rng = np.random.default_rng(20261001)
    x = rng.normal(size=(BATCH, T, D))
    r = rng.normal(size=(BATCH, T, D))
    out: dict[str, object] = {"x": x, "r": r, "dModel": D, "heads": HEADS, "hidden": HIDDEN, "blocks": []}
    for placement, norm, causal in [("pre", "layer", False), ("post", "layer", True), ("pre", "rms", True)]:
        p = params(rng)
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        y = block(p, xt, placement, norm, causal)
        (y * torch.tensor(r, dtype=torch.float64)).sum().backward()
        assert xt.grad is not None
        out["blocks"].append(  # type: ignore[union-attr]
            {
                "placement": placement,
                "norm": norm,
                "causal": causal,
                "params": p,
                "y": y.detach().numpy(),
                "gradX": xt.grad.numpy(),
            }
        )
    return out
