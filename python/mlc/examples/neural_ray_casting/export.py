"""Export trained models to ONNX for the browser, with one contract for every class:

    input  rays:     float32 [N, 3]  (x, y, φ) in cells and radians
    output distance: float32 [N]     distance to the first wall along each ray, in cells

Everything class-specific (the encoding, the wall-line mask and argmax, the volume rendering) is inside the graph, so
the page runs every model the same way.
"""

from pathlib import Path

import numpy as np
import onnx  # pyright: ignore[reportMissingImports]  (uv group "onnx", for exporting only)
import onnxruntime as ort  # pyright: ignore[reportMissingImports]
import torch
from torch import nn


def to_onnx(model: nn.Module, example_rays: torch.Tensor, path: Path, metadata: dict[str, str]) -> float:
    """Write the model to `path` and return the largest difference between ONNX Runtime and PyTorch on the rays."""
    model.eval()
    torch.onnx.export(
        model,
        (example_rays[:48],),
        str(path),
        input_names=["rays"],
        output_names=["distance"],
        dynamic_axes={"rays": {0: "n"}, "distance": {0: "n"}},
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(str(path))
    for key, value in metadata.items():
        graph.metadata_props.add(key=key, value=value)
    onnx.save(graph, str(path))
    with torch.no_grad():
        ref = model(example_rays).numpy()
    got = run(path, example_rays.numpy())
    return float(np.abs(got - ref).max())


def run(path: Path, rays: np.ndarray) -> np.ndarray:
    """Distances from an exported model, with ONNX Runtime."""
    out = ort.InferenceSession(str(path)).run(None, {"rays": rays.astype(np.float32)})[0]
    return np.asarray(out)
