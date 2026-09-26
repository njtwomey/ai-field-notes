"""Build figure data registered with ``@figure`` and cache it under ``generated/figures/``."""

import importlib
import json
import pkgutil
from dataclasses import dataclass
from pathlib import Path

import mlc.figures
from mlc.build.hashing import fingerprint
from mlc.core.figures import REGISTRY, FigureBuilder
from mlc.core.paths import FIGURES_DIR


@dataclass(frozen=True)
class FigureResult:
    builder: FigureBuilder
    path: Path
    cached: bool


def discover() -> dict[str, FigureBuilder]:
    """Import every module under ``mlc.figures`` so that its ``@figure`` decorators register."""
    for info in pkgutil.walk_packages(mlc.figures.__path__, prefix="mlc.figures."):
        importlib.import_module(info.name)
    return REGISTRY


def figure_path(figure_id: str) -> Path:
    return FIGURES_DIR / f"{figure_id}.json"


def build(builder: FigureBuilder, *, force: bool = False) -> FigureResult:
    digest = fingerprint([builder.source], builder.id)
    target = figure_path(builder.id)
    stamp = target.with_suffix(".hash")
    if not force and target.exists() and stamp.exists() and stamp.read_text() == digest:
        return FigureResult(builder, target, cached=True)
    data = builder.build().model_dump(mode="json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    stamp.write_text(digest)
    return FigureResult(builder, target, cached=False)
