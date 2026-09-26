"""Registry of figure-data builders.

A figure builder computes data that an interactive diagram on the site needs but should not compute in the browser,
e.g. a loss surface on a fine grid. Builders live in ``mlc.figures.<topic>`` and return a pydantic model: a
primitive from ``mlc.core.contracts`` (``PointCloud2d``, ``Grid2d``, ``CurveSet``, ...) or a small model composed of
them. The build writes each result to ``generated/figures/<id>.json``; the site loads it with ``useFigure(id)``.
"""

import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from mlc.core.contracts import FigureId

_FIGURE_ID: TypeAdapter[str] = TypeAdapter(FigureId)


@dataclass(frozen=True)
class FigureBuilder:
    id: str
    title: str
    build: Callable[[], BaseModel]
    model: type[BaseModel]
    """The builder's return type. Exported to JSON Schema so the site gets generated TypeScript types for it."""
    source: Path


REGISTRY: dict[str, FigureBuilder] = {}


def figure[M: BaseModel](id: str, *, title: str) -> Callable[[Callable[[], M]], Callable[[], M]]:
    """Register a zero-argument builder under ``id`` (``<note-slug>/<name>``)."""
    _FIGURE_ID.validate_python(id)

    def register(fn: Callable[[], M]) -> Callable[[], M]:
        if id in REGISTRY:
            raise ValueError(f"figure {id!r} registered twice")
        model = typing.get_type_hints(fn).get("return")
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError(f"figure {id!r}: annotate the return type with a pydantic model")
        REGISTRY[id] = FigureBuilder(id=id, title=title, build=fn, model=model, source=Path(fn.__code__.co_filename))
        return fn

    return register
