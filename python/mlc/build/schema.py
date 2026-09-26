"""Export the contracts as JSON Schema. The site's TypeScript types are compiled from this file, never hand-written."""

import json
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from mlc.build.figures import discover
from mlc.core.contracts import (
    ContingencyTable,
    Curve,
    CurveSet,
    Grid2d,
    Manifest,
    Point2d,
    PointCloud2d,
    Registry,
    RocCurve,
    RunRecord,
    Samples1d,
)
from mlc.core.paths import REPO_ROOT

SCHEMA_FILE = REPO_ROOT / "site" / "src" / "generated" / "contracts.schema.json"

ROOTS: tuple[type[BaseModel], ...] = (
    Manifest,
    RunRecord,
    Registry,
    Point2d,
    PointCloud2d,
    Samples1d,
    Curve,
    CurveSet,
    Grid2d,
    ContingencyTable,
    RocCurve,
)
"""Contract models the site reads. Registered figure models are added on top. ``$defs`` covers their references."""


def _strip_property_titles(node: Any) -> Any:
    """Drop field-level ``title`` keys. They otherwise become noise aliases (``Title1``, ``Series1``) in TypeScript."""
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():  # pyright: ignore[reportUnknownVariableType]
            if key == "properties" and isinstance(value, dict):
                props: dict[str, Any] = value  # pyright: ignore[reportUnknownVariableType]
                out[key] = {name: _strip_title(_strip_property_titles(prop)) for name, prop in props.items()}
            else:
                out[str(key)] = _strip_property_titles(value)  # pyright: ignore[reportUnknownArgumentType]
        return out
    if isinstance(node, list):
        return [_strip_property_titles(item) for item in node]  # pyright: ignore[reportUnknownVariableType]
    return node


def _strip_title(node: Any) -> Any:
    return {k: v for k, v in node.items() if k != "title"} if isinstance(node, dict) else node  # pyright: ignore


def _roots() -> list[type[BaseModel]]:
    figure_models = {b.model for b in discover().values()} - set(ROOTS)
    return [*ROOTS, *sorted(figure_models, key=lambda m: m.__name__)]


def contracts_schema() -> dict[str, Any]:
    roots = _roots()
    _, schema = models_json_schema([(m, "serialization") for m in roots], ref_template="#/$defs/{model}")
    defs: dict[str, Any] = _strip_property_titles(schema["$defs"])
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Contracts",
        "description": "Generated from python/mlc/core/contracts.py by `uv run mlc schema`. Do not edit.",
        "type": "object",
        "properties": {m.__name__: {"$ref": f"#/$defs/{m.__name__}"} for m in roots},
        "$defs": defs,
    }


def write_schema() -> None:
    SCHEMA_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_FILE.write_text(json.dumps(contracts_schema(), indent=2) + "\n")
