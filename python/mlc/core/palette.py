"""Data colours from design/palette.json, for any Python-rendered asset."""

import json
from functools import cache

from pydantic import BaseModel

from mlc.core.paths import PALETTE_FILE


class _Modes(BaseModel):
    light: list[str]
    dark: list[str]


class Palette(BaseModel):
    categorical: _Modes
    sequential: list[str]


@cache
def palette() -> Palette:
    return Palette.model_validate(json.loads(PALETTE_FILE.read_text()))
