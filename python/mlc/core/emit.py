"""Record structured outputs from an example CLI.

Every call prints a human-readable form to stdout, so examples behave like normal programs when run by hand. Under
``mlc build`` the runner sets ``MLC_RUN_DIR`` and each call also appends one typed record to ``outputs.jsonl``. The site
renders those records in the note's Outputs tab.
"""

import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from mlc.core.contracts import (
    OUTPUT_ADAPTER,
    Cell,
    ChartOutput,
    FileOutput,
    HeatmapOutput,
    MetricsOutput,
    Output,
    Series,
    TableOutput,
    TextOutput,
)
from mlc.core.paths import RUN_DIR_ENV

__all__ = ["chart", "file", "heatmap", "line", "metrics", "scatter", "table", "text"]


def _run_dir() -> Path | None:
    value = os.environ.get(RUN_DIR_ENV)
    return Path(value) if value else None


def _record(output: Output) -> None:
    run_dir = _run_dir()
    if run_dir is None:
        return
    with (run_dir / "outputs.jsonl").open("ab") as fh:
        fh.write(OUTPUT_ADAPTER.dump_json(output) + b"\n")


def _heading(title: str | None) -> None:
    if title:
        print(f"\n## {title}")


def _floats(values: ArrayLike) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def text(body: str, *, title: str | None = None) -> None:
    _heading(title)
    print(body)
    _record(TextOutput(title=title, text=body))


def metrics(values: Mapping[str, float | int | str], *, title: str | None = None) -> None:
    clean = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in values.items()}
    _heading(title)
    width = max((len(k) for k in clean), default=0)
    for key, value in clean.items():
        print(f"{key:<{width}}  {value:.6g}" if isinstance(value, float) else f"{key:<{width}}  {value}")
    _record(MetricsOutput(title=title, values=clean))


def table(columns: Sequence[str], rows: Sequence[Sequence[Cell]], *, title: str | None = None) -> None:
    _heading(title)
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join("" if c is None else f"{c:.6g}" if isinstance(c, float) else str(c) for c in row))
    _record(TableOutput(title=title, columns=list(columns), rows=[list(r) for r in rows]))


def scatter(name: str, x: ArrayLike, y: ArrayLike, group: ArrayLike | None = None) -> Series:
    groups = None if group is None else [int(g) for g in np.asarray(group).ravel()]
    return Series(name=name, type="scatter", x=_floats(x), y=_floats(y), group=groups)


def line(name: str, x: ArrayLike, y: ArrayLike) -> Series:
    return Series(name=name, type="line", x=_floats(x), y=_floats(y))


def chart(*series: Series, title: str | None = None, x_label: str = "x", y_label: str = "y") -> None:
    _heading(title)
    print(f"[chart: {', '.join(f'{s.name} ({len(s.x)} pts)' for s in series)}]")
    _record(ChartOutput(title=title, x_label=x_label, y_label=y_label, series=list(series)))


def heatmap(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    *,
    title: str | None = None,
    x_label: str = "x",
    y_label: str = "y",
    overlay: Sequence[Series] = (),
) -> None:
    grid = np.asarray(z, dtype=float)
    _heading(title)
    print(f"[heatmap: {grid.shape[0]}x{grid.shape[1]}, range {grid.min():.4g} .. {grid.max():.4g}]")
    _record(
        HeatmapOutput(
            title=title,
            x=_floats(x),
            y=_floats(y),
            z=[_floats(row) for row in grid],
            x_label=x_label,
            y_label=y_label,
            overlay=list(overlay),
        )
    )


def file(source: Path, *, mime: str, title: str | None = None) -> None:
    """Copy ``source`` into the run directory, e.g. a PNG produced by another library."""
    _heading(title)
    print(f"[file: {source.name} ({mime})]")
    run_dir = _run_dir()
    if run_dir is None:
        return
    shutil.copyfile(source, run_dir / source.name)
    _record(FileOutput(title=title, mime=mime, path=source.name))
