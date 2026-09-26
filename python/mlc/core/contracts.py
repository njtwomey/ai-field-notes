"""Typed contracts shared by the example runner, the figure builder and the site.

Everything the site reads from ``site/public/generated/`` is serialised from these models. ``make contracts`` exports
them as JSON Schema and compiles ``site/src/generated/contracts.ts``. Never write the TypeScript side by hand.
"""

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter, model_validator

type Slug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
type FigureId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:/[a-z0-9]+(?:-[a-z0-9]+)*)+$")]


class Strict(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, use_attribute_docstrings=True, json_schema_serialization_defaults_required=True
    )


# --------------------------------------------------------------------------------------------------------------------
# Registry: python/runs.toml
# --------------------------------------------------------------------------------------------------------------------


class RunSpec(Strict):
    """One CLI invocation of an example. Its outputs are cached as static assets."""

    name: Slug
    title: str
    description: str = ""
    args: list[str] = Field(default_factory=list[str])


class ExampleSpec(Strict):
    """A runnable implementation attached to a note. ``id`` is referenced by a note's ``code`` frontmatter field."""

    id: Slug
    title: str
    module: str = Field(pattern=r"^mlc\.examples\.[a-z_][a-z0-9_]*$")
    runs: list[RunSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_runs(self) -> Self:
        names = [r.name for r in self.runs]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate run names in example {self.id!r}: {names}")
        return self


class Registry(Strict):
    examples: list[ExampleSpec]

    @model_validator(mode="after")
    def _unique_examples(self) -> Self:
        ids = [e.id for e in self.examples]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate example ids: {ids}")
        return self


# --------------------------------------------------------------------------------------------------------------------
# Figure-data primitives: generic shapes that figure builders return or compose. Prefer these over one-off models so
# that site components (XYChart, Heatmap, ...) can consume them directly.
# --------------------------------------------------------------------------------------------------------------------


def _check_lengths(owner: str, **columns: list[float] | list[int] | None) -> None:
    lengths = {name: len(col) for name, col in columns.items() if col is not None}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"{owner}: column lengths differ: {lengths}")


class Point2d(Strict):
    """A single point, e.g. an optimum in parameter space. Use instead of a tuple: tuples lose their types in the
    generated TypeScript."""

    x: float
    y: float


class PointCloud2d(Strict):
    """Points in the plane, optionally labelled with a categorical group per point."""

    x: list[float]
    y: list[float]
    group: list[int] | None = None
    """Categorical palette slot per point."""
    group_names: list[str] | None = None
    """Display name per group index."""

    @model_validator(mode="after")
    def _lengths(self) -> Self:
        _check_lengths("PointCloud2d", x=self.x, y=self.y, group=self.group)
        return self


class Samples1d(Strict):
    """Scalar samples, optionally labelled with a categorical group per sample, e.g. a 1-D classification dataset."""

    x: list[float]
    group: list[int] | None = None
    """Categorical palette slot per sample."""
    group_names: list[str] | None = None
    """Display name per group index."""

    @model_validator(mode="after")
    def _lengths(self) -> Self:
        _check_lengths("Samples1d", x=self.x, group=self.group)
        return self


class Curve(Strict):
    """One named y-against-x line, e.g. a training-loss curve."""

    name: str
    x: list[float]
    y: list[float]

    @model_validator(mode="after")
    def _lengths(self) -> Self:
        _check_lengths(f"Curve {self.name!r}", x=self.x, y=self.y)
        return self


class CurveSet(Strict):
    """Curves that share axes, e.g. train and validation loss."""

    x_label: str
    y_label: str
    curves: list[Curve] = Field(min_length=1)


class Grid2d(Strict):
    """Values on a regular grid, e.g. a loss surface or a probability map."""

    x: list[float]
    y: list[float]
    z: list[list[float]]
    """Row-major: ``z[i][j]`` is the value at ``(x[j], y[i])``."""
    x_label: str = "x"
    y_label: str = "y"
    z_label: str = "value"

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if len(self.z) != len(self.y) or any(len(row) != len(self.x) for row in self.z):
            raise ValueError("z must have shape (len(y), len(x))")
        return self


class ContingencyTable(Strict):
    """Counts cross-tabulated by two categorical variables. A confusion matrix is the special case rows=actual."""

    row_label: str
    col_label: str
    rows: list[str]
    cols: list[str]
    counts: list[list[int]]

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if len(self.counts) != len(self.rows) or any(len(r) != len(self.cols) for r in self.counts):
            raise ValueError("counts must have shape (len(rows), len(cols))")
        return self


class RocCurve(Strict):
    """Receiver operating characteristic: true-positive rate against false-positive rate over thresholds."""

    fpr: list[float]
    tpr: list[float]
    thresholds: list[float]
    auc: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _lengths(self) -> Self:
        _check_lengths("RocCurve", fpr=self.fpr, tpr=self.tpr, thresholds=self.thresholds)
        return self


# --------------------------------------------------------------------------------------------------------------------
# Output records: one JSON line per ``emit.*`` call
# --------------------------------------------------------------------------------------------------------------------


class _Output(Strict):
    title: str | None = None


class TextOutput(_Output):
    kind: Literal["text"] = "text"
    mime: Literal["text/plain"] = "text/plain"
    text: str


class MetricsOutput(_Output):
    kind: Literal["metrics"] = "metrics"
    mime: Literal["application/vnd.mlc.metrics+json"] = "application/vnd.mlc.metrics+json"
    values: dict[str, float | int | str]


type Cell = float | int | str | None


class TableOutput(_Output):
    kind: Literal["table"] = "table"
    mime: Literal["application/vnd.mlc.table+json"] = "application/vnd.mlc.table+json"
    columns: list[str]
    rows: list[list[Cell]]

    @model_validator(mode="after")
    def _rectangular(self) -> Self:
        width = len(self.columns)
        if any(len(row) != width for row in self.rows):
            raise ValueError("every table row must have one cell per column")
        return self


class Series(Strict):
    name: str
    type: Literal["scatter", "line"]
    x: list[float]
    y: list[float]
    group: list[int] | None = None
    """Optional categorical slot index per point (scatter only). Colours follow design/palette.json."""

    @model_validator(mode="after")
    def _same_length(self) -> Self:
        if len(self.x) != len(self.y):
            raise ValueError(f"series {self.name!r}: x and y differ in length")
        if self.group is not None and len(self.group) != len(self.x):
            raise ValueError(f"series {self.name!r}: group and x differ in length")
        return self


class ChartOutput(_Output):
    kind: Literal["chart"] = "chart"
    mime: Literal["application/vnd.mlc.chart+json"] = "application/vnd.mlc.chart+json"
    x_label: str = "x"
    y_label: str = "y"
    series: list[Series] = Field(min_length=1)


class HeatmapOutput(_Output):
    kind: Literal["heatmap"] = "heatmap"
    mime: Literal["application/vnd.mlc.heatmap+json"] = "application/vnd.mlc.heatmap+json"
    x: list[float]
    y: list[float]
    z: list[list[float]]
    """Row-major: ``z[i][j]`` is the value at ``(x[j], y[i])``."""
    x_label: str = "x"
    y_label: str = "y"
    overlay: list[Series] = Field(default_factory=list[Series])

    @model_validator(mode="after")
    def _grid_shape(self) -> Self:
        if len(self.z) != len(self.y) or any(len(row) != len(self.x) for row in self.z):
            raise ValueError("z must have shape (len(y), len(x))")
        return self


class FileOutput(_Output):
    """A binary or large asset written next to the run record, e.g. a PNG."""

    kind: Literal["file"] = "file"
    mime: str
    path: str
    """Relative to the run directory."""


type Output = Annotated[
    TextOutput | MetricsOutput | TableOutput | ChartOutput | HeatmapOutput | FileOutput,
    Field(discriminator="kind"),
]


OUTPUT_ADAPTER: TypeAdapter[Output] = TypeAdapter(Output)
"""Parses and serialises one line of ``outputs.jsonl``."""


# --------------------------------------------------------------------------------------------------------------------
# Build artefacts
# --------------------------------------------------------------------------------------------------------------------


class RunRecord(Strict):
    """Written to ``generated/runs/<example>/<run>/run.json``."""

    example: Slug
    run: Slug
    title: str
    description: str
    command: str
    exit_code: int
    duration_s: float
    hash: str
    created: str
    stdout: str
    stderr: str
    outputs: list[Output]


class RunIndex(Strict):
    name: Slug
    title: str
    description: str
    command: str
    exit_code: int
    path: str
    """Relative to ``generated/``."""


class ExampleIndex(Strict):
    id: Slug
    title: str
    module: str
    sources: list[str]
    """Repo-relative paths of every source file, in display order."""
    runs: list[RunIndex]


class FigureIndex(Strict):
    id: FigureId
    title: str
    path: str
    """Relative to ``generated/``."""


class Manifest(Strict):
    """Written to ``generated/manifest.json``. The site's single entry point to generated assets."""

    examples: dict[str, ExampleIndex]
    figures: dict[str, FigureIndex]
