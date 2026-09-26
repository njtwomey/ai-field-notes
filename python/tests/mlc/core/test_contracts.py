import pytest
from pydantic import ValidationError

from mlc.core.contracts import (
    OUTPUT_ADAPTER,
    ContingencyTable,
    ExampleSpec,
    Grid2d,
    HeatmapOutput,
    PointCloud2d,
    Registry,
    RocCurve,
    RunSpec,
    Samples1d,
    Series,
    TableOutput,
    TextOutput,
)


def _example(id: str = "demo", runs: list[RunSpec] | None = None) -> ExampleSpec:
    return ExampleSpec(id=id, title="Demo", module="mlc.examples.demo", runs=runs or [RunSpec(name="a", title="A")])


def test_output_roundtrip_uses_discriminator() -> None:
    raw = OUTPUT_ADAPTER.dump_json(TextOutput(text="hi"))
    parsed = OUTPUT_ADAPTER.validate_json(raw)
    assert isinstance(parsed, TextOutput)
    assert parsed.mime == "text/plain"


def test_unknown_output_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        OUTPUT_ADAPTER.validate_python({"kind": "video", "text": "x"})


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        RunSpec.model_validate({"name": "a", "title": "A", "unexpected": 1})


@pytest.mark.parametrize("slug", ["Upper", "under_score", "-leading", "trailing-", "double--dash"])
def test_bad_slugs_rejected(slug: str) -> None:
    with pytest.raises(ValidationError):
        RunSpec(name=slug, title="A")


def test_module_must_live_under_examples() -> None:
    with pytest.raises(ValidationError):
        ExampleSpec(id="demo", title="Demo", module="os.path", runs=[RunSpec(name="a", title="A")])


def test_duplicate_run_names_rejected() -> None:
    with pytest.raises(ValidationError):
        _example(runs=[RunSpec(name="a", title="A"), RunSpec(name="a", title="B")])


def test_duplicate_example_ids_rejected() -> None:
    with pytest.raises(ValidationError):
        Registry(examples=[_example(), _example()])


def test_table_must_be_rectangular() -> None:
    with pytest.raises(ValidationError):
        TableOutput(columns=["a", "b"], rows=[[1, 2], [3]])


def test_series_lengths_must_match() -> None:
    with pytest.raises(ValidationError):
        Series(name="s", type="line", x=[0, 1], y=[0])
    with pytest.raises(ValidationError):
        Series(name="s", type="scatter", x=[0, 1], y=[0, 1], group=[0])


def test_heatmap_grid_shape() -> None:
    HeatmapOutput(x=[0, 1, 2], y=[0, 1], z=[[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValidationError):
        HeatmapOutput(x=[0, 1, 2], y=[0, 1], z=[[0, 0], [1, 1]])


def test_primitive_shapes_validated() -> None:
    PointCloud2d(x=[0, 1], y=[0, 1], group=[0, 1])
    with pytest.raises(ValidationError):
        Samples1d(x=[0, 1], group=[0])
    with pytest.raises(ValidationError):
        PointCloud2d(x=[0, 1], y=[0])
    with pytest.raises(ValidationError):
        Grid2d(x=[0, 1], y=[0], z=[[0]])
    with pytest.raises(ValidationError):
        ContingencyTable(row_label="actual", col_label="predicted", rows=["a"], cols=["a", "b"], counts=[[1]])
    with pytest.raises(ValidationError):
        RocCurve(fpr=[0, 1], tpr=[0, 1], thresholds=[1], auc=0.5)
