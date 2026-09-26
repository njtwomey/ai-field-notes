from pathlib import Path

import pytest

from mlc.core import emit
from mlc.core.contracts import OUTPUT_ADAPTER, ChartOutput, MetricsOutput
from mlc.core.paths import RUN_DIR_ENV


def test_emit_prints_without_run_dir(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv(RUN_DIR_ENV, raising=False)
    emit.metrics({"mse": 0.25}, title="Fit")
    assert "mse" in capsys.readouterr().out


def test_emit_records_typed_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(RUN_DIR_ENV, str(tmp_path))
    emit.metrics({"mse": 0.25})
    emit.chart(emit.line("fit", [0, 1], [1, 2]), title="Line")
    lines = (tmp_path / "outputs.jsonl").read_text().splitlines()
    parsed = [OUTPUT_ADAPTER.validate_json(line) for line in lines]
    assert isinstance(parsed[0], MetricsOutput)
    assert isinstance(parsed[1], ChartOutput)
    assert parsed[1].series[0].y == [1.0, 2.0]
