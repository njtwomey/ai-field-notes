"""Execute registered example runs and cache their outputs under ``generated/runs/``."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mlc.build.hashing import fingerprint
from mlc.build.registry import example_sources
from mlc.core.contracts import OUTPUT_ADAPTER, ExampleSpec, RunRecord, RunSpec
from mlc.core.paths import REPO_ROOT, RUN_DIR_ENV, RUNS_DIR

TIMEOUT_S = 600


@dataclass(frozen=True)
class RunResult:
    record: RunRecord
    cached: bool


def run_dir(example: ExampleSpec, run: RunSpec) -> Path:
    return RUNS_DIR / example.id / run.name


def display_command(example: ExampleSpec, run: RunSpec) -> str:
    return shlex.join(["uv", "run", "python", "-m", example.module, *run.args])


def _cached(target: Path, digest: str) -> RunRecord | None:
    record_file = target / "run.json"
    if not record_file.exists():
        return None
    record = RunRecord.model_validate_json(record_file.read_text())
    return record if record.hash == digest and record.exit_code == 0 else None


def execute(example: ExampleSpec, run: RunSpec, *, force: bool = False) -> RunResult:
    digest = fingerprint(example_sources(example), example.module, *run.args)
    target = run_dir(example, run)
    if not force and (record := _cached(target, digest)):
        return RunResult(record, cached=True)

    with tempfile.TemporaryDirectory(prefix="mlc-run-") as tmp:
        scratch = Path(tmp)
        env = {**os.environ, RUN_DIR_ENV: str(scratch), "PYTHONHASHSEED": "0"}
        started = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", example.module, *run.args],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            check=False,
        )
        duration = time.perf_counter() - started
        outputs_file = scratch / "outputs.jsonl"
        lines = outputs_file.read_text().splitlines() if outputs_file.exists() else []
        record = RunRecord(
            example=example.id,
            run=run.name,
            title=run.title,
            description=run.description,
            command=display_command(example, run),
            exit_code=proc.returncode,
            duration_s=round(duration, 3),
            hash=digest,
            created=datetime.now(UTC).isoformat(timespec="seconds"),
            stdout=proc.stdout,
            stderr=proc.stderr,
            outputs=[OUTPUT_ADAPTER.validate_json(line) for line in lines if line.strip()],
        )
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        for asset in scratch.iterdir():
            if asset.name != "outputs.jsonl":
                shutil.copy2(asset, target / asset.name)
        (target / "run.json").write_text(json.dumps(record.model_dump(mode="json"), indent=1) + "\n")
    return RunResult(record, cached=False)
