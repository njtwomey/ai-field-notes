import json

from mlc.build.figures import figure_path
from mlc.build.registry import example_sources
from mlc.build.runs import display_command, run_dir
from mlc.core.contracts import ExampleIndex, FigureIndex, Manifest, Registry, RunIndex, RunRecord
from mlc.core.figures import FigureBuilder
from mlc.core.paths import GENERATED_DIR, MANIFEST_FILE, REPO_ROOT


def write_manifest(registry: Registry, figures: dict[str, FigureBuilder]) -> Manifest:
    examples: dict[str, ExampleIndex] = {}
    for example in registry.examples:
        runs: list[RunIndex] = []
        for run in example.runs:
            record_file = run_dir(example, run) / "run.json"
            exit_code = RunRecord.model_validate_json(record_file.read_text()).exit_code if record_file.exists() else -1
            runs.append(
                RunIndex(
                    name=run.name,
                    title=run.title,
                    description=run.description,
                    command=display_command(example, run),
                    exit_code=exit_code,
                    path=record_file.relative_to(GENERATED_DIR).as_posix(),
                )
            )
        examples[example.id] = ExampleIndex(
            id=example.id,
            title=example.title,
            module=example.module,
            sources=[p.relative_to(REPO_ROOT).as_posix() for p in example_sources(example)],
            runs=runs,
        )
    manifest = Manifest(
        examples=examples,
        figures={
            fid: FigureIndex(id=fid, title=b.title, path=figure_path(fid).relative_to(GENERATED_DIR).as_posix())
            for fid, b in sorted(figures.items())
        },
    )
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(manifest.model_dump(mode="json"), indent=1) + "\n")
    return manifest
