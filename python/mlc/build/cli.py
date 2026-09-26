"""``uv run mlc <command>``: build the generated assets the site reads."""

import sys

from pydantic import Field
from pydantic_settings import CliApp, CliSubCommand

from mlc.build import figures as figure_build
from mlc.build import runs as run_build
from mlc.build.manifest import write_manifest
from mlc.build.registry import check_registry, load_registry
from mlc.build.schema import SCHEMA_FILE, write_schema
from mlc.core.cli import Command

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


def _say(ok: bool, message: str) -> None:
    print(f"{GREEN if ok else RED}{'✓' if ok else '✗'} {message}{RESET}")


class Selection(Command):
    """Options shared by the build commands."""

    force: bool = Field(False, description="Ignore the cache and rebuild everything selected.")
    only: list[str] = Field(
        default_factory=list[str],
        description="Example id or figure id prefix to build. Repeatable. Default: everything.",
    )

    def selected(self, key: str) -> bool:
        return not self.only or any(key == o or key.startswith(f"{o}/") for o in self.only)


class Check(Command):
    """Validate python/runs.toml against the example packages."""

    def cli_cmd(self) -> None:
        problems = check_registry(load_registry())
        for problem in problems:
            _say(False, problem)
        if problems:
            sys.exit(1)
        _say(True, "registry ok")


class List(Command):
    """List registered examples, their runs and the figure builders."""

    def cli_cmd(self) -> None:
        for example in load_registry().examples:
            print(f"{example.id}  ({example.module})")
            for run in example.runs:
                print(f"  - {run.name}: {run_build.display_command(example, run)}")
        for fid in sorted(figure_build.discover()):
            print(f"figure {fid}")


class Runs(Selection):
    """Execute example runs (cached by source hash) and refresh the manifest."""

    def cli_cmd(self) -> None:
        registry = load_registry()
        failed = 0
        for example in registry.examples:
            if not self.selected(example.id):
                continue
            for run in example.runs:
                result = run_build.execute(example, run, force=self.force)
                ok = result.record.exit_code == 0
                failed += not ok
                _say(ok, f"{example.id}/{run.name}  [{'cached' if result.cached else f'{result.record.duration_s}s'}]")
                if not ok:
                    print(result.record.stderr[-2000:])
        write_manifest(registry, figure_build.discover())
        if failed:
            sys.exit(1)


class Figures(Selection):
    """Build figure data (cached by source hash) and refresh the manifest."""

    def cli_cmd(self) -> None:
        builders = figure_build.discover()
        for fid, builder in sorted(builders.items()):
            if self.selected(fid):
                result = figure_build.build(builder, force=self.force)
                _say(True, f"{fid}  [{'cached' if result.cached else 'built'}]")
        write_manifest(load_registry(), builders)


class Schema(Command):
    """Write the contracts JSON Schema. `make contracts` compiles it to TypeScript."""

    def cli_cmd(self) -> None:
        write_schema()
        _say(True, f"wrote {SCHEMA_FILE.name}")


class Build(Selection):
    """Check the registry, write the schema, then build figures and runs."""

    def cli_cmd(self) -> None:
        Check().cli_cmd()
        Schema().cli_cmd()
        Figures(force=self.force, only=self.only).cli_cmd()
        Runs(force=self.force, only=self.only).cli_cmd()


class Mlc(Command):
    """Build the generated assets (example outputs and figure data) for the ML concepts site."""

    check: CliSubCommand[Check] = Field(description=Check.__doc__)
    list: CliSubCommand[List] = Field(description=List.__doc__)
    runs: CliSubCommand[Runs] = Field(description=Runs.__doc__)
    figures: CliSubCommand[Figures] = Field(description=Figures.__doc__)
    schema_: CliSubCommand[Schema] = Field(alias="schema", description=Schema.__doc__)
    build: CliSubCommand[Build] = Field(description=Build.__doc__)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


def app() -> None:
    Mlc.main()
