"""Load and check ``python/runs.toml``: the central record of which example backs which note."""

import importlib.util
import tomllib
from pathlib import Path

from mlc.core.contracts import ExampleSpec, Registry
from mlc.core.paths import PACKAGE_ROOT, REGISTRY_FILE, REPO_ROOT


def load_registry(path: Path = REGISTRY_FILE) -> Registry:
    with path.open("rb") as fh:
        return Registry.model_validate(tomllib.load(fh))


def example_dir(example: ExampleSpec) -> Path:
    """The package directory of an example. Every ``.py`` file in it is shown on the site."""
    return PACKAGE_ROOT.joinpath(*example.module.split(".")[1:])


def example_sources(example: ExampleSpec) -> list[Path]:
    directory = example_dir(example)
    if not (directory / "__main__.py").is_file():
        raise FileNotFoundError(f"example {example.id!r}: expected {directory.relative_to(REPO_ROOT)}/__main__.py")
    files = sorted(directory.rglob("*.py"), key=lambda p: (p.name != "__main__.py", p.as_posix()))
    return [f for f in files if "__pycache__" not in f.parts and f.stat().st_size > 0]


def check_registry(registry: Registry) -> list[str]:
    """Return human-readable problems. An empty list means the registry is consistent with the code."""
    problems: list[str] = []
    for example in registry.examples:
        if importlib.util.find_spec(example.module) is None:
            problems.append(f"{example.id}: module {example.module} not found")
            continue
        try:
            example_sources(example)
        except FileNotFoundError as err:
            problems.append(str(err))
    return problems
