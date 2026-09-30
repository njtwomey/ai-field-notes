"""Write golden test values for aifn-js from Python references (numpy, scipy, scikit-learn, torch).

Each module `gen/<name>.py` defines `cases() -> dict`; its result is written to `<name>.json` beside its `gen` folder.
The generators live in both packages: `aifn-js/core/test/fixtures/gen` and `aifn-js/applications/test/fixtures/gen`.
Run with `make fixtures` (or `uv run python aifn-js/core/test/fixtures/generate.py [name ...]`).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[3]
FIXTURE_DIRS = [HERE, ROOT / "aifn-js" / "applications" / "test" / "fixtures"]


def to_json(value: object) -> object:
    """Convert numpy values to JSON; non-finite floats become the strings "inf", "-inf" and "nan"."""
    if isinstance(value, np.ndarray):
        return to_json(value.tolist())
    if isinstance(value, np.generic):
        return to_json(value.item())
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def main() -> None:
    wanted = set(sys.argv[1:])
    for path in sorted(p for d in FIXTURE_DIRS for p in (d / "gen").glob("*.py")):
        name = path.stem
        if name.startswith("_") or (wanted and name not in wanted):
            continue
        spec = importlib.util.spec_from_file_location(f"fixtures_{name}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = path.parent.parent / f"{name}.json"
        out.write_text(json.dumps(to_json(module.cases()), indent=1) + "\n")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
