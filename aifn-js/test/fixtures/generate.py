"""Write golden test values for aifn-js from Python references (numpy, scipy, scikit-learn, torch).

Each module `gen/<name>.py` defines `cases() -> dict`; its result is written to `<name>.json` next to this file.
Run with `make fixtures` (or `uv run python aifn-js/test/fixtures/generate.py [name ...]`).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


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
    for path in sorted((HERE / "gen").glob("*.py")):
        name = path.stem
        if name.startswith("_") or (wanted and name not in wanted):
            continue
        spec = importlib.util.spec_from_file_location(f"fixtures_{name}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out = HERE / f"{name}.json"
        out.write_text(json.dumps(to_json(module.cases()), indent=1) + "\n")
        print(f"wrote {out.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    main()
