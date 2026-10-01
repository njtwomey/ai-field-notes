"""Write golden test values for aifn-js from Python references (numpy, scipy, scikit-learn, torch).

Fixtures mirror the module tree (aifn-js/modules.json). The generator `gen/<node path>.py` (e.g.
`gen/numerics/linalg.py`) defines `cases() -> dict`; its result is written to `<node path>.json` beside the `gen`
folder (`numerics/linalg.json`), which tests load with `fixture('numerics/linalg')`. The generators live in both
packages: `aifn-js/core/test/fixtures/gen` and `aifn-js/applications/test/fixtures/gen`. Run with `make fixtures`
(or `uv run python aifn-js/core/test/fixtures/generate.py [name ...]`); a name is a node path (`numerics/linalg`), a
prefix of one (`numerics`) or a last segment (`linalg`).
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
    wanted = [w.strip("/") for w in sys.argv[1:]]
    found = False
    for fixtures in FIXTURE_DIRS:
        for path in sorted((fixtures / "gen").rglob("*.py")):
            name = path.relative_to(fixtures / "gen").with_suffix("").as_posix()
            if any(part.startswith("_") for part in name.split("/")):
                continue
            if wanted and not any(name == w or name.startswith(f"{w}/") or path.stem == w for w in wanted):
                continue
            found = True
            spec = importlib.util.spec_from_file_location(f"fixtures_{name.replace('/', '_').replace('-', '_')}", path)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            out = fixtures / f"{name}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(to_json(module.cases()), indent=1) + "\n")
            print(f"wrote {out.relative_to(ROOT)}")
    if wanted and not found:
        sys.exit(f"no fixture generator matches {' '.join(wanted)}")


if __name__ == "__main__":
    main()
