"""Repository locations. All paths are absolute."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "python"
PACKAGE_ROOT = PYTHON_ROOT / "mlc"
CORE_ROOT = PACKAGE_ROOT / "core"
REGISTRY_FILE = PYTHON_ROOT / "runs.toml"
PALETTE_FILE = REPO_ROOT / "design" / "palette.json"
GENERATED_DIR = REPO_ROOT / "site" / "public" / "generated"
RUNS_DIR = GENERATED_DIR / "runs"
FIGURES_DIR = GENERATED_DIR / "figures"
MANIFEST_FILE = GENERATED_DIR / "manifest.json"

RUN_DIR_ENV = "MLC_RUN_DIR"
"""Set by the runner. When present, ``emit`` also appends structured records to ``$MLC_RUN_DIR/outputs.jsonl``."""
