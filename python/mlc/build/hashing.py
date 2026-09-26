import hashlib
from collections.abc import Iterable
from pathlib import Path

from mlc.core.paths import CORE_ROOT, REPO_ROOT


def fingerprint(files: Iterable[Path], *extra: str) -> str:
    """Stable hash of file contents plus extra strings. ``mlc.core`` and ``uv.lock`` are always included."""
    digest = hashlib.sha256()
    lock = REPO_ROOT / "uv.lock"
    core = sorted(p for p in CORE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    paths: set[Path] = {*files, *core, *([lock] if lock.exists() else [])}
    for path in sorted(paths, key=Path.as_posix):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    for item in extra:
        digest.update(b"\0" + item.encode())
    return digest.hexdigest()[:16]
