import pytest
from pydantic import Field

from mlc.core.cli import Command


class _Echo(Command):
    """Echo options."""

    k_max: int = Field(3, description="Largest k.")
    verbose: bool = Field(False, description="Print more.")
    result: list[str] = Field(default_factory=list[str], description="Captured calls.")

    def cli_cmd(self) -> None:
        self.result.append(f"{self.k_max}:{self.verbose}")


def test_kebab_case_and_implicit_flags() -> None:
    cmd = _Echo.main(["--k-max", "5", "--verbose"])
    assert (cmd.k_max, cmd.verbose, cmd.result) == (5, True, ["5:True"])


def test_environment_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K_MAX", "9")
    assert _Echo.main([]).k_max == 3


def test_undocumented_field_rejected() -> None:
    with pytest.raises(TypeError, match="description"):

        class _Bad(Command):  # pyright: ignore[reportUnusedClass]
            """Bad."""

            k: int = 3


def test_missing_docstring_rejected() -> None:
    with pytest.raises(TypeError, match="docstring"):

        class _Bad(Command):  # pyright: ignore[reportUnusedClass]
            k: int = Field(3, description="k.")
