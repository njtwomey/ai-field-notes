import pytest
from pydantic import BaseModel, ValidationError

from mlc.core.figures import REGISTRY, figure


class _Data(BaseModel):
    value: int


def test_figure_registers_and_rejects_duplicates() -> None:
    @figure("test-note/registered", title="t")
    def build() -> _Data:
        return _Data(value=1)

    assert REGISTRY["test-note/registered"].build() == _Data(value=1)
    with pytest.raises(ValueError, match="twice"):
        figure("test-note/registered", title="t")(build)
    del REGISTRY["test-note/registered"]


@pytest.mark.parametrize("bad", ["no-slash", "Upper/case", "a/b_c"])
def test_figure_id_validated(bad: str) -> None:
    with pytest.raises(ValidationError):
        figure(bad, title="t")


def test_figure_requires_model_return_annotation() -> None:
    with pytest.raises(TypeError, match="return type"):

        @figure("test-note/unannotated", title="t")
        def build():  # type: ignore[no-untyped-def]  # pyright: ignore[reportUnusedFunction]
            return _Data(value=1)
