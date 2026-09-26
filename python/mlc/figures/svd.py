import numpy as np

from mlc.core.contracts import Strict
from mlc.core.figures import figure
from mlc.examples.svd.model import make_image


class ImageSvd(Strict):
    """A test image and its thin SVD. The site rebuilds any rank-k approximation from these factors."""

    image: list[list[float]]
    """Row-major, first row at the top, values in [0, 1]."""
    u: list[list[float]]
    """Shape (size, size): left singular vectors as columns."""
    s: list[float]
    """Singular values, largest first."""
    vt: list[list[float]]
    """Shape (size, size): right singular vectors as rows."""


@figure("singular-value-decomposition/image", title="A 64 × 64 test image and its SVD")
def image_svd() -> ImageSvd:
    image = make_image(64)
    u, s, vt = np.linalg.svd(image, full_matrices=False)
    return ImageSvd(
        image=image.round(4).tolist(), u=u.round(5).tolist(), s=s.round(6).tolist(), vt=vt.round(5).tolist()
    )
