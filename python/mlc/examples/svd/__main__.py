"""Low-rank approximation of an image by truncated SVD, checked against the Eckart–Young theorem."""

import numpy as np
from pydantic import Field

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.svd.model import low_rank, make_image


class LowRank(Command):
    """Approximate a test image at several ranks and compare the errors with the singular values."""

    size: int = Field(64, ge=8, description="Side length of the square test image, in pixels.")

    def cli_cmd(self) -> None:
        image = make_image(self.size)
        u, s, vt = np.linalg.svd(image, full_matrices=False)
        ks = [1, 2, 4, 8, 16, 32]
        rows = []
        for k in ks:
            error = image - low_rank(u, s, vt, k)
            rows.append(
                [
                    k,
                    float(np.linalg.norm(error, 2)),
                    float(s[k]),
                    float(np.linalg.norm(error, "fro")),
                    float(np.sqrt(np.sum(s[k:] ** 2))),
                ]
            )
        emit.table(
            ["rank k", "spectral error", "σ_{k+1}", "Frobenius error", "√(Σ_{i>k} σᵢ²)"],
            rows,
            title="Eckart–Young: the truncated SVD's error equals the discarded singular values",
        )
        emit.chart(
            emit.line("σᵢ", np.arange(1, len(s) + 1), np.log10(np.maximum(s, 1e-12))),
            title="Singular values of the test image",
            x_label="index i",
            y_label="log₁₀ σᵢ",
        )
        emit.heatmap(
            np.arange(self.size),
            np.arange(self.size),
            low_rank(u, s, vt, 8)[::-1],
            title="Rank-8 approximation",
            x_label="column",
            y_label="row (flipped)",
        )


if __name__ == "__main__":
    LowRank.main()
