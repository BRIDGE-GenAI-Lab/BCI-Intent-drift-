"""Supplement eFigure 2: the pooled 36-by-36 P300 character confusion matrix.

Reproducible builder for the figure that was previously rendered by a one-off
script (only its PDF was committed). It reads the two materialized artifacts
that ``idrift.data.materialize_confusion`` writes -- the row-normalized 36x36
matrix ``output/intermediate/confusion_overall.npy`` and the fixed 36-symbol
alphabet order (``idrift.data.grid.GRID_ALPHABET``: A-Z, 1-9, then space) --
and draws the heatmap of P(selected | intended).

Round-4 reviewer fix: the 36th symbol is a literal space, so its tick label
rendered blank in the previous version. It is now shown as ``SP`` on both
axes, with a one-line in-figure note, so all 36 symbols are labeled. (The
conventional visible-space glyphs U+2423/U+2420 are not in the figure font,
so a two-letter mnemonic is used instead of a missing-glyph box.)

Run:  uv run python -m idrift.figures.fig_confusion
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from idrift.data.grid import GRID_ALPHABET
from idrift.figures.fig_hochberg import INK, _apply_style, _save

# Mnemonic for the space symbol: a short label that renders in the figure
# font (Arial), unlike the U+2423/U+2420 space glyphs, disambiguated by the
# in-figure note below.
_SPACE_GLYPH = "SP"


def _tick_labels(alphabet=GRID_ALPHABET) -> list[str]:
    """Axis tick labels for the fixed alphabet, with the literal space
    replaced by a visible glyph so the 36th symbol is not a blank tick."""
    return [_SPACE_GLYPH if ch == " " else ch for ch in alphabet]


def make_efigure_confusion(
    matrix_path: str = "output/intermediate/confusion_overall.npy",
    out_stem: str = "output/figures/eFigure2",
) -> dict:
    """Render eFigure 2: pooled P300 character confusion heatmap.

    ``matrix``: a row-normalized 36x36 array, ``matrix[i, j] = P(selected j |
    intended i)`` over the fixed ``GRID_ALPHABET`` order. Returns the plotted
    matrix and labels for inspection/testing.
    """
    matrix = np.load(matrix_path)
    if matrix.shape != (len(GRID_ALPHABET), len(GRID_ALPHABET)):
        raise ValueError(
            f"confusion matrix is {matrix.shape}, expected "
            f"{(len(GRID_ALPHABET), len(GRID_ALPHABET))} for the 36-symbol grid"
        )
    labels = _tick_labels()

    _apply_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2))
    im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=6.0)
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.tick_params(length=2)
    ax.set_xlabel("Selected symbol")
    ax.set_ylabel("Intended (target) symbol")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("P(selected | intended)")
    cbar.ax.tick_params(labelsize=6.5)

    # One-line note so the space mnemonic is unambiguous.
    ax.text(
        1.0, -0.085, f"{_SPACE_GLYPH} = space (36th symbol)",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.4, color=INK,
    )

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.98)
    _save(fig, out_stem)
    return {"matrix": matrix, "labels": labels}


def main(root: str = ".") -> None:
    root_p = Path(root)
    result = make_efigure_confusion(
        matrix_path=str(root_p / "output/intermediate/confusion_overall.npy"),
        out_stem=str(root_p / "output/figures/eFigure2"),
    )
    print("eFigure 2 confusion:", result["matrix"].shape, "labels[-1]=",
          repr(result["labels"][-1]))


if __name__ == "__main__":
    main()
