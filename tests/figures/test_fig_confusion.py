"""Tests for the reproducible eFigure 2 confusion-matrix builder
(``idrift.figures.fig_confusion``), added in round 4 so the previously
one-off figure is regenerable and its 36th symbol (space) is labeled."""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from idrift.data.grid import GRID_ALPHABET
from idrift.figures.fig_confusion import _tick_labels, make_efigure_confusion


def _synthetic_matrix() -> np.ndarray:
    """A valid row-normalized 36x36 matrix (identity-ish) for structure
    tests -- the builder must not depend on the real cached .npy."""
    n = len(GRID_ALPHABET)
    m = np.full((n, n), 0.02)
    np.fill_diagonal(m, 1.0)
    return m / m.sum(axis=1, keepdims=True)


def test_tick_labels_cover_all_36_symbols_with_space_labeled():
    labels = _tick_labels()
    assert len(labels) == 36 == len(GRID_ALPHABET)
    # No blank/space tick: the literal space is replaced by a visible mnemonic.
    assert " " not in labels
    assert labels[-1] == "SP"  # the 36th symbol (space)
    # The 35 non-space symbols are passed through verbatim.
    assert labels[:35] == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789")


def test_writes_pdf_and_png_from_a_matrix_path(tmp_path):
    mpath = tmp_path / "confusion.npy"
    np.save(mpath, _synthetic_matrix())
    out_stem = tmp_path / "eFigure2"
    result = make_efigure_confusion(matrix_path=str(mpath), out_stem=str(out_stem))
    assert out_stem.with_suffix(".pdf").exists()
    assert out_stem.with_suffix(".png").exists()
    assert result["matrix"].shape == (36, 36)
    assert result["labels"][-1] == "SP"


def test_rejects_wrong_shape_matrix(tmp_path):
    mpath = tmp_path / "bad.npy"
    np.save(mpath, np.eye(35))
    with pytest.raises(ValueError):
        make_efigure_confusion(matrix_path=str(mpath), out_stem=str(tmp_path / "x"))


@pytest.mark.skipif(
    shutil.which("pdffonts") is None, reason="pdffonts (poppler) not on PATH"
)
def test_pdf_is_truetype_not_type3(tmp_path):
    mpath = tmp_path / "confusion.npy"
    np.save(mpath, _synthetic_matrix())
    out_stem = tmp_path / "eFigure2"
    make_efigure_confusion(matrix_path=str(mpath), out_stem=str(out_stem))
    proc = subprocess.run(
        ["pdffonts", str(out_stem.with_suffix(".pdf"))],
        capture_output=True, text=True, check=True,
    )
    assert "Type 3" not in proc.stdout
    assert "TrueType" in proc.stdout
