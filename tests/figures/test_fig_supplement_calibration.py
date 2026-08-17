"""Tests for the calibration figure arrangement, after the round-4 reviewer
swap:

  * main-text Figure 3 is now the PER-MODEL ECE/AUROC dot plots
    (``make_figure3``, a thin wrapper on ``fig_v2.make_figure2``);
  * the 20-panel per-model reliability grid renders as standalone
    SUPPLEMENT eFigure 3 (``make_efigure_reliability_grid``);
  * every POOLED calibration display -- the pooled confidence-outcome curve
    (descriptive, with an in-panel "not any single model's calibration"
    caveat) and the pooled meta-analytic ECE/AUROC forest -- is demoted to
    SUPPLEMENT eFigure 4 (``make_efigure_pooled_calibration``);
  * the old automated zero-CER cause-taxonomy bar chart is REMOVED: its
    builder no longer exists in ``fig_v2``/``fig_v3``.

Everything here runs on a small hand-built digest + attempts DataFrame that
mirrors the real schema (same columns ``_load_attempts`` reads, same
``_MODEL_ORDER``/20-model calibration schema the production code iterates
over), never on the multi-GB real cache -- this suite is about figure
STRUCTURE, not about re-deriving the manuscript's numbers.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from idrift.figures.fig_hochberg import _MODEL_ORDER
import idrift.figures.fig_v2 as fig_v2
import idrift.figures.fig_v3 as fig_v3
from idrift.figures.fig_supplement_calibration import (
    make_efigure_pooled_calibration,
    make_efigure_reliability_grid,
    make_figure3,
)

ROOT = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def digest() -> dict:
    per_model = {}
    for i, k in enumerate(_MODEL_ORDER):
        per_model[k] = {
            "ece": 0.15 + 0.02 * i,
            "auroc": 0.9 - 0.01 * i,
        }
    return {
        "calibration": {
            "dropped_nan_confidence": 3,
            "per_model": per_model,
            "meta_ece": {
                "pooled": 0.34, "ci": [0.29, 0.39], "tau2": 0.01,
                "q": 100.0, "i2": 99.9, "k_models": len(_MODEL_ORDER),
                "caveat": "not on a common scale",
            },
            "meta_auroc": {
                "pooled": 0.82, "ci": [0.79, 0.85], "tau2": 0.004,
                "q": 80.0, "i2": 99.8, "k_models": len(_MODEL_ORDER),
                "caveat": "not on a common scale",
            },
        },
        "zero_cer_audit": {
            "total_zero_cer_drift": 17971,
            "by_cause": {
                "critical_substitution": 500,
                "unresolved_or_mixed": 9000,
                "formatting": 3000,
                "overgenerative": 2500,
                "paraphrase": 2971,
            },
        },
    }


@pytest.fixture
def attempts() -> pd.DataFrame:
    """20 models, each with a distinct overconfident-then-correct pattern
    across a 0-100 stated-confidence range (matching the real column's
    0-100 scale, not a pre-divided 0-1 fraction) and a real spread of
    faithful/degraded/drift labels, so ``_reliability_bins`` has real bins
    to compute for both the pooled curve and every per-model mini-panel."""
    rng = np.random.default_rng(0)
    rows = []
    for i, model in enumerate(_MODEL_ORDER):
        for _ in range(120):
            conf = float(rng.uniform(0, 100))
            # Higher confidence bins are deliberately NOT proportionally
            # more often faithful -- the same overconfidence pattern the
            # real pooled curve shows -- so the reliability curve has a
            # real (non-diagonal) shape to plot.
            p_faithful = min(0.9, (conf / 100.0) * 0.5 + 0.02 * i)
            label = "faithful" if rng.uniform() < p_faithful else "drift"
            rows.append(dict(model=model, corpus="AUTH", cer_target=0.1,
                              confidence=conf, label=label))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Old automated zero-CER cause-taxonomy chart removed: no builder anywhere.
# --------------------------------------------------------------------------
def test_zero_cer_causes_builder_removed_from_fig_v2():
    assert not hasattr(fig_v2, "make_efigure_zero_cer_causes")
    assert not hasattr(fig_v2, "_e_zero_cer_causes")
    assert not hasattr(fig_v2, "_CAUSE_DISPLAY")


def test_fig_v3_no_longer_imports_or_builds_zero_cer_causes():
    assert not hasattr(fig_v3, "make_efigure_zero_cer_causes")
    # fig_v3 also no longer builds the OLD-style Figure 2 (ECE/AUROC dots)
    # at the "Figure2" path -- that path belongs to
    # fig_validated_examples.py's physician-validated examples now, and
    # fig_v3.main() must never clobber it.
    assert not hasattr(fig_v3, "make_figure2")


# --------------------------------------------------------------------------
# Supplement eFigure 3: 20-panel reliability grid.
# --------------------------------------------------------------------------
def test_reliability_grid_writes_pdf_and_png(tmp_path, digest, attempts):
    out_stem = tmp_path / "eFigure_reliability_grid"
    result = make_efigure_reliability_grid(digest, attempts, out_stem=str(out_stem))
    pdf = out_stem.with_suffix(".pdf")
    png = out_stem.with_suffix(".png")
    assert pdf.exists() and pdf.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0

    from idrift.figures.fig_hochberg import _MODEL_DISPLAY
    expected_labels = {_MODEL_DISPLAY.get(k, k) for k in _MODEL_ORDER}
    assert set(result.keys()) == expected_labels


def test_reliability_grid_covers_all_sixteen_models(tmp_path, digest, attempts):
    out_stem = tmp_path / "eFigure_reliability_grid"
    result = make_efigure_reliability_grid(digest, attempts, out_stem=str(out_stem))
    assert len(result) == len(_MODEL_ORDER) == 20


def test_reliability_grid_has_no_panel_letter(tmp_path, digest, attempts, monkeypatch):
    """Standalone supplementary figure -> no bold lowercase panel letter
    (matches the house convention already used for eFigure 3/4), unlike
    the same panel builder's "c" when embedded inside fig_v2.make_figure1."""
    import matplotlib.pyplot as plt

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    out_stem = tmp_path / "eFigure_reliability_grid"
    make_efigure_reliability_grid(digest, attempts, out_stem=str(out_stem))

    fig = plt.gcf()
    letters = {t.get_text() for t in fig.texts if t.get_text() in ("a", "b", "c")}
    assert letters == set()
    real_close(fig)


@pytest.mark.skipif(
    shutil.which("pdffonts") is None, reason="pdffonts (poppler) not on PATH"
)
def test_reliability_grid_pdf_is_truetype_not_type3(tmp_path, digest, attempts):
    out_stem = tmp_path / "eFigure_reliability_grid"
    make_efigure_reliability_grid(digest, attempts, out_stem=str(out_stem))
    proc = subprocess.run(
        ["pdffonts", str(out_stem.with_suffix(".pdf"))],
        capture_output=True, text=True, check=True,
    )
    assert "Type 3" not in proc.stdout
    assert "TrueType" in proc.stdout


# --------------------------------------------------------------------------
# Main-text Figure 3: per-model ECE/AUROC dot plot.
# --------------------------------------------------------------------------
def test_figure3_dots_writes_pdf_and_png_at_custom_stem(tmp_path, digest):
    out_stem = tmp_path / "Figure3"
    result = make_figure3(digest, out_stem=str(out_stem))
    pdf = out_stem.with_suffix(".pdf")
    png = out_stem.with_suffix(".png")
    assert pdf.exists() and pdf.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0
    assert set(result.keys()) == {"a", "b"}
    assert len(result["a"]["ece"]) == len(_MODEL_ORDER)
    assert len(result["b"]["auroc"]) == len(_MODEL_ORDER)


def test_figure3_png_is_600_dpi(tmp_path, digest):
    from PIL import Image

    out_stem = tmp_path / "Figure3"
    make_figure3(digest, out_stem=str(out_stem))
    im = Image.open(out_stem.with_suffix(".png"))
    dpi = im.info.get("dpi")
    assert dpi is not None
    assert abs(dpi[0] - 600) < 1.0
    assert abs(dpi[1] - 600) < 1.0


# --------------------------------------------------------------------------
# Supplement eFigure 4: pooled confidence-outcome calibration.
# --------------------------------------------------------------------------
def test_efigure4_pooled_writes_pdf_and_png(tmp_path, digest, attempts):
    out_stem = tmp_path / "eFigure4"
    result = make_efigure_pooled_calibration(digest, attempts, out_stem=str(out_stem))
    pdf = out_stem.with_suffix(".pdf")
    png = out_stem.with_suffix(".png")
    assert pdf.exists() and pdf.stat().st_size > 0
    assert png.exists() and png.stat().st_size > 0
    assert set(result.keys()) == {"a", "b"}


def test_efigure4_pooled_png_is_600_dpi(tmp_path, digest, attempts):
    from PIL import Image

    out_stem = tmp_path / "eFigure4"
    make_efigure_pooled_calibration(digest, attempts, out_stem=str(out_stem))
    im = Image.open(out_stem.with_suffix(".png"))
    dpi = im.info.get("dpi")
    assert dpi is not None
    assert abs(dpi[0] - 600) < 1.0
    assert abs(dpi[1] - 600) < 1.0


@pytest.mark.skipif(
    shutil.which("pdffonts") is None, reason="pdffonts (poppler) not on PATH"
)
def test_efigure4_pooled_pdf_is_truetype_not_type3(tmp_path, digest, attempts):
    out_stem = tmp_path / "eFigure4"
    make_efigure_pooled_calibration(digest, attempts, out_stem=str(out_stem))
    proc = subprocess.run(
        ["pdffonts", str(out_stem.with_suffix(".pdf"))],
        capture_output=True, text=True, check=True,
    )
    assert "Type 3" not in proc.stdout
    assert "TrueType" in proc.stdout


def test_efigure4_pooled_panel_letters_are_exactly_a_and_b(tmp_path, digest, attempts, monkeypatch):
    import matplotlib.pyplot as plt

    real_close = plt.close
    monkeypatch.setattr(plt, "close", lambda *a, **k: None)

    out_stem = tmp_path / "eFigure4"
    make_efigure_pooled_calibration(digest, attempts, out_stem=str(out_stem))

    fig = plt.gcf()
    # ``_panel`` draws via ax.text(transform=ax.transAxes), not fig.text --
    # collect from both, matching the convention in
    # tests/figures/test_fig_v3.py's analogous check.
    letters = {t.get_text() for t in fig.texts if t.get_text() in ("a", "b", "c")}
    for ax in fig.axes:
        letters |= {t.get_text() for t in ax.texts if t.get_text() in ("a", "b", "c")}
    assert letters == {"a", "b"}
    real_close(fig)


def test_efigure4_pooled_values_come_from_the_digest_not_reinvented(tmp_path, digest, attempts):
    """No new inference: panel (b)'s plotted pooled point/CI must be
    literally the digest's ``meta_ece``/``meta_auroc`` values, not a
    separately recomputed statistic."""
    out_stem = tmp_path / "eFigure4"
    result = make_efigure_pooled_calibration(digest, attempts, out_stem=str(out_stem))
    assert result["b"]["ece"]["pooled"] == digest["calibration"]["meta_ece"]["pooled"]
    assert result["b"]["ece"]["ci"] == digest["calibration"]["meta_ece"]["ci"]
    assert result["b"]["auroc"]["pooled"] == digest["calibration"]["meta_auroc"]["pooled"]
    assert result["b"]["auroc"]["ci"] == digest["calibration"]["meta_auroc"]["ci"]


def test_figure2_is_never_touched_by_this_module():
    """This whole relocation must never write to the "Figure2" output
    stem -- that belongs to fig_validated_examples.py's physician-validated
    examples. Static guard: the string "Figure2" must not appear as an
    output path anywhere in the new calibration module's source."""
    import idrift.figures.fig_supplement_calibration as mod
    src = Path(mod.__file__).read_text()
    assert "figures/Figure2" not in src
