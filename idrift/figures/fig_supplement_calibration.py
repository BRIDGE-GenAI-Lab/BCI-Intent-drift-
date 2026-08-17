"""Calibration figures for the 20-model v3plus corpus. After the round-4
reviewer critique that a POOLED confidence-reliability curve is a fragile
main-text headline (verbalized confidence is not on a common scale across
models, so a single pooled curve is not interpretable as any one model's
calibration), the main-text calibration figure is now the PER-MODEL ECE and
AUROC dot plots, and every pooled display is relocated to the Supplement.

This module adds NO new computation. It reuses, verbatim and unmodified,
the calibration panel-builders that already live in ``idrift.figures.fig_v2``
(``_f1c_reliability_small_multiples``, ``_f2_calibration_dotplots`` via
``make_figure2``, ``_reliability_bins``) and the same
``output/stats_v3plus_digest.json`` /
``output/intermediate/attempts_v3plus_labeled.parquet`` inputs every other
v3plus figure reads. It only changes WHERE those numbers are drawn:

  Main-text Figure 3 (per-model calibration)
    ``make_figure3`` -- the per-model ECE (a) / AUROC (b) dot plots (each
    model's own within-model calibration error and discrimination, sorted
    by ECE). A thin wrapper around ``fig_v2.make_figure2`` at the Figure 3
    output path; no plotting logic is duplicated. This is the defensible
    main-text calibration display: it makes no cross-model pooling claim,
    showing instead that overconfidence (high ECE) is broad while
    discrimination (AUROC) holds, with real per-model spread.

  Supplement eFigures (relocated, unmodified computation)
    ``make_efigure_reliability_grid`` (eFigure 3) -- the 20-panel per-model
    confidence-reliability small multiples, standalone (was embedded as
    panel (c) of the old ``fig_v2.make_figure1``). Rendered with
    ``panel_letter=None``: a standalone single-block supplementary figure,
    per this repo's convention, carries no panel letter.
    ``make_efigure_pooled_calibration`` (eFigure 4) -- the POOLED displays
    demoted here from the round-3 main Figure 3: (a) the pooled
    confidence-outcome curve (the same real 10-bin pooled curve, drawn
    large and annotated, but titled descriptively and carrying an in-panel
    caveat that it is Not interpretable as calibration for any individual
    model) and (b) the pooled meta-analytic ECE and AUROC (random-effects,
    DerSimonian-Laird, with 95% CI -- straight from
    ``digest["calibration"]["meta_ece"/"meta_auroc"]``, already reported in
    Methods/Supplement/eTable 11) as a compact two-row forest summary with
    each row's between-model heterogeneity (I-squared) printed alongside.

Nothing here is random: a given pair of inputs renders byte-for-byte the
same output.

Run:  uv run python -m idrift.figures.fig_supplement_calibration
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from idrift.figures.fig_hochberg import (
    KEY,
    ERROR,
    MUTED,
    _apply_style,
    _panel,
    _save,
    _despine,
)
from idrift.figures.fig_v2 import (
    _f1c_reliability_small_multiples,
    _fmt_dp,
    _load_attempts,
    _load_digest,
    _reliability_bins,
    make_figure2,
)


# --------------------------------------------------------------------------
# Supplement eFigures (relocated, unmodified computation)
# --------------------------------------------------------------------------
def make_efigure_reliability_grid(
    digest: dict,
    attempts: pd.DataFrame,
    out_stem: str = "output/figures/eFigure_reliability_grid",
) -> dict:
    """Standalone supplementary eFigure: the 20-panel per-model
    confidence-reliability small multiples, moved out of main Figure 1
    (round-3 reviewer critique) into the Supplement.

    Wraps ``fig_v2._f1c_reliability_small_multiples`` in its own
    single-cell GridSpec/figure rather than nesting it inside a larger
    lettered figure; the panel-building logic itself (binning, per-model
    colors/markers, the muted pooled context curve, the ECE/AUROC text in
    each mini-panel, the meta-analytic legend cell) is untouched.
    """
    _apply_style()
    # Portrait-ish canvas for the 4-column, 5-row grid (round-4 reviewer):
    # wider per-panel cells than the old 6-column layout at the same page
    # width, so each mini reliability panel is legible.
    fig = plt.figure(figsize=(9.6, 10.0))
    outer = gridspec.GridSpec(
        1, 1, figure=fig, left=0.085, right=0.985, top=0.965, bottom=0.07,
    )
    result = _f1c_reliability_small_multiples(
        fig, outer[0, 0], attempts, digest, panel_letter=None,
    )
    _save(fig, out_stem)
    return result


def make_figure3(
    digest: dict, out_stem: str = "output/figures/Figure3",
) -> dict:
    """Main-text Figure 3: per-model ECE (a) and AUROC (b) dot plots -- each
    of the 20 models' own within-model expected calibration error and
    confidence->faithful discrimination, sorted by ECE. Thin wrapper around
    ``fig_v2.make_figure2``, which already renders exactly this two-panel
    figure; no plotting logic is reimplemented here.

    Promoted to the main text in round 4 (the pooled confidence-reliability
    curve that previously held the Figure 3 slot moved to the Supplement --
    see ``make_efigure_pooled_calibration`` -- because a single pooled curve
    over non-common-scale confidences is not interpretable as any one
    model's calibration). This per-model view carries no cross-model pooling
    claim: it shows that overconfidence (high ECE) is broad while
    discrimination (AUROC) holds, with real per-model spread.
    """
    return make_figure2(digest, out_stem=out_stem)


# --------------------------------------------------------------------------
# Supplement eFigure 4: pooled confidence-outcome calibration (demoted from
# the round-3 main Figure 3 -- a single pooled curve over non-common-scale
# confidences is descriptive, not interpretable as any one model's
# calibration).
# --------------------------------------------------------------------------
def _f3a_pooled_reliability(ax, attempts: pd.DataFrame) -> dict:
    """(a) Pooled confidence-outcome curve, all 20 models combined.

    Same ``_reliability_bins`` computation already used (per model, and
    pooled as muted context) inside the supplement grid -- not a new
    binning or a new statistic, just drawn large, bold, and annotated
    instead of small and muted. The gap between the curve and the identity
    diagonal is shaded to make the systematic overconfidence visible at a
    glance: through most of the confidence range, models report far higher
    confidence than their observed faithful rate supports.
    """
    conf, acc, n = _reliability_bins(attempts)
    x = np.asarray(conf, dtype=float)
    y = np.asarray(acc, dtype=float)
    nn = np.asarray(n, dtype=float)

    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.fill_between(
        x, y, x, where=(y <= x), interpolate=True, color=ERROR, alpha=0.20,
        zorder=2, linewidth=0,
    )

    # Marker area scaled by log10(n): bin populations span 2.6e3-2.3e6,
    # so a linear scale would make the smallest bins invisible.
    log_n = np.log10(np.clip(nn, 1, None))
    span = max(log_n.max() - log_n.min(), 1e-9)
    sizes = 34.0 + 300.0 * (log_n - log_n.min()) / span

    ax.plot(x, y, color=KEY, lw=2.6, zorder=3, solid_capstyle="round")
    ax.scatter(x, y, s=sizes, color=KEY, edgecolors="white", linewidths=0.9, zorder=4)
    # Value labels are offset from each marker by a fixed 8pt PLUS that
    # marker's own radius (sqrt(size/pi), since ``s`` is a marker area in
    # points^2) -- a flat 8pt offset put the label inside the marker disc
    # for the largest (highest-n) bins, e.g. the x~0.97/y~0.78 and x~0.87/
    # y~0.28 points that carry the paper's headline "confidently wrong"
    # claim, because those markers' radii (~9-10pt) exceed the flat offset
    # on their own.
    #
    # The offset direction is the local curve normal (perpendicular to the
    # secant through the neighboring points), not always straight up: with
    # ``set_aspect("equal")`` and equal x/y data ranges, a unit vector in
    # data space is isotropic on screen, so this stays a true screen-space
    # perpendicular. A plain vertical push is enough where the curve is
    # locally shallow, but for a point straddling one shallow and one steep
    # segment (e.g. x~0.87, immediately followed by the steep run up to
    # x~0.97), pushing straight up drives the label into the path of the
    # outgoing line to the next point -- the normal direction routes it out
    # to the side instead, clear of both neighboring segments.
    n_pts = len(x)
    for i, (xi, yi, si) in enumerate(zip(x, y, sizes)):
        offset_pts = 8.0 + np.sqrt(si / np.pi)
        if n_pts > 1:
            j0 = max(i - 1, 0)
            j1 = min(i + 1, n_pts - 1)
            tangent = np.array([x[j1] - x[j0], y[j1] - y[j0]])
        else:
            tangent = np.array([0.0, 0.0])
        norm = float(np.hypot(*tangent))
        if norm < 1e-9:
            unit_perp = np.array([0.0, 1.0])
        else:
            t = tangent / norm
            unit_perp = np.array([-t[1], t[0]])
            if unit_perp[1] < 0:
                unit_perp = -unit_perp
        dx_pt, dy_pt = unit_perp * offset_pts
        ax.annotate(
            _fmt_dp(yi, 2), (xi, yi), textcoords="offset points",
            xytext=(dx_pt, dy_pt),
            ha="center", fontsize=6.6, fontweight="bold", color=KEY,
        )

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Mean stated confidence")
    ax.set_ylabel("Observed faithful rate")
    ax.legend(
        handles=[
            Line2D([], [], color=MUTED, lw=1.1, ls=(0, (4, 3)),
                   label="Perfect calibration"),
            Line2D([], [], color=KEY, lw=2.6, marker="o", ms=6, mfc=KEY,
                   mec="white",
                   label=f"Pooled, {attempts['model'].nunique()} models (n = {int(nn.sum()):,})"),
            Line2D([], [], color=ERROR, lw=7, alpha=0.20,
                   label="Overconfidence gap"),
        ],
        loc="upper left", handlelength=2.2, fontsize=6.8, borderaxespad=0.3,
    )
    ax.text(
        0.98, 0.03, "Marker area scales with log(bin n).",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0,
        color=MUTED,
    )
    # In-panel caveat (round-4 reviewer): the pooled curve mixes confidence
    # scores that are not on a common scale across models, so it is a
    # descriptive cross-model summary, not any single model's calibration.
    # Placed in the empty overconfidence-gap region (clear of the upper-left
    # legend and the low pooled curve), in a white callout box so it reads
    # on the shaded background.
    ax.text(
        0.50, 0.31,
        "Descriptive pooled summary.\nNot interpretable as calibration\n"
        "for any individual model.",
        transform=ax.transAxes, ha="center", va="center", fontsize=6.6,
        fontweight="bold", color=ERROR,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=ERROR, lw=0.6,
                  alpha=0.92),
    )
    _despine(ax)
    return {"conf": conf, "acc": acc, "n": n}


def _f3b_pooled_meta_summary(ax, digest: dict) -> dict:
    """(b) Pooled meta-analytic ECE and AUROC, random-effects
    (DerSimonian-Laird) pooling across all 20 models, each with its 95% CI
    -- straight from ``digest["calibration"]["meta_ece"/"meta_auroc"]``,
    the same pooled numbers already reported in Methods/Supplement/
    eTable 11. Drawn as a compact two-row forest summary rather than
    repeating all 20 per-model dots (that detail is the Supplement's
    eFigure), with each row's between-model heterogeneity (I-squared)
    printed alongside its point estimate as an explicit caveat that a
    single pooled number hides real per-model spread.
    """
    meta_auroc = digest["calibration"]["meta_auroc"]
    meta_ece = digest["calibration"]["meta_ece"]
    rows = [
        ("AUROC\n(confidence→faithful;\nhigher is better)", meta_auroc, KEY, 1.0),
        ("ECE\n(lower is better)", meta_ece, ERROR, 0.0),
    ]
    y = np.arange(len(rows))

    for yi, (label, m, color, ideal) in zip(y, rows):
        lo, hi = float(m["ci"][0]), float(m["ci"][1])
        pt = float(m["pooled"])
        ax.axvline(ideal, color=color, lw=0.8, ls=(0, (1, 2)), alpha=0.5, zorder=1)
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.4, zorder=3,
                 solid_capstyle="round")
        ax.scatter([pt], [yi], s=95, color=color, marker="D",
                   edgecolors="white", linewidths=1.0, zorder=4)
        ax.annotate(
            f"{_fmt_dp(pt, 2)} [{_fmt_dp(lo, 2)}-{_fmt_dp(hi, 2)}]\n"
            f"I² {_fmt_dp(float(m['i2']), 0)}%, k = {int(m['k_models'])} models",
            (hi, yi), textcoords="offset points", xytext=(10, 0),
            ha="left", va="center", fontsize=6.8, fontweight="bold", color=color,
        )

    ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(0.5, -0.62, "chance\n(AUROC = 0.5)", ha="center", va="top",
             fontsize=5.8, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.4)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.05, len(rows) - 0.15)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Pooled value\n(random-effects, DerSimonian-Laird)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    return {"auroc": meta_auroc, "ece": meta_ece}


def make_efigure_pooled_calibration(
    digest: dict, attempts: pd.DataFrame,
    out_stem: str = "output/figures/eFigure4",
) -> dict:
    """Render Supplement eFigure 4: pooled confidence-outcome calibration.
    (a) pooled confidence-outcome curve (descriptive; carries an in-panel
    caveat that it is not any single model's calibration); (b) pooled
    meta-analytic ECE/AUROC forest summary. Demoted from the round-3 main
    Figure 3 in round 4: the per-model ECE/AUROC dots (now the main Figure 3,
    ``make_figure3``) are the interpretable calibration display, and every
    pooled view lives here in the Supplement.
    """
    _apply_style()
    fig = plt.figure(figsize=(11.4, 5.9))
    outer = gridspec.GridSpec(
        1, 2, width_ratios=[1.0, 1.05], wspace=0.42, figure=fig,
        left=0.075, right=0.97, top=0.94, bottom=0.30,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])

    a = _f3a_pooled_reliability(ax_a, attempts)
    b = _f3b_pooled_meta_summary(ax_b, digest)
    _panel(ax_a, "a")
    _panel(ax_b, "b", dx=-0.30)

    # One shared caveat line under both panels (not repeated per-panel):
    # heterogeneity applies to the pooling in (b), and the same "Methods"
    # non-shared-scale caveat governs both the pooled curve in (a) and the
    # pooled points in (b).
    fig.text(
        0.075, 0.045,
        "Verbalized confidence is not on a common scale across models "
        "(Methods); pooling is descriptive (I² > 99% for both metrics), "
        "not a shared-scale claim. Per-model ECE and AUROC are the main-text "
        "Figure 3; per-model reliability curves are in eFigure 3.",
        ha="left", va="bottom", fontsize=6.4, color=MUTED,
    )

    _save(fig, out_stem)
    return {"a": a, "b": b}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(root: str = ".", build_figure3: bool = True) -> None:
    root_p = Path(root)
    digest = _load_digest(str(root_p / "output/stats_v3plus_digest.json"))
    attempts = _load_attempts(
        str(root_p / "output/intermediate/attempts_v3plus_labeled.parquet")
    )

    # Main-text Figure 3: per-model ECE/AUROC dots.
    if build_figure3:
        f3 = make_figure3(digest, str(root_p / "output/figures/Figure3"))
        print("Figure 3 (per-model dots):", sorted(f3.keys()))

    # Supplement eFigure 3: per-model reliability grid.
    grid = make_efigure_reliability_grid(
        digest, attempts, str(root_p / "output/figures/eFigure3")
    )
    print("eFigure 3 (reliability grid):", sorted(grid.keys()))

    # Supplement eFigure 4: pooled confidence-outcome calibration.
    pooled = make_efigure_pooled_calibration(
        digest, attempts, str(root_p / "output/figures/eFigure4")
    )
    print("eFigure 4 (pooled calibration):", sorted(pooled.keys()))


if __name__ == "__main__":
    main()
