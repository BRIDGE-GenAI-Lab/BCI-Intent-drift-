"""v2 publication figures for the Intent-Drift LLM-BCI in-silico benchmark,
rendered in the same Leigh Hochberg / BrainGate figure aesthetic as
``fig_hochberg.py`` (bold saturated conditions, thick lines over shaded
context, printed key numbers, quiet chrome, lowercase panel letters, no
in-figure titles, Arial, 600 DPI vector + raster with TrueType-embedded
fonts). This module reuses the house-style helpers and palette constants
from ``fig_hochberg`` rather than reimplementing them; it does not modify
that module.

Two figures, rendered from ``output/stats_v2_digest.json`` and the labeled
attempt-level table ``output/intermediate/attempts_v2_labeled.parquet`` so
the plotted numbers mirror the reframed v2 Results ("detected drift rate",
in-silico, evaluated models):

  Figure 1
    (a) detected drift rate vs decoder CER, by corpus (AUTH / CRIT / CTRL).
    (b) per-model detected drift vs CER, authentic corpus (AUTH) only, with
        a distinct marker per model and a 95% Wilson-score band; checkpoint
        and quantization differ across the panel (eTable 14), so curves are
        not a strictly like-for-like architecture sweep.
    (c) per-model confidence-reliability small multiples (one mini-panel per
        model, each against the identity diagonal and a muted pooled
        reference curve). Reviewer-driven redesign: verbalized confidence is
        not calibrated on a common scale across models (Methods), so a single
        pooled reliability curve is misleading; the pooled curve is now
        contextual only, not the headline. As of the round-3 reviewer pass
        this panel is no longer part of the current Figure 1 (see
        ``fig_v3.make_figure1``); it is now the current SUPPLEMENT reliability
        grid, rendered standalone by
        ``idrift.figures.fig_supplement_calibration.make_efigure_reliability_grid``.

  Figure 2 (per-model calibration; single topic after this redesign)
    (a) per-model expected calibration error (ECE, lower is better).
    (b) per-model AUROC, confidence -> faithful (higher is better), same
        model order as (a). ECE and AUROC are separate metrics with opposite
        preferred directions, so they are two aligned dot plots rather than
        one dumbbell plot joined by a line. As of the round-3 reviewer pass
        this is no longer the main-text Figure 2 (that slot now holds the
        physician-validated faithful/degraded/drift examples, built by
        ``fig_validated_examples.py``); ``make_figure2`` in this module is
        reused unmodified, at a different output path, as the current
        SUPPLEMENT calibration-dots eFigure, via
        ``idrift.figures.fig_supplement_calibration.make_efigure_calibration_dots``.

The former eFigure 3 (automated, rule-based causes of drift detected at zero
realized decoder corruption) has been REMOVED (round-3 reviewer): targeted
physician review of the dedicated zero-CER adjudication sheet rejected the
automated drift label on most of these items and the finer rule-based causes
were never adjudicated (see ``output/human_rating_v3plus`` and
``manuscript/supplement.md``'s zero-CER note). Its builder
(``make_efigure_zero_cer_causes`` / ``_e_zero_cer_causes``) has been deleted
from this module along with its output files; the underlying zero-CER counts
and the adjudication result remain reported in text (Results/Supplement),
just not as a figure.

Nothing here is random: a given pair of inputs renders byte-for-byte the
same output.

Run:  uv run python idrift/figures/fig_v2.py
"""
from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from idrift.figures.fig_hochberg import (
    INK,
    KEY,
    ERROR,
    CONTEXT,
    MUTED,
    _OKABE_ITO,
    _MODEL_DISPLAY,
    _MODEL_ORDER,
    _CER_GRID,
    _apply_style,
    _panel,
    _save,
    _despine,
    _curve_pct,
)

_CORPUS_ORDER = ["AUTH", "CRIT", "CTRL"]

# Twenty distinct marker shapes paired 1:1 with _MODEL_ORDER (and its tiled
# _OKABE_ITO colors), so every per-model curve is distinguishable by shape as
# well as color -- house-style colorblind safety plus print/grayscale safety.
# With eight hues tiled across twenty models, two models share a color only
# when they carry different markers, so each series stays a unique
# (color, marker) pair. The last four entries are tuple markers (4- and
# 3-pointed stars, plus/x line markers), distinct from the 5-pointed "*".
_MARKERS = [
    "o", "s", "^", "D", "v", "P", "X", "<",
    ">", "p", "h", "H", "d", "*", "8", (4, 1, 0),
    "x", "+", (3, 1, 0), "1",
]


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def _load_digest(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def _load_attempts(path: str) -> pd.DataFrame:
    # message_id is needed for the message-clustered bootstrap CI on the
    # per-model slopes (Figure 1b); it is the same resampling unit the paper
    # clusters on. The other figures ignore it.
    cols = ["model", "corpus", "cer_target", "confidence", "label", "message_id"]
    return pd.read_parquet(path, columns=cols)


# --------------------------------------------------------------------------
# Numeric-formatting helpers
# --------------------------------------------------------------------------
def _fmt_dp(v: float, dp: int) -> str:
    """Format ``v`` to ``dp`` decimal places using round-half-up on its exact
    decimal representation, rather than a naive ``f"{v:.{dp}f}"``.

    A naive format is vulnerable to binary-float representation artefacts at
    exact rounding boundaries: e.g. ``f"{48.15:.1f}"`` renders "48.1" because
    the float literal 48.15 is actually stored as
    48.14999999999999857..., even though the correctly-rounded value (and
    the one the manuscript's eTable reports) is 48.2. Quantizing the
    ``repr()`` string through ``decimal.Decimal`` avoids that trap.
    """
    q = Decimal(1).scaleb(-dp)
    d = Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP)
    return str(d)


def _fmt1(v: float) -> str:
    return _fmt_dp(v, 1)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion k/n (fractions, 0-1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _reliability_bins(df: pd.DataFrame) -> tuple[list[float], list[float], list[int]]:
    """Bin a (sub)set of attempts into 10 equal-width stated-confidence bins.

    Returns (mean confidence, observed faithful rate, n) per non-empty bin,
    sorted by confidence. Shared by the pooled and per-model reliability
    curves so both are computed identically.
    """
    d = df.dropna(subset=["confidence"]).copy()
    if d.empty:
        return [], [], []
    d["p"] = d["confidence"].astype(float) / 100.0
    d["faithful"] = (d["label"] == "faithful").astype(float)
    edges = np.linspace(0.0, 1.0, 11)
    d["bin"] = pd.cut(d["p"], bins=edges, include_lowest=True)
    grp = (
        d.groupby("bin", observed=True)
        .agg(conf=("p", "mean"), acc=("faithful", "mean"), n=("p", "size"))
    )
    grp = grp[grp["n"] > 0].sort_values("conf")
    return grp["conf"].tolist(), grp["acc"].tolist(), grp["n"].astype(int).tolist()


# --------------------------------------------------------------------------
# Figure 1 panels
# --------------------------------------------------------------------------
def _wilson_drift_ci_by_corpus(
    attempts: pd.DataFrame,
) -> dict[str, tuple[list[float], list[float]]]:
    """Wilson-score 95% CIs for the detected-drift proportion in each
    (corpus, target-CER) cell, computed from the real attempt-level rows.

    Returns ``{corpus: (lo_pct_list, hi_pct_list)}`` aligned to ``_CER_GRID``
    (values in percent), so they can be drawn as vertical caps on the
    dose-response points in ``_f1a_drift_by_corpus``. The point estimate
    k/n these bound equals the digest ``drift_by_corpus_x_targetcer`` value
    (verified in tests); at these cell sizes (tens of thousands) the
    intervals are tight, which is the honest picture.
    """
    z = 1.959963984540054  # 95% normal quantile
    out: dict[str, tuple[list[float], list[float]]] = {}
    is_drift = (attempts["label"] == "drift").astype(int)
    grp = attempts.assign(_d=is_drift).groupby(["corpus", "cer_target"])["_d"]
    k_by = grp.sum()
    n_by = grp.size()
    for corpus in ("AUTH", "CRIT", "CTRL"):
        lo_list: list[float] = []
        hi_list: list[float] = []
        for c in _CER_GRID:
            n = int(n_by.get((corpus, c), 0))
            k = int(k_by.get((corpus, c), 0))
            if n == 0:
                lo_list.append(float("nan"))
                hi_list.append(float("nan"))
                continue
            p = k / n
            denom = 1.0 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
            lo_list.append((center - half) * 100.0)
            hi_list.append((center + half) * 100.0)
        out[corpus] = (lo_list, hi_list)
    return out


def _f1a_drift_by_corpus(ax, digest: dict, attempts: "pd.DataFrame | None" = None) -> dict:
    """(a) Detected drift rate vs CER, by corpus (AUTH / CRIT / CTRL).

    ``drift_by_corpus_x_targetcer`` values in the digest are already in
    percent; divide by 100 before handing them to ``_curve_pct``, which
    multiplies fractions by 100 to build the CER-grid-ordered percent list.
    """
    x = [c * 100 for c in _CER_GRID]
    raw = digest["drift_curve"]["drift_by_corpus_x_targetcer"]

    def _frac(name: str) -> list[float]:
        pct = raw[name]
        return _curve_pct({k: float(v) / 100.0 for k, v in pct.items()})

    auth = _frac("AUTH")
    crit = _frac("CRIT")
    ctrl = _frac("CTRL")

    # Wilson 95% CIs (real, from the attempt-level rows) drawn as vertical
    # caps under each corpus's points. Tight at these cell sizes -- honest,
    # and shows the CRIT elevation over CTRL is not sampling noise.
    ci = _wilson_drift_ci_by_corpus(attempts) if attempts is not None else None

    def _yerr(name: str, pts: list[float]):
        # In production the plotted point (digest k/n, in %) is bracketed by
        # the Wilson CI computed from the same rows, so both arms are >= 0;
        # clip guards only the case of a synthetic fixture whose digest and
        # attempts disagree (matplotlib rejects any negative err arm).
        lo, hi = ci[name]
        return [np.clip(np.array(pts) - np.array(lo), 0, None),
                np.clip(np.array(hi) - np.array(pts), 0, None)]

    # CTRL (context gray, squares) drawn first so the accented lines sit above it.
    ax.plot(x, ctrl, color=CONTEXT, lw=2.2, zorder=2, solid_capstyle="round")
    if ci is not None:
        ax.errorbar(x, ctrl, yerr=_yerr("CTRL", ctrl), fmt="none", ecolor=CONTEXT,
                    elinewidth=0.8, capsize=1.6, capthick=0.8, alpha=0.9, zorder=3)
    ax.scatter(x, ctrl, s=38, color=CONTEXT, edgecolors="white", linewidths=0.8,
               marker="s", zorder=4)

    # CRIT (the message-critical probe set -> reserved error red, dashed).
    ax.plot(x, crit, color=ERROR, lw=2.6, ls="--", zorder=3, solid_capstyle="round")
    if ci is not None:
        ax.errorbar(x, crit, yerr=_yerr("CRIT", crit), fmt="none", ecolor=ERROR,
                    elinewidth=0.8, capsize=1.6, capthick=0.8, alpha=0.9, zorder=3)
    ax.scatter(x, crit, s=42, color=ERROR, edgecolors="white", linewidths=0.9,
               marker="^", zorder=4)

    # AUTH (the primary authentic corpus -> key blue, circles).
    ax.plot(x, auth, color=KEY, lw=2.6, zorder=3, solid_capstyle="round")
    if ci is not None:
        ax.errorbar(x, auth, yerr=_yerr("AUTH", auth), fmt="none", ecolor=KEY,
                    elinewidth=0.8, capsize=1.6, capthick=0.8, alpha=0.9, zorder=3)
    ax.scatter(x, auth, s=42, color=KEY, edgecolors="white", linewidths=0.9,
               marker="o", zorder=4)

    # Print the value on every datum (house style: numbers on the data),
    # rounded with _fmt1 (not a naive f"{v:.1f}") so every printed label
    # matches the manuscript eTables exactly at every CER level. The three
    # corpora track closely, so labels are stacked in three fixed rows (CRIT
    # above the markers, AUTH just below, CTRL further below) rather than
    # nudged sideways, which stays legible even where the curves touch.
    for xi, yi in zip(x, crit):
        ax.annotate(_fmt1(yi), (xi, yi), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=6.2, fontweight="bold", color=ERROR)
    for xi, yi in zip(x, auth):
        ax.annotate(_fmt1(yi), (xi, yi), textcoords="offset points", xytext=(0, -10),
                    ha="center", fontsize=6.2, fontweight="bold", color=KEY)
    for xi, yi in zip(x, ctrl):
        ax.annotate(_fmt1(yi), (xi, yi), textcoords="offset points", xytext=(0, -20),
                    ha="center", fontsize=6.2, fontweight="bold", color=MUTED)

    ax.legend(
        handles=[
            Line2D([], [], color=KEY, lw=2.6, marker="o", ms=6, mfc=KEY,
                   mec="white", label="Authentic corpus (AUTH)"),
            Line2D([], [], color=ERROR, lw=2.6, ls="--", marker="^", ms=6, mfc=ERROR,
                   mec="white", label="Message-critical (CRIT)"),
            Line2D([], [], color=CONTEXT, lw=2.2, marker="s", ms=6, mfc=CONTEXT,
                   mec="white", label="Matched controls (CTRL)"),
        ],
        loc="upper left", handlelength=2.2, borderaxespad=0.3,
    )
    ax.set_xlabel("Decoder character-error rate (%)")
    ax.set_ylabel("Detected drift rate (%)")
    ax.set_xticks(x)
    ax.set_ylim(-11, 80)
    _despine(ax)
    out = {"cer": x, "AUTH": auth, "CRIT": crit, "CTRL": ctrl}
    if ci is not None:
        out["ci"] = {k: {"lo": ci[k][0], "hi": ci[k][1]} for k in ci}
    return out


def _f1b_per_model_drift(ax, attempts: pd.DataFrame) -> dict:
    """(b) Per-model detected drift rate vs CER, authentic corpus (AUTH).

    Each model gets a distinct color (Okabe-Ito) *and* a distinct marker
    shape, plus a shaded 95% Wilson-score band computed from the real
    per-model, per-CER counts -- reviewer critique: 7 lines with identical
    circular markers were hard to tell apart. The seven models differ in
    checkpoint and quantization (eTable 14: three checkpoints served at Q8_0,
    two at Q4_K_M), so these curves are a panel of available open models
    stress-tested identically, not a controlled like-for-like architecture
    sweep; that caveat is repeated in the figure legend.
    """
    x = [c * 100 for c in _CER_GRID]
    auth = attempts[attempts["corpus"] == "AUTH"]

    plotted = {}
    slopes = {}
    handles = []
    for i, k in enumerate(_MODEL_ORDER):
        sub = auth[auth["model"] == k]
        n_by = sub.groupby("cer_target")["label"].size()
        k_by = sub.groupby("cer_target")["label"].apply(lambda s: int((s == "drift").sum()))

        y, lo, hi = [], [], []
        for c in _CER_GRID:
            ni = int(n_by.get(c, 0))
            ki = int(k_by.get(c, 0))
            if ni == 0:
                y.append(float("nan")); lo.append(float("nan")); hi.append(float("nan"))
                continue
            rate = ki / ni
            l, h = _wilson_ci(ki, ni)
            y.append(rate * 100.0); lo.append(l * 100.0); hi.append(h * 100.0)

        color = _OKABE_ITO[i % len(_OKABE_ITO)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.fill_between(x, lo, hi, color=color, alpha=0.15, zorder=2, linewidth=0)
        ax.plot(x, y, color=color, lw=1.8, zorder=3, solid_capstyle="round")
        ax.scatter(x, y, s=24, color=color, edgecolors="white", linewidths=0.6,
                   marker=marker, zorder=4)
        label = _MODEL_DISPLAY.get(k, k)
        handles.append(Line2D([], [], color=color, lw=1.8, marker=marker, ms=5,
                              mfc=color, mec="white", label=label))
        plotted[label] = y

        # Least-squares slope of this model's own curve, in percentage points
        # of drift per 10-point rise in CER -- the compact summary statistic
        # the reviewer offered as an alternative to (or alongside) a shaded
        # uncertainty band, feeding the slope-summary inset below.
        xs = np.array(x, dtype=float)
        ys = np.array(y, dtype=float)
        finite = ~np.isnan(ys)
        if finite.sum() >= 2:
            m, _b0 = np.polyfit(xs[finite], ys[finite], 1)
            slopes[label] = (float(m) * 10.0, color)

    # Sixteen models: a two-column legend keeps the key from running off the
    # bottom of the panel while the rising curves leave the upper-left clear.
    ax.legend(handles=handles, loc="upper left", ncol=2, handlelength=1.4,
              columnspacing=1.0, labelspacing=0.26, borderaxespad=0.25,
              fontsize=6.0)
    ax.set_xlabel("Decoder character-error rate (%)")
    ax.set_ylabel("Detected drift rate (%)")
    ax.set_xticks(x)
    ax.set_ylim(-3, 82)
    _despine(ax)

    # Compact slope-summary inset: each model's own least-squares slope
    # (percentage points of drift per 10-point rise in CER), sorted, as a
    # quick single-number complement to the full curves -- reviewer's
    # "compact slope summary" option. Taller here to seat all sixteen bars.
    ax_in = ax.inset_axes([0.60, 0.045, 0.38, 0.56])
    order = sorted(slopes, key=lambda lbl: slopes[lbl][0])
    yy = np.arange(len(order))
    vals = [slopes[lbl][0] for lbl in order]
    cols = [slopes[lbl][1] for lbl in order]
    ax_in.barh(yy, vals, color=cols, height=0.74, edgecolor="white", linewidth=0.3)
    ax_in.set_yticks(yy)
    ax_in.set_yticklabels(order, fontsize=3.9)
    ax_in.tick_params(axis="y", length=1.5, pad=1.2)
    ax_in.tick_params(axis="x", labelsize=4.8, length=2)
    ax_in.set_xlabel("Slope (pp / 10pp CER)", fontsize=5.0, labelpad=1.5)
    ax_in.set_facecolor("white")
    ax_in.patch.set_alpha(0.92)
    for spine in ("top", "right"):
        ax_in.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax_in.spines[spine].set_linewidth(0.5)

    return plotted


def _f1c_reliability_small_multiples(
    fig, subplot_spec, attempts: pd.DataFrame, digest: dict,
    panel_letter: str | None = "c",
) -> dict:
    """(c) Per-model confidence-reliability small multiples.

    ``panel_letter``: the bold lowercase letter drawn above this block when
    it is embedded as one panel of a larger lettered figure (the original
    use, "c", still the default so the one existing caller,
    ``fig_v2.make_figure1``, is byte-identical). Pass ``None`` when this
    grid is instead rendered as its own standalone supplementary figure
    (``idrift.figures.fig_supplement_calibration``), which -- like the
    other single-block supplementary eFigures in this repo -- carries no
    panel letter at all.

    Reviewer critique: pooling raw stated confidence across all 7 models
    into one reliability curve is misleading, because Methods states
    verbalized confidence is not calibrated on a common scale across models.
    Each model therefore gets its own mini reliability panel (against the
    identity diagonal), colored and marker-matched to panel (b); a muted
    pooled curve is drawn in each mini panel purely as visual context, not
    as a claim of a shared calibration scale (explained in the legend cell).
    """
    # 4 columns (round-4 reviewer: a 6-wide grid renders the mini-panels too
    # small to read). 20 models fill the first five rows exactly (5x4); the
    # legend/context cell takes the first cell of the sixth row, the other
    # three staying blank.
    nrows, ncols = 6, 4  # 24 cells: 20 model panels + 1 legend cell + 3 blank
    inner = gridspec.GridSpecFromSubplotSpec(
        nrows, ncols, subplot_spec=subplot_spec, wspace=0.55, hspace=0.62
    )
    pooled_conf, pooled_acc, _ = _reliability_bins(attempts)

    cal_per_model = digest["calibration"]["per_model"]
    meta_ece = digest["calibration"]["meta_ece"]
    meta_auroc = digest["calibration"]["meta_auroc"]

    plotted = {}
    for i, k in enumerate(_MODEL_ORDER):
        r, c = divmod(i, ncols)
        ax = fig.add_subplot(inner[r, c])

        sub = attempts[attempts["model"] == k]
        conf, acc, n = _reliability_bins(sub)
        color = _OKABE_ITO[i % len(_OKABE_ITO)]
        marker = _MARKERS[i % len(_MARKERS)]

        ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.plot(pooled_conf, pooled_acc, color=CONTEXT, lw=1.1, zorder=2, alpha=0.9)
        ax.plot(conf, acc, color=color, lw=1.5, zorder=3, solid_capstyle="round")
        ax.scatter(conf, acc, s=13, color=color, marker=marker, edgecolors="white",
                   linewidths=0.5, zorder=4)

        cal = cal_per_model.get(k, {})
        ece = float(cal.get("ece", float("nan")))
        auroc = float(cal.get("auroc", float("nan")))
        label = _MODEL_DISPLAY.get(k, k)
        ax.set_title(label, fontsize=7.0, fontweight="bold", color=color, pad=3)
        ax.text(0.05, 0.95, f"ECE {_fmt_dp(ece, 2)}\nAUROC {_fmt_dp(auroc, 2)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.6, color=INK)

        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=6, length=2)
        ax.set_aspect("equal", adjustable="box")
        _despine(ax)
        plotted[label] = {"conf": conf, "acc": acc, "n": n, "ece": ece, "auroc": auroc}

    # First free grid cell after the model panels: a legend/context note
    # rather than a blank panel (any remaining cells stay empty).
    leg_r, leg_c = divmod(len(_MODEL_ORDER), ncols)
    ax_leg = fig.add_subplot(inner[leg_r, leg_c])
    ax_leg.axis("off")
    ax_leg.text(
        0.0, 1.0,
        "- - - perfect calibration\n"
        "— pooled (context only)\n"
        "— per-model curve\n\n"
        f"Pooled meta-analytic:\n"
        f"ECE {_fmt_dp(meta_ece['pooled'], 2)} "
        f"[{_fmt_dp(meta_ece['ci'][0], 2)}-{_fmt_dp(meta_ece['ci'][1], 2)}]\n"
        f"AUROC {_fmt_dp(meta_auroc['pooled'], 2)} "
        f"[{_fmt_dp(meta_auroc['ci'][0], 2)}-{_fmt_dp(meta_auroc['ci'][1], 2)}]\n\n"
        "Verbalized confidence is not on a\n"
        "common scale across models\n"
        "(Methods); the pooled curve and\n"
        "meta-analytic values are shown for\n"
        "context only, not a shared-scale claim.",
        transform=ax_leg.transAxes, ha="left", va="top", fontsize=5.7, color=INK,
    )

    bbox = subplot_spec.get_position(fig)
    x_center = (bbox.x0 + bbox.x1) / 2.0
    fig.text(x_center, bbox.y0 - 0.045, "Mean stated confidence", ha="center",
              va="top", fontsize=8.5, fontweight="bold", color=INK)
    y_center = (bbox.y0 + bbox.y1) / 2.0
    fig.text(bbox.x0 - 0.035, y_center, "Observed faithful rate", ha="center",
              va="center", rotation=90, fontsize=8.5, fontweight="bold", color=INK)

    # Panel letter for the whole nested block (not the first mini axis, whose
    # transAxes corner sits under its own model-name title): placed in figure
    # coordinates just above-left of the panel-c block, matching the bold
    # lowercase style _panel uses for (a) and (b). Skipped entirely when this
    # grid is rendered as its own standalone (unlettered) supplementary
    # figure -- see ``panel_letter`` above.
    if panel_letter is not None:
        fig.text(bbox.x0 - 0.045, bbox.y1 + 0.035, panel_letter, fontsize=13,
                  fontweight="bold", va="top", ha="left", color=INK)
    return plotted


def make_figure1(
    digest: dict, attempts: pd.DataFrame, out_stem: str = "output/figures/Figure1"
) -> dict:
    """Render the 3-panel v2 flagship figure. Returns plotted values.

    Layout: top row holds (a) drift-by-corpus and (b) per-model drift (AUTH),
    two ordinary axes; the bottom row is a full-width nested 3x6 grid holding
    (c), the 20 per-model reliability small multiples plus a legend cell, since
    a single 1-of-3 column would be too narrow for 20 legible mini panels.
    """
    _apply_style()
    fig = plt.figure(figsize=(13.2, 11.6))
    outer = gridspec.GridSpec(
        2, 1, height_ratios=[1.0, 2.25], hspace=0.30, figure=fig,
        left=0.058, right=0.99, top=0.965, bottom=0.05,
    )
    top = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.30)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])

    a = _f1a_drift_by_corpus(ax_a, digest)
    b = _f1b_per_model_drift(ax_b, attempts)
    _panel(ax_a, "a")
    _panel(ax_b, "b")

    c = _f1c_reliability_small_multiples(fig, outer[1], attempts, digest)

    _save(fig, out_stem)
    return {"a": a, "b": b, "c": c}


# --------------------------------------------------------------------------
# Figure 2 panels
# --------------------------------------------------------------------------
def _f2_calibration_dotplots(ax_ece, ax_auroc, digest: dict) -> dict:
    """(a) Per-model ECE (lower is better) and (b) per-model AUROC (higher
    is better), sorted by ECE and sharing the same model row order, on two
    separate aligned axes. ECE and AUROC are distinct metrics with opposite
    preferred directions, so unlike the earlier dumbbell design they are
    never joined by a connecting line: a visible "gap" between two unrelated
    scales is not a meaningful quantity.
    """
    per_model = digest["calibration"]["per_model"]
    order = sorted(per_model.keys(), key=lambda k: per_model[k]["ece"])
    labels = [_MODEL_DISPLAY.get(k, k) for k in order]
    eces = [float(per_model[k]["ece"]) for k in order]
    aurocs = [float(per_model[k]["auroc"]) for k in order]
    # 95% CIs are precomputed per model in the digest (message-clustered;
    # each model carries its own ``n_messages``); drawn here as horizontal
    # caps so the per-model estimates show their real precision rather than
    # reading as bare points. At n_messages ~2,300 per model the intervals
    # are tight -- that is honest and is itself the point (the per-model
    # spread, not sampling noise, drives the ECE range).
    # Fall back to a zero-width interval when a (test) digest predates the
    # per-model CI fields; the production v3plus digest always carries them.
    def _ci(k: str, field: str, point: float) -> list:
        return [float(v) for v in per_model[k].get(field, [point, point])]

    ece_ci = [[_ci(k, "ece_ci", float(per_model[k]["ece"]))[0] for k in order],
              [_ci(k, "ece_ci", float(per_model[k]["ece"]))[1] for k in order]]
    auroc_ci = [[_ci(k, "auroc_ci", float(per_model[k]["auroc"]))[0] for k in order],
                [_ci(k, "auroc_ci", float(per_model[k]["auroc"]))[1] for k in order]]
    y = np.arange(len(order))

    # (a) ECE.
    ax_ece.errorbar(
        eces, y, xerr=[np.array(eces) - np.array(ece_ci[0]),
                       np.array(ece_ci[1]) - np.array(eces)],
        fmt="none", ecolor=ERROR, elinewidth=1.2, capsize=3.0, capthick=1.2,
        alpha=0.9, zorder=3,
    )
    ax_ece.scatter(eces, y, s=40, color=ERROR, edgecolors="white", linewidths=0.8,
                   marker="D", zorder=4)
    for yi, v in zip(y, eces):
        ax_ece.annotate(_fmt_dp(v, 2), (v, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=6.4, fontweight="bold",
                        color=ERROR)
    ax_ece.set_yticks(y)
    ax_ece.set_yticklabels(labels)
    ax_ece.invert_yaxis()
    lo, hi = min(ece_ci[0]), max(ece_ci[1])
    pad = max((hi - lo) * 0.35, 0.05)
    ax_ece.set_xlim(0, hi + pad)
    ax_ece.set_xlabel("Expected calibration error\n(lower is better)")
    _despine(ax_ece)

    # (b) AUROC, same row order as (a); no connecting line between the two panels.
    ax_auroc.errorbar(
        aurocs, y, xerr=[np.array(aurocs) - np.array(auroc_ci[0]),
                         np.array(auroc_ci[1]) - np.array(aurocs)],
        fmt="none", ecolor=KEY, elinewidth=1.2, capsize=3.0, capthick=1.2,
        alpha=0.9, zorder=3,
    )
    ax_auroc.scatter(aurocs, y, s=44, color=KEY, edgecolors="white", linewidths=0.8,
                     marker="o", zorder=4)
    for yi, v in zip(y, aurocs):
        ax_auroc.annotate(_fmt_dp(v, 2), (v, yi), textcoords="offset points",
                          xytext=(0, 9), ha="center", fontsize=6.4, fontweight="bold",
                          color=KEY)
    ax_auroc.set_yticks(y)
    ax_auroc.set_yticklabels([])  # row order shared with (a); not repeated here
    ax_auroc.set_ylim(ax_ece.get_ylim())
    lo, hi = min(auroc_ci[0]), max(auroc_ci[1])
    pad = max((hi - lo) * 0.35, 0.03)
    ax_auroc.set_xlim(max(0.0, lo - pad), min(1.0, hi + pad))
    ax_auroc.set_xlabel("AUROC, confidence→faithful\n(higher is better)")
    _despine(ax_auroc)

    return {"model": labels, "ece": eces, "auroc": aurocs,
            "ece_ci": [list(c) for c in zip(*ece_ci)],
            "auroc_ci": [list(c) for c in zip(*auroc_ci)]}


def make_figure2(digest: dict, out_stem: str = "output/figures/Figure2") -> dict:
    """Render the 2-panel Figure 2. Both panels are per-model calibration
    (one topic): (a) ECE, (b) AUROC, split into two aligned dot plots with
    no connecting line (see ``_f2_calibration_dotplots``). The zero-CER-
    causes panel formerly at (b) has moved to eFigure 3
    (``make_efigure_zero_cer_causes``) per reviewer request, pending
    physician adjudication of the rule-based categorisation.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 6.8))
    result = _f2_calibration_dotplots(axes[0], axes[1], digest)
    _panel(axes[0], "a", dx=-0.34)
    _panel(axes[1], "b", dx=-0.10)
    fig.subplots_adjust(left=0.26, right=0.97, bottom=0.11, top=0.955, wspace=0.14)
    _save(fig, out_stem)
    return {
        "a": {"model": result["model"], "ece": result["ece"]},
        "b": {"model": result["model"], "auroc": result["auroc"]},
    }


# --------------------------------------------------------------------------
# Entry point
#
# (The former "Supplementary eFigure (moved out of Figure 2)" section --
# ``_e_zero_cer_causes`` / ``make_efigure_zero_cer_causes``, which rendered
# eFigure 3 -- has been deleted (round-3 reviewer): targeted physician
# review of the dedicated zero-CER adjudication sheet rejected the
# automated drift label on most of these items, and the finer rule-based
# causes were never physician-adjudicated. See the module docstring above.
# The underlying zero-CER data (``digest["zero_cer_audit"]``) is untouched
# and still backs the text-only reporting in Results/Supplement.)
# --------------------------------------------------------------------------
def main(root: str = ".") -> None:
    root_p = Path(root)
    digest = _load_digest(str(root_p / "output/stats_v2_digest.json"))
    attempts = _load_attempts(
        str(root_p / "output/intermediate/attempts_v2_labeled.parquet")
    )

    f1 = make_figure1(digest, attempts, str(root_p / "output/figures/Figure1"))
    f2 = make_figure2(digest, str(root_p / "output/figures/Figure2"))

    print(json.dumps({"figure1": f1, "figure2": f2}, indent=2, default=float))


if __name__ == "__main__":
    main()
