"""Render Figure 1 from the full v3 corpus (Task E1; figure redesign per
reviewer critique in Task C7; Figure 1 further simplified per the round-3
reviewer critique, Task 5).

Figure 1 is built HERE, not reused from ``fig_v2``, because its composition
genuinely differs from the v2 design (see ``make_figure1`` below) --
``fig_v2.make_figure1`` (3 panels: corpus dose-response, 20-line per-model
drift with an overlapping slope inset, and a 20-panel reliability
small-multiples grid) is left completely untouched for reuse elsewhere.

This module used to also build Figure 2 and eFigure 3 by calling
``fig_v2.make_figure2`` / ``fig_v2.make_efigure_zero_cer_causes`` directly
from ``main()`` below; both calls have been removed (round-3 reviewer pass):

  * Figure 2 is now the physician-validated faithful/degraded/drift
    examples, built by ``fig_validated_examples.py`` -- calling
    ``fig_v2.make_figure2`` at the "Figure2" output path here would
    CLOBBER that with the old per-model ECE/AUROC dot plots. That reuse
    now lives, at a different (eFigure) output path, in
    ``idrift.figures.fig_supplement_calibration``, alongside the relocated
    20-panel reliability grid.
  * eFigure 3 (automated zero-CER cause taxonomy) has been deleted
    entirely, builder and outputs, because targeted physician review
    rejected most of the underlying automated drift labels; see
    ``fig_v2.py``'s module docstring and
    ``idrift.figures.fig_supplement_calibration`` for the replacement
    calibration eFigures.

Sections/figures are regenerated, not hand-edited.

Run: ``python -m idrift.figures.fig_v3`` (builds ONLY Figure 1; it no longer
touches Figure 2 or eFigure 3/eFigures).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from idrift.figures.fig_hochberg import (
    INK,
    KEY,
    CONTEXT,
    MUTED,
    _MODEL_DISPLAY,
    _MODEL_ORDER,
    _CER_GRID,
    _apply_style,
    _panel,
    _save,
    _despine,
)
from idrift.figures.fig_v2 import (
    _f1a_drift_by_corpus,
    _fmt1,
    _load_attempts,
    _load_digest,
)


# --------------------------------------------------------------------------
# Figure 1, panel (b): model heterogeneity (Task 5 redesign)
# --------------------------------------------------------------------------
def _per_model_auth_drift_curves(attempts: pd.DataFrame) -> dict[str, list[float]]:
    """Real detected-drift-rate-vs-CER curve for each model, AUTH corpus only.

    Identical computation to the rate half of the removed
    ``fig_v2._f1b_per_model_drift`` (same groupby, same AUTH filter, same
    CER grid) -- this module does not re-derive or re-model anything, it
    only changes how the same numbers are drawn.
    """
    auth = attempts[attempts["corpus"] == "AUTH"]
    curves: dict[str, list[float]] = {}
    for k in _MODEL_ORDER:
        sub = auth[auth["model"] == k]
        n_by = sub.groupby("cer_target")["label"].size()
        k_by = sub.groupby("cer_target")["label"].apply(lambda s: int((s == "drift").sum()))
        y = []
        for c in _CER_GRID:
            ni = int(n_by.get(c, 0))
            ki = int(k_by.get(c, 0))
            y.append((ki / ni) * 100.0 if ni else float("nan"))
        curves[k] = y
    return curves


def _model_slopes(curves: dict[str, list[float]]) -> dict[str, float]:
    """Least-squares slope of each model's own curve, in percentage points of
    detected drift per 10-point rise in target CER -- the same summary
    statistic the removed inset computed, kept byte-for-byte."""
    x = np.array([c * 100 for c in _CER_GRID], dtype=float)
    slopes: dict[str, float] = {}
    for k, y in curves.items():
        ys = np.array(y, dtype=float)
        finite = ~np.isnan(ys)
        if finite.sum() >= 2:
            m, _b0 = np.polyfit(x[finite], ys[finite], 1)
            slopes[k] = float(m) * 10.0
    return slopes


def _bootstrap_auth_slope_ci(
    attempts: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 20260804,
    cache_path: str = "output/fig1b_slope_ci.json",
) -> dict[str, list[float]]:
    """Message-clustered bootstrap 95% CI for each model's own drift-vs-CER
    slope (AUTH corpus), consistent with the paper's message x model
    clustering. The resampling unit is the AUTH *message* (each message is
    evaluated by every model): one shared multiset of messages is drawn per
    replicate and reused across all 20 models, then each model's slope is
    refit on the resampled messages.

    Deterministic: fixed seed, and the result is cached to ``cache_path`` so
    re-renders reproduce the same intervals without re-bootstrapping.

    Returns ``{model_key: [lo, hi]}`` in slope units (pp drift / 10pp CER).
    """
    import json

    cp = Path(cache_path)
    if cp.exists():
        cached = json.loads(cp.read_text())
        cached_models = set(cached.keys()) - {"_meta"}
        if (cached.get("_meta", {}).get("n_boot") == n_boot
                and cached.get("_meta", {}).get("seed") == seed
                and cached_models == set(_MODEL_ORDER)):
            return {k: v for k, v in cached.items() if k != "_meta"}

    auth = attempts[attempts["corpus"] == "AUTH"]
    msg_ids = auth["message_id"].unique()
    m_index = {mid: i for i, mid in enumerate(msg_ids)}
    M = len(msg_ids)
    x = np.array([c * 100 for c in _CER_GRID], dtype=float)
    xc = x - x.mean()
    sxx = float((xc * xc).sum())

    # Per (model, cer, message): drift count k and total n, as [20, 5, M].
    n_models = len(_MODEL_ORDER)
    n_cer = len(_CER_GRID)
    K3 = np.zeros((n_models, n_cer, M), dtype=np.float64)
    N3 = np.zeros((n_models, n_cer, M), dtype=np.float64)
    cer_index = {c: j for j, c in enumerate(_CER_GRID)}
    model_index = {k: i for i, k in enumerate(_MODEL_ORDER)}
    sub = auth[["model", "cer_target", "message_id", "label"]].copy()
    sub["is_drift"] = (sub["label"] == "drift").astype(np.int64)
    g = sub.groupby(["model", "cer_target", "message_id"])["is_drift"].agg(["sum", "size"])
    for (mdl, cer, mid), row in g.iterrows():
        mi = model_index.get(mdl)
        cj = cer_index.get(round(float(cer), 3))
        if mi is None or cj is None:
            continue
        idx = m_index[mid]
        K3[mi, cj, idx] = row["sum"]
        N3[mi, cj, idx] = row["size"]

    rng = np.random.default_rng(seed)
    # Weight matrix W [n_boot, M]: each row is a bootstrap draw of M messages.
    slopes_boot = np.empty((n_models, n_boot), dtype=np.float64)
    for b in range(n_boot):
        draw = rng.integers(0, M, M)
        w = np.bincount(draw, minlength=M).astype(np.float64)
        num = K3 @ w  # [20, 5]
        den = N3 @ w  # [20, 5]
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(den > 0, num / den * 100.0, np.nan)  # [20, 5]
        ratebar = np.nanmean(rate, axis=1, keepdims=True)
        centered = rate - ratebar
        slope = (centered * xc[None, :]).sum(axis=1) / sxx * 10.0
        slopes_boot[:, b] = slope

    ci = {}
    for k in _MODEL_ORDER:
        i = model_index[k]
        lo, hi = np.nanpercentile(slopes_boot[i], [2.5, 97.5])
        ci[k] = [float(lo), float(hi)]

    payload = dict(ci)
    payload["_meta"] = {"n_boot": n_boot, "seed": seed, "n_messages": int(M),
                        "unit": "pp drift per 10pp CER, AUTH, message-clustered bootstrap"}
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(payload, indent=1))
    return ci


def _f1b_model_heterogeneity(fig, subplot_spec, attempts: pd.DataFrame) -> dict:
    """(b) Model heterogeneity in per-model detected drift vs CER (AUTH),
    redesigned per reviewer critique of the 20-line/legend/overlapping-inset
    original: panel (b) is now one labeled block holding two ordinary,
    side-by-side, NON-overlapping sub-axes rather than one axis with an
    ``inset_axes`` slope summary drawn on top of the curves.

    Left sub-axis: all 20 individual model curves, drawn as thin low-alpha
    context-gray lines with NO per-model color and NO per-model legend --
    the point of this half is the visible *spread*, not identifying any one
    curve by eye. The bold key-blue line is the across-model median at each
    CER grid point (value printed on the point, house style), with a light
    shaded band for the 25th-75th percentile across models.

    Right sub-axis: a compact, ordered dot plot of each model's own
    least-squares slope (percentage points of drift per 10-point rise in
    CER; the same statistic the removed inset computed), one dot per model,
    sorted ascending, with model display names as the y-axis labels. This is
    the only place per-model identity appears in this panel -- a single
    ranked list replacing the old 20-entry boxed legend.

    Every number plotted is a real detected-drift rate (or a least-squares
    slope of real detected-drift rates) computed from the AUTH-corpus
    attempt-level rows; nothing here is invented or newly modeled.
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=subplot_spec, width_ratios=[1.5, 1.0], wspace=0.62,
    )
    ax_lines = fig.add_subplot(inner[0, 0])
    ax_dots = fig.add_subplot(inner[0, 1])

    x = [c * 100 for c in _CER_GRID]
    curves = _per_model_auth_drift_curves(attempts)
    slopes = _model_slopes(curves)

    # --- left sub-axis: faint individual lines + bold across-model median ---
    for k in _MODEL_ORDER:
        ax_lines.plot(x, curves[k], color=CONTEXT, lw=1.1, alpha=0.42, zorder=2,
                      solid_capstyle="round")

    ys_matrix = np.array([curves[k] for k in _MODEL_ORDER], dtype=float)
    median = np.nanmedian(ys_matrix, axis=0)
    q25 = np.nanpercentile(ys_matrix, 25, axis=0)
    q75 = np.nanpercentile(ys_matrix, 75, axis=0)

    ax_lines.fill_between(x, q25, q75, color=KEY, alpha=0.13, zorder=3, linewidth=0)
    ax_lines.plot(x, median, color=KEY, lw=2.8, zorder=4, solid_capstyle="round")
    ax_lines.scatter(x, median, s=40, color=KEY, edgecolors="white", linewidths=0.9,
                      marker="o", zorder=5)
    for xi, yi in zip(x, median):
        ax_lines.annotate(_fmt1(yi), (xi, yi), textcoords="offset points",
                           xytext=(0, 9), ha="center", fontsize=6.6,
                           fontweight="bold", color=KEY)

    ax_lines.legend(
        handles=[
            Line2D([], [], color=KEY, lw=2.8, marker="o", ms=6, mfc=KEY,
                   mec="white", label=f"Median across {len(_MODEL_ORDER)} models"),
            Line2D([], [], color=KEY, lw=7, alpha=0.28,
                   label="Model 25th–75th percentile"),
            Line2D([], [], color=CONTEXT, lw=1.6, alpha=0.65,
                   label=f"Individual model (n = {len(_MODEL_ORDER)})"),
        ],
        loc="upper left", handlelength=2.0, borderaxespad=0.3, fontsize=6.6,
    )
    ax_lines.set_xlabel("Decoder character-error rate (%)")
    ax_lines.set_ylabel("Detected drift rate (%)")
    ax_lines.set_xticks(x)
    ax_lines.set_ylim(-3, 85)
    # Corpus label (round-4 reviewer): this whole panel is the AUTH corpus
    # only; a white callout box keeps it legible over the curves.
    ax_lines.text(
        0.985, 0.045, "AUTH corpus", transform=ax_lines.transAxes,
        ha="right", va="bottom", fontsize=7.2, fontweight="bold", color=INK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CONTEXT, lw=0.6,
                  alpha=0.92),
    )
    _despine(ax_lines)

    # --- right sub-axis: compact per-model slope dot plot, sorted ---
    order = sorted(slopes, key=lambda kk: slopes[kk])
    labels = [_MODEL_DISPLAY.get(k, k) for k in order]
    vals = [slopes[k] for k in order]
    yy = np.arange(len(order))

    # Message-clustered bootstrap 95% CI per model slope (real resampling of
    # AUTH messages; deterministic + cached). Drawn as a thin horizontal line
    # behind each diamond so the per-model differences are read with their
    # uncertainty, not as bare points.
    slope_ci = _bootstrap_auth_slope_ci(attempts)
    ci_lo = [slope_ci[k][0] for k in order]
    ci_hi = [slope_ci[k][1] for k in order]

    ax_dots.axvline(0, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    for yi, lo_i, hi_i in zip(yy, ci_lo, ci_hi):
        ax_dots.plot([lo_i, hi_i], [yi, yi], color=KEY, lw=1.3, alpha=0.55,
                     solid_capstyle="round", zorder=3)
    ax_dots.scatter(vals, yy, s=32, color=KEY, edgecolors="white", linewidths=0.7,
                     marker="D", zorder=4)
    for yi, v, hi_i in zip(yy, vals, ci_hi):
        ax_dots.annotate(
            _fmt1(v), (max(v, hi_i), yi), textcoords="offset points",
            xytext=(6, 0), ha="left", va="center",
            fontsize=5.6, color=INK,
        )
    ax_dots.set_yticks(yy)
    ax_dots.set_yticklabels(labels, fontsize=6.0)
    ax_dots.tick_params(axis="y", length=2)
    ax_dots.set_ylim(-0.7, len(order) - 0.3)
    lo, hi = min(ci_lo), max(ci_hi)
    pad = max((hi - lo) * 0.20, 1.0)
    ax_dots.set_xlim(lo - pad, hi + pad + 2.0)
    ax_dots.set_xlabel("Model's own slope\n(pp drift / 10pp CER)", fontsize=7.0)
    _despine(ax_dots)

    bbox = subplot_spec.get_position(fig)
    fig.text(bbox.x0 - 0.028, bbox.y1 + 0.015, "b", fontsize=13, fontweight="bold",
              va="top", ha="left", color=INK)

    return {
        "cer": x,
        "curves": {_MODEL_DISPLAY.get(k, k): v for k, v in curves.items()},
        "median": [float(v) for v in median],
        "q25": [float(v) for v in q25],
        "q75": [float(v) for v in q75],
        "slope_order": labels,
        "slopes": [float(v) for v in vals],
        "slope_ci": [[float(lo_i), float(hi_i)] for lo_i, hi_i in zip(ci_lo, ci_hi)],
    }


def make_figure1(
    digest: dict, attempts: pd.DataFrame, out_stem: str = "output/figures/Figure1"
) -> dict:
    """Render the redesigned 2-panel Figure 1 (reviewer #3, Task 5).

    (a) Corpus dose-response (AUTH/CRIT/CTRL detected drift vs target CER),
    reused verbatim from ``fig_v2._f1a_drift_by_corpus`` -- unchanged,
    already publication-ready.

    (b) Model heterogeneity, rebuilt in this module (see
    ``_f1b_model_heterogeneity``) to replace the removed 20-line/legend/
    overlapping-inset panel with 20 faint individual-model lines, one bold
    across-model median line, and a non-overlapping compact per-model slope
    dot plot.

    The 20-panel per-model reliability small-multiples grid that used to sit
    at (c) (``fig_v2._f1c_reliability_small_multiples``) is intentionally
    NOT called here: unreadable at print size and overlapping Figure 2 per
    reviewer critique. That builder (and the per-model ECE/AUROC dot plots)
    is left completely untouched in ``fig_v2.py`` for reuse in the
    Supplement.
    """
    _apply_style()
    fig = plt.figure(figsize=(13.2, 6.6))
    outer = gridspec.GridSpec(
        1, 2, width_ratios=[1.0, 1.62], wspace=0.34, figure=fig,
        left=0.055, right=0.985, top=0.93, bottom=0.13,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    a = _f1a_drift_by_corpus(ax_a, digest, attempts)
    _panel(ax_a, "a")

    b = _f1b_model_heterogeneity(fig, outer[0, 1], attempts)

    _save(fig, out_stem)
    return {"a": a, "b": b}


def main(root: str = ".") -> None:
    """Builds ONLY Figure 1. Figure 2 (physician-validated examples) is owned
    by ``fig_validated_examples.py``; the calibration eFigures (and the
    optional pooled-calibration Figure 3) are owned by
    ``idrift.figures.fig_supplement_calibration``. Running this ``main()``
    is therefore safe -- unlike the old version, it will not clobber
    ``output/figures/Figure2.pdf``.
    """
    root_p = Path(root)
    digest = _load_digest(str(root_p / "output/stats_v3plus_digest.json"))
    attempts = _load_attempts(str(root_p / "output/intermediate/attempts_v3plus_labeled.parquet"))
    f1 = make_figure1(digest, attempts, str(root_p / "output/figures/Figure1"))
    print("Figure 1:", f1)


if __name__ == "__main__":
    main()
