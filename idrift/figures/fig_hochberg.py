"""Publication figures for the Intent-Drift LLM-BCI device-safety manuscript,
rendered in the Leigh Hochberg / BrainGate neural-decoding figure aesthetic
(bold saturated conditions, thick lines over shaded context, printed key
numbers, quiet chrome, lowercase panel letters, reference lines, no in-figure
titles, Arial, 600 DPI vector + raster with TrueType-embedded fonts).

Two figures, both rendered purely from the analysis digests so the plotted
numbers always mirror the manuscript text:

  Figure 1 (current-generation flagship, results_digest.json)
    (a) drift rate vs CER: overall + message-critical, 1% tolerance marker.
    (b) confidence reliability curve (mean stated confidence vs observed
        faithful rate) against the identity diagonal; ECE and AUROC annotated;
        points scaled by bin count.
    (c) per-model drift vs CER for the seven current-generation models.

  Figure 2 (generational comparison, both digests)
    (a) overall drift vs CER, current vs prior generation.
    (b) grouped bars of ECE and AUROC by generation.

The Hochberg house style is inlined (not imported) so this module is
self-contained inside the repo and carries no dependency on a skill directory
outside the project tree. Nothing here is random: a given pair of digests
renders byte-for-byte the same output.

Run:  uv run python idrift/figures/fig_hochberg.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import

import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Palette (Hochberg/BrainGate grammar + Okabe-Ito colorblind-safe categoricals)
# --------------------------------------------------------------------------
INK = "#1A1A1A"        # axes, text, primary near-black
KEY = "#2C5FAA"        # the one accent (BrainGate blue): overall / current gen
ERROR = "#C0392B"      # RESERVED for the worse outcome: message-critical drift
CONTEXT = "#9AA0A6"    # secondary / context / prior generation (gray)
MUTED = "#6B7280"

# Okabe & Ito (2008) colorblind-safe qualitative base palette. The canonical
# pale yellow is replaced with a deeper gold (#B69B00) so its thin line and
# marker stay legible on white. Eight hues; for the twenty-model panel the
# base is tiled three times (below, truncated to twenty), and every model is
# additionally given its own distinct marker shape (see fig_v2._MARKERS), so
# each series is a unique (color, marker) pair -- colorblind-safe and
# grayscale/print-safe even where models share a hue.
_OKABE_ITO_BASE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#000000",  # black
    "#B69B00",  # deep gold (Okabe-Ito yellow, darkened for white-bg contrast)
]
# Twenty entries (base tiled three times, truncated), one per model in
# _MODEL_ORDER. Models eight (or sixteen) apart share a hue but always differ
# in marker shape (fig_v2._MARKERS), keeping every model a unique
# (color, marker) pair.
_OKABE_ITO = (_OKABE_ITO_BASE * 3)[:20]

# Pretty display names for the model-class keys in the digest (20-model panel).
_MODEL_DISPLAY = {
    # Original current-generation seven.
    "gemma4:e4b": "Gemma-4 E4B",
    "gemma4:12b": "Gemma-4 12B",
    "gemma4:31b": "Gemma-4 31B",
    "phi4:mini": "Phi-4 mini",
    "phi4:14b": "Phi-4 14B",
    "qwen3.5:27b-q4": "Qwen-3.5 27B",
    "mistral-small:24b": "Mistral-Small 24B",
    # Nine added in the first panel expansion (7 -> 16).
    "command-r:latest": "Command-R",
    "gemma2:9b": "Gemma-2 9B",
    "gemma3:27b": "Gemma-3 27B",
    "llama3.1:8b": "Llama-3.1 8B",
    "phi3:mini": "Phi-3 mini",
    "qwen2.5:7b": "Qwen-2.5 7B",
    "llama4:17b-scout-16e-instruct-q4_K_M": "Llama-4 Scout",
    "qwen3.6:27b": "Qwen-3.6 27B",
    "qwen3.6:35b-a3b": "Qwen-3.6 35B-A3B",
    # Four added in the final panel expansion (16 -> 20).
    "granite3.3:8b": "Granite-3.3 8B",
    "gpt-oss:20b": "GPT-OSS 20B",
    "gpt-oss:120b": "GPT-OSS 120B",
    "deepseek-r1:32b": "DeepSeek-R1 32B",
}
# Stable plotting order, independent of dict order.
_MODEL_ORDER = [
    "gemma4:e4b", "gemma4:12b", "gemma4:31b",
    "phi4:mini", "phi4:14b",
    "qwen3.5:27b-q4", "mistral-small:24b",
    "command-r:latest",
    "gemma2:9b", "gemma3:27b",
    "llama3.1:8b", "phi3:mini", "qwen2.5:7b",
    "llama4:17b-scout-16e-instruct-q4_K_M",
    "qwen3.6:27b", "qwen3.6:35b-a3b",
    "granite3.3:8b", "gpt-oss:20b", "gpt-oss:120b", "deepseek-r1:32b",
]

_CER_GRID = [0.0, 0.1, 0.2, 0.3, 0.4]  # the manipulated independent variable
_TOLERANCE = 1.0                       # message-critical tolerance, in percent


# --------------------------------------------------------------------------
# Style + IO helpers (inlined Hochberg house style)
# --------------------------------------------------------------------------
def _ensure_arial() -> bool:
    """Register system Arial TTFs so matplotlib resolves 'Arial' (not DejaVu)."""
    for p in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if os.path.exists(p):
            try:
                _fm.fontManager.addfont(p)
            except Exception:
                pass
    return "Arial" in {f.name for f in _fm.fontManager.ttflist}


def _apply_style() -> None:
    """Shared rcParams: Arial sans, quiet chrome, TrueType-embedded vector fonts."""
    _ensure_arial()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8.5,
            "axes.labelweight": "bold",
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,   # embed TrueType, not Type-3
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 600,
            "figure.dpi": 120,
        }
    )


def _despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _panel(ax, letter: str, dx: float = -0.17, dy: float = 1.05) -> None:
    """Bold lowercase panel letter, top-left (Nature / Lancet Digital Health case)."""
    ax.text(
        dx, dy, letter, transform=ax.transAxes, fontsize=13, fontweight="bold",
        va="top", ha="left", color=INK,
    )


def _save(fig, path_no_ext: str, dpi: int = 600) -> str:
    p = Path(path_no_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{p}.{ext}", dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return f"{p}.pdf"


def _curve_pct(curve: dict) -> list[float]:
    """Read a {cer: rate} mapping (string or float keys) into a percent list on
    the fixed CER grid, in grid order."""
    out = []
    for c in _CER_GRID:
        # tolerate both "0.0" and 0.0 keys from a JSON round-trip
        v = curve.get(str(c), curve.get(c))
        if v is None and c == 0.0:
            v = curve.get("0", curve.get(0))
        out.append(float(v) * 100.0)
    return out


# --------------------------------------------------------------------------
# Figure 1 panels
# --------------------------------------------------------------------------
def _f1a_drift_vs_cer(ax, digest: dict) -> dict:
    """(a) Overall + message-critical drift vs CER, with the 1% tolerance marker."""
    x = [c * 100 for c in _CER_GRID]
    overall = _curve_pct(digest["drift_curve_overall"])
    critical = _curve_pct(digest["critical_curve"])

    # 1% message-critical tolerance (light dashed reference), drawn first (behind).
    ax.axhline(_TOLERANCE, ls=(0, (4, 3)), lw=0.9, color=MUTED, zorder=1)
    ax.text(
        40, _TOLERANCE + 1.2, "1% message-critical tolerance", ha="right", va="bottom",
        fontsize=6.6, color=MUTED, style="italic",
    )

    # Message-critical (the worse outcome -> reserved red) and overall (KEY blue).
    ax.plot(x, critical, color=ERROR, lw=2.6, ls="--", zorder=3, solid_capstyle="round")
    ax.scatter(x, critical, s=42, color=ERROR, edgecolors="white", linewidths=0.9,
               marker="^", zorder=4)
    ax.plot(x, overall, color=KEY, lw=2.6, zorder=3, solid_capstyle="round")
    ax.scatter(x, overall, s=42, color=KEY, edgecolors="white", linewidths=0.9,
               marker="o", zorder=4)

    # Print the value on every datum (Hochberg: put the number on the data).
    for xi, yi in zip(x, overall):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(1, -11),
                    ha="center", fontsize=6.6, fontweight="bold", color=KEY)
    for xi, yi in zip(x, critical):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(-2, 7),
                    ha="center", fontsize=6.6, fontweight="bold", color=ERROR)

    # Mark where message-critical drift first exceeds the 1% tolerance. The
    # callout sits in the empty region below the curves (right of the crossing).
    thr = digest.get("critical_threshold_tol0.01")
    if thr is not None:
        xthr = float(thr) * 100
        ax.axvline(xthr, ls=":", lw=0.9, color=ERROR, zorder=1)
        ax.annotate(
            "crosses 1% tolerance\nat 10% CER", xy=(xthr, 10.6), xytext=(21.5, 13.5),
            fontsize=6.4, color=ERROR, ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=ERROR, lw=0.8),
        )

    ax.legend(
        handles=[
            Line2D([], [], color=KEY, lw=2.6, marker="o", ms=6, mfc=KEY,
                   mec="white", label="Overall drift"),
            Line2D([], [], color=ERROR, lw=2.6, ls="--", marker="^", ms=6, mfc=ERROR,
                   mec="white", label="Message-critical drift"),
        ],
        loc="upper left", handlelength=2.2, borderaxespad=0.3,
    )
    ax.set_xlabel("Decoder character-error rate (%)")
    ax.set_ylabel("Intent drift rate (%)")
    ax.set_xticks(x)
    ax.set_ylim(-2, 55)
    _despine(ax)
    return {"cer": x, "overall": overall, "critical": critical}


def _f1b_reliability(ax, digest: dict) -> dict:
    """(b) Reliability curve: mean stated confidence vs observed faithful rate,
    against the identity diagonal; points scaled by bin count; ECE + AUROC noted."""
    cal = digest["calibration"]
    rel = cal["reliability"]
    conf = [float(r["conf"]) for r in rel]
    acc = [float(r["acc"]) for r in rel]
    n = [int(r["n"]) for r in rel]
    ece = float(cal["ece"])
    auroc = float(cal["auroc_conf_faithful"])

    # Identity (perfect calibration) diagonal.
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(0.97, 0.99, "perfect\ncalibration", ha="right", va="top", fontsize=6.4,
            color=MUTED, style="italic", rotation=0)

    # Overconfidence lives below the diagonal: shade the gap for the two dominant
    # top bins so the "confidently wrong" region is visible, not asserted.
    ax.fill_between([0, 1], [0, 1], [0, 0], color=KEY, alpha=0.05, zorder=0)

    # Reliability curve: thick line + points scaled by bin count (sqrt for area).
    ax.plot(conf, acc, color=KEY, lw=2.2, zorder=3, solid_capstyle="round")
    sizes = [18 + (ni ** 0.5) * 2.6 for ni in n]
    ax.scatter(conf, acc, s=sizes, color=KEY, edgecolors="white", linewidths=1.0,
               zorder=4)

    # Annotate the two dominant high-confidence bins (the overconfidence headline),
    # placed clear of their (large) markers, and separated so each label plainly
    # belongs to its own point: n=6172 centered below its marker, n=1228 upper-left.
    _nlabel_offset = {6172: (2, -24, "top", "center"), 1228: (-11, 9, "bottom", "right")}
    for r in rel:
        if r["n"] in _nlabel_offset:
            dx_, dy_, va_, ha_ = _nlabel_offset[r["n"]]
            ax.annotate(
                f"n={r['n']}\n{r['acc']*100:.0f}% faithful",
                (float(r["conf"]), float(r["acc"])),
                textcoords="offset points", xytext=(dx_, dy_), ha=ha_, va=va_,
                fontsize=6.2, color=INK,
            )

    # ECE + AUROC callout.
    ax.text(
        0.03, 0.965, f"ECE = {ece:.3f}\nAUROC = {auroc:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.4, fontweight="bold",
        color=INK,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=CONTEXT, lw=0.7, alpha=0.9),
    )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean stated confidence")
    ax.set_ylabel("Observed faithful rate")
    ax.set_aspect("equal", adjustable="box")
    _despine(ax)
    return {"conf": conf, "acc": acc, "n": n, "ece": ece, "auroc": auroc}


def _f1c_per_model(ax, digest: dict) -> dict:
    """(c) Per-model drift vs CER for the seven current-generation models."""
    x = [c * 100 for c in _CER_GRID]
    by_class = digest["curve_by_class"]
    keys = [k for k in _MODEL_ORDER if k in by_class]
    keys += [k for k in sorted(by_class) if k not in keys]  # any unexpected keys last

    plotted = {}
    handles = []
    for i, k in enumerate(keys):
        y = _curve_pct(by_class[k])
        color = _OKABE_ITO[i % len(_OKABE_ITO)]
        ax.plot(x, y, color=color, lw=1.8, zorder=3, solid_capstyle="round")
        ax.scatter(x, y, s=20, color=color, edgecolors="white", linewidths=0.6,
                   zorder=4)
        label = _MODEL_DISPLAY.get(k, k)
        handles.append(Line2D([], [], color=color, lw=1.8, marker="o", ms=4.5,
                              mfc=color, mec="white", label=label))
        plotted[label] = y

    ax.legend(handles=handles, loc="upper left", ncol=1, handlelength=1.6,
              labelspacing=0.28, borderaxespad=0.25, fontsize=6.4)
    ax.set_xlabel("Decoder character-error rate (%)")
    ax.set_ylabel("Intent drift rate (%)")
    ax.set_xticks(x)
    ax.set_ylim(-2, 55)
    _despine(ax)
    return plotted


def make_figure1(digest: dict, out_stem: str = "output/figures/Figure1") -> dict:
    """Render the 3-panel current-generation flagship figure. Returns plotted values."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.5))
    a = _f1a_drift_vs_cer(axes[0], digest)
    b = _f1b_reliability(axes[1], digest)
    c = _f1c_per_model(axes[2], digest)
    for letter, ax in zip("abc", axes):
        _panel(ax, letter)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.16, top=0.93, wspace=0.34)
    _save(fig, out_stem)
    return {"a": a, "b": b, "c": c}


# --------------------------------------------------------------------------
# Figure 2 panels (generational comparison)
# --------------------------------------------------------------------------
def _f2a_generational_drift(ax, current: dict, prior: dict) -> dict:
    """(a) Overall drift vs CER, current vs prior generation."""
    x = [c * 100 for c in _CER_GRID]
    cur = _curve_pct(current["drift_curve_overall"])
    pri = _curve_pct(prior["drift_curve_overall"])

    ax.plot(x, pri, color=CONTEXT, lw=2.6, ls="--", zorder=3, solid_capstyle="round")
    ax.scatter(x, pri, s=42, color=CONTEXT, edgecolors="white", linewidths=0.9,
               marker="s", zorder=4)
    ax.plot(x, cur, color=KEY, lw=2.6, zorder=3, solid_capstyle="round")
    ax.scatter(x, cur, s=42, color=KEY, edgecolors="white", linewidths=0.9,
               marker="o", zorder=4)

    for xi, yi in zip(x, cur):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, -11),
                    ha="center", fontsize=6.6, fontweight="bold", color=KEY)
    for xi, yi in zip(x, pri):
        ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.6, fontweight="bold", color=MUTED)

    ax.legend(
        handles=[
            Line2D([], [], color=KEY, lw=2.6, marker="o", ms=6, mfc=KEY, mec="white",
                   label="Current generation (7 models)"),
            Line2D([], [], color=CONTEXT, lw=2.6, ls="--", marker="s", ms=6,
                   mfc=CONTEXT, mec="white", label="Prior generation (4 models)"),
        ],
        loc="upper left", handlelength=2.2, borderaxespad=0.3,
    )
    ax.set_xlabel("Decoder character-error rate (%)")
    ax.set_ylabel("Overall intent drift rate (%)")
    ax.set_xticks(x)
    ax.set_ylim(-2, 55)
    _despine(ax)
    return {"cer": x, "current": cur, "prior": pri}


def _f2b_calibration_bars(ax, current: dict, prior: dict) -> dict:
    """(b) Grouped bars of ECE and AUROC by generation."""
    ece_cur = float(current["calibration"]["ece"])
    ece_pri = float(prior["calibration"]["ece"])
    auc_cur = float(current["calibration"]["auroc_conf_faithful"])
    auc_pri = float(prior["calibration"]["auroc_conf_faithful"])

    groups = ["ECE", "AUROC"]
    prior_vals = [ece_pri, auc_pri]
    cur_vals = [ece_cur, auc_cur]
    xpos = [0, 1]
    w = 0.34

    bars_p = ax.bar([p - w / 2 for p in xpos], prior_vals, w, color=CONTEXT,
                    edgecolor="white", linewidth=0.6, label="Prior generation")
    bars_c = ax.bar([p + w / 2 for p in xpos], cur_vals, w, color=KEY,
                    edgecolor="white", linewidth=0.6, label="Current generation")

    for bars in (bars_p, bars_c):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.3f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        va="bottom", fontsize=6.8, fontweight="bold", color=INK)

    ax.set_xticks(xpos)
    ax.set_xticklabels(
        ["ECE\n(lower is better)", "AUROC\n(higher is better)"], fontsize=7.5
    )
    ax.set_ylabel("Calibration metric")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", borderaxespad=0.3)
    _despine(ax)
    return {"ece_prior": ece_pri, "ece_current": ece_cur,
            "auroc_prior": auc_pri, "auroc_current": auc_cur}


def make_figure2(
    current: dict, prior: dict, out_stem: str = "output/figures/Figure2"
) -> dict:
    """Render the 2-panel generational comparison figure. Returns plotted values."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    a = _f2a_generational_drift(axes[0], current, prior)
    b = _f2b_calibration_bars(axes[1], current, prior)
    for letter, ax in zip("ab", axes):
        _panel(ax, letter, dx=-0.19)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.16, top=0.93, wspace=0.30)
    _save(fig, out_stem)
    return {"a": a, "b": b}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def _load(path: str) -> dict:
    with open(path, "r") as fh:
        return json.load(fh)


def main(root: str = ".") -> None:
    root_p = Path(root)
    digest = _load(str(root_p / "output/results_digest.json"))
    baseline = _load(str(root_p / "output/results_digest_baseline.json"))

    f1 = make_figure1(digest, str(root_p / "output/figures/Figure1"))
    f2 = make_figure2(digest, baseline, str(root_p / "output/figures/Figure2"))

    print(json.dumps({"figure1": f1, "figure2": f2}, indent=2))


if __name__ == "__main__":
    main()
