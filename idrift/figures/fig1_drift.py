"""Figure 1: CER -> drift-rate curves (with a message-critical overlay), a
reliability/calibration panel, and an optional drift-taxonomy panel.

Journal styling -- Helvetica/DejaVu Sans, the Okabe-Ito colorblind-safe
palette, panel labels a/b/c, no in-figure title, top/right spines off -- is
implemented inline rather than importing an external `figure_kit` module,
since such a module is not guaranteed to be importable from this package
(this task's brief explicitly calls for a self-contained implementation).

`make_fig1` is pure rendering: every number it draws (curve points,
reliability bins, taxonomy shares) is read directly out of the
`results_digest` dict handed in -- computed upstream by
`idrift.analysis.drift_curve`/`calibration`/`digests` -- so the figure
always mirrors whatever numbers the manuscript text quotes. Nothing here
is random, so a given digest always renders byte-for-byte the same PDF.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import

import matplotlib.pyplot as plt

# Okabe & Ito (2008) colorblind-safe qualitative palette.
_OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]
_CRITICAL_COLOR = "#D55E00"  # vermillion, reserved for the message-critical overlay
_RELIABILITY_COLOR = "#0072B2"  # blue


def _apply_style():
    """Set the shared rcParams for the figure: fonts, sizes, vector fonts."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,  # embed as TrueType, not Type-3
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _style_axis(ax):
    """Apply the shared journal look to one axis: spines and tick direction."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _curve_points(curve):
    """Coerce a `{cer: rate}` (or `{cer: {"rate", "lo", "hi"}}`) mapping into
    sorted point arrays, tolerating string keys/values from a JSON round-trip.

    Args:
        curve: mapping of CER grid value -> either a scalar drift rate, or a
            dict with a "rate" entry and optional "lo"/"hi" bootstrap-CI
            bounds.

    Returns:
        tuple[list, list, list|None, list|None]: (xs, ys, los, his) sorted by
            x. `los`/`his` are `None` (not lists of `None`) unless every
            point in `curve` supplies both bounds.
    """
    rows = []
    for k, v in curve.items():
        x = float(k)
        if isinstance(v, dict):
            y = float(v.get("rate", v.get("y", 0.0)))
            lo = v.get("lo")
            hi = v.get("hi")
        else:
            y, lo, hi = float(v), None, None
        rows.append((x, y, lo, hi))
    rows.sort(key=lambda r: r[0])
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    has_ci = bool(rows) and all(r[2] is not None and r[3] is not None for r in rows)
    los = [float(r[2]) for r in rows] if has_ci else None
    his = [float(r[3]) for r in rows] if has_ci else None
    return xs, ys, los, his


def _panel_drift_curves(ax, digest):
    """Panel (a): CER -> drift-rate curve per model class, with the
    message-critical curve overlaid.

    Args:
        ax: matplotlib Axes to draw into.
        digest: the `results_digest` dict; reads `curve_by_class` and
            `critical_curve`.
    """
    curve_by_class = digest.get("curve_by_class") or {}
    any_curve = False
    for i, cls in enumerate(sorted(curve_by_class)):
        curve = curve_by_class[cls]
        if not curve:
            continue
        xs, ys, los, his = _curve_points(curve)
        color = _OKABE_ITO[i % len(_OKABE_ITO)]
        if los is not None:
            ax.fill_between(xs, los, his, color=color, alpha=0.2, linewidth=0)
        ax.plot(xs, ys, marker="o", ms=3, lw=1.4, color=color, label=str(cls))
        any_curve = True

    critical = digest.get("critical_curve") or {}
    if critical:
        xs, ys, los, his = _curve_points(critical)
        if los is not None:
            ax.fill_between(xs, los, his, color=_CRITICAL_COLOR, alpha=0.2, linewidth=0)
        ax.plot(
            xs, ys, marker="^", ms=4, lw=1.6, color=_CRITICAL_COLOR,
            linestyle="--", label="message-critical",
        )
        any_curve = True

    ax.set_xlabel("Character error rate (CER)")
    ax.set_ylabel("Drift rate")
    ax.set_ylim(-0.02, 1.02)
    if any_curve:
        ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)


def _panel_reliability(ax, digest):
    """Panel (b): reliability/calibration diagram -- mean accuracy vs. mean
    confidence per bin, against the identity (perfect-calibration) diagonal.

    Args:
        ax: matplotlib Axes to draw into.
        digest: the `results_digest` dict; reads `reliability`, a list of
            `{"bin_mid", "acc", "conf"}` dicts (per
            `idrift.analysis.calibration.reliability`).
    """
    ax.plot([0, 1], [0, 1], color="0.6", lw=1.0, linestyle=":", zorder=1)
    reliability = digest.get("reliability") or []
    if reliability:
        confs = [float(r["conf"]) for r in reliability]
        accs = [float(r["acc"]) for r in reliability]
        ax.plot(
            confs, accs, marker="o", ms=4, lw=1.2, color=_RELIABILITY_COLOR, zorder=2,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean stated confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_aspect("equal", adjustable="box")
    _style_axis(ax)


def _panel_taxonomy(ax, digest):
    """Panel (c) [optional]: stacked bar of drift-type share by CER.

    Args:
        ax: matplotlib Axes to draw into.
        digest: the `results_digest` dict; reads `taxonomy`, a mapping of
            `{cer: {drift_type: share}}`.

    Returns:
        bool: True if taxonomy data was present and drawn.
    """
    taxonomy = digest.get("taxonomy") or {}
    if not taxonomy:
        ax.axis("off")
        return False

    cers = sorted(taxonomy, key=lambda c: float(c))
    xs = [float(c) for c in cers]
    types = sorted({t for row in taxonomy.values() for t in row})
    width = (max(xs) - min(xs)) / (len(xs) - 1) * 0.6 if len(xs) > 1 else 0.15

    bottoms = [0.0] * len(xs)
    for i, t in enumerate(types):
        vals = [float(taxonomy[c].get(t, 0.0)) for c in cers]
        ax.bar(
            xs, vals, bottom=bottoms, width=width,
            color=_OKABE_ITO[i % len(_OKABE_ITO)], label=t,
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xlabel("Character error rate (CER)")
    ax.set_ylabel("Drift-type share")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="upper left")
    _style_axis(ax)
    return True


def make_fig1(results_digest: dict, out_pdf) -> Path:
    """Render Figure 1: CER -> drift-rate curves, calibration/reliability,
    and (if present) drift taxonomy.

    Args:
        results_digest: the parsed contents of `results_digest.json` (see
            `idrift.analysis.digests.write_digests`), expected to hold
            `curve_by_class` (`{model_class: {cer: rate}}`), `critical_curve`
            (`{cer: rate}`), `reliability` (list of `{bin_mid, acc, conf}`
            dicts), and optionally `taxonomy` (`{cer: {drift_type: share}}`).
            Missing optional keys degrade gracefully rather than raising.
        out_pdf: path to write the PDF to (parent directories are created if
            needed); a sibling `.png` with the same stem is written too.

    Returns:
        Path: `out_pdf`, coerced to a `pathlib.Path`.
    """
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    _apply_style()

    has_taxonomy = bool(results_digest.get("taxonomy"))
    ncols = 3 if has_taxonomy else 2
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.0))
    axes = [axes] if ncols == 1 else list(axes)

    _panel_drift_curves(axes[0], results_digest)
    _panel_reliability(axes[1], results_digest)
    if has_taxonomy:
        _panel_taxonomy(axes[2], results_digest)

    for label, ax in zip("abc", axes):
        ax.text(
            -0.15, 1.05, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="right",
        )

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=600)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=600)
    plt.close(fig)
    return out_pdf
