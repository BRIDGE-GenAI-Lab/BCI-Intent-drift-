"""Figure 2: physician-validated faithful, degraded, and drift reconstruction
examples (round-3 LDH review point: eFigure 4's examples contradicted the
taxonomy it was supposed to illustrate -- e.g. an output captioned "degraded"
that reads as fluent drift, and a plain misspelling captioned with an
automated "numeral/dose change" subtype flag that was never itself
physician-adjudicated). This figure fixes both problems at the root:

1. Every row's 3-class outcome (faithful / degraded / drift) comes from the
   completed 16-model physician panel (``output/human_rating_v3plus/``), and
   is used ONLY where BOTH independent physician raters, blinded to model
   identity and corruption level, independently gave the item the SAME
   3-class label (see ``idrift.adjudicate.analyze_panel16.map_human_label``
   for the 4-label -> 3-class collapse -- "Message-critical" folds into
   "drift"). No automated label, and no single-rater or adjudicator-only
   label, is ever eligible: a disagreed item is excluded even if a third
   adjudicator later resolved it, because this figure's entire purpose is to
   show only the least-ambiguous cases under the taxonomy.
2. No subtype caption (negation flip / numeral or dose change / recipient
   change / urgency change / actionable omission) is rendered anywhere. Those
   five flags are a RULE-BASED detector applied only to the automated
   "drift"-labeled subset (see ``fig_examples.py``'s ``_CRIT_COLS``) and were
   never themselves physician-adjudicated -- captioning a physician-agreed
   row with an unvalidated rule-based subtype, as the old eFigure 4 did, is
   exactly the mismatch the reviewer flagged. Every displayed row here
   carries only the validated 3-class outcome, the model, the corpus, the
   target CER, and the realized CER.

Selection rule (deterministic, see ``select_validated_examples``)
-------------------------------------------------------------------------
Two examples per class (faithful, degraded, drift), chosen from the clean,
both-rater-agreed pool by a fixed rule -- never a random sample, never a
hand-picked ``item_id``:

  * faithful and degraded rows draw ONE example at each of two moderate
    target-CER levels (``_ALIGNED_TARGET_LEVELS`` = 20% and 30%), and only
    rows whose realized CER lands within ``_ALIGN_TOL`` of that level are
    eligible (round-4 LDH reviewer point: the earlier version sorted these
    two classes by realized CER **descending**, which drew the single most
    extreme outlier in the pool -- e.g. a degraded row at target 20% whose
    realized CER was 80% and whose fluent output read as a different
    message rather than as broken text, and a faithful "recovery" from an
    input more corrupted than any level actually studied. Restricting to a
    moderate, target-aligned band fixes both: the displayed "target CER
    X% / realized CER Y%" caption now reads as a matched pair, and the two
    rows show a moderate and a heavier-but-still-moderate case rather than
    a cherry-picked extreme). Within each level the winner is chosen by a
    class-appropriate clarity rule, ties broken by ``item_id`` ascending:
    a FAITHFUL winner is the agreed row whose model output most closely
    reproduces the intended message (smallest normalized edit distance --
    the clearest recovery), and a DEGRADED winner is the agreed row whose
    realized CER is closest to the target level (the cleanest
    illustration of that corruption level, and -- unlike a
    maximum-departure rule -- one that does not drift into fluent,
    interpretable substitutions). The two picked rows are ordered by
    realized CER ascending so the panel reads moderate -> heavier.
  * drift rows **span the realized-CER range** (``_select_span_extremes``):
    one row is the LOWEST realized CER in the clean agreed pool and the
    other is the HIGHEST, ties broken by ``item_id`` ascending at each end
    independently. This replaced an earlier version of this figure that
    picked the two lowest-CER drift rows (both landed at exactly 0.0%);
    that both undersold the corruption-drift dose-response that is a
    co-primary finding of the study, and sat awkwardly next to the
    manuscript's separate finding that most AUTOMATED zero-CER drift
    labels are physician false positives (these two are real,
    both-rater-confirmed drift, but drawing both examples from that same
    contested bucket invited "why these?" scrutiny). The low-CER row is
    kept (drift is not merely propagated character noise -- a model can
    assert a different, fluent, wrong intent from an input with ZERO
    measured corruption) and paired with a high-CER row so the figure
    also shows the dose-response directly: a model can still produce a
    fluent, confidently wrong reconstruction even from a majority-corrupted
    input, rather than visibly breaking down (contrast with the degraded
    rows above, which DO visibly break down at similarly high corruption).

"Clean" (``_clean_mask``) excludes rows that would not typeset legibly (too
long or too short to read as a compact table cell) or that contain a
formatting artifact (a code fence) or a trailing ellipsis (an incomplete
AAC-vocabulary phrase whose own truncation, not the model's output, would
be what a reader notices -- excluded so every row's ambiguity, if any, is
about the MODEL's behavior, not the source phrase). No text is edited,
paraphrased, or authored for this figure: every intended/noisy/output cell
is copied verbatim from the rater sheet or the unblinding key.

Run: uv run python idrift/figures/fig_validated_examples.py
     (writes output/figures/Figure2.pdf / .png)
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path

import Levenshtein
import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, Ellipse, Polygon

_ASSETS = Path(__file__).resolve().parent / "assets"

from idrift.adjudicate.analyze_panel16 import _load_sheets, _mapped_series
from idrift.figures.fig_hochberg import (
    INK,
    KEY,
    ERROR,
    CONTEXT,
    MUTED,
    _apply_style,
    _save,
)

# --------------------------------------------------------------------------
# Default real-data paths
# --------------------------------------------------------------------------
_DEFAULT_RATER_A = [
    "output/human_rating_v3plus/panel_stratified_rated.csv",
    "output/human_rating_v3plus/panel_zerocer_rated.csv",
]
_DEFAULT_RATER_B = [
    "output/human_rating_v3plus/sheet_rated_modelrater_2.csv",
    "output/human_rating_v3plus/zerocer_rated_modelrater_2.csv",
]
_DEFAULT_KEY = "output/human_rating_v3plus/_KEY_DO_NOT_SHARE/key.csv"

_CLASS_ORDER = ("faithful", "degraded", "drift")
_N_PER_CLASS = 2

# Moderate target-CER levels the faithful and degraded rows are drawn from --
# one example per level. See the module docstring for why these two classes
# use a moderate, target-aligned band rather than the most extreme realized
# CER. The figure always requests n_per_class == len(_ALIGNED_TARGET_LEVELS).
_ALIGNED_TARGET_LEVELS = (0.2, 0.3)

# A faithful/degraded row is eligible for a level only if its realized CER is
# within this tolerance of the level, so the displayed target/realized pair
# reads as matched (never target 20% shown next to a realized 80% outlier).
_ALIGN_TOL = 0.12

# Classes selected by _select_span_extremes (one lowest-CER + one
# highest-CER row across the full clean pool) instead of the aligned-level
# rule: only drift, whose corruption dose-response -- including fluent,
# confidently wrong output from ZERO measured corruption -- is the finding
# the full span is meant to show.
_SPAN_CLASSES = frozenset({"drift"})

# Good-vs-bad outcome palette, muted (dusty, not neon triage): faithful is a
# calm cool teal-blue = GOOD (meaning preserved); the two failure modes share a
# warm family = BAD, escalating from a muted ochre (degraded) to a muted
# brick-red (drift, the worst). Cool/warm carries the good/bad split and the
# hue difference makes the three clearly separable at a glance, while the low
# saturation keeps it journal-grade. Local to this figure; other figures use
# the house KEY/ERROR for different quantities.
_LABEL_COLOR = {"faithful": "#3f7d86", "degraded": "#a37a44", "drift": "#9a473c"}
_LABEL_DISPLAY = {"faithful": "Faithful", "degraded": "Degraded", "drift": "Drift"}

# One-line plain-language descriptor per outcome class, shown in each section
# header so the figure teaches the taxonomy at a glance.
_LABEL_DESC = {
    "faithful": "meaning preserved",
    "degraded": "output visibly breaks down",
    "drift": "fluent, but a different message",
}

_PAPER = "#fdfcfa"        # warm off-white example card
_CARD_EC = "#e6e3dd"      # warm hairline card border
_CHIP_FC = "#f1f2f4"      # monospace "raw decoder signal" chip
_INK_SOFT = "#2b3034"     # intended text and preserved output words
_NOISE_CLEAN = "#b6bcc2"  # uncorrupted decoder characters (recede)
_NOISE_CORRUPT = "#1d2226"  # corrupted decoder characters (stand out)
_NODE_FC = "#f4f3ef"      # schematic node fill
_NODE_EC = "#a7acb2"      # schematic node / arrow stroke
_GAUGE_BG = "#e9eaec"     # CER gauge track
_GAUGE_FILL = "#7d848a"   # CER gauge fill (signal corruption)


# --------------------------------------------------------------------------
# Drawn schematic elements (the graphical layer: an actual P300 speller grid,
# decoder / language-model nodes, flow arrows, a spoken-message bubble, and a
# per-example corruption gauge) so the figure carries meaning beyond its text.
# --------------------------------------------------------------------------
def _draw_speller_grid(ax, cx, cy, cell_w=0.011, highlight="H") -> None:
    """A 6x6 BCI2000-style P300 speller grid centred at ``(cx, cy)`` with one
    attended cell highlighted -- the iconic input device, drawn to scale with
    square cells."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789 "
    # data-y units per data-x unit that render square (12.6 in wide over x in
    # [0,1]; _UNIT_IN inches tall per y unit).
    cell_h = cell_w * (12.6 / _UNIT_IN)
    x0, y0 = cx - 3 * cell_w, cy - 3 * cell_h
    for idx, ltr in enumerate(letters):
        r, c = divmod(idx, 6)
        gx, gy = x0 + c * cell_w, y0 + (5 - r) * cell_h
        hot = ltr == highlight
        ax.add_patch(Rectangle((gx + cell_w * 0.07, gy + cell_h * 0.07),
                               cell_w * 0.86, cell_h * 0.86,
                               facecolor=_LABEL_COLOR["faithful"] if hot else "white",
                               edgecolor="#cdd0d4", lw=0.4, zorder=6))
        if ltr.strip():
            ax.text(gx + cell_w * 0.5, gy + cell_h * 0.5, ltr, fontsize=3.7,
                    ha="center", va="center", zorder=7,
                    color="white" if hot else "#5c6167",
                    fontweight="bold" if hot else "normal")


def _draw_node(ax, cx, cy, w, h, label, fc=_NODE_FC, ec=_NODE_EC, tc=INK, fs=6.3) -> None:
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, facecolor=fc,
                           edgecolor=ec, lw=1.0, zorder=6))
    ax.text(cx, cy, label, fontsize=fs, fontweight="bold", color=tc,
            ha="center", va="center", zorder=7, linespacing=1.0)


def _draw_flow_arrow(ax, x0, x1, y, label=None) -> None:
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="#8b9096", lw=1.7,
                                shrinkA=1.5, shrinkB=1.5), zorder=6)
    if label:
        ax.text((x0 + x1) / 2, y + 0.42, label, fontsize=6.2, color="#6b7075",
                ha="center", va="bottom", zorder=7, style="italic")


def _draw_cer_gauge(ax, x, y, w, frac, pct_text) -> None:
    """A slim horizontal gauge encoding realized CER (how corrupted the decode
    was) as a filled fraction, with the percentage printed after it."""
    h = 0.18
    frac = min(max(frac, 0.0), 1.0)
    ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor=_GAUGE_BG,
                           edgecolor="none", zorder=4))
    ax.add_patch(Rectangle((x, y - h / 2), w * frac, h, facecolor=_GAUGE_FILL,
                           edgecolor="none", zorder=5))
    ax.add_patch(Rectangle((x, y - h / 2), w, h, facecolor="none",
                           edgecolor="#cfd1d4", lw=0.5, zorder=6))
    ax.text(x + w + 0.008, y, pct_text, fontsize=6.6, color="#6b7075",
            ha="left", va="center", zorder=6)


# --------------------------------------------------------------------------
# Drawn pipeline-stage icons (vector primitives, aspect-corrected for the
# anisotropic axes so circles read round). The neural decoder and the language
# model become actual glyphs, not text boxes -- the schematic reads as a real
# device -> decoder -> model -> physicians pipeline. Drawn in a neutral ink so
# the banner stays muted infrastructure and the outcome colours below carry the
# meaning.
# --------------------------------------------------------------------------
_ICON_INK = "#3f454b"


def _ar() -> float:
    """data-y units per data-x unit that render square in this figure's axes
    (12.6 in wide over x in [0,1]; _UNIT_IN in tall per y unit)."""
    return 12.6 / _UNIT_IN


def _icon_decoder(ax, cx, cy, w, color=_ICON_INK) -> None:
    """A neural-signal decoder: an IC 'chip' with pin ticks and an EEG trace
    running through it (raw brain signal being turned into text)."""
    ar = _ar()
    h = w * ar
    bw, bh = w * 0.84, h * 0.66
    ax.add_patch(Rectangle((cx - bw / 2, cy - bh / 2), bw, bh, facecolor="white",
                           edgecolor=color, lw=1.4, zorder=6, joinstyle="round"))
    for t in (-0.26, 0.0, 0.26):
        px = cx + t * bw
        ax.plot([px, px], [cy + bh / 2, cy + bh / 2 + h * 0.12], color=color, lw=1.2,
                zorder=6, solid_capstyle="round")
        ax.plot([px, px], [cy - bh / 2, cy - bh / 2 - h * 0.12], color=color, lw=1.2,
                zorder=6, solid_capstyle="round")
    xs = [-0.32, -0.20, -0.12, -0.02, 0.08, 0.20, 0.32]
    ys = [0.0, 0.20, -0.03, 0.26, -0.22, 0.07, 0.0]
    ax.plot([cx + t * bw for t in xs], [cy + u * bh for u in ys], color=color,
            lw=1.4, zorder=7, solid_capstyle="round", solid_joinstyle="round")


def _icon_llm(ax, cx, cy, w, color=_ICON_INK) -> None:
    """A language model: a chat/speech bubble with an ellipsis and a small
    generative 'spark', signalling text produced by a model."""
    ar = _ar()
    h = w * ar
    bw, bh = w * 0.88, h * 0.60
    by = cy + h * 0.07
    ax.add_patch(Rectangle((cx - bw / 2, by - bh / 2), bw, bh, facecolor="white",
                           edgecolor=color, lw=1.4, zorder=6, joinstyle="round"))
    # tail (drawn white-filled over the bubble's bottom edge to read as one shape)
    ax.add_patch(Polygon([[cx - bw * 0.10, by - bh / 2 + h * 0.012],
                          [cx - bw * 0.30, by - bh / 2 - h * 0.16],
                          [cx + bw * 0.06, by - bh / 2 + h * 0.012]],
                         closed=True, facecolor="white", edgecolor=color, lw=1.4,
                         zorder=7))
    ax.add_patch(Rectangle((cx - bw * 0.10, by - bh / 2 - h * 0.004),
                           bw * 0.16, h * 0.03, facecolor="white", edgecolor="none",
                           zorder=8))  # cover the seam where the tail meets the body
    for t in (-0.24, 0.0, 0.24):
        ax.add_patch(Ellipse((cx + t * bw, by), width=w * 0.11, height=w * 0.11 * ar,
                             facecolor=color, edgecolor="none", zorder=8))
    sx, sy = cx + bw * 0.40, by + bh * 0.5 + h * 0.10
    ax.plot([sx - w * 0.07, sx + w * 0.07], [sy, sy], color=color, lw=1.2,
            zorder=8, solid_capstyle="round")
    ax.plot([sx, sx], [sy - h * 0.10, sy + h * 0.10], color=color, lw=1.2,
            zorder=8, solid_capstyle="round")


def _place_image(ax, name, cx, cy, w) -> None:
    """Place a neutralized PNG logo (author-supplied, prepared in
    ``figures/assets/``) centred at ``(cx, cy)`` with data-x width ``w``,
    preserving the image's aspect ratio in the anisotropic axes (y-height =
    w * (12.6/_UNIT_IN) * img_h/img_w). Used for the language-model
    (brain-circuit) and physician-panel logos; the P300 grid and decoder chip
    stay vector line-art."""
    img = mpimg.imread(str(_ASSETS / name))
    ih, iw = img.shape[0], img.shape[1]
    h = w * (12.6 / _UNIT_IN) * (ih / iw)
    ax.imshow(img, extent=[cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2],
              aspect="auto", zorder=6, interpolation="antialiased")


def _icon_physicians(ax, cx, cy, w, color=_ICON_INK) -> None:
    """Two clinician glyphs side by side: the two-physician adjudication panel
    that assigns the concordant label."""
    ar = _ar()
    h = w * ar
    for dx, z in ((-w * 0.19, 6), (w * 0.19, 8)):
        px = cx + dx
        sw = w * 0.40
        ax.add_patch(Polygon([[px - sw / 2, cy - h * 0.34], [px + sw / 2, cy - h * 0.34],
                              [px + sw * 0.32, cy + h * 0.04], [px - sw * 0.32, cy + h * 0.04]],
                             closed=True, facecolor="white", edgecolor=color, lw=1.4,
                             zorder=z))
        ax.add_patch(Ellipse((px, cy + h * 0.20), width=w * 0.25, height=w * 0.25 * ar,
                             facecolor="white", edgecolor=color, lw=1.4, zorder=z + 1))


# Vertical layout budget in abstract "row units"; the figure height is scaled
# from the total (``_UNIT_IN`` inches per unit) so each example card keeps a
# fixed on-page height regardless of how many rows a class contributes.
# Vertical budget tuned so the whole figure stays close to landscape/square
# (about 12.6 x 9 in) and reads without scrolling, while keeping the enlarged
# text and the two-row cards.
_LAYOUT = {"header": 2.9, "sechead": 0.85, "card": 1.5, "cardgap": 0.2, "secgap": 0.45}
_UNIT_IN = 0.56


def _total_units(by_class: dict, class_order: tuple) -> float:
    """Total vertical extent (in row units) of the header plus every section,
    used both to set ``ylim`` and to size the figure so cards render at a
    constant on-page height."""
    t = _LAYOUT["header"]
    for c in class_order:
        n = len(by_class[c])
        t += _LAYOUT["sechead"] + n * _LAYOUT["card"] + max(n - 1, 0) * _LAYOUT["cardgap"] + _LAYOUT["secgap"]
    return t


# --------------------------------------------------------------------------
# Inline diff highlighting (what makes this a figure, not a table): the
# corrupted decoder characters and the words the model actually changed are
# rendered in their own colour, aligned by a real edit-distance diff.
# --------------------------------------------------------------------------
def _corruption_flags(intended: str, noisy: str) -> list[bool]:
    """Per-character flags over ``noisy``: True where the character is not part
    of an aligned 'equal' run with ``intended`` (a substitution or insertion
    the P300 corruption introduced), False where it matches the intent."""
    sm = difflib.SequenceMatcher(a=intended, b=noisy, autojunk=False)
    flags = [True] * len(noisy)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for j in range(j1, j2):
                flags[j] = False
    return flags


def _changed_word_flags(intended: str, output: str) -> list[bool]:
    """Per-word flags over ``output``'s whitespace tokens: True where the word
    was replaced or inserted relative to the intended message (case-folded for
    the comparison), False where it was preserved verbatim."""
    iw = intended.upper().split()
    ow = output.split()
    sm = difflib.SequenceMatcher(a=iw, b=[w.upper() for w in ow], autojunk=False)
    flags = [True] * len(ow)
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for j in range(j1, j2):
                flags[j] = False
    return flags


def _output_word_specs(intended: str, output: str, changed_color: str) -> list:
    """Per-word ``(text, color, weight)`` specs for the model output, with the
    opening/closing quotes attached to the first/last word: a word the model
    changed relative to the intent is shown bold in the outcome colour, a
    preserved word in quiet ink."""
    wflags = _changed_word_flags(intended, output)
    words = output.split()
    specs = []
    for k, w in enumerate(words):
        disp = ("“" if k == 0 else "") + w + ("”" if k == len(words) - 1 else "")
        changed = wflags[k] if k < len(wflags) else False
        specs.append((disp, changed_color if changed else _INK_SOFT,
                      "bold" if changed else "normal"))
    return specs or [("“”", _INK_SOFT, "normal")]


def _space_width_data(ax, renderer, fontsize) -> float:
    """Width of one space, in data-x units, for a proportional run at
    ``fontsize`` (measured as width('n n') - width('nn'))."""
    a = ax.text(0, 0, "n n", fontsize=fontsize, alpha=0, ha="left", va="bottom")
    b = ax.text(0, 0, "nn", fontsize=fontsize, alpha=0, ha="left", va="bottom")
    d = a.get_window_extent(renderer).width - b.get_window_extent(renderer).width
    a.remove()
    b.remove()
    x0 = ax.transData.transform((0, 0))[0]
    x1 = ax.transData.transform((1, 0))[0]
    return d / (x1 - x0)


def _draw_words(ax, renderer, x, y, word_specs, fontsize, space_w, zorder=5) -> float:
    """Draw whole words left-to-right (kerning preserved) with one measured
    space between them, each in its own colour/weight. Returns the trailing x."""
    cur = x
    x0 = ax.transData.transform((0, 0))[0]
    per_px = 1.0 / (ax.transData.transform((1, 0))[0] - x0)
    for i, (word, color, weight) in enumerate(word_specs):
        t = ax.text(cur, y, word, fontsize=fontsize, color=color, fontweight=weight,
                    ha="left", va="center", zorder=zorder)
        cur += t.get_window_extent(renderer).width * per_px + (space_w if i < len(word_specs) - 1 else 0.0)
    return cur


def _draw_spans(ax, renderer, x, y, spans, fontsize, family=None, zorder=5) -> float:
    """Draw a run of styled text spans left-to-right starting at data ``x``,
    advancing by each span's measured on-screen width so mixed colours/weights
    sit flush like one line. ``spans`` is a list of ``(text, color, weight)``.
    Returns the data-x coordinate just past the last span."""
    cur = x
    trans, inv = ax.transData, ax.transData.inverted()
    for text, color, weight in spans:
        t = ax.text(cur, y, text, fontsize=fontsize, color=color, fontweight=weight,
                    family=family, ha="left", va="center", zorder=zorder)
        w_px = t.get_window_extent(renderer).width
        px, py = trans.transform((cur, y))
        cur = float(inv.transform((px + w_px, py))[0])
    return cur

# Legibility bounds for a compact table cell. A lower bound excludes
# near-empty rows that would not illustrate anything; an upper bound keeps
# every cell readable at print size (same purpose as fig_examples.py's
# _MAX_TEXT_LEN, tightened slightly because this figure shows six rows in
# three two-row sections rather than one flat list of five).
_MIN_TEXT_LEN = 6
_MAX_TEXT_LEN = 40


# --------------------------------------------------------------------------
# IO: build the (not yet agreement-filtered) merged pool
# --------------------------------------------------------------------------
def load_validated_pool(
    rater_a_paths=None, rater_b_paths=None, key_path=None
) -> pd.DataFrame:
    """Load both raters' completed sheets + the unblinding key and return one
    merged, item-level DataFrame -- NOT yet filtered to agreement (that is
    ``select_validated_examples``'s job, so it can be unit-tested against a
    fixture that also contains disagreement rows).

    Columns: item_id, class_a, class_b, model, corpus, cer_target,
    realized_cer, intended_text, noisy_text, output_message.

    Args:
        rater_a_paths, rater_b_paths: each a single path or list of paths
            accepted by ``idrift.adjudicate.analyze_panel16._load_sheets``
            (defaults to the real panel_stratified + panel_zerocer sheets
            for each rater).
        key_path: path to the unblinding key, or an already-loaded
            DataFrame (defaults to the real
            ``_KEY_DO_NOT_SHARE/key.csv``).
    """
    rater_a_paths = rater_a_paths if rater_a_paths is not None else _DEFAULT_RATER_A
    rater_b_paths = rater_b_paths if rater_b_paths is not None else _DEFAULT_RATER_B
    key_path = key_path if key_path is not None else _DEFAULT_KEY

    a = _load_sheets(rater_a_paths)
    b = _load_sheets(rater_b_paths)
    ma = _mapped_series(a)[["item_id", "class3"]].rename(columns={"class3": "class_a"})
    mb = _mapped_series(b)[["item_id", "class3"]].rename(columns={"class3": "class_b"})

    # Blinded display text is carried verbatim on rater A's own sheet (both
    # raters see byte-identical intended/noisy/output text for a given
    # item_id -- neither rater can edit it), so rater A's copy is the
    # canonical source here.
    text = a[["item_id", "intended_text", "noisy_text", "output_message"]]

    key = key_path if isinstance(key_path, pd.DataFrame) else pd.read_csv(key_path)
    key = key[["item_id", "model", "corpus", "cer_target", "realized_cer"]]

    merged = (
        ma.merge(mb, on="item_id", how="inner")
        .merge(text, on="item_id", how="left")
        .merge(key, on="item_id", how="left")
    )
    return merged


# --------------------------------------------------------------------------
# Deterministic selection
# --------------------------------------------------------------------------
def _agreed_only(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where both raters' mapped 3-class labels are the same non-null
    value. Renames the shared value to ``class3``."""
    agreed = df[
        df["class_a"].notna() & df["class_b"].notna() & (df["class_a"] == df["class_b"])
    ].copy()
    agreed["class3"] = agreed["class_a"]
    return agreed


def _clean_mask(df: pd.DataFrame) -> pd.Series:
    """Rows short enough to typeset legibly, long enough to illustrate
    something, free of a raw code-fence formatting artifact, and free of a
    trailing ellipsis in either the intended or the output text (an
    incomplete source phrase, not the model's behavior, is what a reader
    would notice -- see module docstring)."""
    intended = df["intended_text"].astype(str)
    output = df["output_message"].astype(str)
    lens_ok = (
        intended.str.len().between(_MIN_TEXT_LEN, _MAX_TEXT_LEN)
        & output.str.len().between(_MIN_TEXT_LEN, _MAX_TEXT_LEN)
    )
    no_artifact = ~intended.str.contains("```", regex=False) & ~output.str.contains(
        "```", regex=False
    )
    no_ellipsis = (
        ~intended.str.contains("...", regex=False)
        & ~output.str.contains("...", regex=False)
        & ~intended.str.contains("…", regex=False)
        & ~output.str.contains("…", regex=False)
    )
    return lens_ok & no_artifact & no_ellipsis


def _norm_edit_distance(output: str, intended: str) -> float:
    """Case- and whitespace-normalized Levenshtein distance between a model
    output and the intended message, divided by the longer length -- 0.0 for
    an exact recovery, up to 1.0 for a fully different string. Used to rank
    faithful candidates (the clearest recovery is the one closest to the
    intended message)."""
    a = str(output).upper().strip()
    b = str(intended).upper().strip()
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b), 1)


def _select_aligned_levels(
    sub: pd.DataFrame,
    class3: str,
    levels: tuple = _ALIGNED_TARGET_LEVELS,
    tol: float = _ALIGN_TOL,
) -> pd.DataFrame:
    """Deterministically pick ONE row of ``sub`` (already restricted to a
    single class) at each target-CER level in ``levels``, from the rows whose
    realized CER is within ``tol`` of that level.

    Within a level the winner is chosen by a class-appropriate clarity rule,
    ties broken by ``item_id`` ascending: for ``"faithful"`` the row whose
    output most closely reproduces the intended message (smallest
    ``_norm_edit_distance``, then closest realized/target alignment); for any
    other class (``"degraded"``) the row whose realized CER is closest to the
    level. Levels are distinct target values, so a row (which has one
    ``cer_target``) can win at most one level; the picks are returned ordered
    by realized CER ascending. A fixed filter + stable (mergesort) sort, never
    a random sample: a given ``sub`` yields byte-identical picks regardless of
    row order.
    """
    picked_idx: list = []
    for level in levels:
        cand = sub[
            (sub["cer_target"] == level)
            & ((sub["realized_cer"] - level).abs() <= tol)
        ].copy()
        cand = cand[~cand.index.isin(picked_idx)]
        if cand.empty:
            continue
        cand["_align"] = (cand["realized_cer"] - level).abs()
        if class3 == "faithful":
            cand["_fid"] = [
                _norm_edit_distance(o, i)
                for o, i in zip(cand["output_message"], cand["intended_text"])
            ]
            cand = cand.sort_values(
                ["_fid", "_align", "item_id"], kind="mergesort"
            )
        else:
            cand = cand.sort_values(["_align", "item_id"], kind="mergesort")
        picked_idx.append(cand.index[0])
    return sub.loc[picked_idx].sort_values(
        ["realized_cer", "item_id"], kind="mergesort"
    )


def _select_span_extremes(sub: pd.DataFrame, n: int) -> pd.DataFrame:
    """Deterministically select up to ``n`` rows of ``sub`` that SPAN its
    ``realized_cer`` range, by alternately taking the lowest remaining and
    the highest remaining value (ties broken by ``item_id`` ascending at
    each end independently). With the default ``n=2`` this always yields
    exactly one lowest-CER row and one highest-CER row -- never two rows
    from the same end or the middle of the range.

    A fixed rule, not a random sample: a given ``sub`` always yields the
    same picks regardless of row order (mergesort is stable and the tie
    break is an explicit column, not row position).
    """
    picked_idx: list = []
    take_low = True
    for _ in range(n):
        remaining = sub[~sub.index.isin(picked_idx)]
        if remaining.empty:
            break
        ordered = remaining.sort_values(
            ["realized_cer", "item_id"], ascending=[take_low, True], kind="mergesort"
        )
        picked_idx.append(ordered.index[0])
        take_low = not take_low
    return sub.loc[picked_idx]


def select_validated_examples(
    df: pd.DataFrame,
    n_per_class: int = _N_PER_CLASS,
    class_order: tuple = _CLASS_ORDER,
) -> list[dict]:
    """Deterministically select ``n_per_class`` real rows per class from the
    both-rater-agreed, clean subset of ``df`` (the shape returned by
    ``load_validated_pool``: item_id, class_a, class_b, model, corpus,
    cer_target, realized_cer, intended_text, noisy_text, output_message).

    A row is eligible only if ``class_a == class_b`` (both raters
    independently gave it the same 3-class label) and it passes
    ``_clean_mask``. Faithful and degraded rows use ``_select_aligned_levels``
    (one example per moderate, target-aligned level; see module docstring),
    which returns exactly ``len(_ALIGNED_TARGET_LEVELS)`` picks -- so for
    those classes ``n_per_class`` must equal that count (2, the figure's
    default) or the ValueError below fires. Classes in ``_SPAN_CLASSES``
    (drift) instead use ``_select_span_extremes``, which alternates the
    lowest and highest remaining realized CER so both ends of the range are
    represented. Either way this is a fixed filter + stable (mergesort)
    sort, never a random sample: a given ``df`` always yields byte-identical
    output regardless of row order.

    Returns a list of dicts, each with keys ``item_id``, ``label``,
    ``model``, ``corpus``, ``cer_target``, ``realized_cer``,
    ``intended_text``, ``noisy_text``, ``output_message``, ``source_index``
    -- and deliberately NO subtype/cause field (see module docstring: no
    unvalidated subtype caption is ever attached to a physician-agreed row).

    Raises:
        ValueError: if fewer than ``n_per_class`` clean, agreed rows exist
            for any requested class.
    """
    agreed = _agreed_only(df)
    agreed = agreed[_clean_mask(agreed)]

    examples: list[dict] = []
    for cls in class_order:
        sub = agreed[agreed["class3"] == cls]
        if cls in _SPAN_CLASSES:
            picked = _select_span_extremes(sub, n_per_class)
        else:
            picked = _select_aligned_levels(sub, cls)
        if len(picked) < n_per_class:
            raise ValueError(
                f"only {len(picked)} clean both-rater-agreed {cls!r} example(s) "
                f"available, need {n_per_class}"
            )
        for _, row in picked.iterrows():
            examples.append(
                {
                    "item_id": row["item_id"],
                    "label": row["class3"],
                    "model": row["model"],
                    "corpus": row["corpus"],
                    "cer_target": float(row["cer_target"]),
                    "realized_cer": float(row["realized_cer"]),
                    "intended_text": row["intended_text"],
                    "noisy_text": row["noisy_text"],
                    "output_message": row["output_message"],
                    "source_index": row.name,
                }
            )
    return examples


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def _examples_panel(ax, examples: list[dict], class_order: tuple = _CLASS_ORDER) -> None:
    """Small-multiples of physician-validated reconstructions, grouped into
    one soft-tinted, colour-keyed section per outcome class (faithful /
    degraded / drift). Each example is a white card that reads left to right
    as the decode pipeline: the intended message, the raw noisy decoder text
    (shown in a monospaced chip to mark it as the corrupted signal), the model
    output (in the outcome colour), and the concordant physician label as a
    pill. A fine-print line under each card carries the model, corpus, target
    CER, and realized CER -- never a subtype caption. Layout only; the six
    displayed rows and their verbatim text are fixed by
    ``select_validated_examples``."""
    ax.axis("off")
    ax.patch.set_visible(False)  # bbox_inches="tight" crops to rendered content

    by_class: dict[str, list[dict]] = {c: [] for c in class_order}
    for ex in examples:
        by_class[ex["label"]].append(ex)

    total = _total_units(by_class, class_order)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total)
    ax.set_autoscale_on(False)  # keep imshow logos from rescaling the panel
    renderer = ax.figure.canvas.get_renderer()
    space_w = _space_width_data(ax, renderer, 9.4)  # output word spacing

    # Horizontal geometry (axes fraction).
    band_x0, band_x1 = 0.006, 0.994        # tinted section band
    card_x0, card_x1 = 0.020, 0.988        # example card
    x_txt = 0.036                          # text left padding inside a card
    x_arrow1, x_noisy = 0.312, 0.352
    x_arrow2, x_output = 0.616, 0.652
    x_badge = 0.955                        # badge centre
    hdr_color = "#6b7075"

    # ---- schematic: each column is a pipeline stage, drawn as an actual glyph
    # (P300 grid, decoder chip, language-model bubble, two-physician panel)
    # centred over its column so the banner and the table read as one grid ----
    sch_cy = total - _LAYOUT["header"] * 0.36
    nx1, nx2, nx3, nx4 = 0.15, 0.50, 0.79, 0.955
    lab_cy = sch_cy - 0.92
    _draw_speller_grid(ax, nx1, sch_cy, cell_w=0.0104, highlight="H")
    _icon_decoder(ax, nx2, sch_cy, 0.052)
    _icon_llm(ax, nx3, sch_cy, 0.056)                        # drawn speech-bubble glyph (preferred)
    _icon_physicians(ax, nx4, sch_cy, 0.052)                 # drawn two-clinician glyph (preferred)
    for nx, lab in ((nx1, "P300 speller"), (nx2, "Neural decoder"),
                    (nx3, "Language model"), (nx4, "Physician panel (2)")):
        ax.text(nx, lab_cy, lab, fontsize=6.6, fontweight="bold", color=hdr_color,
                ha="center", va="center")
    _draw_flow_arrow(ax, nx1 + 0.052, nx2 - 0.055, sch_cy, "decode")
    _draw_flow_arrow(ax, nx2 + 0.055, nx3 - 0.058, sch_cy, "post-edit")
    _draw_flow_arrow(ax, nx3 + 0.058, nx4 - 0.05, sch_cy, "adjudicate")

    # ---- editorial column headers, aligned under the schematic stages ----
    hdr_cy = total - _LAYOUT["header"] + 0.52
    ax.text(x_txt, hdr_cy, "INTENDED MESSAGE", fontsize=7.4, fontweight="bold",
            color=hdr_color, ha="left", va="center")
    ax.text(x_noisy, hdr_cy, "NOISY DECODER TEXT", fontsize=7.4, fontweight="bold",
            color=hdr_color, ha="left", va="center")
    ax.text(x_output, hdr_cy, "MODEL OUTPUT", fontsize=7.4, fontweight="bold",
            color=hdr_color, ha="left", va="center")
    ax.text(band_x1, hdr_cy, "CONCORDANT PHYSICIAN LABEL", fontsize=7.4,
            fontweight="bold", color=hdr_color, ha="right", va="center")
    ax.plot([band_x0, band_x1], [total - _LAYOUT["header"]] * 2, color="#3a3f44",
            lw=1.0, zorder=3)

    y = total - _LAYOUT["header"]
    for cls in class_order:
        rows = by_class[cls]
        color = _LABEL_COLOR.get(cls, INK)
        sec_h = _LAYOUT["sechead"] + len(rows) * _LAYOUT["card"] + max(len(rows) - 1, 0) * _LAYOUT["cardgap"]
        sec_top, sec_bot = y, y - sec_h

        # Soft section wash + a saturated left accent rail, colour-keyed.
        ax.add_patch(Rectangle((band_x0, sec_bot), band_x1 - band_x0, sec_h,
                               facecolor=color, alpha=0.05, edgecolor="none", zorder=0))
        ax.add_patch(Rectangle((band_x0, sec_bot), 0.008, sec_h,
                               facecolor=color, alpha=0.92, edgecolor="none", zorder=1))

        # Section header: a solid colour tag + a plain-language descriptor.
        sh_cy = sec_top - _LAYOUT["sechead"] * 0.5
        ax.text(x_txt, sh_cy, " " + _LABEL_DISPLAY.get(cls, cls).upper() + " ",
                fontsize=10.0, fontweight="bold", color="white", ha="left", va="center",
                zorder=4, bbox=dict(boxstyle="round,pad=0.36", fc=color, ec="none"))
        ax.text(band_x1 - 0.006, sh_cy, _LABEL_DESC.get(cls, ""), fontsize=8.6,
                color=MUTED, ha="right", va="center", style="italic")

        cy = sec_top - _LAYOUT["sechead"]
        for ex in rows:
            card_bot = cy - _LAYOUT["card"]
            pl_cy = card_bot + _LAYOUT["card"] * 0.62   # pipeline line
            fp_cy = card_bot + _LAYOUT["card"] * 0.26   # fine print

            ax.add_patch(Rectangle(
                (card_x0, card_bot + 0.08), card_x1 - card_x0, _LAYOUT["card"] - 0.18,
                facecolor=_PAPER, edgecolor=_CARD_EC, lw=0.8, zorder=2))

            # Intended message: the reference, quiet dark ink.
            ax.text(x_txt, pl_cy, ex["intended_text"], fontsize=9.6, color=_INK_SOFT,
                    ha="left", va="center", zorder=5)

            # Noisy decode: monospaced, with the corrupted characters standing
            # out from the recessive clean ones, on a light "signal" chip.
            cflags = _corruption_flags(ex["intended_text"], ex["noisy_text"])
            nspans = [(ch, _NOISE_CORRUPT if f else _NOISE_CLEAN,
                       "bold" if f else "normal")
                      for ch, f in zip(ex["noisy_text"], cflags)]
            n_end = _draw_spans(ax, renderer, x_noisy, pl_cy, nspans, fontsize=8.6,
                                family="monospace", zorder=5)
            ax.add_patch(Rectangle((x_noisy - 0.008, pl_cy - 0.235),
                                   (n_end - x_noisy) + 0.016, 0.47,
                                   facecolor=_CHIP_FC, edgecolor="none", zorder=3))

            # Model output: preserved words in quiet ink, the words the model
            # changed in the outcome colour -- so drift shows the exact flipped
            # word and faithful reads as intact.
            _draw_words(ax, renderer, x_output, pl_cy,
                        _output_word_specs(ex["intended_text"], ex["output_message"], color),
                        fontsize=9.6, space_w=space_w, zorder=5)

            # Per-specimen flow: thin arrows tracing intended -> noisy -> output
            # and a dotted lead into the concordant-label pill, so each row is
            # read as one pass through the pipeline (not four disconnected cells).
            for ax_c in (0.331, 0.633):
                ax.annotate("", xy=(ax_c + 0.013, pl_cy), xytext=(ax_c - 0.013, pl_cy),
                            arrowprops=dict(arrowstyle="-|>", color="#c3c8cd", lw=1.2,
                                            shrinkA=0, shrinkB=0), zorder=4)
            ax.plot([0.874, x_badge - 0.032], [pl_cy, pl_cy], color="#d4d8dc",
                    lw=0.9, ls=(0, (1.4, 1.7)), zorder=3)

            ax.text(x_badge, pl_cy, _LABEL_DISPLAY.get(ex["label"], ex["label"]),
                    fontsize=8.2, fontweight="bold", color=color, ha="center",
                    va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.40", fc=color, ec=color, lw=0.8,
                              alpha=0.16))
            # Fine print (model/corpus/target) under the intended column, and a
            # realized-CER corruption gauge under the decode column: a data mark
            # per row, so the amount of corruption is seen, not just read.
            ax.text(x_txt, fp_cy,
                    f"{ex['model']} · {ex['corpus']} · target CER "
                    f"{ex['cer_target'] * 100:.0f}%",
                    fontsize=7.0, color=MUTED, ha="left", va="center", style="italic",
                    zorder=5)
            _draw_cer_gauge(ax, x_noisy, fp_cy, 0.11, ex["realized_cer"],
                            f"{ex['realized_cer'] * 100:.0f}% realized CER")
            cy = card_bot - _LAYOUT["cardgap"]

        y = sec_bot - _LAYOUT["secgap"]


def make_figure2(
    pool: pd.DataFrame,
    out_stem: str = "output/figures/Figure2",
    n_per_class: int = _N_PER_CLASS,
    class_order: tuple = _CLASS_ORDER,
) -> dict:
    """Render Figure 2 (no in-figure title, per house style: the caption
    carries the title) and save PDF + PNG at 600 DPI, TrueType-embedded
    (``pdf.fonttype=42`` from ``_apply_style``, not the default Type-3).
    Returns the selected examples for downstream inspection/testing."""
    examples = select_validated_examples(pool, n_per_class=n_per_class, class_order=class_order)

    _apply_style()
    by_class: dict[str, list[dict]] = {c: [] for c in class_order}
    for ex in examples:
        by_class[ex["label"]].append(ex)
    fig_h = _total_units(by_class, class_order) * _UNIT_IN
    fig, ax = plt.subplots(1, 1, figsize=(12.6, fig_h))
    # Set the final axes geometry BEFORE building the panel: _examples_panel
    # measures text width against the axes' data<->pixel scale to place the
    # diff-highlighted spans, so the axes must already be at its rendered size.
    fig.subplots_adjust(left=0.015, right=0.99, top=0.995, bottom=0.005)
    fig.canvas.draw()
    _examples_panel(ax, examples, class_order=class_order)
    _save(fig, out_stem)

    return {"examples": examples}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(root: str = ".") -> None:
    root_p = Path(root)
    pool = load_validated_pool(
        rater_a_paths=[str(root_p / p) for p in _DEFAULT_RATER_A],
        rater_b_paths=[str(root_p / p) for p in _DEFAULT_RATER_B],
        key_path=str(root_p / _DEFAULT_KEY),
    )
    result = make_figure2(pool, str(root_p / "output/figures/Figure2"))
    printable = [
        {k: v for k, v in ex.items() if k != "source_index"} for ex in result["examples"]
    ]
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
