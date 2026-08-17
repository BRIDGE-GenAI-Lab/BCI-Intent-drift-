"""eFigure 4: representative faithful, degraded, and drift reconstructions
(Task 13; reviewer minor 7 -- readers need to see the construct, not just
aggregate rates), rendered in the same house style as ``fig_hochberg.py`` /
``fig_v2.py`` (Arial, 600 DPI vector + raster with TrueType-embedded fonts,
no in-figure title, quiet chrome).

Every displayed (intended, noisy, model-output) triple is copied verbatim
from the cached labeled attempts table
(``output/intermediate/attempts_v3plus_labeled.parquet``) -- none of the
text is authored, paraphrased, or edited for the figure. Five rows are
selected, one per fixed spec in ``_EXAMPLE_SPECS``: >=1 faithful, >=1
degraded, and three message-critical drift examples (negation flip, numeral
change, recipient change) spanning four target-CER levels (0.1-0.4), well
past the required minimum of one of each outcome and two CER levels.
Message-critical drift rows are preferred per the reviewer's request, to show
the stakes of a fluent-but-wrong reconstruction.

Selection is a deterministic filter + stable sort (see ``_select_example``),
never a random sample: for a drift spec, only rows where the named
message-critical flag fired *and no other flag also fired* are eligible (so
the flag named in the panel is unambiguously the one responsible), then rows
whose intended/output text would not typeset legibly, or which contain a raw
JSON/tool-call formatting artifact instead of plain text, are excluded; the
alphabetically-first surviving row (by message_id, model, corpus,
replicate_idx) is picked. Specs are filled in the fixed order given in
``_EXAMPLE_SPECS``, and each spec additionally excludes every message_id
already claimed by an earlier spec in the same call (see ``select_examples``),
so the five displayed rows always illustrate five distinct intended
messages rather than repeatedly reusing whichever message happens to sort
first. A given input table therefore always yields byte-identical output.

Run: uv run python idrift/figures/fig_examples.py  (writes eFigure4.pdf/.png)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede pyplot import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from idrift.figures.fig_hochberg import (
    INK,
    KEY,
    ERROR,
    CONTEXT,
    MUTED,
    _apply_style,
    _save,
)

_CRIT_COLS = [
    "crit_negation_flip",
    "crit_numeral_change",
    "crit_recipient_change",
    "crit_urgency_change",
    "crit_actionable_omission",
]

_CRIT_DISPLAY = {
    # Wording matches the five rule-based detector categories named in
    # Methods ("negation flip, numeral or dose change, recipient change,
    # urgency change, and actionable omission").
    "crit_negation_flip": "negation flip",
    "crit_numeral_change": "numeral or dose change",
    "crit_recipient_change": "recipient change",
    "crit_urgency_change": "urgency change",
    "crit_actionable_omission": "actionable omission",
}

_LABEL_COLOR = {"faithful": KEY, "degraded": MUTED, "drift": ERROR}
_LABEL_DISPLAY = {"faithful": "Faithful", "degraded": "Degraded", "drift": "Drift"}

# Panel row order and selection criteria. See module docstring for the
# selection rule. Three distinct message-critical categories and four CER
# levels are used across the three drift rows -- more than the brief's
# minimum of one drift example at two CER levels.
_EXAMPLE_SPECS = [
    dict(key="faithful", label="faithful", crit=None, cer_target=0.1),
    dict(key="degraded", label="degraded", crit=None, cer_target=0.2),
    dict(key="drift_negation", label="drift", crit="crit_negation_flip", cer_target=0.2),
    dict(key="drift_numeral", label="drift", crit="crit_numeral_change", cer_target=0.3),
    dict(key="drift_recipient", label="drift", crit="crit_recipient_change", cer_target=0.4),
]

# Keeps every cell legible in the small-multiples table and excludes the rare
# raw JSON/tool-call formatting artifact some models emit instead of plain
# text. A row failing this check is simply skipped in favor of the next real
# row in sorted order -- never replaced with fabricated or edited text.
_MAX_TEXT_LEN = 45


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def _load_attempts(path: str) -> pd.DataFrame:
    cols = [
        "model", "message_id", "corpus", "cer_target", "realized_cer",
        "replicate_idx", "intended_text", "noisy_text", "output_message",
        "label",
    ] + _CRIT_COLS
    return pd.read_parquet(path, columns=cols)


# --------------------------------------------------------------------------
# Deterministic selection
# --------------------------------------------------------------------------
def _clean_mask(df: pd.DataFrame) -> pd.Series:
    """Rows short enough to typeset legibly and free of raw JSON/tool-call
    formatting artifacts (some models emit a fenced code block instead of a
    plain-text answer)."""
    return (
        ~df["output_message"].str.contains("```", regex=False)
        & (df["output_message"].str.len() <= _MAX_TEXT_LEN)
        & (df["intended_text"].str.len() <= _MAX_TEXT_LEN)
    )


def _select_example(
    df: pd.DataFrame, spec: dict, exclude_message_ids: frozenset[str] = frozenset()
) -> dict:
    sub = df[(df["label"] == spec["label"]) & (df["cer_target"] == spec["cer_target"])]
    if exclude_message_ids:
        sub = sub[~sub["message_id"].isin(exclude_message_ids)]
    crit = spec.get("crit")
    if crit is not None:
        sub = sub[sub[crit].astype(bool)]
        for other in _CRIT_COLS:
            if other != crit:
                sub = sub[~sub[other].astype(bool)]
    sub = sub[_clean_mask(sub)]
    if sub.empty:
        raise ValueError(
            f"no cached row matches spec {spec['key']!r} "
            f"(label={spec['label']!r}, crit={crit!r}, cer_target={spec['cer_target']}, "
            f"excluding {len(exclude_message_ids)} already-used message_id(s))"
        )
    # Stable (mergesort) so ties resolve identically regardless of input row
    # order -- a fixed rule, not a random sample.
    sub = sub.sort_values(
        ["message_id", "model", "corpus", "replicate_idx"], kind="mergesort"
    )
    row = sub.iloc[0]
    return {
        "key": spec["key"],
        "label": row["label"],
        "crit": crit,
        "cer_target": float(row["cer_target"]),
        "realized_cer": float(row["realized_cer"]),
        "model": row["model"],
        "message_id": row["message_id"],
        "corpus": row["corpus"],
        "intended_text": row["intended_text"],
        "noisy_text": row["noisy_text"],
        "output_message": row["output_message"],
        "source_index": row.name,
    }


def select_examples(df: pd.DataFrame, specs: list[dict] | None = None) -> list[dict]:
    """Deterministically select one real cached row per spec (default
    ``_EXAMPLE_SPECS``): filter on outcome label / target CER / (for drift) an
    exclusive message-critical flag, then take the alphabetically-first
    surviving row. Specs are processed in order, and each spec additionally
    excludes every ``message_id`` already picked by an earlier spec in this
    call, so the resulting rows always cover distinct intended messages
    (avoids five panels illustrating the same underlying sentence). Raises
    ``ValueError`` if a spec has no match. Never samples at random -- a given
    ``df`` always yields byte-identical output, independent of row order (see
    the shuffle-invariance test)."""
    specs = specs if specs is not None else _EXAMPLE_SPECS
    examples: list[dict] = []
    used_message_ids: set[str] = set()
    for spec in specs:
        ex = _select_example(df, spec, exclude_message_ids=frozenset(used_message_ids))
        used_message_ids.add(ex["message_id"])
        examples.append(ex)
    return examples


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def _e4_examples_panel(ax, examples: list[dict]) -> None:
    """Small-multiples table: one row per example, columns for target CER,
    intended message, noisy decoder text, model output, and outcome (colored
    by label, using the same KEY/MUTED/ERROR grammar as faithful/degraded/
    drift elsewhere in the house style)."""
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    n = len(examples)
    top, bottom = 0.94, 0.03
    header_y = 0.985
    row_ys = np.linspace(top, bottom, n)
    row_h = (top - bottom) / max(n - 1, 1)

    col_cer, col_intended, col_arrow1 = 0.005, 0.075, 0.315
    col_noisy, col_arrow2, col_output, col_label = 0.345, 0.615, 0.645, 0.945

    headers = [
        (col_cer, "Target\nCER"),
        (col_intended, "Intended message"),
        (col_noisy, "Noisy decoder text"),
        (col_output, "Model output"),
        (col_label, "Outcome"),
    ]
    for x, txt in headers:
        ha = "center" if x == col_label else "left"
        ax.text(x, header_y, txt, fontsize=7.6, fontweight="bold", color=INK,
                ha=ha, va="top")
    ax.plot([0.0, 1.0], [header_y - 0.075, header_y - 0.075], color=INK, lw=0.9,
            zorder=2)

    for i, ex in enumerate(examples):
        y = row_ys[i]
        color = _LABEL_COLOR.get(ex["label"], INK)

        if i > 0:
            ax.plot([0.0, 1.0], [y + row_h * 0.5, y + row_h * 0.5], color=CONTEXT,
                    lw=0.5, alpha=0.6, zorder=1)

        ax.text(col_cer, y, f"{ex['cer_target'] * 100:.0f}%", fontsize=8.4,
                fontweight="bold", color=INK, ha="left", va="center")
        ax.text(col_intended, y, ex["intended_text"], fontsize=7.6, color=INK,
                ha="left", va="center")
        ax.text(col_arrow1, y, "→", fontsize=9, color=MUTED, ha="center",
                va="center")
        ax.text(col_noisy, y, ex["noisy_text"], fontsize=7.6, color=MUTED,
                ha="left", va="center")
        ax.text(col_arrow2, y, "→", fontsize=9, color=MUTED, ha="center",
                va="center")
        ax.text(col_output, y, f'“{ex["output_message"]}”', fontsize=7.8,
                fontweight="bold", color=color, ha="left", va="center")

        label_txt = _LABEL_DISPLAY.get(ex["label"], ex["label"])
        if ex.get("crit"):
            label_txt += f"\n({_CRIT_DISPLAY.get(ex['crit'], ex['crit'])})"
        ax.text(col_label, y, label_txt, fontsize=6.9, fontweight="bold",
                color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none", alpha=0.13))

        ax.text(
            col_intended, y - row_h * 0.34,
            f"{ex['model']} · {ex['message_id']} · realized CER "
            f"{ex['realized_cer'] * 100:.1f}%",
            fontsize=5.6, color=MUTED, ha="left", va="center", style="italic",
        )


def make_efigure_examples(
    attempts: pd.DataFrame, out_stem: str = "output/figures/eFigure4",
    specs: list[dict] | None = None,
) -> dict:
    """Render the single-panel eFigure 4 (no panel letter, consistent with
    eFigure 1-3: a single-panel supplementary figure) and save PDF + PNG at
    600 DPI. Returns the selected examples for downstream inspection/testing."""
    examples = select_examples(attempts, specs=specs)

    _apply_style()
    n = len(examples)
    fig, ax = plt.subplots(1, 1, figsize=(12.4, 1.35 + 0.92 * n))
    _e4_examples_panel(ax, examples)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    _save(fig, out_stem)

    return {"examples": examples}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main(root: str = ".") -> None:
    root_p = Path(root)
    attempts = _load_attempts(
        str(root_p / "output/intermediate/attempts_v3plus_labeled.parquet")
    )
    result = make_efigure_examples(attempts, str(root_p / "output/figures/eFigure4"))
    printable = [
        {k: v for k, v in ex.items() if k != "source_index"} for ex in result["examples"]
    ]
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
