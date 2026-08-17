import pandas as pd
from sklearn.metrics import cohen_kappa_score

REQUIRED_RATER_COLUMNS = {"item_id", "rating"}


def reconcile(nli, judge, human):
    """Reconcile three label sources with priority: human > judge > nli.

    Presence is checked with `pandas.notna`, not `is not None`: rows produced
    by a left-merge of a partial human/judge rating table onto the full
    attempts frame carry pandas NaN (float) for unmatched rows, not Python
    None, and `NaN is not None` is True -- so a naive `is not None` check
    would wrongly treat an unrated row as rated.

    Returns:
        tuple: (final_label, source) where source is "human", "judge", or "nli"
    """
    if pd.notna(human):
        return human, "human"
    if pd.notna(judge):
        return judge, "judge"
    return nli, "nli"


def cohen_kappa(a, b):
    """Compute Cohen's kappa between two sequences.

    Args:
        a: First sequence of labels
        b: Second sequence of labels

    Returns:
        float: Cohen's kappa score
    """
    return float(cohen_kappa_score(a, b))


def assign_final(df):
    """Add final_label, label_source, and critical flag to dataframe.

    Args:
        df: DataFrame with columns nli_label, judge_label, human_label, category

    Returns:
        DataFrame: Copy with new columns added
    """
    df = df.copy()
    fl, src = zip(*[reconcile(r.nli_label, r.judge_label, r.human_label) for r in df.itertuples()])
    df["final_label"] = fl
    df["label_source"] = src
    df["critical"] = (df["final_label"] == "drift") & (df["category"] == "message_critical")
    return df


def load_rater_csv(path) -> pd.DataFrame:
    """Load one completed rater CSV (as produced by
    `adjudicate.build_rating_sheet` and filled in by a human rater).

    Args:
        path: Path to a rater's `rating_sheet_rater{A,B}.csv`.

    Returns:
        DataFrame: the raw rater CSV, unmodified (columns at minimum
        `item_id`, `rating`; `notes` if present).

    Raises:
        ValueError: if `item_id` or `rating` is missing, so a malformed or
        wrong sheet fails loudly instead of silently merging garbage.
    """
    df = pd.read_csv(path)
    missing = REQUIRED_RATER_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"rater CSV {path} missing required column(s): {sorted(missing)}")
    return df


def merge_ratings(rater_a_df: pd.DataFrame, rater_b_df: pd.DataFrame) -> pd.DataFrame:
    """Align two independent human raters' completed sheets on `item_id` and
    flag disagreements, for the human dual-rater layer (manuscript_facts.md
    Sec. 7 layer 3). Distinct from `assign_final`, which reconciles
    nli/judge/human label PRIORITY for a single already-merged `human_label`
    column; this merges the two RAW human raters against each other, which
    nothing upstream does.

    An item rated by only one rater (an incomplete pair) is kept with the
    missing side as NaN and `agree=False`, since agreement cannot be
    established without both ratings; it should route to the third
    adjudicator like any other disagreement.

    Args:
        rater_a_df: Rater A's completed sheet (`item_id`, `rating`, optional
            `notes`), e.g. from `load_rater_csv`.
        rater_b_df: Rater B's completed sheet, same shape.

    Returns:
        DataFrame: one row per `item_id` (outer join), with `rating_a`,
        `rating_b`, `notes_a`, `notes_b` (if present), and a boolean `agree`
        column (case/whitespace-insensitive comparison).
    """
    a = rater_a_df.rename(columns={"rating": "rating_a", "notes": "notes_a"})
    b = rater_b_df.rename(columns={"rating": "rating_b", "notes": "notes_b"})
    merged = a.merge(b, on="item_id", how="outer", suffixes=("", ""))

    def _norm(s):
        return s.str.strip().str.lower() if hasattr(s, "str") else s

    merged["agree"] = (
        merged["rating_a"].notna()
        & merged["rating_b"].notna()
        & (_norm(merged["rating_a"]) == _norm(merged["rating_b"]))
    )
    return merged


def validation_frame(
    human_df: pd.DataFrame,
    auto_df: pd.DataFrame,
    on_missing: str = "raise",
    auto_col: str = "auto_label",
) -> pd.DataFrame:
    """LEFT-join every human-consensus-labeled item onto its automated label,
    for the automated-vs-human agreement comparison (rev Task 4.1).

    This is the structural guard against the reviewer-identified 395-vs-378
    discrepancy: the human panel rated 395 items (17 of which needed
    third-adjudicator resolution after an initial rater disagreement), but a
    prior ad hoc comparison silently computed automated-vs-human agreement on
    only 378 of them -- exactly the 395 minus those 17 disagreement-resolved
    items -- because some upstream step filtered to already-agreed rows
    before joining to the automated labels. That silently dropped precisely
    the hardest, most informative items and inflated the reported agreement.

    The fix here is structural, not a promise to "remember to include
    everyone": this function always keeps every row of `human_df` (the LEFT
    side of an unconditional `how="left"` merge on `item_id`), independent of
    whether that item's rating history involved a disagreement, so
    `len(validation_frame(human_df, auto_df)) == len(human_df)` always holds
    (enforced by an assertion). A stray automated label that never matches a
    human-rated item is dropped (correct: it was never part of the rated
    sample), but a human-rated item can never vanish from the comparison.

    Args:
        human_df: One row per item with the resolved consensus human label
            (must include `item_id`) -- every rated item, including ones
            that needed third-adjudicator resolution after an initial
            disagreement.
        auto_df: One row per item with the automated pipeline's label (must
            include `item_id` and `auto_col`).
        on_missing: What to do if a human-labeled item has no automated
            counterpart. `"raise"` (default) raises `ValueError` naming the
            missing item_ids -- a join gap fails loudly instead of quietly
            shrinking the comparison denominator, exactly the failure mode
            behind the 395-vs-378 discrepancy. `"mark"` instead keeps the
            row (with a NaN `auto_col`) and adds a boolean
            `has_automated_label` column, so a caller who intentionally
            wants to inspect the gap can do so without an exception -- but
            the row is never dropped either way.
        auto_col: Name of the automated-label column in `auto_df` (and in
            the returned frame).

    Returns:
        DataFrame: one row per item in `human_df` (row count always equals
        `len(human_df)`, regardless of `on_missing`).

    Raises:
        ValueError: if `on_missing="raise"` (the default) and at least one
            human-labeled item has no matching automated label.
    """
    if on_missing not in ("raise", "mark"):
        raise ValueError(f"on_missing must be 'raise' or 'mark', got {on_missing!r}")

    merged = human_df.merge(auto_df, on="item_id", how="left", suffixes=("", "_auto"))
    missing_mask = merged[auto_col].isna()

    if on_missing == "raise" and missing_mask.any():
        missing_ids = sorted(merged.loc[missing_mask, "item_id"].tolist())
        raise ValueError(
            f"{int(missing_mask.sum())} human-labeled item(s) have no automated label and "
            f"would be silently dropped from the automated-vs-human comparison "
            f"(the 395-vs-378 bug): {missing_ids}"
        )

    if on_missing == "mark":
        merged["has_automated_label"] = ~missing_mask

    assert len(merged) == len(human_df), (
        "validation_frame must never change the number of human-labeled items "
        f"(got {len(merged)}, expected {len(human_df)})"
    )
    return merged
