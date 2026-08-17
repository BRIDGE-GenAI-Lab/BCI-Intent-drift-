"""Physician-panel classifier diagnostics, STRATIFIED (revision Task B5).

`idrift.adjudicate.analyze_panel.auto_vs_human` (eTable 6) reports the
automated ensemble's class-specific agreement against the physician
consensus POOLED over the whole 1,295-item panel. Reviewers asked whether
that pooled number hides heterogeneity: maybe the ensemble is reliable on
the authentic corpus but weaker on the message-critical challenge set,
or weaker at high corruption, or weaker for one model than another --
pooling could mask any of that. This module recomputes
`idrift.adjudicate.validation_metrics.class_metrics` (sensitivity,
specificity, PPV, NPV, F1, support, per class) independently within each
level of a stratifying variable, plus that level's own overall percent
agreement, Cohen's kappa, and n.

Three stratifications (`by`):

  * ``"corpus"``   -- AUTH / CRIT / CTRL, from the key's ``category``
                      column (``"message_critical"`` renamed ``"CRIT"``,
                      matching `analyze_panel.corrected_drift_by_corpus`'s
                      existing convention; ``"AUTH"``/``"CTRL"`` pass
                      through unchanged).
  * ``"cer_band"`` -- ``"low"`` (``cer_target <= 0.2``) vs ``"high"``
                      (``cer_target > 0.2``).
  * ``"model"``    -- the key's ``model_id`` column directly, one stratum
                      per model.

Thin strata are FLAGGED, never dropped or hidden: any stratum with fewer
than `SMALL_STRATUM_N` (30) resolved items carries `small_stratum=True` in
its output, but its metrics are still computed and reported -- silently
hiding an unreliable-looking number for a thin corpus/CER-band/model cell
would be worse than reporting it with its caveat attached.

Reuses, rather than re-derives:
  * `idrift.adjudicate.reconcile.validation_frame` for the join from
    resolved human-consensus labels to the automated label (the same
    structural guard `analyze_panel.auto_vs_human` uses, so a
    human-rated item can never be silently dropped from a stratum).
  * `idrift.adjudicate.reconcile.cohen_kappa` for the per-stratum kappa.
  * `idrift.adjudicate.validation_metrics.class_metrics` /
    `CLASS_ORDER` for the per-class table, so every stratum's class rows
    are in the same fixed (faithful, degraded, drift) order and use the
    same documented zero-division convention as the pooled eTable 6.

Deterministic; American spelling; no em dashes (double hyphens for asides,
per house style).
"""
from __future__ import annotations

import pandas as pd

from idrift.adjudicate.reconcile import cohen_kappa, validation_frame
from idrift.adjudicate.validation_metrics import CLASS_ORDER, class_metrics

# Below this many resolved items, a stratum's class-specific metrics are
# reported (never dropped) but flagged unreliable via `small_stratum=True`.
SMALL_STRATUM_N = 30

# cer_target <= this value bands "low"; strictly above bands "high".
CER_BAND_THRESHOLD = 0.2

BY_CHOICES = ("corpus", "cer_band", "model")

# Columns pulled from the key alongside the automated label, so any of the
# three `by` groupings can be computed after a single join.
_KEY_STRATUM_SOURCE_COLUMNS = ["category", "cer_target", "model_id"]


def _corpus_column(df: pd.DataFrame) -> pd.Series:
    """AUTH / CRIT / CTRL, from `category` (``message_critical`` -> ``CRIT``).

    Mirrors `analyze_panel.corrected_drift_by_corpus`'s existing corpus
    convention exactly, so corpus labels agree between eTable 6's pooled
    view and this stratified one.
    """
    return df["category"].where(df["category"] != "message_critical", "CRIT")


def _cer_band_column(df: pd.DataFrame) -> pd.Series:
    """``"low"`` (``cer_target <= CER_BAND_THRESHOLD``) vs ``"high"``."""
    is_low = df["cer_target"] <= CER_BAND_THRESHOLD
    return is_low.map({True: "low", False: "high"})


def _stratum_column(df: pd.DataFrame, by: str) -> pd.Series:
    if by == "corpus":
        return _corpus_column(df)
    if by == "cer_band":
        return _cer_band_column(df)
    if by == "model":
        return df["model_id"]
    raise ValueError(f"by must be one of {BY_CHOICES}, got {by!r}")


def build_validation_frame(consensus_df: pd.DataFrame, key_df: pd.DataFrame) -> pd.DataFrame:
    """Resolved human-consensus labels joined to the automated label + the
    columns needed to compute any of the three `by` stratifications.

    Args:
        consensus_df: one row per item with a `human` column (the resolved
            3-class consensus label; NaN for an unresolved item) and
            `item_id` -- e.g. `idrift.adjudicate.analyze_panel.
            consensus_labels`'s output. Unresolved items (`human` is NaN)
            are dropped here, matching `analyze_panel.auto_vs_human`'s own
            convention (an unresolved item cannot be scored against the
            automated label because it has no reference label).
        key_df: the unblinding key (`item_id`, `automated_final_label`,
            `category`, `cer_target`, `model_id`, ...).

    Returns:
        DataFrame: one row per resolved item, with `human`, `auto_label`,
        and the stratum source columns (`category`, `cer_target`,
        `model_id`) attached. Uses
        `idrift.adjudicate.reconcile.validation_frame` (`on_missing="raise"`)
        for the join, so a resolved item missing from the key fails loudly
        rather than silently shrinking a stratum's denominator.
    """
    resolved = consensus_df.dropna(subset=["human"]).copy()
    auto = key_df.rename(columns={"automated_final_label": "auto_label"})[
        ["item_id", "auto_label"] + _KEY_STRATUM_SOURCE_COLUMNS
    ]
    return validation_frame(resolved, auto, on_missing="raise", auto_col="auto_label")


def stratified_class_metrics(consensus_df: pd.DataFrame, key_df: pd.DataFrame, by: str) -> dict:
    """Class-specific automated-vs-human agreement, recomputed independently
    within each level of `by`, with thin strata flagged rather than dropped.

    Args:
        consensus_df: resolved human-consensus labels (see
            `build_validation_frame`).
        key_df: the unblinding key (see `build_validation_frame`).
        by: one of `BY_CHOICES` -- `"corpus"`, `"cer_band"`, or `"model"`.

    Returns:
        dict: ``{"by": by, "small_stratum_threshold": SMALL_STRATUM_N,
        "strata": {level: {"n", "small_stratum", "overall_percent_agreement",
        "cohen_kappa_overall", "class_metrics"}}}``. `class_metrics` is a
        list of per-class dicts (class, sensitivity, specificity, ppv, npv,
        f1, support) in the fixed `CLASS_ORDER`. Strata are actual
        partition-and-recompute subsets (each stratum's `class_metrics`
        equals `validation_metrics.class_metrics` called directly on that
        stratum's own human/auto subset) -- never a pooled metric repeated
        per level.

    Raises:
        ValueError: if `by` is not one of `BY_CHOICES`.
    """
    if by not in BY_CHOICES:
        raise ValueError(f"by must be one of {BY_CHOICES}, got {by!r}")

    vf = build_validation_frame(consensus_df, key_df)
    vf = vf.assign(_stratum=_stratum_column(vf, by))

    strata_out = {}
    for level, group in vf.groupby("_stratum", sort=True):
        human = group["human"]
        auto = group["auto_label"]
        n = int(len(group))
        strata_out[level] = {
            "n": n,
            "small_stratum": n < SMALL_STRATUM_N,
            "overall_percent_agreement": float((human.values == auto.values).mean()),
            "cohen_kappa_overall": cohen_kappa(human, auto),
            "class_metrics": class_metrics(human, auto, labels=CLASS_ORDER).to_dict(orient="records"),
        }

    return {
        "by": by,
        "small_stratum_threshold": SMALL_STRATUM_N,
        "strata": strata_out,
    }
