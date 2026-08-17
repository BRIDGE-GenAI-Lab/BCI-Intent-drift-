"""Class-specific automated-vs-human validation metrics (revision Task 4.2).

Reviewers rejected Cohen's kappa alone as sufficient justification for
treating the automated pipeline's labels as ground truth for the full
cohort: a single scalar agreement statistic cannot show whether the
pipeline is, say, systematically insensitive to `drift` while agreeing well
on `faithful`. This module characterizes the automated pipeline AGAINST the
human-consensus labels -- `human` is always the reference, `auto` is always
the classifier being evaluated, never the reverse -- with per-class
one-vs-rest sensitivity (recall), specificity, PPV (precision), NPV, F1, and
support, plus the full confusion matrix, optionally broken down by one or
more stratum columns (CER target, model, critical-subtype, ...).

Outcome taxonomy and fixed label order
---------------------------------------
The study's outcome taxonomy is exactly {faithful, degraded, drift}
(`idrift.adjudicate.taxonomy`). `CLASS_ORDER` below fixes that order for
every confusion matrix, `class_metrics` frame, and `by_stratum` frame this
module produces, so results are always directly comparable across strata
and across tables -- a class never silently shifts row/column position
because of how a particular subset happened to be ordered.

By default (`labels=None`), `human`/`auto` values are validated against
exactly `CLASS_ORDER`: an unexpected value (e.g., a typo, or a leaked
fourth class) raises `ValueError` naming it, rather than sklearn's
`confusion_matrix(labels=...)` silently excluding it from every row/column
sum. Pass `labels=` explicitly to use a different, fixed label universe
(e.g., in a test fixture that is not the study's real taxonomy).

Optional per-item weights (Task 7, reviewer #1 follow-up)
----------------------------------------------------------
`confusion`, `class_metrics`, and `by_stratum` all accept an optional
`weights=` sequence, one value per item, threaded straight through to
`sklearn.metrics.confusion_matrix`'s own `sample_weight=`. This exists for
the 16-model stratified human-rating panel (`build_panel16_sheet.py`),
whose items are drawn from a disproportionate-by-design stratified sample
(every nonempty model x corpus x cer_target x label cell contributes up to
`n_per_cell` items regardless of that cell's population) and each carry a
`sampling_weight = stratum population / stratum sample size` -- so a
reweighted confusion matrix/class-metrics table reflects the FULL labeled
population each stratum was drawn from, not the raw (disproportionate)
rated-sample counts.

`weights=None` (the default, and the value every pre-Task-7 caller uses
implicitly) is a hard backward-compatibility requirement: passing
`sample_weight=None` to `confusion_matrix` is byte-identical to omitting
the argument entirely (same values, same `int64` dtype), so every existing
caller and test is completely unaffected. When per-item weights ARE
supplied, the confusion matrix becomes a weighted-sum matrix (`float64`)
instead of an integer count matrix, and every downstream ratio
(sensitivity, specificity, PPV, NPV, F1) and `support` is computed from
that weighted matrix by the exact same formulas as the unweighted case.
Because those ratios are all built from sums of the SAME per-item weight
scale, giving every item an equal (even non-unit) weight leaves every
ratio unchanged from the plain unweighted call -- only `support` (a
weighted sum, not a ratio) scales with the constant weight.

Zero-division convention (the reviewer will check this)
-----------------------------------------------------------
Every per-class ratio below is derived directly from that class's
one-vs-rest 2x2 confusion cell (TP, FN, FP, TN), computed from the full
confusion matrix -- not composed from other already-rounded ratios, and not
delegated to two different scikit-learn calls that could in principle
disagree with each other.

An UNDEFINED ratio (both numerator and denominator are structurally zero,
i.e. a 0/0) is reported as `0.0` by convention (matching scikit-learn's own
`zero_division=0` default for precision/recall/F1), NOT as `NaN`. This is a
documented, tested convention, not a silent fallback: a class entirely
absent from `human` (support 0) has sensitivity 0/0 -> 0.0 and PPV may
still be a REAL, well-defined 0.0 (e.g. 0 true positives over 1 false
positive) rather than a 0/0 case -- both land on the same numeric value by
different paths, and the tests in `tests/adjudicate/test_validation_metrics.py`
verify each path by hand arithmetic separately so the convention cannot be
mistaken for silently corrupting a real rate. Specificity and NPV are
frequently well-defined even for a zero-support class (e.g. specificity is
1.0 when a class was never predicted at all, since there are then zero
false positives), and are computed for real in that case, not forced to a
fixed value.

Deterministic; American spelling; no em dashes (double hyphens for asides,
per house style).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

CLASS_ORDER = ("faithful", "degraded", "drift")

_METRIC_COLUMNS = ["class", "sensitivity", "specificity", "ppv", "npv", "f1", "support"]


def _safe_div(numerator: float, denominator: float) -> float:
    """Return `numerator / denominator`, or `0.0` if `denominator == 0`
    (the documented zero-division convention -- see module docstring)."""
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _resolve_labels(human: Sequence, auto: Sequence, labels: Sequence | None) -> list:
    """Return the fixed label order to use for a confusion matrix / class
    metrics frame. `labels=None` (the default) means "the study's own fixed
    3-class taxonomy" (`CLASS_ORDER`), and validates that `human`/`auto`
    contain nothing else -- an unexpected label raises `ValueError` naming
    it, rather than sklearn's `confusion_matrix` silently excluding it from
    every row/column sum it participates in. Pass `labels=` explicitly to
    use a different fixed label universe (e.g. a non-taxonomy test
    fixture)."""
    if labels is not None:
        return list(labels)

    observed = set(pd.unique(pd.Series(list(human)))) | set(pd.unique(pd.Series(list(auto))))
    unexpected = observed - set(CLASS_ORDER)
    if unexpected:
        raise ValueError(
            f"human/auto contain label(s) outside the fixed 3-class taxonomy "
            f"{CLASS_ORDER}: {sorted(unexpected)}. Pass labels= explicitly to "
            f"use a different label universe."
        )
    return list(CLASS_ORDER)


def _check_equal_length(human: Sequence, auto: Sequence) -> None:
    if len(human) != len(auto):
        raise ValueError(
            f"human and auto must be the same length (got {len(human)} vs {len(auto)})"
        )


def _resolve_weights(length: int, weights: Sequence | None):
    """Validate and convert an optional per-item `weights=` sequence to a
    plain float `numpy.ndarray`, or return `None` unchanged.

    `None` (the default) is returned as-is, not converted to an all-ones
    array: passing `sample_weight=None` to `sklearn.metrics.confusion_matrix`
    is byte-identical to omitting the argument, which is exactly the
    backward-compatibility guarantee this module makes (see module
    docstring). Converting `None` to an explicit all-ones array here would
    silently change the confusion matrix's dtype (int64 -> float64) for
    every existing caller.
    """
    if weights is None:
        return None
    w = np.asarray(list(weights), dtype=float)
    if len(w) != length:
        raise ValueError(
            f"weights must be the same length as human/auto (got {len(w)} vs {length})"
        )
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    return w


def confusion(
    human: Sequence, auto: Sequence, labels: Sequence | None = None, weights: Sequence | None = None
) -> pd.DataFrame:
    """Full confusion matrix, human (rows) x auto (columns), fixed label
    order (see module docstring). `human[i]`/`auto[i]` are the reference and
    automated labels for the same item i (already aligned; joining a raw
    human-consensus table to an automated-label table on `item_id` is
    `idrift.adjudicate.reconcile.validation_frame`'s job, upstream of this
    function).

    Args:
        human: reference (human-consensus) labels, one per item.
        auto: automated-pipeline labels, one per item, same order as human.
        labels: fixed label order. Defaults to the study's 3-class taxonomy
            (`CLASS_ORDER`); see `_resolve_labels`.
        weights: optional per-item weights (e.g. a stratified panel's
            `sampling_weight`), same order as `human`/`auto`. `None` (the
            default) is byte-identical to the pre-Task-7 unweighted
            behavior (see module docstring); non-`None` produces a
            weighted-sum (`float64`) matrix instead of an integer count
            matrix.

    Returns:
        DataFrame: NxN matrix (N = len(labels)), index named "human",
        columns named "auto", cell [i, j] = count (or, if `weights` is
        given, sum of weights) of items with human label labels[i] and auto
        label labels[j].
    """
    human = list(human)
    auto = list(auto)
    _check_equal_length(human, auto)
    resolved = _resolve_labels(human, auto, labels)
    weights_arr = _resolve_weights(len(human), weights)

    cm = confusion_matrix(human, auto, labels=resolved, sample_weight=weights_arr)
    return pd.DataFrame(
        cm,
        index=pd.Index(resolved, name="human"),
        columns=pd.Index(resolved, name="auto"),
    )


def class_metrics(
    human: Sequence, auto: Sequence, labels: Sequence | None = None, weights: Sequence | None = None
) -> pd.DataFrame:
    """Per-class one-vs-rest sensitivity, specificity, PPV, NPV, F1, and
    support, with `human` as the reference labels and `auto` as the
    classifier being evaluated against it.

    Each class's metrics are derived directly from that class's one-vs-rest
    2x2 cell of the full confusion matrix (TP = human==c & auto==c, FN =
    human==c & auto!=c, FP = human!=c & auto==c, TN = human!=c & auto!=c):

        sensitivity (recall) = TP / (TP + FN)   [= TP / support]
        specificity           = TN / (TN + FP)
        ppv (precision)       = TP / (TP + FP)
        npv                   = TN / (TN + FN)
        f1                    = 2*TP / (2*TP + FP + FN)

    An undefined (0/0) ratio is `0.0` by documented convention; see the
    module docstring for the zero-division rule and why specificity/NPV are
    frequently well-defined (not convention-forced) even for a zero-support
    class.

    Args:
        human: reference (human-consensus) labels, one per item.
        auto: automated-pipeline labels, one per item, same order as human.
        labels: fixed label order. Defaults to the study's 3-class taxonomy
            (`CLASS_ORDER`); see `_resolve_labels`.
        weights: optional per-item weights, same order as `human`/`auto`.
            `None` (the default) reproduces the exact pre-Task-7 unweighted
            computation, including `support`'s Python-`int` dtype (see
            module docstring for the backward-compatibility guarantee).
            When given, every ratio is computed from the weighted
            confusion cells instead of raw counts, and `support` is the
            weighted (float) sum rather than an item count.

    Returns:
        DataFrame with columns ["class", "sensitivity", "specificity",
        "ppv", "npv", "f1", "support"], one row per label in `labels`
        (or the resolved default), in that fixed order.
    """
    human = list(human)
    auto = list(auto)
    _check_equal_length(human, auto)
    resolved = _resolve_labels(human, auto, labels)
    weights_arr = _resolve_weights(len(human), weights)

    cm = confusion_matrix(human, auto, labels=resolved, sample_weight=weights_arr)

    # `_cast` is `int` for weights_arr is None (the exact pre-Task-7 dtype,
    # so that path is byte-identical to the original implementation) and
    # `float` when weighted (weighted cells are already float64; `float()`
    # is then a no-op cast, kept for symmetry/clarity).
    _cast = int if weights_arr is None else float
    total = _cast(cm.sum())

    rows = []
    for i, cls in enumerate(resolved):
        tp = _cast(cm[i, i])
        fn = _cast(cm[i, :].sum()) - tp
        fp = _cast(cm[:, i].sum()) - tp
        tn = total - tp - fn - fp
        rows.append(
            {
                "class": cls,
                "sensitivity": _safe_div(tp, tp + fn),
                "specificity": _safe_div(tn, tn + fp),
                "ppv": _safe_div(tp, tp + fp),
                "npv": _safe_div(tn, tn + fn),
                "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
                "support": tp + fn,
            }
        )

    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)


def by_stratum(
    human: Sequence, auto: Sequence, strata, labels: Sequence | None = None, weights: Sequence | None = None
) -> pd.DataFrame:
    """`class_metrics`, recomputed independently within each level of one or
    more stratum columns (e.g. cer_target, model_id, critical-subtype),
    returned tidy (stratum column(s) + class + metrics). Each stratum
    level's numbers are computed from that level's own subset of
    human/auto -- an actual partition-and-recompute, not one global metric
    repeated for every level (see the by_stratum tests, which assert each
    stratum's rows equal `class_metrics` called directly on that stratum's
    subset, and that distinct strata produce distinct numbers).

    Args:
        human: reference (human-consensus) labels, one per item.
        auto: automated-pipeline labels, one per item, same order as human.
        strata: a `pandas.Series` (single stratum column; its `.name` is
            used as the column name), a `pandas.DataFrame` (one or more
            stratum columns), or anything `pandas.DataFrame(...)` accepts
            (e.g. a dict of column_name -> sequence). Must have the same
            length as `human`/`auto`.
        labels: fixed label order, applied identically within every
            stratum level so a class absent from one level's subset still
            gets an explicit (zero-support) row rather than silently
            vanishing from that level's table. Defaults to the study's
            3-class taxonomy (`CLASS_ORDER`).
        weights: optional per-item weights, same order as `human`/`auto`.
            `None` (the default) reproduces the exact pre-Task-7 unweighted
            computation. When given, each stratum level's own weight slice
            is passed to `class_metrics` for that level -- an actual
            partition-and-reweight, matching the unweighted partition-and-
            recompute contract above.

    Returns:
        DataFrame: columns = stratum column(s) (in the order given) +
        ["class", "sensitivity", "specificity", "ppv", "npv", "f1",
        "support"], one row per (stratum level, class) combination.
    """
    human_s = pd.Series(list(human)).reset_index(drop=True)
    auto_s = pd.Series(list(auto)).reset_index(drop=True)
    _check_equal_length(human_s, auto_s)

    if isinstance(strata, pd.Series):
        strata_df = strata.to_frame(name=strata.name or "stratum").reset_index(drop=True)
    elif isinstance(strata, pd.DataFrame):
        strata_df = strata.reset_index(drop=True)
    else:
        strata_df = pd.DataFrame(strata).reset_index(drop=True)

    if len(strata_df) != len(human_s):
        raise ValueError(
            f"strata must have the same length as human/auto "
            f"(got {len(strata_df)} vs {len(human_s)})"
        )

    resolved = _resolve_labels(human_s, auto_s, labels)
    weights_arr = _resolve_weights(len(human_s), weights)
    stratum_cols = list(strata_df.columns)

    combined = strata_df.copy()
    combined["_human"] = human_s
    combined["_auto"] = auto_s
    if weights_arr is not None:
        combined["_weight"] = weights_arr

    pieces = []
    for key, group in combined.groupby(stratum_cols, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        group_weights = group["_weight"] if weights_arr is not None else None
        metrics_df = class_metrics(group["_human"], group["_auto"], labels=resolved, weights=group_weights)
        prefix = pd.DataFrame([key_tuple] * len(metrics_df), columns=stratum_cols)
        piece = pd.concat([prefix.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)
        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True)


def macro_weighted(class_df: pd.DataFrame) -> dict:
    """Macro-F1 (unweighted mean of per-class F1) and support-weighted F1,
    from a `class_metrics`-shaped frame.

    Args:
        class_df: a DataFrame with at least "f1" and "support" columns
            (typically the output of `class_metrics`).

    Returns:
        dict: {"macro_f1": float, "weighted_f1": float}. `weighted_f1` is
        `0.0` (documented convention, not NaN) if total support is 0.
    """
    required = {"f1", "support"}
    missing = required - set(class_df.columns)
    if missing:
        raise ValueError(f"class_df missing required column(s): {sorted(missing)}")

    macro_f1 = float(class_df["f1"].mean())
    total_support = float(class_df["support"].sum())
    weighted_f1 = _safe_div((class_df["f1"] * class_df["support"]).sum(), total_support)

    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1}
