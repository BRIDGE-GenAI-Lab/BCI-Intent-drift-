"""Misclassification-corrected prevalence (revision Task 4.3, the last
offline instrument of Phase 4).

Reviewers required that the automated pipeline be EVALUATED against
human-consensus reference labels on a validation subset, never treated as
ground truth for the full cohort (see `idrift.adjudicate.validation_metrics`,
Task 4.2). That validation subset characterizes the classifier under
evaluation's own error pattern -- its class-specific sensitivity and
cross-class leakage -- but by itself does not tell us the TRUE prevalence
of each outcome class over the full cohort, where we only ever observe the
classifier's (naive) labels. This module closes that gap: every full-cohort
drift estimate can be reported both naively (the classifier's raw label
proportions) and corrected for the classifier's own measured
misclassification, via a multiclass generalization of the Rogan-Gladen
correction.

The math (matrix inversion, q = M^T p)
---------------------------------------
Fix the study's 3-class taxonomy order, `CLASS_ORDER = ("faithful",
"degraded", "drift")` (`idrift.adjudicate.validation_metrics.CLASS_ORDER`).
Let:

    p    = the TRUE prevalence vector over CLASS_ORDER (unknown, what we
           want), p_i = P(true class == CLASS_ORDER[i]).
    q    = the OBSERVED (naive) prevalence vector, q_j = the classifier's
           own label proportions over the full cohort, CLASS_ORDER[j].
    M    = the classifier's confusion matrix, M[i, j] = P(auto == j |
           true == i), estimated from the validation subset (human labels
           standing in for true class there).

For any full-cohort item, the law of total probability over its (unknown)
true class gives:

    q_j = P(auto == j) = sum_i P(true == i) * P(auto == j | true == i)
        = sum_i p_i * M[i, j]

which is exactly `q = M^T p` in matrix form (the transpose because M's
rows are indexed by true class i but the sum for q_j runs over i for a
FIXED j, i.e. over M's column j -- equivalently, over M^T's row j). Given
an ESTIMATED M (from the validation confusion matrix) and an OBSERVED q
(from the full-cohort naive labels), we recover the corrected prevalence
by solving this linear system for p: `numpy.linalg.solve(M.T, q)`, falling
back to `numpy.linalg.lstsq` if `M.T` is (numerically) singular.

A true class with ZERO support in the validation subset (no validation
item was ever labeled that class by the human-consensus reference) cannot
have its confusion-matrix row estimated at all -- there is no data on what
the classifier does to items of that true class. In that case this module
falls back to an IDENTITY row for that class (`M[i, i] = 1.0`, i.e. "if a
cohort item were truly class i, assume the classifier reports it as i
too"): an unobserved true class is assumed perfectly classified, the
conservative no-information choice (it does not manufacture a leakage
pattern that was never observed, and it correctly returns the naive count
unchanged if there is truly nothing else to correct). Any such class is
recorded in the returned `identity_fallback_classes`, never silently
assumed.

Because M is only ESTIMATED (from a finite validation subset) and q is
only OBSERVED (from a finite cohort), solving the linear system can return
a `p` with a small negative component or a component slightly above 1,
even though every true p_i must lie in the probability simplex. This
module projects any such solution back onto the simplex by clipping
negative components to 0 and renormalizing the remainder to sum to 1
(documented, not silent); `clipped` records whether this projection
actually changed anything for a given point estimate.

Confidence intervals (message-clustered bootstrap, or a multinomial
fallback)
-------------------------------------------------------------------------
Generation errors within the same source message are correlated (the same
underlying decode noise or LLM behavior can push every replicate of one
message the same way), so a naive item-level bootstrap would understate
uncertainty. When BOTH `cohort_frame` (one row per full-cohort generation,
columns `["message_id", "auto_label"]`) and `val_frame` (one row per
human-labeled validation item, columns `["message_id", "human", "auto"]`)
are supplied, this module runs a GENUINE message-clustered bootstrap:
each iteration independently resamples the distinct `message_id`s (with
replacement) in each frame, pulls every row belonging to each resampled
message ("cluster bootstrap"), rebuilds `auto_counts` and
`confusion_from_human` (via `idrift.adjudicate.validation_metrics.confusion`
-- reused, never reimplemented, per DRY) from the resampled data, and
recomputes the corrected target prevalence. `ci_method` is then
"message-clustered".

When only the aggregate `auto_counts` / `confusion_from_human` are
available (no `cohort_frame`/`val_frame`, e.g. a downstream caller that
only has the summary counts, not the underlying per-message rows), this
module instead falls back to a PARAMETRIC multinomial bootstrap: each
iteration draws `auto_counts ~ Multinomial(total, q_hat)` and, independently
for each true-class row, `confusion_counts_row ~
Multinomial(row_total, row_prob_hat)`, and recomputes the corrected target
from those simulated counts. `ci_method` is then "multinomial-parametric".
This is a strictly weaker uncertainty model (it assumes item-level
independence, so it will typically understate uncertainty relative to the
message-clustered bootstrap when clustering is present) and is used only
as a fallback when the per-message data is not available.

Both paths draw every random number from a single
`numpy.random.default_rng(seed)` constructed once per call, never the
global `numpy.random` state, so the same inputs and seed always reproduce
bit-for-bit identical output.

Weighted confusion matrices (Task 7, reviewer #1 follow-up)
-------------------------------------------------------------
`confusion_from_human` is just a human(rows) x auto(cols) DataFrame to
`_build_confusion_matrix`, which row-normalizes it into `M` -- it never
assumed the cells were integer item COUNTS. This means a WEIGHTED
confusion matrix (e.g. from `validation_metrics.confusion(human, auto,
weights=sampling_weight)`, for the 16-model stratified panel's
disproportionate-by-design sample) can be passed as `confusion_from_human`
with no code change at all: the row-normalization step divides each
weighted cell by its own weighted row sum, exactly as it already divided
each count cell by its count row sum. Because a UNIFORM per-item weight
(every item weighted by the same constant, even a non-unit one) scales
every cell of a given row by that same constant, the row-normalized `M`
-- and therefore the corrected point estimate -- is unaffected: equal
weights reduce to the current unweighted correction exactly.

The bootstrap CI, however, is a real code path that has to be told about
weights explicitly: `_bootstrap_message_clustered` rebuilds a confusion
matrix from the resampled `val_frame` rows each iteration via
`validation_metrics.confusion`, and previously always called it
unweighted. `val_frame` may now optionally carry a `weight` column (one
value per validation item, e.g. `sampling_weight`); when present, each
bootstrap iteration's confusion rebuild is itself reweighted by it, so the
CI uses the SAME (weighted) error model as the point estimate rather than
silently reverting to an unweighted resample -- matching this module's
own "ONE labeler error model" principle (see `analyze_panel.py`'s
`corrected_drift_by_corpus`). A `val_frame` with no `weight` column (every
pre-Task-7 caller) is unaffected: the rebuild is called with
`weights=None`, byte-identical to before.

House style: American spelling, no em dashes (double hyphens for asides).
The automated pipeline is always "the classifier under evaluation," and
its validation-subset reference is always "human-consensus reference
labels" -- never "ground truth."
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idrift.adjudicate.validation_metrics import CLASS_ORDER, confusion

_N_CLASSES = len(CLASS_ORDER)
_PCTILES = (2.5, 97.5)


def _counts_to_vector(auto_counts) -> np.ndarray:
    """Reindex `auto_counts` (a `{class: int}` mapping or a `pd.Series`
    indexed by class) to `CLASS_ORDER`, filling a missing class with 0, and
    return it as a plain float `numpy.ndarray` in that fixed order."""
    if isinstance(auto_counts, pd.Series):
        counts = auto_counts
    else:
        counts = pd.Series(dict(auto_counts))
    counts = counts.reindex(CLASS_ORDER).fillna(0.0)
    return counts.to_numpy(dtype=float)


def _build_confusion_matrix(confusion_df: pd.DataFrame):
    """Row-normalize a human(rows) x auto(cols) confusion count DataFrame
    (the shape `validation_metrics.confusion` produces) into the
    classifier confusion matrix `M[i, j] = P(auto == j | true == i)`.

    A row with zero total support (the true class never appeared in the
    validation subset at all) cannot be normalized -- it is replaced with
    an identity row (`M[i, i] = 1.0`), the conservative "assume perfectly
    classified" no-information fallback documented in the module
    docstring, and that class is recorded.

    Returns:
        (M, identity_fallback_classes): `M` is an `(n_classes, n_classes)`
        float ndarray in `CLASS_ORDER` row/column order.
        `identity_fallback_classes` is a list of the class names (in
        `CLASS_ORDER` order) that hit the zero-support fallback.
    """
    cm = (
        confusion_df.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
        .to_numpy(dtype=float)
    )
    row_sums = cm.sum(axis=1)
    m_mat = np.zeros((_N_CLASSES, _N_CLASSES))
    fallback_classes = []
    for i in range(_N_CLASSES):
        if row_sums[i] <= 0:
            m_mat[i, i] = 1.0
            fallback_classes.append(CLASS_ORDER[i])
        else:
            m_mat[i, :] = cm[i, :] / row_sums[i]
    return m_mat, fallback_classes


def _solve_true_prevalence(m_mat: np.ndarray, q: np.ndarray):
    """Solve `M^T p = q` for `p` (the generative relation derived in the
    module docstring). Falls back to `numpy.linalg.lstsq` if `M.T` is
    singular (or exactly-singular detection raises).

    Returns:
        (p, singular): `p` is the (possibly off-simplex) solution vector;
        `singular` is True iff the `lstsq` fallback was used.
    """
    try:
        p = np.linalg.solve(m_mat.T, q)
        singular = False
    except np.linalg.LinAlgError:
        p, _residuals, _rank, _sv = np.linalg.lstsq(m_mat.T, q, rcond=None)
        singular = True
    return p, singular


def _project_to_simplex_nonneg(p: np.ndarray):
    """Clip negative components of `p` to 0 and renormalize the remainder
    to sum to 1 (the documented simplex projection -- see module
    docstring for why an exact linear solve can land slightly outside the
    simplex). Returns `(p_projected, clipped)`, where `clipped` is True
    iff any component of `p` was negative."""
    clipped = bool(np.any(p < 0))
    p_nonneg = np.clip(p, 0.0, None)
    total = p_nonneg.sum()
    if total <= 0:
        # Degenerate only if every component was <= 0 (e.g. a pathological
        # M/q pair); fall back to a uniform vector rather than dividing by
        # zero. Not expected to occur for any realistic validation subset.
        p_proj = np.full(_N_CLASSES, 1.0 / _N_CLASSES)
    else:
        p_proj = p_nonneg / total
    return p_proj, clipped


def _corrected_target(auto_vec: np.ndarray, confusion_df: pd.DataFrame, target_idx: int) -> float:
    """Run the full point-estimate pipeline (row-normalize -> solve ->
    project) for one set of (auto_vec, confusion_df) and return only the
    projected prevalence of `CLASS_ORDER[target_idx]`. Shared by the main
    point estimate and every bootstrap iteration (both CI methods) so the
    correction logic lives in exactly one place."""
    total = auto_vec.sum()
    if total <= 0:
        return 0.0
    q = auto_vec / total
    m_mat, _fallback = _build_confusion_matrix(confusion_df)
    p_raw, _singular = _solve_true_prevalence(m_mat, q)
    p_proj, _clipped = _project_to_simplex_nonneg(p_raw)
    return float(p_proj[target_idx])


def _bootstrap_message_clustered(cohort_frame: pd.DataFrame, val_frame: pd.DataFrame,
                                  target_idx: int, n_boot: int, rng: np.random.Generator):
    """Message-clustered bootstrap CI for the corrected target prevalence.

    Each iteration independently resamples the distinct `message_id`s (with
    replacement) in `cohort_frame` and in `val_frame`, pulling every row of
    each resampled message (a cluster bootstrap, so within-message
    correlation in the corruption/classification process is preserved),
    rebuilds the naive `auto_counts` from the resampled cohort rows and the
    `human x auto` confusion counts from the resampled validation rows
    (via `validation_metrics.confusion`, reused rather than reimplemented),
    and recomputes the corrected target prevalence.

    If `val_frame` has an optional `weight` column (one value per
    validation item, e.g. a stratified panel's `sampling_weight`), the
    per-iteration confusion rebuild is itself reweighted by it (see module
    docstring), so the CI reflects the same weighted error model as a
    weighted point estimate. A `val_frame` with no `weight` column
    (every pre-Task-7 caller) rebuilds the confusion matrix unweighted,
    byte-identical to before.

    Returns:
        (lo, hi): the 2.5th/97.5th percentile of the bootstrap corrected-
        target distribution.
    """
    cohort_groups = {
        message_id: group["auto_label"].tolist()
        for message_id, group in cohort_frame.groupby("message_id")
    }
    has_weight = "weight" in val_frame.columns
    if has_weight:
        val_groups = {
            message_id: (group["human"].tolist(), group["auto"].tolist(), group["weight"].tolist())
            for message_id, group in val_frame.groupby("message_id")
        }
    else:
        val_groups = {
            message_id: (group["human"].tolist(), group["auto"].tolist(), None)
            for message_id, group in val_frame.groupby("message_id")
        }
    cohort_ids = np.array(list(cohort_groups.keys()), dtype=object)
    val_ids = np.array(list(val_groups.keys()), dtype=object)
    n_cohort_ids = len(cohort_ids)
    n_val_ids = len(val_ids)
    if n_cohort_ids == 0 or n_val_ids == 0:
        raise ValueError("cohort_frame and val_frame must each contain at least one message_id")

    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_cohort_ids = cohort_ids[rng.integers(0, n_cohort_ids, size=n_cohort_ids)]
        resampled_auto_labels = []
        for message_id in sampled_cohort_ids:
            resampled_auto_labels.extend(cohort_groups[message_id])
        auto_counts = pd.Series(resampled_auto_labels, dtype=object).value_counts()
        auto_vec = _counts_to_vector(auto_counts)

        sampled_val_ids = val_ids[rng.integers(0, n_val_ids, size=n_val_ids)]
        resampled_human = []
        resampled_auto = []
        resampled_weights = [] if has_weight else None
        for message_id in sampled_val_ids:
            human_labels, auto_labels, item_weights = val_groups[message_id]
            resampled_human.extend(human_labels)
            resampled_auto.extend(auto_labels)
            if has_weight:
                resampled_weights.extend(item_weights)
        confusion_boot = confusion(
            resampled_human, resampled_auto, labels=list(CLASS_ORDER), weights=resampled_weights
        )

        draws[b] = _corrected_target(auto_vec, confusion_boot, target_idx)

    lo, hi = np.percentile(draws, _PCTILES)
    return float(lo), float(hi)


def _bootstrap_multinomial_parametric(auto_vec: np.ndarray, confusion_counts: np.ndarray,
                                       target_idx: int, n_boot: int, rng: np.random.Generator):
    """Parametric multinomial bootstrap CI for the corrected target
    prevalence, used when no per-message frames are available (only the
    aggregate `auto_counts` / `confusion_from_human` counts).

    Each iteration draws `auto_counts ~ Multinomial(total, q_hat)` and,
    independently for each true-class row with nonzero support,
    `confusion_row ~ Multinomial(row_total, row_prob_hat)`, then
    recomputes the corrected target prevalence from those simulated
    counts. A true-class row with zero support in the real data stays all-
    zero in every draw (consistent with the identity-row fallback it
    triggers in `_build_confusion_matrix`).

    Returns:
        (lo, hi): the 2.5th/97.5th percentile of the bootstrap corrected-
        target distribution.
    """
    total = int(round(auto_vec.sum()))
    q_hat = auto_vec / auto_vec.sum()

    row_totals = confusion_counts.sum(axis=1)
    row_probs = np.zeros((_N_CLASSES, _N_CLASSES))
    for i in range(_N_CLASSES):
        if row_totals[i] > 0:
            row_probs[i, :] = confusion_counts[i, :] / row_totals[i]

    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        auto_draw = rng.multinomial(total, q_hat).astype(float)

        confusion_draw = np.zeros((_N_CLASSES, _N_CLASSES))
        for i in range(_N_CLASSES):
            if row_totals[i] > 0:
                confusion_draw[i, :] = rng.multinomial(int(round(row_totals[i])), row_probs[i, :])
        confusion_draw_df = pd.DataFrame(
            confusion_draw, index=list(CLASS_ORDER), columns=list(CLASS_ORDER)
        )

        draws[b] = _corrected_target(auto_draw, confusion_draw_df, target_idx)

    lo, hi = np.percentile(draws, _PCTILES)
    return float(lo), float(hi)


def corrected_prevalence(
    auto_counts,
    confusion_from_human: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
    target: str = "drift",
    *,
    cohort_frame: pd.DataFrame | None = None,
    val_frame: pd.DataFrame | None = None,
) -> dict:
    """Misclassification-corrected prevalence of `target` over the full
    cohort, via multiclass Rogan-Gladen matrix inversion, with a bootstrap
    CI (message-clustered if per-message frames are supplied, otherwise a
    parametric multinomial fallback -- see module docstring for both).

    Args:
        auto_counts: `{class: int}` mapping or `pd.Series` indexed by
            class -- the classifier's naive label counts over the FULL
            cohort. Reindexed to `CLASS_ORDER` (a missing class -> 0).
        confusion_from_human: the human(rows) x auto(cols) confusion count
            DataFrame from `validation_metrics.confusion`, computed on the
            validation subset (human-consensus reference labels vs. the
            classifier under evaluation).
        n_boot: number of bootstrap iterations.
        seed: seed for the single `numpy.random.default_rng` driving every
            random draw in this call.
        target: which taxonomy class's prevalence to report (naive,
            corrected, and CI). Must be one of `CLASS_ORDER`.
        cohort_frame: optional. One row per full-cohort generation, columns
            `["message_id", "auto_label"]`. Supplying this AND `val_frame`
            switches the CI to a genuine message-clustered bootstrap.
        val_frame: optional. One row per human-labeled validation item,
            columns `["message_id", "human", "auto"]`. See `cohort_frame`.
            May optionally include a `"weight"` column (one value per item,
            e.g. a stratified panel's `sampling_weight`); if present, the
            per-iteration bootstrap confusion rebuild is itself reweighted
            by it (see module docstring). Absent (the pre-Task-7 default),
            the rebuild is unweighted, byte-identical to before.

    Returns:
        dict: {
            "naive": float, the observed (uncorrected) prevalence of
                `target` in `auto_counts`,
            "corrected": float, the misclassification-corrected (simplex-
                projected) prevalence of `target`,
            "ci": [lo, hi], the bootstrap 95% CI on the corrected estimate,
            "ci_method": "message-clustered" or "multinomial-parametric",
            "n_boot": int, echoed back,
            "target": str, echoed back,
            "clipped": bool, whether the point-estimate solve landed
                outside the simplex and had to be clipped/renormalized,
            "singular": bool, whether the point-estimate solve fell back
                to `numpy.linalg.lstsq` because `M.T` was singular,
            "identity_fallback_classes": list[str], any true class(es)
                with zero support in `confusion_from_human` whose row was
                replaced with an identity row (see module docstring).
        }

    Raises:
        ValueError: if `target` is not one of `CLASS_ORDER`, or if
            `auto_counts` sums to zero (there is nothing to correct).
    """
    if target not in CLASS_ORDER:
        raise ValueError(f"target must be one of {CLASS_ORDER}, got {target!r}")
    target_idx = CLASS_ORDER.index(target)

    auto_vec = _counts_to_vector(auto_counts)
    total = auto_vec.sum()
    if total <= 0:
        raise ValueError("auto_counts must sum to a positive total across CLASS_ORDER")
    q = auto_vec / total

    m_mat, identity_fallback_classes = _build_confusion_matrix(confusion_from_human)
    p_raw, singular = _solve_true_prevalence(m_mat, q)
    p_proj, clipped = _project_to_simplex_nonneg(p_raw)

    naive = float(q[target_idx])
    corrected = float(p_proj[target_idx])

    rng = np.random.default_rng(seed)

    if cohort_frame is not None and val_frame is not None:
        ci_lo, ci_hi = _bootstrap_message_clustered(cohort_frame, val_frame, target_idx, n_boot, rng)
        ci_method = "message-clustered"
    else:
        confusion_counts = (
            confusion_from_human.reindex(index=CLASS_ORDER, columns=CLASS_ORDER, fill_value=0)
            .to_numpy(dtype=float)
        )
        ci_lo, ci_hi = _bootstrap_multinomial_parametric(
            auto_vec, confusion_counts, target_idx, n_boot, rng
        )
        ci_method = "multinomial-parametric"

    return {
        "naive": naive,
        "corrected": corrected,
        "ci": [ci_lo, ci_hi],
        "ci_method": ci_method,
        "n_boot": n_boot,
        "target": target,
        "clipped": clipped,
        "singular": singular,
        "identity_fallback_classes": identity_fallback_classes,
    }
