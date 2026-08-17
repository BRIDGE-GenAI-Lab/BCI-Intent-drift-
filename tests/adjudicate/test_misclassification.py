"""Tests for misclassification-corrected prevalence (revision Task 4.3, the
last offline instrument of Phase 4).

Reviewers required that every FULL-COHORT drift estimate carry a
misclassification-corrected sensitivity analysis alongside the naive one:
the automated pipeline is EVALUATED against human-consensus reference
labels on a validation subset, never treated as ground truth for the full
cohort, so the pipeline's own known classification error (from that
validation subset) must be propagated into the cohort-level prevalence
estimate, not ignored.

The correction is a multiclass Rogan-Gladen matrix inversion:

    q = M^T p

where `q` is the observed (naive) prevalence vector over
`idrift.adjudicate.validation_metrics.CLASS_ORDER` (from the classifier's
full-cohort labels), `M[i, j] = P(auto == j | true == i)` is the
classifier's own confusion matrix (row-normalized human-vs-auto counts
from the validation subset), and `p` is the TRUE prevalence vector we want
to recover. Solving `M^T p = q` for `p` corrects the naive prevalence for
the classifier's measured sensitivity/specificity per class.

The headline test below (`test_recovery_vs_bias_...`) is a full,
hand-designed simulation: a KNOWN true prevalence, a KNOWN, deliberately
imperfect confusion matrix (drift recall well below 1, some faithful ->
drift leakage), a large synthetic cohort generated from those two
ingredients with a seeded rng, and the assertion that the CORRECTED
estimate recovers the known truth while the NAIVE estimate does not.
"""
import numpy as np
import pandas as pd
import pytest

from idrift.adjudicate.validation_metrics import CLASS_ORDER, confusion
from idrift.adjudicate.misclassification import corrected_prevalence


# ---------------------------------------------------------------------------
# Shared simulation ingredients for the headline recovery-vs-bias test.
#
# True prevalence p (order faithful, degraded, drift):
#   p = [0.80, 0.15, 0.05]
#
# Classifier confusion M[i, j] = P(auto == j | true == i), fixed order
# (faithful, degraded, drift) for both true class i (rows) and predicted
# class j (columns), DELIBERATELY IMPERFECT:
#   M[faithful] = [0.85, 0.10, 0.05]   (5% of true-faithful mislabeled drift)
#   M[degraded] = [0.10, 0.75, 0.15]
#   M[drift]    = [0.30, 0.30, 0.40]   (drift recall only 0.40 -- well below 1)
#
# Hand-computed observed (naive) prevalence q = M^T p:
#   q_faithful = 0.80*0.85 + 0.15*0.10 + 0.05*0.30 = 0.68 + 0.015 + 0.015 = 0.71
#   q_degraded = 0.80*0.10 + 0.15*0.75 + 0.05*0.30 = 0.08 + 0.1125 + 0.015 = 0.2075
#   q_drift    = 0.80*0.05 + 0.15*0.15 + 0.05*0.40 = 0.04 + 0.0225 + 0.02 = 0.0825
#   (sums to 1.0)
#
# So the naive drift estimate (0.0825) is biased upward by 65% relative to
# the true 0.05 -- almost entirely because 80% of the cohort is faithful and
# even a small 5% faithful->drift leak rate outweighs drift's own poor
# recall in absolute terms. The corrected estimate must recover ~0.05.
# ---------------------------------------------------------------------------

TRUE_PREVALENCE = {"faithful": 0.80, "degraded": 0.15, "drift": 0.05}

CONFUSION_M = {
    "faithful": {"faithful": 0.85, "degraded": 0.10, "drift": 0.05},
    "degraded": {"faithful": 0.10, "degraded": 0.75, "drift": 0.15},
    "drift": {"faithful": 0.30, "degraded": 0.30, "drift": 0.40},
}

EXPECTED_NAIVE_DRIFT = 0.0825


def _simulate(n_cohort, n_val_per_class, seed):
    """Generate a synthetic full cohort (auto labels only, as in real full-
    cohort deployment) and a separate, STRATIFIED validation subset (equal
    draws per TRUE class, `n_val_per_class` each) from TRUE_PREVALENCE /
    CONFUSION_M, both via a single seeded rng.

    The validation subset is stratified by true class (not a simple random
    sample at the overall prevalence p) so the confusion matrix -- in
    particular the rare "drift" true-class row -- can be estimated with
    adequate precision from a tractable number of validation items. This
    mirrors how a real human-adjudication validation panel is deliberately
    enriched for rare/critical classes rather than drawn as a simple
    random sample of the cohort.

    Returns:
        auto_counts (pd.Series over CLASS_ORDER): full-cohort naive label
            counts.
        confusion_from_human (pd.DataFrame): human(rows) x auto(cols)
            confusion counts from the validation subset, exactly the shape
            `validation_metrics.confusion` produces.
    """
    rng = np.random.default_rng(seed)
    p_vec = np.array([TRUE_PREVALENCE[c] for c in CLASS_ORDER])
    m_mat = np.array([[CONFUSION_M[i][j] for j in CLASS_ORDER] for i in CLASS_ORDER])

    # Full cohort: true labels ~ p (unknown to the correction -- only the
    # resulting auto labels are ever observed), auto labels ~ M[true].
    true_labels_cohort = rng.choice(CLASS_ORDER, size=n_cohort, p=p_vec)
    cohort_auto = np.empty(n_cohort, dtype=object)
    for i, cls in enumerate(CLASS_ORDER):
        mask = true_labels_cohort == cls
        n_cls = int(mask.sum())
        if n_cls:
            cohort_auto[mask] = rng.choice(CLASS_ORDER, size=n_cls, p=m_mat[i])

    # Validation subset: n_val_per_class true items drawn directly from
    # EACH class, auto labels ~ M[true class] -- a stratified panel.
    val_human = []
    val_auto = []
    for i, cls in enumerate(CLASS_ORDER):
        auto_for_cls = rng.choice(CLASS_ORDER, size=n_val_per_class, p=m_mat[i])
        val_human.extend([cls] * n_val_per_class)
        val_auto.extend(auto_for_cls.tolist())

    auto_counts = pd.Series(cohort_auto).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    confusion_from_human = confusion(val_human, val_auto, labels=list(CLASS_ORDER))
    return auto_counts, confusion_from_human


def test_recovery_vs_bias_corrected_recovers_truth_naive_is_biased():
    auto_counts, confusion_from_human = _simulate(n_cohort=60_000, n_val_per_class=5_000, seed=20260719)

    result = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=500, seed=1, target="drift"
    )

    # (a) corrected recovers the true drift prevalence (0.05) within its CI,
    # and lands close to the truth as a point estimate.
    lo, hi = result["ci"]
    assert lo <= 0.05 <= hi
    assert result["corrected"] == pytest.approx(0.05, abs=0.02)

    # (b) naive is biased: materially off the truth, and worse than corrected.
    assert result["naive"] == pytest.approx(EXPECTED_NAIVE_DRIFT, abs=0.01)
    naive_error = abs(result["naive"] - 0.05)
    corrected_error = abs(result["corrected"] - 0.05)
    assert naive_error > corrected_error
    assert naive_error > 0.02  # materially off, not just noisy


def test_recovery_vs_bias_deviation_is_large_relative_to_correction():
    # Same simulation, phrased as the brief's exact requirement: naive point
    # value must be MATERIALLY off truth relative to corrected's error.
    auto_counts, confusion_from_human = _simulate(n_cohort=60_000, n_val_per_class=5_000, seed=20260719)

    result = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=500, seed=1, target="drift"
    )

    naive_error = abs(result["naive"] - 0.05)
    corrected_error = abs(result["corrected"] - 0.05)
    # naive error should be at least 3x the corrected error in this
    # deliberately-biased scenario.
    assert naive_error > 3 * corrected_error


# ---------------------------------------------------------------------------
# Perfect classifier: M = identity -> corrected == naive == q for the
# target, and the CI (either method) brackets that shared value.
# ---------------------------------------------------------------------------

def test_perfect_classifier_is_a_no_op():
    human = ["faithful"] * 70 + ["degraded"] * 20 + ["drift"] * 10
    auto = list(human)  # auto == human exactly -> M is the identity matrix
    confusion_from_human = confusion(human, auto, labels=list(CLASS_ORDER))

    auto_counts = pd.Series({"faithful": 700, "degraded": 200, "drift": 100})

    result = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=300, seed=3, target="drift"
    )

    expected_q_drift = 100 / 1000
    assert result["naive"] == pytest.approx(expected_q_drift)
    assert result["corrected"] == pytest.approx(expected_q_drift)
    assert result["clipped"] is False
    assert result["singular"] is False
    assert result["identity_fallback_classes"] == []

    lo, hi = result["ci"]
    assert lo <= expected_q_drift <= hi


# ---------------------------------------------------------------------------
# Determinism: identical inputs + seed -> identical dict, across two calls.
# ---------------------------------------------------------------------------

def test_determinism_same_inputs_and_seed_reproduce_identical_result():
    auto_counts, confusion_from_human = _simulate(n_cohort=5_000, n_val_per_class=500, seed=99)

    result_a = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=200, seed=7, target="drift"
    )
    result_b = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=200, seed=7, target="drift"
    )

    assert result_a["naive"] == result_b["naive"]
    assert result_a["corrected"] == result_b["corrected"]
    assert result_a["ci"] == result_b["ci"]
    assert result_a["ci_method"] == result_b["ci_method"]


# ---------------------------------------------------------------------------
# Clustered (message_id) vs. parametric (multinomial) CI method selection.
# ---------------------------------------------------------------------------

def _simulate_message_frames(n_messages_cohort, replicates_per_message, n_messages_val, seed):
    """Build (auto_counts, confusion_from_human, cohort_frame, val_frame)
    with an explicit `message_id` clustering structure: each message has
    ONE true class, and multiple (correlated) generations/validation items
    drawn from that message's own confusion-matrix row -- exactly the
    per-message-row shape `corrected_prevalence`'s clustered bootstrap
    expects.
    """
    rng = np.random.default_rng(seed)
    p_vec = np.array([TRUE_PREVALENCE[c] for c in CLASS_ORDER])
    m_mat = np.array([[CONFUSION_M[i][j] for j in CLASS_ORDER] for i in CLASS_ORDER])

    cohort_rows = []
    true_per_cohort_message = rng.choice(CLASS_ORDER, size=n_messages_cohort, p=p_vec)
    for msg_num, true_cls in enumerate(true_per_cohort_message):
        i = CLASS_ORDER.index(true_cls)
        auto_labels = rng.choice(CLASS_ORDER, size=replicates_per_message, p=m_mat[i])
        for auto_label in auto_labels:
            cohort_rows.append({"message_id": f"cohort_{msg_num}", "auto_label": auto_label})
    cohort_frame = pd.DataFrame(cohort_rows)

    val_rows = []
    true_per_val_message = rng.choice(CLASS_ORDER, size=n_messages_val, p=p_vec)
    for msg_num, true_cls in enumerate(true_per_val_message):
        i = CLASS_ORDER.index(true_cls)
        n_items = int(rng.integers(1, 3))  # 1-2 human-labeled items per message
        auto_labels = rng.choice(CLASS_ORDER, size=n_items, p=m_mat[i])
        for auto_label in auto_labels:
            val_rows.append({"message_id": f"val_{msg_num}", "human": true_cls, "auto": auto_label})
    val_frame = pd.DataFrame(val_rows)

    auto_counts = cohort_frame["auto_label"].value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    confusion_from_human = confusion(
        val_frame["human"].tolist(), val_frame["auto"].tolist(), labels=list(CLASS_ORDER)
    )
    return auto_counts, confusion_from_human, cohort_frame, val_frame


def test_message_clustered_ci_used_when_frames_supplied():
    auto_counts, confusion_from_human, cohort_frame, val_frame = _simulate_message_frames(
        n_messages_cohort=1_000, replicates_per_message=3, n_messages_val=200, seed=42
    )

    result = corrected_prevalence(
        auto_counts,
        confusion_from_human,
        n_boot=200,
        seed=2,
        target="drift",
        cohort_frame=cohort_frame,
        val_frame=val_frame,
    )

    assert result["ci_method"] == "message-clustered"
    lo, hi = result["ci"]
    assert lo <= result["corrected"] <= hi


def test_multinomial_parametric_ci_used_when_frames_absent():
    auto_counts, confusion_from_human, _cohort_frame, _val_frame = _simulate_message_frames(
        n_messages_cohort=1_000, replicates_per_message=3, n_messages_val=200, seed=42
    )

    result = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=200, seed=2, target="drift"
    )

    assert result["ci_method"] == "multinomial-parametric"
    lo, hi = result["ci"]
    assert lo <= result["corrected"] <= hi


# ---------------------------------------------------------------------------
# Zero-support true class: the validation subset never contains a human
# label of "drift" at all -> that row cannot be normalized -> identity
# fallback, recorded, no crash.
# ---------------------------------------------------------------------------

def test_zero_support_true_class_triggers_identity_fallback_without_crashing():
    # human never contains "drift"; auto predicts "drift" a few times
    # (false positives against truly faithful/degraded items).
    human = ["faithful"] * 80 + ["degraded"] * 20
    auto = (["faithful"] * 75 + ["drift"] * 5) + (["degraded"] * 15 + ["drift"] * 5)
    confusion_from_human = confusion(human, auto, labels=list(CLASS_ORDER))

    assert confusion_from_human.loc["drift"].sum() == 0  # zero support, by construction

    auto_counts = pd.Series({"faithful": 700, "degraded": 250, "drift": 50})

    result = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=200, seed=5, target="drift"
    )

    assert result["identity_fallback_classes"] == ["drift"]
    assert np.isfinite(result["naive"])
    assert np.isfinite(result["corrected"])
    lo, hi = result["ci"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


# ---------------------------------------------------------------------------
# Clipping: an observed q picked outside the achievable region for a given
# M solves to a `p` with negative components (and one > 1), which must be
# clipped to 0 and renormalized rather than returned as-is.
#
# M (exact, hand-built counts, no sampling noise), same as CONFUSION_M:
#   M^T = [[0.85, 0.10, 0.30],
#          [0.10, 0.75, 0.30],
#          [0.05, 0.15, 0.40]]
# q = [0.05, 0.05, 0.90] (auto_counts 50/50/900 of 1000) solves (verified
# by hand with numpy.linalg.solve outside this test) to
#   p_raw ~ [-0.783, -0.904, 2.687]  (sums to 1.0, but two components < 0)
# Clip negatives to 0 -> [0, 0, 2.687], renormalize by the new sum 2.687 ->
# [0, 0, 1.0]. So the corrected drift prevalence must land at exactly 1.0
# and "clipped" must be True.
# ---------------------------------------------------------------------------

def test_clipping_projects_an_out_of_simplex_solution_back_onto_it():
    human = ["faithful"] * 1000 + ["degraded"] * 1000 + ["drift"] * 1000
    auto = (
        ["faithful"] * 850 + ["degraded"] * 100 + ["drift"] * 50
        + ["faithful"] * 100 + ["degraded"] * 750 + ["drift"] * 150
        + ["faithful"] * 300 + ["degraded"] * 300 + ["drift"] * 400
    )
    confusion_from_human = confusion(human, auto, labels=list(CLASS_ORDER))

    auto_counts = pd.Series({"faithful": 50, "degraded": 50, "drift": 900})

    result = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=50, seed=11, target="drift"
    )

    assert result["naive"] == pytest.approx(0.90)
    assert result["clipped"] is True
    assert result["corrected"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# weights= (Task 7, reviewer #1 follow-up): the Rogan-Gladen correction must
# accept a WEIGHTED confusion matrix (built by
# `validation_metrics.confusion(..., weights=sampling_weight)`) as
# `confusion_from_human`, and equal (uniform, even if non-unit) per-item
# weights must reduce to the current unweighted correction. `confusion_from_
# human` is just a human(rows) x auto(cols) DataFrame to `_build_confusion_
# matrix`, which row-normalizes it -- so this is mostly a documented
# capability + regression pin, plus (below) an extension so the
# message-clustered bootstrap CI also honors an optional per-item weight
# column on `val_frame`, keeping the point estimate and the CI on the SAME
# (weighted) error model, matching this module's own "ONE labeler error
# model" principle (see module docstring / analyze_panel.py).
#
# Hand-built weighted pattern: item 3 (a faithful item the classifier
# mislabels "drift") is given weight 10 instead of 1, so its leak into
# "drift" dominates the weighted confusion matrix's faithful row far more
# than in the unweighted matrix -- a corpus-representative validation draw
# would use such weights to correct for a stratified (not simple-random)
# sample.
# ---------------------------------------------------------------------------

_RG_WEIGHT_HUMAN = ["faithful"] * 4 + ["degraded"] * 2 + ["drift"] * 2
_RG_WEIGHT_AUTO = ["faithful", "faithful", "faithful", "drift", "degraded", "degraded", "faithful", "drift"]


def test_weighted_confusion_matrix_changes_the_correction_vs_unweighted():
    cm_unweighted = confusion(_RG_WEIGHT_HUMAN, _RG_WEIGHT_AUTO, labels=list(CLASS_ORDER))
    cm_upweighted_leak = confusion(
        _RG_WEIGHT_HUMAN, _RG_WEIGHT_AUTO, labels=list(CLASS_ORDER),
        weights=[1, 1, 1, 10, 1, 1, 1, 1],
    )

    auto_counts = pd.Series({"faithful": 700, "degraded": 200, "drift": 100})

    result_unweighted = corrected_prevalence(
        auto_counts, cm_unweighted, n_boot=200, seed=1, target="drift"
    )
    result_weighted = corrected_prevalence(
        auto_counts, cm_upweighted_leak, n_boot=200, seed=1, target="drift"
    )

    # same naive value (auto_counts is identical in both calls) ...
    assert result_unweighted["naive"] == result_weighted["naive"]
    # ...but materially different corrected values: the weighted confusion
    # matrix (not the unweighted one) drove the correction.
    assert abs(result_unweighted["corrected"] - result_weighted["corrected"]) > 0.05


def test_equal_weights_confusion_reduces_to_unweighted_correction():
    cm_plain = confusion(_RG_WEIGHT_HUMAN, _RG_WEIGHT_AUTO, labels=list(CLASS_ORDER))  # weights=None
    cm_equal_weighted = confusion(
        _RG_WEIGHT_HUMAN, _RG_WEIGHT_AUTO, labels=list(CLASS_ORDER),
        weights=[3.7] * len(_RG_WEIGHT_HUMAN),  # uniform, non-unit weight
    )

    auto_counts = pd.Series({"faithful": 700, "degraded": 200, "drift": 100})

    result_plain = corrected_prevalence(auto_counts, cm_plain, n_boot=200, seed=9, target="drift")
    result_equal_weighted = corrected_prevalence(
        auto_counts, cm_equal_weighted, n_boot=200, seed=9, target="drift"
    )

    assert result_plain["naive"] == result_equal_weighted["naive"]
    assert result_plain["corrected"] == pytest.approx(result_equal_weighted["corrected"])


def test_message_clustered_bootstrap_respects_weight_column_when_present():
    auto_counts, confusion_from_human, cohort_frame, val_frame = _simulate_message_frames(
        n_messages_cohort=1_000, replicates_per_message=3, n_messages_val=200, seed=42
    )

    # Upweight only the faithful->drift LEAK cell (not the whole "faithful"
    # true-class row uniformly, which would cancel exactly under row-
    # normalization and prove nothing -- see the module docstring on why a
    # per-class-uniform weight is a no-op). Upweighting one cell within a
    # row changes that row's normalized proportions, so it must move the
    # weighted bootstrap's confusion rebuild (and therefore its CI) away
    # from the unweighted rebuild's.
    val_frame_weighted = val_frame.copy()
    val_frame_weighted["weight"] = 1.0
    leak_cell = (val_frame_weighted["human"] == "faithful") & (val_frame_weighted["auto"] == "drift")
    assert leak_cell.sum() > 0
    val_frame_weighted.loc[leak_cell, "weight"] = 15.0

    result_unweighted = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=200, seed=2, target="drift",
        cohort_frame=cohort_frame, val_frame=val_frame,
    )
    result_weighted = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=200, seed=2, target="drift",
        cohort_frame=cohort_frame, val_frame=val_frame_weighted,
    )

    assert result_unweighted["ci_method"] == "message-clustered"
    assert result_weighted["ci_method"] == "message-clustered"
    # up-weighting the drift-true validation rows shifts the bootstrap CI:
    # proves the "weight" column, when present, is actually consumed by the
    # per-iteration confusion rebuild, not silently ignored.
    assert result_unweighted["ci"] != result_weighted["ci"]


def test_message_clustered_bootstrap_without_weight_column_is_unchanged():
    # Backward compatibility: a val_frame with no "weight" column (every
    # existing caller, including analyze_panel.py) must reproduce byte-
    # identical output to before this task's change, for the same seed.
    auto_counts, confusion_from_human, cohort_frame, val_frame = _simulate_message_frames(
        n_messages_cohort=1_000, replicates_per_message=3, n_messages_val=200, seed=42
    )
    assert "weight" not in val_frame.columns

    result_a = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=200, seed=2, target="drift",
        cohort_frame=cohort_frame, val_frame=val_frame,
    )
    result_b = corrected_prevalence(
        auto_counts, confusion_from_human, n_boot=200, seed=2, target="drift",
        cohort_frame=cohort_frame, val_frame=val_frame,
    )

    assert result_a == result_b
