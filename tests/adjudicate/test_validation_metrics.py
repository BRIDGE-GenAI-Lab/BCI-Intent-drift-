"""Tests for class-specific automated-vs-human validation metrics
(revision Task 4.2).

Reviewers rejected Cohen's kappa alone as sufficient justification for
treating the automated pipeline's labels as ground truth for the full
cohort; they demanded class-specific sensitivity, specificity, PPV, NPV, F1,
and a full confusion matrix, broken down by stratum. Every confusion pattern
below is HAND-BUILT: the expected sensitivity/specificity/PPV/NPV/F1 values
are computed by manual arithmetic in each test's comment (not derived by
calling the module under test), so a bug in the module's own logic cannot
also produce the "expected" value.

`human` is always the reference label; `auto` is always the pipeline's
label being evaluated against it -- consistent with `class_metrics(human,
auto)` treating human as ground truth for this comparison, never the
reverse.
"""
import math

import pandas as pd
import pytest

from idrift.adjudicate.validation_metrics import (
    CLASS_ORDER,
    class_metrics,
    confusion,
    by_stratum,
    macro_weighted,
)


# ---------------------------------------------------------------------------
# Perfect classifier: every metric must be exactly 1.0 for every class that
# has nonzero support and at least one other class present (so TN > 0).
# ---------------------------------------------------------------------------

def test_perfect_classifier_all_metrics_are_one():
    human = ["faithful", "faithful", "degraded", "drift", "drift", "drift"]
    auto = list(human)  # auto == human exactly

    result = class_metrics(human, auto)

    assert set(result["class"]) == {"faithful", "degraded", "drift"}
    for _, row in result.iterrows():
        assert row["sensitivity"] == 1.0
        assert row["specificity"] == 1.0
        assert row["ppv"] == 1.0
        assert row["npv"] == 1.0
        assert row["f1"] == 1.0
    supports = dict(zip(result["class"], result["support"]))
    assert supports == {"faithful": 2, "degraded": 1, "drift": 3}


def test_perfect_classifier_confusion_is_diagonal():
    human = ["faithful", "faithful", "degraded", "drift", "drift", "drift"]
    auto = list(human)

    cm = confusion(human, auto)

    assert list(cm.index) == list(CLASS_ORDER)
    assert list(cm.columns) == list(CLASS_ORDER)
    assert cm.loc["faithful", "faithful"] == 2
    assert cm.loc["degraded", "degraded"] == 1
    assert cm.loc["drift", "drift"] == 3
    # off-diagonal all zero
    off_diag_sum = cm.values.sum() - (2 + 1 + 3)
    assert off_diag_sum == 0


# ---------------------------------------------------------------------------
# Specific confusion pattern, hand-computed.
#
# Confusion matrix (rows = human truth, cols = auto prediction), fixed order
# (faithful, degraded, drift):
#
#            auto:faithful  auto:degraded  auto:drift
# human:faithful      4              1            0      (support 5)
# human:degraded      0              2            1      (support 3)
# human:drift         0              0            2      (support 2)
#
# total = 10
#
# class "faithful": TP=4, FN=5-4=1, FP=(4-4)=0 (col sum 4), TN=10-4-1-0=5
#   sensitivity = 4/5 = 0.8
#   specificity = 5/(5+0) = 1.0
#   ppv         = 4/(4+0) = 1.0
#   npv         = 5/(5+1) = 5/6 = 0.833333...
#   f1          = 2*4/(2*4+0+1) = 8/9 = 0.888888...
#
# class "degraded": TP=2, FN=3-2=1, FP=3-2=1 (col sum 3), TN=10-2-1-1=6
#   sensitivity = 2/3 = 0.666666...
#   specificity = 6/(6+1) = 6/7 = 0.857142...
#   ppv         = 2/(2+1) = 2/3 = 0.666666...
#   npv         = 6/(6+1) = 6/7 = 0.857142...
#   f1          = 2*2/(2*2+1+1) = 4/6 = 0.666666...
#
# class "drift": TP=2, FN=2-2=0, FP=3-2=1 (col sum 3), TN=10-2-0-1=7
#   sensitivity = 2/2 = 1.0
#   specificity = 7/(7+1) = 0.875
#   ppv         = 2/(2+1) = 2/3 = 0.666666...
#   npv         = 7/(7+0) = 1.0
#   f1          = 2*2/(2*2+1+0) = 4/5 = 0.8
# ---------------------------------------------------------------------------

def _hand_built_case():
    human = ["faithful"] * 5 + ["degraded"] * 3 + ["drift"] * 2
    auto = (
        ["faithful", "faithful", "faithful", "faithful", "degraded"]
        + ["degraded", "degraded", "drift"]
        + ["drift", "drift"]
    )
    return human, auto


def test_hand_built_confusion_pattern_confusion_matrix():
    human, auto = _hand_built_case()
    cm = confusion(human, auto)

    expected = pd.DataFrame(
        [[4, 1, 0], [0, 2, 1], [0, 0, 2]],
        index=pd.Index(list(CLASS_ORDER), name="human"),
        columns=pd.Index(list(CLASS_ORDER), name="auto"),
    )
    pd.testing.assert_frame_equal(cm, expected, check_dtype=False)
    assert cm.values.sum() == 10


def test_hand_built_confusion_pattern_class_metrics():
    human, auto = _hand_built_case()
    result = class_metrics(human, auto).set_index("class")

    faithful = result.loc["faithful"]
    assert faithful["sensitivity"] == pytest.approx(0.8)
    assert faithful["specificity"] == pytest.approx(1.0)
    assert faithful["ppv"] == pytest.approx(1.0)
    assert faithful["npv"] == pytest.approx(5 / 6)
    assert faithful["f1"] == pytest.approx(8 / 9)
    assert faithful["support"] == 5

    degraded = result.loc["degraded"]
    assert degraded["sensitivity"] == pytest.approx(2 / 3)
    assert degraded["specificity"] == pytest.approx(6 / 7)
    assert degraded["ppv"] == pytest.approx(2 / 3)
    assert degraded["npv"] == pytest.approx(6 / 7)
    assert degraded["f1"] == pytest.approx(2 / 3)
    assert degraded["support"] == 3

    drift = result.loc["drift"]
    assert drift["sensitivity"] == pytest.approx(1.0)
    assert drift["specificity"] == pytest.approx(0.875)
    assert drift["ppv"] == pytest.approx(2 / 3)
    assert drift["npv"] == pytest.approx(1.0)
    assert drift["f1"] == pytest.approx(0.8)
    assert drift["support"] == 2


def test_class_metrics_column_order_and_fixed_label_order():
    human, auto = _hand_built_case()
    result = class_metrics(human, auto)

    assert list(result.columns) == [
        "class",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "support",
    ]
    # fixed label order, not alphabetical-by-accident or data-order-dependent
    assert list(result["class"]) == list(CLASS_ORDER)


# ---------------------------------------------------------------------------
# macro_weighted: derived directly from the hand-built case above.
#
# macro_f1 = mean(8/9, 2/3, 4/5) = (0.888888... + 0.666666... + 0.8) / 3
#          = 2.355555.../3 = 0.785185...
# weighted_f1 = (5*8/9 + 3*2/3 + 2*4/5) / 10
#             = (4.444444... + 2.0 + 1.6) / 10 = 8.044444.../10 = 0.804444...
# ---------------------------------------------------------------------------

def test_macro_weighted_hand_computed():
    human, auto = _hand_built_case()
    class_df = class_metrics(human, auto)

    result = macro_weighted(class_df)

    assert result["macro_f1"] == pytest.approx(0.7851851851851852)
    assert result["weighted_f1"] == pytest.approx(0.8044444444444444)


# ---------------------------------------------------------------------------
# Zero-support / small-support edge cases: a class absent from human, or
# absent from auto, must not raise and must not produce a silent NaN.
# Convention: an undefined (0/0) ratio -> 0.0. A ratio that is well-defined
# even though one side is zero (e.g., 0 false positives -> PPV real 0.0, or
# specificity 1.0 because there were zero false positives) is computed for
# real, not forced to a fixed convention value.
# ---------------------------------------------------------------------------

def test_class_absent_from_human_convention_is_documented_not_nan():
    # "drift" never appears in human (ground truth), but auto predicts it
    # once (a false positive against a truly "faithful" item).
    #
    # Confusion (rows=human, cols=auto), fixed order:
    #            auto:faithful  auto:degraded  auto:drift
    # faithful         1              0             1        (support 2)
    # degraded         0              1             0        (support 1)
    # drift            0              0             0        (support 0)
    # total = 3
    #
    # class "drift": TP=0, FN=0-0=0 (row sum 0), FP=1-0=1 (col sum 1),
    #   TN=3-0-0-1=2
    #   sensitivity = 0/0 -> undefined -> convention 0.0
    #   specificity = 2/(2+1) = 2/3 = 0.666666...  (well-defined, real)
    #   ppv         = 0/(0+1) = 0.0                (well-defined, real zero)
    #   npv         = 2/(2+0) = 1.0                (well-defined, real)
    #   f1          = 2*0/(0+1+0) = 0.0             (well-defined, real zero)
    human = ["faithful", "faithful", "degraded"]
    auto = ["faithful", "drift", "degraded"]

    result = class_metrics(human, auto).set_index("class")
    drift = result.loc["drift"]

    assert drift["support"] == 0
    assert drift["sensitivity"] == 0.0
    assert not math.isnan(drift["sensitivity"])
    assert drift["specificity"] == pytest.approx(2 / 3)
    assert drift["ppv"] == 0.0
    assert drift["npv"] == pytest.approx(1.0)
    assert drift["f1"] == 0.0
    # no metric anywhere in the frame is NaN
    assert not result.isna().any().any()


def test_class_absent_from_auto_convention_is_documented_not_nan():
    # "drift" appears once in human (ground truth) but auto never predicts
    # it (the one drift item is misclassified as "faithful").
    #
    # Confusion (rows=human, cols=auto), fixed order:
    #            auto:faithful  auto:degraded  auto:drift
    # faithful         2              0             0        (support 2)
    # drift            1              0             0        (support 1)
    # degraded         0              0             0        (support 0, not in data at all)
    # total = 3
    #
    # class "drift": TP=0, FN=1-0=1 (row sum 1), FP=0-0=0 (col sum 0),
    #   TN=3-0-1-0=2
    #   sensitivity = 0/(0+1) = 0.0   (well-defined, real zero)
    #   specificity = 2/(2+0) = 1.0   (well-defined, real)
    #   ppv         = 0/0 -> undefined -> convention 0.0
    #   npv         = 2/(2+1) = 2/3 = 0.666666...
    #   f1          = 2*0/(0+0+1) = 0.0
    human = ["faithful", "faithful", "drift"]
    auto = ["faithful", "faithful", "faithful"]

    result = class_metrics(human, auto).set_index("class")
    drift = result.loc["drift"]

    assert drift["support"] == 1
    assert drift["sensitivity"] == 0.0
    assert drift["specificity"] == pytest.approx(1.0)
    assert drift["ppv"] == 0.0
    assert not math.isnan(drift["ppv"])
    assert drift["npv"] == pytest.approx(2 / 3)
    assert drift["f1"] == 0.0

    degraded = result.loc["degraded"]
    assert degraded["support"] == 0
    assert not result.isna().any().any()


def test_unexpected_label_raises_instead_of_silently_dropping():
    human = ["faithful", "typo_label"]
    auto = ["faithful", "faithful"]

    with pytest.raises(ValueError, match="typo_label"):
        class_metrics(human, auto)


def test_explicit_labels_override_the_fixed_default():
    human = ["a", "b", "a"]
    auto = ["a", "a", "b"]

    result = class_metrics(human, auto, labels=["a", "b"])
    assert list(result["class"]) == ["a", "b"]


# ---------------------------------------------------------------------------
# by_stratum: two strata levels with different confusion patterns must
# produce different per-stratum metrics, and each stratum's numbers must
# equal class_metrics() computed on that stratum's subset alone (i.e. an
# actual partition-and-recompute, not the global metric repeated).
# ---------------------------------------------------------------------------

def test_by_stratum_partitions_and_recomputes_not_global_repeated():
    # stratum "low": perfect classifier (all 1.0)
    human_low = ["faithful", "faithful", "degraded", "drift"]
    auto_low = list(human_low)

    # stratum "high": the hand-built imperfect case from above
    human_high, auto_high = _hand_built_case()

    human = human_low + human_high
    auto = auto_low + auto_high
    strata = pd.DataFrame(
        {"cer_target": ["low"] * len(human_low) + ["high"] * len(human_high)}
    )

    result = by_stratum(human, auto, strata)

    assert list(result.columns) == [
        "cer_target",
        "class",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
        "support",
    ]
    assert set(result["cer_target"]) == {"low", "high"}

    low_rows = result[result["cer_target"] == "low"].drop(columns="cer_target").reset_index(drop=True)
    high_rows = result[result["cer_target"] == "high"].drop(columns="cer_target").reset_index(drop=True)

    expected_low = class_metrics(human_low, auto_low).reset_index(drop=True)
    expected_high = class_metrics(human_high, auto_high).reset_index(drop=True)

    pd.testing.assert_frame_equal(low_rows, expected_low)
    pd.testing.assert_frame_equal(high_rows, expected_high)

    # the two strata must actually differ (guards against a bug that just
    # repeats one global computation for every stratum level)
    low_faithful_f1 = low_rows.set_index("class").loc["faithful", "f1"]
    high_faithful_f1 = high_rows.set_index("class").loc["faithful", "f1"]
    assert low_faithful_f1 != high_faithful_f1
    assert low_faithful_f1 == pytest.approx(1.0)
    assert high_faithful_f1 == pytest.approx(8 / 9)


def test_by_stratum_supports_multiple_stratum_columns():
    human_a = ["faithful", "faithful", "degraded", "drift"]
    auto_a = list(human_a)
    human_b, auto_b = _hand_built_case()

    human = human_a + human_b
    auto = auto_a + auto_b
    strata = pd.DataFrame(
        {
            "cer_target": ["low"] * len(human_a) + ["high"] * len(human_b),
            "model_id": ["m1"] * len(human_a) + ["m2"] * len(human_b),
        }
    )

    result = by_stratum(human, auto, strata)

    assert list(result.columns)[:2] == ["cer_target", "model_id"]
    combos = set(zip(result["cer_target"], result["model_id"]))
    assert combos == {("low", "m1"), ("high", "m2")}


def test_by_stratum_accepts_a_series_as_a_single_stratum_column():
    human_low = ["faithful", "faithful", "degraded", "drift"]
    auto_low = list(human_low)
    human_high, auto_high = _hand_built_case()

    human = human_low + human_high
    auto = auto_low + auto_high
    strata = pd.Series(["low"] * len(human_low) + ["high"] * len(human_high), name="cer_target")

    result = by_stratum(human, auto, strata)

    assert "cer_target" in result.columns
    assert set(result["cer_target"]) == {"low", "high"}


def test_by_stratum_length_mismatch_raises():
    human = ["faithful", "degraded"]
    auto = ["faithful", "degraded"]
    strata = pd.DataFrame({"cer_target": ["low"]})

    with pytest.raises(ValueError):
        by_stratum(human, auto, strata)


def test_class_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        class_metrics(["faithful", "drift"], ["faithful"])


# ---------------------------------------------------------------------------
# weights= (Task 7, reviewer #1 follow-up): per-item sampling weights ->
# weighted sensitivity/specificity/PPV/NPV/F1 and the weighted confusion
# matrix that feeds them. `weights=None` (the default) MUST be byte-identical
# to the pre-Task-7 behavior -- every test above this section is re-run
# unmodified against the same functions and must still pass unchanged; the
# tests below additionally pin the identity explicitly.
#
# Hand-built weighted confusion pattern (human rows, auto cols, fixed order):
#   item  human      auto       weight
#   0     faithful   faithful   2.0
#   1     faithful   degraded   1.0
#   2     degraded   degraded   3.0
#   3     drift      drift      4.0
#
# Weighted confusion matrix:
#            auto:faithful  auto:degraded  auto:drift
# faithful        2.0            1.0           0.0      (row sum 3.0)
# degraded        0.0            3.0           0.0      (row sum 3.0)
# drift           0.0            0.0           4.0      (row sum 4.0)
# total = 10.0
#
# class "faithful": TP=2, FN=3-2=1, FP=(col sum 2)-2=0, TN=10-2-1-0=7
#   sensitivity = 2/3 = 0.666666...
#   specificity = 7/(7+0) = 1.0
#   ppv         = 2/(2+0) = 1.0
#   npv         = 7/(7+1) = 7/8 = 0.875
#   f1          = 2*2/(2*2+0+1) = 4/5 = 0.8
#
# class "degraded": TP=3, FN=3-3=0, FP=(col sum 1+3=4)-3=1, TN=10-3-0-1=6
#   sensitivity = 3/3 = 1.0
#   specificity = 6/(6+1) = 6/7 = 0.857142...
#   ppv         = 3/(3+1) = 3/4 = 0.75
#   npv         = 6/(6+0) = 1.0
#   f1          = 2*3/(2*3+1+0) = 6/7 = 0.857142...
#
# class "drift": TP=4, FN=4-4=0, FP=(col sum 4)-4=0, TN=10-4-0-0=6
#   sensitivity = specificity = ppv = npv = f1 = 1.0
# ---------------------------------------------------------------------------

_WEIGHTED_HUMAN = ["faithful", "faithful", "degraded", "drift"]
_WEIGHTED_AUTO = ["faithful", "degraded", "degraded", "drift"]
_WEIGHTS = [2.0, 1.0, 3.0, 4.0]


def test_weighted_confusion_hand_computed():
    cm = confusion(_WEIGHTED_HUMAN, _WEIGHTED_AUTO, weights=_WEIGHTS)

    expected = pd.DataFrame(
        [[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
        index=pd.Index(list(CLASS_ORDER), name="human"),
        columns=pd.Index(list(CLASS_ORDER), name="auto"),
    )
    pd.testing.assert_frame_equal(cm, expected, check_dtype=False)
    assert cm.values.sum() == pytest.approx(10.0)


def test_weighted_class_metrics_hand_computed():
    result = class_metrics(_WEIGHTED_HUMAN, _WEIGHTED_AUTO, weights=_WEIGHTS).set_index("class")

    faithful = result.loc["faithful"]
    assert faithful["sensitivity"] == pytest.approx(2 / 3)
    assert faithful["specificity"] == pytest.approx(1.0)
    assert faithful["ppv"] == pytest.approx(1.0)
    assert faithful["npv"] == pytest.approx(7 / 8)
    assert faithful["f1"] == pytest.approx(4 / 5)
    assert faithful["support"] == pytest.approx(3.0)

    degraded = result.loc["degraded"]
    assert degraded["sensitivity"] == pytest.approx(1.0)
    assert degraded["specificity"] == pytest.approx(6 / 7)
    assert degraded["ppv"] == pytest.approx(3 / 4)
    assert degraded["npv"] == pytest.approx(1.0)
    assert degraded["f1"] == pytest.approx(6 / 7)
    assert degraded["support"] == pytest.approx(3.0)

    drift = result.loc["drift"]
    assert drift["sensitivity"] == pytest.approx(1.0)
    assert drift["specificity"] == pytest.approx(1.0)
    assert drift["ppv"] == pytest.approx(1.0)
    assert drift["npv"] == pytest.approx(1.0)
    assert drift["f1"] == pytest.approx(1.0)
    assert drift["support"] == pytest.approx(4.0)


def test_weights_none_is_byte_identical_to_previous_unweighted_behavior():
    # weights=None must reproduce the exact pre-Task-7 output: same values
    # AND same dtypes (not just numerically close), on both the existing
    # hand-built case and a fresh call with no weights= argument at all.
    human, auto = _hand_built_case()

    cm_no_arg = confusion(human, auto)
    cm_explicit_none = confusion(human, auto, weights=None)
    pd.testing.assert_frame_equal(cm_no_arg, cm_explicit_none)  # dtype-strict

    cls_no_arg = class_metrics(human, auto)
    cls_explicit_none = class_metrics(human, auto, weights=None)
    pd.testing.assert_frame_equal(cls_no_arg, cls_explicit_none)  # dtype-strict


def test_equal_weights_reduce_to_unweighted_ratios():
    # Every item given the SAME nonunit weight (5.0): the confusion matrix
    # scales uniformly, but every sensitivity/specificity/ppv/npv/f1 ratio
    # is scale-invariant to a uniform per-item weight, so it must reproduce
    # the plain unweighted class_metrics() ratios exactly.
    human, auto = _hand_built_case()

    unweighted = class_metrics(human, auto).set_index("class")
    equal_weighted = class_metrics(human, auto, weights=[5.0] * len(human)).set_index("class")

    for cls in CLASS_ORDER:
        for metric in ("sensitivity", "specificity", "ppv", "npv", "f1"):
            assert equal_weighted.loc[cls, metric] == pytest.approx(unweighted.loc[cls, metric]), (
                f"{cls}/{metric} differed under a uniform nonunit weight"
            )
    # support scales by the constant weight (3 real items * weight 5 = 15, etc.)
    for cls in CLASS_ORDER:
        assert equal_weighted.loc[cls, "support"] == pytest.approx(unweighted.loc[cls, "support"] * 5.0)


def test_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        class_metrics(["faithful", "drift"], ["faithful", "drift"], weights=[1.0])
    with pytest.raises(ValueError):
        confusion(["faithful", "drift"], ["faithful", "drift"], weights=[1.0, 2.0, 3.0])


def test_negative_weights_raise():
    with pytest.raises(ValueError):
        class_metrics(["faithful", "drift"], ["faithful", "drift"], weights=[1.0, -1.0])


def test_by_stratum_weighted_partitions_and_recomputes():
    # Same two-stratum shape as test_by_stratum_partitions_and_recomputes_not_
    # global_repeated, but with per-item weights: each stratum's weighted
    # rows must equal class_metrics(..., weights=...) called directly on
    # that stratum's own subset (an actual partition-and-reweight, not a
    # global weighted metric repeated for every stratum level).
    human_low = ["faithful", "faithful", "degraded", "drift"]
    auto_low = list(human_low)
    weights_low = [1.0, 2.0, 3.0, 4.0]

    human_high, auto_high = _hand_built_case()
    weights_high = _WEIGHTS + [1.0] * (len(human_high) - len(_WEIGHTS))

    human = human_low + human_high
    auto = auto_low + auto_high
    weights = weights_low + weights_high
    strata = pd.DataFrame(
        {"cer_target": ["low"] * len(human_low) + ["high"] * len(human_high)}
    )

    result = by_stratum(human, auto, strata, weights=weights)

    low_rows = result[result["cer_target"] == "low"].drop(columns="cer_target").reset_index(drop=True)
    high_rows = result[result["cer_target"] == "high"].drop(columns="cer_target").reset_index(drop=True)

    expected_low = class_metrics(human_low, auto_low, weights=weights_low).reset_index(drop=True)
    expected_high = class_metrics(human_high, auto_high, weights=weights_high).reset_index(drop=True)

    pd.testing.assert_frame_equal(low_rows, expected_low)
    pd.testing.assert_frame_equal(high_rows, expected_high)


def test_by_stratum_weights_none_is_byte_identical_to_previous_unweighted_behavior():
    human_low = ["faithful", "faithful", "degraded", "drift"]
    auto_low = list(human_low)
    human_high, auto_high = _hand_built_case()
    human = human_low + human_high
    auto = auto_low + auto_high
    strata = pd.DataFrame(
        {"cer_target": ["low"] * len(human_low) + ["high"] * len(human_high)}
    )

    no_arg = by_stratum(human, auto, strata)
    explicit_none = by_stratum(human, auto, strata, weights=None)
    pd.testing.assert_frame_equal(no_arg, explicit_none)
