"""Tests for the CRIT/CTRL matching balance table (revision Task B4).

Reviewers said "matched controls" is asserted (`idrift.data.matched_controls`
builds one non-critical CONTROL per CRIT item) but no balance diagnostic is
shown proving the match actually equated CRIT and CTRL on the matching
covariates. This module reports the standard covariate-balance statistic
-- the standardized mean difference (SMD) -- both BEFORE matching (full CRIT
pool vs. full CTRL candidate pool, i.e. what the imbalance would be if no
matching had been done) and AFTER matching (each CRIT item vs. its own
realized matched-CTRL partner, from `idrift.data.matched_controls.
build_controls`'s output), in a single tidy long-form table -- Love-plot
ready.

All tests run on small hand-built synthetic frames (never the real corpus),
so the SMD arithmetic can be verified by hand.
"""
import math

import numpy as np
import pandas as pd
import pytest

from idrift.analysis.balance import DEFAULT_COVARIATES, balance_table


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ---------------------------------------------------------------------------
# Core SMD arithmetic: known mean gap, hand-computed pooled SD.
# ---------------------------------------------------------------------------


def test_smd_before_matches_hand_formula_for_known_mean_gap():
    # crit "score": mean 13, population var 5 -- ctrl "score": mean 9, population var 5
    crit_frame = pd.DataFrame({"score": [10, 12, 14, 16]})
    ctrl_frame = pd.DataFrame({"score": [6, 8, 10, 12]})

    out = balance_table(crit_frame, ctrl_frame, covariates=["score"])

    assert list(out["covariate"]) == ["score"]
    pooled_sd = math.sqrt((5.0 + 5.0) / 2)
    expected = (13.0 - 9.0) / pooled_sd
    assert out.loc[0, "smd_before"] == pytest.approx(expected)
    assert expected == pytest.approx(1.7888543819998317)


def test_smd_after_near_zero_for_balanced_matched_subset_despite_unbalanced_pools():
    # Same unbalanced before-pools as above (SMD_before ~ 1.79), but the
    # REALIZED matched pairs have equal means on each side -> SMD_after ~ 0,
    # demonstrating the matching corrected the imbalance the before-column
    # shows existed in the raw candidate pools.
    crit_frame = pd.DataFrame({"score": [10, 12, 14, 16]})
    ctrl_frame = pd.DataFrame({"score": [6, 8, 10, 12]})
    matched_pairs = pd.DataFrame({
        "crit_score": [10, 12, 14, 16],  # mean 13
        "score": [11, 11, 15, 15],       # mean 13 (matched CTRL's own values)
    })

    out = balance_table(crit_frame, ctrl_frame, matched_pairs, covariates=["score"])

    assert out.loc[0, "smd_before"] == pytest.approx(1.7888543819998317)
    assert out.loc[0, "smd_after"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc[0, "max_abs_smd_after"] == pytest.approx(0.0, abs=1e-9)


def test_pooled_sd_zero_guard_returns_zero_not_nan_or_error():
    # Covariate constant (no variance) in both groups -- pooled_sd == 0.
    crit_frame = pd.DataFrame({"score": [5, 5, 5]})
    ctrl_frame = pd.DataFrame({"score": [5, 5, 5]})

    out = balance_table(crit_frame, ctrl_frame, covariates=["score"])

    assert out.loc[0, "smd_before"] == 0.0


# ---------------------------------------------------------------------------
# "Do not fabricate" -- before/after unavailability is marked null with a
# note, never silently guessed at.
# ---------------------------------------------------------------------------


def test_before_matching_pool_unavailable_is_null_not_fabricated():
    crit_frame = pd.DataFrame({"score": [10, 12, 14, 16]})
    matched_pairs = pd.DataFrame({"crit_score": [10, 12], "score": [10, 12]})

    out = balance_table(crit_frame, None, matched_pairs, covariates=["score"])

    assert _is_nan(out.loc[0, "smd_before"])
    assert "not supplied" in out.loc[0, "note"]
    # after is still computed -- unavailability of one side must not blank the other
    assert out.loc[0, "smd_after"] == pytest.approx(0.0, abs=1e-9)


def test_matched_pairs_unavailable_is_null_not_fabricated():
    crit_frame = pd.DataFrame({"score": [10, 12, 14, 16]})
    ctrl_frame = pd.DataFrame({"score": [6, 8, 10, 12]})

    out = balance_table(crit_frame, ctrl_frame, covariates=["score"])  # matched_pairs defaults to None

    assert _is_nan(out.loc[0, "smd_after"])
    assert "not supplied" in out.loc[0, "note"]
    assert out.loc[0, "smd_before"] == pytest.approx(1.7888543819998317)
    assert _is_nan(out.loc[0, "max_abs_smd_after"])


def test_empty_ctrl_frame_treated_same_as_none():
    crit_frame = pd.DataFrame({"score": [10, 12, 14, 16]})
    empty_ctrl = pd.DataFrame({"score": []})

    out = balance_table(crit_frame, empty_ctrl, covariates=["score"])

    assert _is_nan(out.loc[0, "smd_before"])


# ---------------------------------------------------------------------------
# Tidy long-form shape: one row per covariate, plus summary columns.
# ---------------------------------------------------------------------------


def test_tidy_long_form_columns_and_max_abs_smd_after():
    crit_frame = pd.DataFrame({
        "message_id": ["p1", "p2"],
        "char_len": [11, 13],
        "word_count": [2, 3],
    })
    ctrl_frame = pd.DataFrame({
        "message_id": ["c1", "c2", "c3", "c4", "c5"],
        "char_len": [10, 12, 8, 20, 15],
        "word_count": [2, 3, 1, 4, 3],
    })
    matched_pairs = pd.DataFrame({
        "crit_message_id": ["p1", "p2"],
        "message_id": ["c1", "c2"],
        "crit_char_len": [11, 13],
        "char_len": [10, 12],
        "crit_word_count": [2, 3],
        "word_count": [2, 3],
    })

    out = balance_table(crit_frame, ctrl_frame, matched_pairs, covariates=["char_len", "word_count"])

    assert list(out.columns) == [
        "covariate", "smd_before", "smd_after", "note", "max_abs_smd_after", "n_unmatched",
    ]
    assert len(out) == 2
    assert set(out["covariate"]) == {"char_len", "word_count"}

    # word_count matched pairs are identical (2,3 vs 2,3) -> smd_after 0;
    # char_len matched pairs (11,13 vs 10,12) -> smd_after 1.0 (equal
    # variances, mean gap 1). max|SMD| after must be the max across rows.
    wc_row = out[out["covariate"] == "word_count"].iloc[0]
    cl_row = out[out["covariate"] == "char_len"].iloc[0]
    assert wc_row["smd_after"] == pytest.approx(0.0, abs=1e-9)
    assert cl_row["smd_after"] == pytest.approx(1.0)
    assert out["max_abs_smd_after"].iloc[0] == pytest.approx(1.0)

    # 5 ctrl candidates, only 2 used as matches -> 3 unmatched, determinable
    # via the shared message_id id column.
    assert (out["n_unmatched"] == 3).all()


def test_n_unmatched_is_none_when_not_determinable():
    # No message_id / id column anywhere -- unmatched count cannot be derived.
    crit_frame = pd.DataFrame({"score": [10, 12]})
    ctrl_frame = pd.DataFrame({"score": [6, 8, 10, 12, 14]})
    matched_pairs = pd.DataFrame({"crit_score": [10, 12], "score": [10, 12]})

    out = balance_table(crit_frame, ctrl_frame, matched_pairs, covariates=["score"])

    assert out.loc[0, "n_unmatched"] is None


def test_default_covariates_infer_the_five_real_matching_features():
    # Mirrors the real idrift.data.matched_controls column convention
    # (char_len, word_count, has_numeral, has_negation, mean_word_freq) with
    # no explicit `covariates=` argument.
    crit_frame = pd.DataFrame({
        "char_len": [20.0, 22.0],
        "word_count": [4.0, 5.0],
        "has_numeral": [False, True],
        "has_negation": [False, False],
        "mean_word_freq": [5.5, 5.7],
    })
    ctrl_frame = pd.DataFrame({
        "char_len": [21.0, 23.0, 19.0],
        "word_count": [4.0, 5.0, 3.0],
        "has_numeral": [False, False, False],
        "has_negation": [False, True, False],
        "mean_word_freq": [5.4, 5.6, 5.8],
    })

    out = balance_table(crit_frame, ctrl_frame)

    assert set(out["covariate"]) == set(DEFAULT_COVARIATES)
    assert len(out) == len(DEFAULT_COVARIATES)
    assert out["smd_before"].notna().all()


def test_no_resolvable_covariates_raises():
    crit_frame = pd.DataFrame({"message_id": ["p1", "p2"]})
    ctrl_frame = pd.DataFrame({"message_id": ["c1", "c2"]})
    with pytest.raises(ValueError):
        balance_table(crit_frame, ctrl_frame)
