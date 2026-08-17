"""Tests for the CRIT-vs-CTRL matched-pair conditional logistic drift
comparison (revision Task 5.3).

All tests run on SIMULATED data or hand-built tiny fixtures constructed
here, never on the real labeled/exposure/ctrl_matched artifacts (a labeling
job was running remotely at the time this task was implemented -- see the
task brief's environment note).
"""
import numpy as np
import pandas as pd
import pytest

from idrift.analysis.matched_compare import (
    build_matched_frame,
    matched_drift,
    risk_difference,
    risk_difference_by_cer,
)

# ---------------------------------------------------------------------------
# Simulation ingredients for the null-gate / positive-control tests.
#
# N_PAIRS matched CRIT-CTRL pairs, N_ROWS_PER_HALF attempt-level rows per
# half (8 rows/pair total). The CONFOUND the reviewers flagged is baked in
# on purpose: the CRIT half of every pair gets a systematically HIGHER
# corrupted_negation/corrupted_numeral rate (+0.30 each) and a longer
# char_len (+6) and higher realized_cer (+0.05) than its own CTRL half --
# exactly "critical items differ in negation/numeral/length" -- while the
# TRUE outcome model is
#
#   logit(P(drift)) = BETA0 + BETA_NEG*corrupted_negation
#                     + BETA_NUM*corrupted_numeral + beta_crit*critical
#
# i.e. char_len/n_errors/realized_cer have NO true causal effect either
# (included only as prespecified adjusters), and `critical` itself only has
# an effect when `beta_crit != 0`.
#
# Confound sanity check (development-time, not asserted in a test): fitting
# the SAME matched-pair conditional model WITHOUT the negation/numeral
# adjusters on this exact seed/beta_crit=0 draw gives critical est=0.636,
# 95% CI (0.400, 0.872) -- a spurious, confidently-wrong positive effect.
# Only once corrupted_negation/corrupted_numeral enter the design (as
# `matched_drift` always includes them) does the CI correctly cover 0 (see
# `test_null_gate_...` below). This proves the simulated confound is real,
# not just a "no effect -> no effect" tautology.
# ---------------------------------------------------------------------------

N_PAIRS = 150
N_ROWS_PER_HALF = 4
BETA0 = -1.6
BETA_NEG = 1.6
BETA_NUM = 1.3
SIM_SEED = 0


def _simulate_matched_pairs(seed: int, beta_crit: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_PAIRS):
        pair_id = f"pair_{i:03d}"
        base_neg = rng.uniform(0.10, 0.35)
        base_num = rng.uniform(0.10, 0.35)
        # (critical, negation_shift, numeral_shift, char_len_shift, cer_shift)
        for critical, neg_shift, num_shift, len_shift, cer_shift in (
            (1, 0.30, 0.30, 6.0, 0.05),
            (0, 0.0, 0.0, 0.0, 0.0),
        ):
            for _rep in range(N_ROWS_PER_HALF):
                negation = int(rng.random() < min(base_neg + neg_shift, 0.95))
                numeral = int(rng.random() < min(base_num + num_shift, 0.95))
                char_len = float(rng.normal(30.0 + len_shift, 4.0))
                n_errors = int(rng.poisson(3))
                realized_cer = float(np.clip(rng.uniform(0.05, 0.35) + cer_shift, 0.0, 1.0))

                lin = BETA0 + BETA_NEG * negation + BETA_NUM * numeral + beta_crit * critical
                p_drift = 1.0 / (1.0 + np.exp(-lin))
                drift = int(rng.random() < p_drift)

                rows.append(
                    dict(
                        pair_id=pair_id,
                        message_id=f"{'crit' if critical else 'ctrl'}_{i:03d}",
                        critical=critical,
                        corrupted_negation=negation,
                        corrupted_numeral=numeral,
                        char_len=char_len,
                        n_errors=n_errors,
                        realized_cer=realized_cer,
                        drift=drift,
                    )
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Null gate (the confound guard).
# ---------------------------------------------------------------------------


def test_null_gate_adjusted_critical_ci_covers_zero_once_negation_numeral_controlled():
    df = _simulate_matched_pairs(SIM_SEED, beta_crit=0.0)
    result = matched_drift(df)

    assert result["engine"] == "conditional_logit"
    assert result["converged"] is True
    assert set(result["coef"].keys()) == {
        "critical", "char_len", "n_errors", "corrupted_negation", "corrupted_numeral", "realized_cer",
    }

    crit = result["coef"]["critical"]
    assert crit["ci_lo"] <= 0.0 <= crit["ci_hi"]

    # The two TRUE drivers should recover positive, non-null coefficients
    # (sanity that the design is well-posed, not just uninformative).
    assert result["coef"]["corrupted_negation"]["est"] > 0
    assert result["coef"]["corrupted_numeral"]["est"] > 0

    assert result["n_pairs"] <= N_PAIRS  # some fully-concordant pairs may be dropped (documented)
    assert result["n_pairs"] > N_PAIRS * 0.9  # but not many, at this design's event rate
    assert "matched pair" in result["notes"]
    assert "conditional logistic regression, not a bug" in result["notes"]


# ---------------------------------------------------------------------------
# 2. Positive control.
# ---------------------------------------------------------------------------


def test_positive_control_adjusted_critical_ci_excludes_zero_with_correct_sign():
    df = _simulate_matched_pairs(SIM_SEED, beta_crit=1.3)
    result = matched_drift(df)

    crit = result["coef"]["critical"]
    assert crit["est"] > 0
    assert crit["ci_lo"] > 0.0
    assert crit["p"] < 0.05


# ---------------------------------------------------------------------------
# 3. Determinism.
# ---------------------------------------------------------------------------


def test_determinism_same_df_reproduces_identical_coefficients():
    df = _simulate_matched_pairs(SIM_SEED, beta_crit=1.3)
    result_a = matched_drift(df)
    result_b = matched_drift(df)

    assert result_a["n_obs"] == result_b["n_obs"]
    assert result_a["n_pairs"] == result_b["n_pairs"]
    for name in result_a["coef"]:
        assert result_a["coef"][name]["est"] == result_b["coef"][name]["est"]
        assert result_a["coef"][name]["se"] == result_b["coef"][name]["se"]
        assert result_a["coef"][name]["p"] == result_b["coef"][name]["p"]


# ---------------------------------------------------------------------------
# 4. Input-contract validation.
# ---------------------------------------------------------------------------


def test_matched_drift_missing_column_raises():
    df = _simulate_matched_pairs(SIM_SEED, beta_crit=0.0).drop(columns=["corrupted_negation"])
    with pytest.raises(ValueError, match="corrupted_negation"):
        matched_drift(df)


def test_matched_drift_missing_value_raises():
    df = _simulate_matched_pairs(SIM_SEED, beta_crit=0.0)
    df.loc[df.index[0], "realized_cer"] = np.nan
    with pytest.raises(ValueError, match=r"1 row\(s\)"):
        matched_drift(df)


# ---------------------------------------------------------------------------
# 5. build_matched_frame join correctness (hand-built fixtures only).
# ---------------------------------------------------------------------------


def _ctrl_matched_fixture():
    # Mirrors idrift.data.matched_controls.build_controls's real output
    # schema (see that module): one row per CRIT item, crit_message_id
    # paired with its own matched control's message_id.
    return pd.DataFrame(
        {
            "crit_message_id": ["c1", "c2"],
            "crit_text": ["DO NOT GIVE WATER", "CALL DOCTOR NOW"],
            "crit_char_len": [18.0, 15.0],
            "crit_word_count": [4.0, 3.0],
            "crit_has_numeral": [False, False],
            "crit_has_negation": [True, False],
            "crit_mean_word_freq": [5.0, 5.2],
            "message_id": ["k1", "k2"],
            "text": ["PLEASE BRING SOME FOOD", "TAKE ME HOME SOON"],
            "char_len": [22.0, 18.0],
            "word_count": [4.0, 4.0],
            "has_numeral": [False, False],
            "has_negation": [False, False],
            "mean_word_freq": [5.1, 5.3],
            "match_distance": [0.42, 0.31],
        }
    )


def _labeled_fixture():
    return pd.DataFrame(
        {
            "message_id": ["c1", "c1", "k1", "k1", "c2", "k2", "a1"],
            "corpus": ["CRIT", "CRIT", "CTRL", "CTRL", "CRIT", "CTRL", "AUTH"],
            "cer_target": [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1],
            "replicate_idx": [0, 0, 0, 0, 0, 0, 0],
            "label": ["drift", "faithful", "faithful", "degraded", "drift", "faithful", "drift"],
            "realized_cer": [0.09, 0.19, 0.11, 0.22, 0.08, 0.09, 0.10],
        }
    )


def _exposure_fixture():
    # 'a1' (AUTH, not part of any Task-1.3 match) deliberately has no
    # exposure row either -- it must never be reached, since the mapping
    # join drops it first.
    return pd.DataFrame(
        {
            "message_id": ["c1", "c1", "k1", "k1", "c2", "k2"],
            "cer_target": [0.1, 0.2, 0.1, 0.2, 0.1, 0.1],
            "replicate_idx": [0, 0, 0, 0, 0, 0],
            "n_errors": [3, 5, 2, 4, 1, 2],
            "corrupted_negation": [1, 0, 0, 1, 1, 0],
            "corrupted_numeral": [0, 0, 0, 0, 0, 1],
            "text": [
                "DO NOT GIVE WATER", "DO NOT GIVE WATER",
                "PLEASE BRING SOME FOOD", "PLEASE BRING SOME FOOD",
                "CALL DOCTOR NOW", "TAKE ME HOME SOON",
            ],
        }
    )


def test_build_matched_frame_join_correctness_no_fanout_and_auth_dropped():
    merged = build_matched_frame(_labeled_fixture(), _ctrl_matched_fixture(), _exposure_fixture())

    # No row multiplication, and the AUTH row ('a1', not part of any
    # Task-1.3 match) is dropped: 6 of the original 7 labeled rows survive.
    assert len(merged) == 6
    assert "a1" not in set(merged["message_id"])

    def _row(message_id, cer_target):
        sub = merged[(merged.message_id == message_id) & (merged.cer_target == cer_target)]
        assert len(sub) == 1
        return sub.iloc[0]

    c1_a = _row("c1", 0.1)
    c1_b = _row("c1", 0.2)
    k1_a = _row("k1", 0.1)
    k1_b = _row("k1", 0.2)
    c2 = _row("c2", 0.1)
    k2 = _row("k2", 0.1)

    # critical / pair_id land correctly: c1 and k1 share pair_id "c1"
    # (the CRIT item's own message_id); c2 and k2 share pair_id "c2".
    assert c1_a["critical"] == 1 and c1_a["pair_id"] == "c1"
    assert c1_b["critical"] == 1 and c1_b["pair_id"] == "c1"
    assert k1_a["critical"] == 0 and k1_a["pair_id"] == "c1"
    assert k1_b["critical"] == 0 and k1_b["pair_id"] == "c1"
    assert c2["critical"] == 1 and c2["pair_id"] == "c2"
    assert k2["critical"] == 0 and k2["pair_id"] == "c2"

    # drift derived correctly (label == 'drift'); non-drift labels
    # ('faithful', 'degraded') both collapse to drift=0.
    assert c1_a["drift"] == 1  # label 'drift'
    assert c1_b["drift"] == 0  # label 'faithful'
    assert k1_a["drift"] == 0  # label 'faithful'
    assert k1_b["drift"] == 0  # label 'degraded'
    assert c2["drift"] == 1
    assert k2["drift"] == 0

    # Exposure predictors land on the right rows (not swapped/averaged).
    assert c1_a["n_errors"] == 3 and c1_a["corrupted_negation"] == 1
    assert c1_b["n_errors"] == 5 and c1_b["corrupted_negation"] == 0
    assert k1_a["n_errors"] == 2 and k1_a["corrupted_negation"] == 0
    assert k1_b["n_errors"] == 4 and k1_b["corrupted_negation"] == 1
    assert k2["corrupted_numeral"] == 1

    # char_len derived from exposure's 'text' length (no literal column).
    assert c1_a["char_len"] == len("DO NOT GIVE WATER")
    assert k1_a["char_len"] == len("PLEASE BRING SOME FOOD")
    assert c2["char_len"] == len("CALL DOCTOR NOW")
    assert k2["char_len"] == len("TAKE ME HOME SOON")

    # realized_cer is carried through from the labeled side untouched.
    assert c1_a["realized_cer"] == 0.09
    assert k2["realized_cer"] == 0.09

    # The frame satisfies matched_drift's full input contract end to end.
    fit = matched_drift(merged.assign(pair_id=merged["pair_id"].astype(str)))
    assert fit["engine"] == "conditional_logit"


def test_build_matched_frame_duplicate_message_id_in_mapping_raises():
    dup = _ctrl_matched_fixture().copy()
    dup.loc[1, "message_id"] = "c1"  # c2's control collides with c1's own CRIT message_id
    with pytest.raises(ValueError, match="message_id"):
        build_matched_frame(_labeled_fixture(), dup, _exposure_fixture())


def test_build_matched_frame_raises_on_missing_join_key():
    labeled = _labeled_fixture().drop(columns=["cer_target"])
    with pytest.raises(ValueError, match="join key"):
        build_matched_frame(labeled, _ctrl_matched_fixture(), _exposure_fixture())


# ---------------------------------------------------------------------------
# 6. Task B6: absolute-scale effect measures (risk difference, drifts-per-
#    1000) and the criticality-by-CER interaction (stratum-specific RDs).
#
# A synthetic matched frame with a KNOWN CRIT-vs-CTRL drift-rate gap, built
# per matched pair so the pair-clustered bootstrap has real clusters to
# resample. Each (cer_target, arm) cell draws Bernoulli(rate) with a fixed
# per-arm rate, so the marginal drift rate in each arm has a KNOWN
# expectation and the CRIT-minus-CTRL risk difference has a KNOWN true value
# both overall and within every CER stratum. The gap GROWS with CER
# (0.06 -> 0.10 -> 0.18), so the by-CER RDs are the reviewer-requested
# criticality-by-CER interaction on the absolute scale.
# ---------------------------------------------------------------------------

_RD_CERS = (0.0, 0.1, 0.2)
_RD_CTRL_RATE = {0.0: 0.05, 0.1: 0.20, 0.2: 0.30}
_RD_CRIT_RATE = {0.0: 0.11, 0.1: 0.30, 0.2: 0.48}  # gap 0.06 / 0.10 / 0.18
_RD_N_PAIRS = 80
_RD_ROWS_PER_CELL = 12  # rows per (pair, cer, arm)


def _simulate_rd_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(_RD_N_PAIRS):
        pair_id = f"P{i:03d}"
        for cer in _RD_CERS:
            for critical, rate in ((1, _RD_CRIT_RATE[cer]), (0, _RD_CTRL_RATE[cer])):
                for _ in range(_RD_ROWS_PER_CELL):
                    rows.append(
                        dict(
                            pair_id=pair_id,
                            critical=critical,
                            cer_target=cer,
                            drift=int(rng.random() < rate),
                        )
                    )
    return pd.DataFrame(rows)


# True marginal rates are the equal-weight mean across the three CER cells
# (equal n per cell by construction).
_RD_TRUE_OVERALL = (
    sum(_RD_CRIT_RATE.values()) - sum(_RD_CTRL_RATE.values())
) / len(_RD_CERS)  # = 0.11333...
_RD_TRUE_BY_CER = {c: _RD_CRIT_RATE[c] - _RD_CTRL_RATE[c] for c in _RD_CERS}


def test_risk_difference_recovers_known_gap_per_1000_and_bracketing_ci():
    df = _simulate_rd_frame()
    res = risk_difference(df, n_boot=400, seed=0)

    assert res["estimand"] == "risk_difference_crit_minus_ctrl"

    # Point RD recovers the known generative gap within Monte-Carlo tolerance.
    assert abs(res["rd"] - _RD_TRUE_OVERALL) < 0.03

    # The clustered bootstrap CI brackets both the point estimate and truth.
    assert res["ci_lo"] <= res["rd"] <= res["ci_hi"]
    assert res["ci_lo"] <= _RD_TRUE_OVERALL <= res["ci_hi"]

    # additional_drifts_per_1000 is exactly RD*1000 on the point and both
    # CI bounds (a pure rescale of the same quantity).
    per1000 = res["additional_drifts_per_1000"]
    assert per1000["est"] == pytest.approx(res["rd"] * 1000.0)
    assert per1000["ci_lo"] == pytest.approx(res["ci_lo"] * 1000.0)
    assert per1000["ci_hi"] == pytest.approx(res["ci_hi"] * 1000.0)

    # Counts describe the real matched design.
    assert res["n_pairs"] == _RD_N_PAIRS
    assert res["n_crit"] == _RD_N_PAIRS * len(_RD_CERS) * _RD_ROWS_PER_CELL
    assert res["n_ctrl"] == _RD_N_PAIRS * len(_RD_CERS) * _RD_ROWS_PER_CELL
    # p_crit - p_ctrl is exactly the reported RD.
    assert res["p_crit"] - res["p_ctrl"] == pytest.approx(res["rd"])


def test_risk_difference_by_cer_returns_one_rd_per_level_matching_gaps():
    df = _simulate_rd_frame()
    res = risk_difference_by_cer(df, n_boot=400, seed=0)

    by = res["by_cer"]
    # Exactly one RD per CER level, in ascending CER order.
    assert [d["cer_target"] for d in by] == list(_RD_CERS)

    for d in by:
        true_gap = _RD_TRUE_BY_CER[d["cer_target"]]
        # Per-stratum tolerance ~2.5x the ~0.022 gap SE at this cell size
        # (960 rows/arm/stratum); the frame is fixed-seed, so deterministic.
        assert abs(d["rd"] - true_gap) < 0.05
        assert d["ci_lo"] <= d["rd"] <= d["ci_hi"]
        assert d["additional_drifts_per_1000"] == pytest.approx(d["rd"] * 1000.0)
        assert d["n_pairs"] == _RD_N_PAIRS

    # The interaction the reviewers asked for: the gap is materially larger
    # at high CER than at low CER (0.18 vs 0.06 true; separation >> noise).
    assert by[-1]["rd"] > by[0]["rd"]


def test_risk_difference_is_deterministic_for_fixed_seed():
    df = _simulate_rd_frame()
    a = risk_difference(df, n_boot=300, seed=0)
    b = risk_difference(df, n_boot=300, seed=0)
    assert a["rd"] == b["rd"]
    assert a["ci_lo"] == b["ci_lo"]
    assert a["ci_hi"] == b["ci_hi"]


def test_risk_difference_missing_column_raises():
    df = _simulate_rd_frame().drop(columns=["critical"])
    with pytest.raises(ValueError, match="critical"):
        risk_difference(df)


def test_risk_difference_by_cer_missing_cer_target_raises():
    df = _simulate_rd_frame().drop(columns=["cer_target"])
    with pytest.raises(ValueError, match="cer_target"):
        risk_difference_by_cer(df)
