"""Tests for the confidently-wrong rate, missing-confidence audit, and
predicted-drift-probability grid (revision Task B3).

Answers a gap in the manuscript: "confidently wrong" is DEFINED (a drift
output emitted at high verbalized confidence) but never actually
QUANTIFIED. This module reports (1) how often that happens, broken down
per model because verbalized-confidence scales differ across models (a
single pooled number would blur a well-calibrated model into a
badly-calibrated one), (2) where verbalized confidence is simply missing,
and (3) a simple, interpretable predicted-probability grid from a
message-clustered logistic fit of drift on realized CER alone.

All tests run on small, hand-built synthetic frames constructed here (never
on the real labeled parquet), so every assertion can be checked by hand
arithmetic.
"""
import numpy as np
import pandas as pd
import pytest

from idrift.analysis.confidence_analysis import (
    CER_GRID,
    confidently_wrong_rate,
    missing_confidence_audit,
    predicted_drift_grid,
)

# ---------------------------------------------------------------------------
# 1. confidently_wrong_rate -- pooled, hand-calculable.
#
# 10 rows, confidence on a 0-100 scale, default conf_threshold=0.9 (i.e. an
# effective threshold of 90 on this scale, inclusive).
#
#   row  label      confidence   high_conf(>=90)?
#   m1   drift      95           yes
#   m2   drift      95           yes
#   m3   drift      50           no
#   m4   faithful   95           yes
#   m5   faithful   92           yes
#   m6   faithful   60           no
#   m7   degraded   91           yes
#   m8   degraded   30           no
#   m9   drift      90           yes  (boundary: >= is inclusive)
#   m10  faithful   89           no   (boundary: just below threshold)
#
# n_drift = 4 (m1, m2, m3, m9); high-confidence AND drift = 3 (m1, m2, m9)
#   -> frac_drift_at_high_confidence = 3 / 4 = 0.75
# high-confidence rows = 6 (m1, m2, m4, m5, m7, m9)
#   not-faithful among those = 4 (m1, m2, m7, m9)
#   -> confidently_wrong_rate = 4 / 6 = 0.6667
# ---------------------------------------------------------------------------


def _row(message_id, label, confidence, model="modelA", cer_target=0.2, realized_cer=0.2):
    return dict(
        message_id=message_id,
        label=label,
        confidence=confidence,
        model=model,
        cer_target=cer_target,
        realized_cer=realized_cer,
    )


def _pooled_hand_calc_df():
    return pd.DataFrame(
        [
            _row("m1", "drift", 95),
            _row("m2", "drift", 95),
            _row("m3", "drift", 50),
            _row("m4", "faithful", 95),
            _row("m5", "faithful", 92),
            _row("m6", "faithful", 60),
            _row("m7", "degraded", 91),
            _row("m8", "degraded", 30),
            _row("m9", "drift", 90),
            _row("m10", "faithful", 89),
        ]
    )


def test_confidently_wrong_rate_matches_hand_calculation_pooled():
    df = _pooled_hand_calc_df()
    result = confidently_wrong_rate(df, conf_threshold=0.9)

    pooled = result["pooled"]
    assert pooled["n_drift"] == 4
    assert pooled["n_drift_at_high_confidence"] == 3
    assert pooled["frac_drift_at_high_confidence"] == pytest.approx(3 / 4)
    assert pooled["n_high_confidence"] == 6
    assert pooled["n_high_confidence_not_faithful"] == 4
    assert pooled["confidently_wrong_rate"] == pytest.approx(4 / 6)


def test_confidently_wrong_rate_reports_scale_and_effective_threshold():
    df = _pooled_hand_calc_df()
    result = confidently_wrong_rate(df, conf_threshold=0.9)

    assert result["scale_detected"] == "0-100"
    assert result["effective_threshold"] == pytest.approx(90.0)
    assert result["conf_threshold"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 2. confidently_wrong_rate -- per model, with pooled != either model's rate
#    (catches a bug where "pooled" silently collapses to one model or where
#    per-model grouping is wrong).
#
#   Model A: (drift,95) (drift,50) (faithful,95)
#     n_drift=2, high_conf&drift=1 -> frac=0.5
#     high_conf rows=2 (drift@95, faithful@95), not_faithful=1 -> rate=0.5
#
#   Model B: (drift,95) (drift,95) (degraded,95)
#     n_drift=2, high_conf&drift=2 -> frac=1.0
#     high_conf rows=3, not_faithful=3 -> rate=1.0
#
#   Pooled (A+B, 6 rows): n_drift=4, high_conf&drift=3 -> frac=0.75
#     high_conf rows=5, not_faithful=4 -> rate=0.8
# ---------------------------------------------------------------------------


def _per_model_df():
    return pd.DataFrame(
        [
            _row("a1", "drift", 95, model="modelA"),
            _row("a2", "drift", 50, model="modelA"),
            _row("a3", "faithful", 95, model="modelA"),
            _row("b1", "drift", 95, model="modelB"),
            _row("b2", "drift", 95, model="modelB"),
            _row("b3", "degraded", 95, model="modelB"),
        ]
    )


def test_confidently_wrong_rate_per_model_distinct_from_pooled():
    df = _per_model_df()
    result = confidently_wrong_rate(df, conf_threshold=0.9)

    per_model = result["per_model"]
    assert per_model["modelA"]["frac_drift_at_high_confidence"] == pytest.approx(0.5)
    assert per_model["modelA"]["confidently_wrong_rate"] == pytest.approx(0.5)
    assert per_model["modelB"]["frac_drift_at_high_confidence"] == pytest.approx(1.0)
    assert per_model["modelB"]["confidently_wrong_rate"] == pytest.approx(1.0)

    pooled = result["pooled"]
    assert pooled["frac_drift_at_high_confidence"] == pytest.approx(0.75)
    assert pooled["confidently_wrong_rate"] == pytest.approx(0.8)
    # Pooled must differ from BOTH per-model rates -- proves it isn't just
    # echoing one model's numbers.
    assert pooled["confidently_wrong_rate"] not in (
        per_model["modelA"]["confidently_wrong_rate"],
        per_model["modelB"]["confidently_wrong_rate"],
    )


# ---------------------------------------------------------------------------
# 3. NaN confidence rows are excluded from every denominator, and the
#    excluded count is reported.
# ---------------------------------------------------------------------------


def test_nan_confidence_rows_excluded_and_counted():
    df = _pooled_hand_calc_df()
    df_with_nan = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    _row("nan1", "drift", np.nan),
                    _row("nan2", "faithful", np.nan),
                ]
            ),
        ],
        ignore_index=True,
    )
    result = confidently_wrong_rate(df_with_nan, conf_threshold=0.9)

    assert result["n_excluded_nan"] == 2
    # Same hand-calculated rates as the no-NaN case -- the NaN rows must not
    # shift the denominators at all.
    pooled = result["pooled"]
    assert pooled["n_drift"] == 4
    assert pooled["frac_drift_at_high_confidence"] == pytest.approx(3 / 4)
    assert pooled["confidently_wrong_rate"] == pytest.approx(4 / 6)


# ---------------------------------------------------------------------------
# 4. Scale handling -- 0-100 vs 0-1 confidence must produce IDENTICAL rates
#    once conf_threshold is interpreted on the detected scale.
# ---------------------------------------------------------------------------


def test_scale_handling_0_100_and_0_1_give_identical_rates():
    df_pct = _pooled_hand_calc_df()
    df_frac = df_pct.copy()
    df_frac["confidence"] = df_frac["confidence"] / 100.0

    result_pct = confidently_wrong_rate(df_pct, conf_threshold=0.9)
    result_frac = confidently_wrong_rate(df_frac, conf_threshold=0.9)

    assert result_pct["scale_detected"] == "0-100"
    assert result_frac["scale_detected"] == "0-1"
    assert result_pct["pooled"]["confidently_wrong_rate"] == pytest.approx(
        result_frac["pooled"]["confidently_wrong_rate"]
    )
    assert result_pct["pooled"]["frac_drift_at_high_confidence"] == pytest.approx(
        result_frac["pooled"]["frac_drift_at_high_confidence"]
    )


# ---------------------------------------------------------------------------
# 5. Missing-column contract.
# ---------------------------------------------------------------------------


def test_confidently_wrong_rate_missing_column_raises():
    df = _pooled_hand_calc_df().drop(columns=["model"])
    with pytest.raises(ValueError, match="model"):
        confidently_wrong_rate(df)


# ---------------------------------------------------------------------------
# 6. missing_confidence_audit -- tabulated by model x cer_target, total
#    equals the injected NaN count.
#
#   modelA / cer 0.0: 2 NaN + 3 present
#   modelA / cer 0.1: 0 NaN + 2 present
#   modelB / cer 0.0: 1 NaN + 1 present
#   modelB / cer 0.1: 3 NaN + 0 present
#   total NaN = 2 + 0 + 1 + 3 = 6
# ---------------------------------------------------------------------------


def _missing_audit_df():
    rows = []
    rows += [_row(f"a0_{i}", "faithful", np.nan, model="modelA", cer_target=0.0) for i in range(2)]
    rows += [_row(f"a0p_{i}", "faithful", 95, model="modelA", cer_target=0.0) for i in range(3)]
    rows += [_row(f"a1p_{i}", "faithful", 95, model="modelA", cer_target=0.1) for i in range(2)]
    rows += [_row(f"b0_{i}", "drift", np.nan, model="modelB", cer_target=0.0) for i in range(1)]
    rows += [_row(f"b0p_{i}", "drift", 95, model="modelB", cer_target=0.0) for i in range(1)]
    rows += [_row(f"b1_{i}", "drift", np.nan, model="modelB", cer_target=0.1) for i in range(3)]
    return pd.DataFrame(rows)


def test_missing_confidence_audit_total_equals_nan_count():
    df = _missing_audit_df()
    result = missing_confidence_audit(df)

    assert result["total_missing"] == 6
    assert result["total_missing"] == int(df["confidence"].isna().sum())


def test_missing_confidence_audit_table_cell_counts():
    df = _missing_audit_df()
    result = missing_confidence_audit(df)

    by_cell = {(r["model"], r["cer_target"]): r["n_missing"] for r in result["table"]}
    assert by_cell[("modelA", 0.0)] == 2
    assert by_cell[("modelB", 0.0)] == 1
    assert by_cell[("modelB", 0.1)] == 3
    # A present-but-zero-missing cell must still appear in the table (an
    # audit should show where it's NOT a problem too).
    assert by_cell[("modelA", 0.1)] == 0


def test_missing_confidence_audit_missing_column_raises():
    df = _missing_audit_df().drop(columns=["cer_target"])
    with pytest.raises(ValueError, match="cer_target"):
        missing_confidence_audit(df)


# ---------------------------------------------------------------------------
# 7. predicted_drift_grid -- message-clustered logistic fit of drift on
#    realized_cer alone; predicted probability at 5 grid points must be
#    monotonically increasing for a planted positive slope.
# ---------------------------------------------------------------------------


def _simulate_drift_frame(seed=0, n_messages=40, n_rep=6, beta0=-2.0, beta1=8.0):
    rng = np.random.default_rng(seed)
    rows = []
    for mi in range(n_messages):
        message_id = f"msg_{mi:03d}"
        for rep in range(n_rep):
            realized_cer = float(rng.uniform(0.0, 0.5))
            lin = beta0 + beta1 * realized_cer
            p_drift = 1 / (1 + np.exp(-lin))
            is_drift = rng.binomial(1, p_drift)
            label = "drift" if is_drift else "faithful"
            rows.append(
                dict(
                    message_id=message_id,
                    label=label,
                    realized_cer=realized_cer,
                    model="modelA",
                    confidence=90.0,
                    cer_target=round(realized_cer, 1),
                )
            )
    return pd.DataFrame(rows)


def test_predicted_drift_grid_monotonically_increasing_for_positive_slope():
    df = _simulate_drift_frame(seed=0)
    result = predicted_drift_grid(df, seed=0)

    probs = [result["predicted_drift_probability"][f"{cer:.1f}"] for cer in CER_GRID]

    assert len(probs) == 5
    for p in probs:
        assert 0.0 <= p <= 1.0
    assert all(p2 > p1 for p1, p2 in zip(probs, probs[1:]))


def test_predicted_drift_grid_returns_expected_grid_and_metadata():
    df = _simulate_drift_frame(seed=1)
    result = predicted_drift_grid(df, seed=1)

    assert result["cer_grid"] == list(CER_GRID)
    assert set(result["predicted_drift_probability"].keys()) == {f"{c:.1f}" for c in CER_GRID}
    assert "realized_cer" in result["coefficients"]
    assert result["n_obs"] == len(df)


def test_predicted_drift_grid_missing_column_raises():
    df = _simulate_drift_frame(seed=0).drop(columns=["message_id"])
    with pytest.raises(ValueError, match="message_id"):
        predicted_drift_grid(df)
