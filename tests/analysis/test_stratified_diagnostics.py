"""Tests for stratified physician-panel classifier diagnostics (Task B5).

eTable 6 (`idrift.adjudicate.analyze_panel.auto_vs_human`) reports POOLED
class-specific agreement between the automated ensemble and the physician
consensus. Reviewers asked whether that pooled agreement hides heterogeneity
by corpus (AUTH/CRIT/CTRL), CER band, or model -- e.g., maybe the ensemble is
fine on AUTH but poor on CRIT, and pooling masks it. This module recomputes
`class_metrics` independently within each stratum level, plus per-stratum
overall agreement, Cohen's kappa, and n -- and flags (never drops) strata
too thin to trust (n < 30).

Every expected number below is HAND-COMPUTED in the test's comment, not
derived by calling the module under test, so a bug in the module's own
arithmetic cannot also produce the "expected" value. The "high" stratum
below reuses the exact hand-built confusion pattern from
`tests/adjudicate/test_validation_metrics.py` (`_hand_built_case`), whose
per-class sensitivity/specificity/ppv/npv/f1 are already hand-verified
there; this test additionally hand-derives the pooled agreement and kappa
for that same pattern (shown below).
"""
import pandas as pd
import pytest

from idrift.adjudicate.validation_metrics import CLASS_ORDER
from idrift.analysis.stratified_diagnostics import (
    SMALL_STRATUM_N,
    stratified_class_metrics,
)


# ---------------------------------------------------------------------------
# Synthetic fixture: two cer_band strata of known composition.
#
# "low" stratum (cer_target = 0.1 <= 0.2 -> band "low"), n = 36: six
# repeats of the perfect-classifier pattern from test_validation_metrics.py
# (human == auto exactly for every item) -> every class_metrics ratio is
# exactly 1.0, overall agreement 1.0, Cohen kappa 1.0 (multiple classes
# present, so kappa is well-defined, not a 0/0 single-label case).
# n = 36 >= 30 -> NOT flagged small.
#
# "high" stratum (cer_target = 0.3 > 0.2 -> band "high"), n = 10: the
# hand-built imperfect confusion pattern from test_validation_metrics.py:
#   human = 5x faithful, 3x degraded, 2x drift
#   auto  = [faithful,faithful,faithful,faithful,degraded,   (faithful block)
#            degraded,degraded,drift,                          (degraded block)
#            drift,drift]                                      (drift block)
# Confusion (rows=human, cols=auto), fixed order (faithful, degraded, drift):
#            auto:faithful  auto:degraded  auto:drift
# faithful        4              1              0        (support 5)
# degraded        0              2              1        (support 3)
# drift           0              0              2        (support 2)
# total = 10, correct = 4+2+2 = 8 -> overall agreement = 0.8
#
# Cohen kappa: p_o = 0.8
#   row marginals: faithful=5, degraded=3, drift=2
#   col marginals: faithful=4, degraded=3, drift=3
#   p_e = (5*4 + 3*3 + 2*3) / 10^2 = (20 + 9 + 6) / 100 = 35/100 = 0.35
#   kappa = (p_o - p_e) / (1 - p_e) = (0.8 - 0.35) / 0.65 = 0.45/0.65
#         = 0.6923076923076923
# n = 10 < 30 -> flagged small.
#
# Per-class metrics for "high" (from test_validation_metrics.py hand calc):
#   faithful: sensitivity 0.8, specificity 1.0, ppv 1.0, npv 5/6, f1 8/9, support 5
#   degraded: sensitivity 2/3, specificity 6/7, ppv 2/3, npv 6/7, f1 2/3, support 3
#   drift:    sensitivity 1.0, specificity 0.875, ppv 2/3, npv 1.0, f1 0.8, support 2
# ---------------------------------------------------------------------------

def _perfect_block():
    human = ["faithful", "faithful", "degraded", "drift", "drift", "drift"]
    auto = list(human)
    return human, auto


def _hand_built_block():
    human = ["faithful"] * 5 + ["degraded"] * 3 + ["drift"] * 2
    auto = (
        ["faithful", "faithful", "faithful", "faithful", "degraded"]
        + ["degraded", "degraded", "drift"]
        + ["drift", "drift"]
    )
    return human, auto


def _make_cer_band_fixture():
    low_human, low_auto = [], []
    for _ in range(6):
        h, a = _perfect_block()
        low_human += h
        low_auto += a
    high_human, high_auto = _hand_built_block()

    human = low_human + high_human
    auto = low_auto + high_auto
    n = len(human)
    item_ids = [f"R{i:04d}" for i in range(n)]
    cer_targets = [0.1] * len(low_human) + [0.3] * len(high_human)
    # category/model_id are irrelevant to this fixture (by="cer_band"), but
    # the key schema requires them.
    categories = ["AUTH"] * n
    model_ids = ["modelX"] * n

    consensus_df = pd.DataFrame({"item_id": item_ids, "human": human})
    key_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "automated_final_label": auto,
            "cer_target": cer_targets,
            "category": categories,
            "model_id": model_ids,
        }
    )
    return consensus_df, key_df


def test_cer_band_low_stratum_matches_hand_calc_and_not_flagged_small():
    consensus_df, key_df = _make_cer_band_fixture()

    result = stratified_class_metrics(consensus_df, key_df, by="cer_band")

    assert result["by"] == "cer_band"
    assert result["small_stratum_threshold"] == SMALL_STRATUM_N == 30
    low = result["strata"]["low"]

    assert low["n"] == 36
    assert low["small_stratum"] is False
    assert low["overall_percent_agreement"] == pytest.approx(1.0)
    assert low["cohen_kappa_overall"] == pytest.approx(1.0)

    cm = {row["class"]: row for row in low["class_metrics"]}
    assert set(cm) == set(CLASS_ORDER)
    for cls in CLASS_ORDER:
        assert cm[cls]["sensitivity"] == pytest.approx(1.0)
        assert cm[cls]["specificity"] == pytest.approx(1.0)
        assert cm[cls]["ppv"] == pytest.approx(1.0)
        assert cm[cls]["npv"] == pytest.approx(1.0)
        assert cm[cls]["f1"] == pytest.approx(1.0)
    supports = {cls: cm[cls]["support"] for cls in CLASS_ORDER}
    assert supports == {"faithful": 12, "degraded": 6, "drift": 18}


def test_cer_band_high_stratum_matches_hand_calc_and_flagged_small():
    consensus_df, key_df = _make_cer_band_fixture()

    result = stratified_class_metrics(consensus_df, key_df, by="cer_band")
    high = result["strata"]["high"]

    assert high["n"] == 10
    assert high["small_stratum"] is True
    assert high["overall_percent_agreement"] == pytest.approx(0.8)
    assert high["cohen_kappa_overall"] == pytest.approx(0.6923076923076923)

    cm = {row["class"]: row for row in high["class_metrics"]}
    assert cm["faithful"]["sensitivity"] == pytest.approx(0.8)
    assert cm["faithful"]["specificity"] == pytest.approx(1.0)
    assert cm["faithful"]["ppv"] == pytest.approx(1.0)
    assert cm["faithful"]["npv"] == pytest.approx(5 / 6)
    assert cm["faithful"]["f1"] == pytest.approx(8 / 9)
    assert cm["faithful"]["support"] == 5

    assert cm["degraded"]["sensitivity"] == pytest.approx(2 / 3)
    assert cm["degraded"]["specificity"] == pytest.approx(6 / 7)
    assert cm["degraded"]["ppv"] == pytest.approx(2 / 3)
    assert cm["degraded"]["npv"] == pytest.approx(6 / 7)
    assert cm["degraded"]["f1"] == pytest.approx(2 / 3)
    assert cm["degraded"]["support"] == 3

    assert cm["drift"]["sensitivity"] == pytest.approx(1.0)
    assert cm["drift"]["specificity"] == pytest.approx(0.875)
    assert cm["drift"]["ppv"] == pytest.approx(2 / 3)
    assert cm["drift"]["npv"] == pytest.approx(1.0)
    assert cm["drift"]["f1"] == pytest.approx(0.8)
    assert cm["drift"]["support"] == 2


def test_class_metrics_rows_are_in_fixed_class_order():
    consensus_df, key_df = _make_cer_band_fixture()
    result = stratified_class_metrics(consensus_df, key_df, by="cer_band")
    for stratum in result["strata"].values():
        assert [row["class"] for row in stratum["class_metrics"]] == list(CLASS_ORDER)


def test_unresolved_items_are_excluded_not_guessed():
    # An item with human=None (e.g. an unresolved disagreement, per
    # analyze_panel.consensus_labels) must not appear in any stratum, and
    # must not silently inflate/deflate a stratum's n.
    consensus_df, key_df = _make_cer_band_fixture()
    extra_id = "R9999"
    consensus_df = pd.concat(
        [consensus_df, pd.DataFrame({"item_id": [extra_id], "human": [None]})],
        ignore_index=True,
    )
    key_df = pd.concat(
        [
            key_df,
            pd.DataFrame(
                {
                    "item_id": [extra_id],
                    "automated_final_label": ["drift"],
                    "cer_target": [0.1],
                    "category": ["AUTH"],
                    "model_id": ["modelX"],
                }
            ),
        ],
        ignore_index=True,
    )

    result = stratified_class_metrics(consensus_df, key_df, by="cer_band")

    total_n = sum(s["n"] for s in result["strata"].values())
    assert total_n == 46  # 36 + 10, the R9999 unresolved row excluded


def test_corpus_grouping_maps_message_critical_category_to_crit():
    # by="corpus": category "message_critical" -> "CRIT"; "AUTH"/"CTRL" pass
    # through unchanged. Each corpus block mixes two classes (not the point
    # of this test, which is the category->corpus mapping and n) so Cohen's
    # kappa is well-defined rather than the single-label nan/warning case.
    block = ["faithful", "faithful", "faithful", "drift", "drift"]
    human = block * 3
    auto = list(human)  # perfect, so metrics aren't the point of this test
    item_ids = [f"C{i:03d}" for i in range(15)]
    categories = ["AUTH"] * 5 + ["message_critical"] * 5 + ["CTRL"] * 5

    consensus_df = pd.DataFrame({"item_id": item_ids, "human": human})
    key_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "automated_final_label": auto,
            "cer_target": [0.1] * 15,
            "category": categories,
            "model_id": ["modelX"] * 15,
        }
    )

    result = stratified_class_metrics(consensus_df, key_df, by="corpus")

    assert set(result["strata"].keys()) == {"AUTH", "CRIT", "CTRL"}
    assert result["strata"]["AUTH"]["n"] == 5
    assert result["strata"]["CRIT"]["n"] == 5
    assert result["strata"]["CTRL"]["n"] == 5


def test_model_grouping_uses_model_id_directly():
    human = ["faithful", "drift", "degraded", "faithful"]
    auto = list(human)
    item_ids = ["M1", "M2", "M3", "M4"]
    consensus_df = pd.DataFrame({"item_id": item_ids, "human": human})
    key_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "automated_final_label": auto,
            "cer_target": [0.1, 0.1, 0.3, 0.3],
            "category": ["AUTH"] * 4,
            "model_id": ["gemma4:e4b", "gemma4:e4b", "phi4:14b", "phi4:14b"],
        }
    )

    result = stratified_class_metrics(consensus_df, key_df, by="model")

    assert set(result["strata"].keys()) == {"gemma4:e4b", "phi4:14b"}
    assert result["strata"]["gemma4:e4b"]["n"] == 2
    assert result["strata"]["phi4:14b"]["n"] == 2


def test_small_stratum_threshold_is_strictly_less_than_30():
    # n=29 -> flagged small; n=30 -> not flagged. Boundary check, using
    # model_id as the stratum column and a 2-class alternating pattern
    # (avoids the single-label cohen_kappa undefined/NaN edge case).
    n_small, n_ok = 29, 30

    def _alt(n):
        return [("faithful" if i % 2 == 0 else "degraded") for i in range(n)]

    human = _alt(n_small) + _alt(n_ok)
    auto = list(human)  # perfect match within each stratum
    item_ids = [f"B{i:03d}" for i in range(n_small + n_ok)]
    model_ids = ["small_model"] * n_small + ["ok_model"] * n_ok

    consensus_df = pd.DataFrame({"item_id": item_ids, "human": human})
    key_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "automated_final_label": auto,
            "cer_target": [0.1] * len(item_ids),
            "category": ["AUTH"] * len(item_ids),
            "model_id": model_ids,
        }
    )

    result = stratified_class_metrics(consensus_df, key_df, by="model")

    assert result["strata"]["small_model"]["n"] == 29
    assert result["strata"]["small_model"]["small_stratum"] is True
    assert result["strata"]["ok_model"]["n"] == 30
    assert result["strata"]["ok_model"]["small_stratum"] is False


def test_cer_band_boundary_at_threshold_is_low():
    # cer_target exactly at the 0.2 threshold banded as "low" (spec: "low:
    # cer_target<=0.2 vs high: >0.2").
    human = ["faithful", "drift"]
    auto = list(human)
    item_ids = ["T1", "T2"]
    consensus_df = pd.DataFrame({"item_id": item_ids, "human": human})
    key_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "automated_final_label": auto,
            "cer_target": [0.2, 0.2],
            "category": ["AUTH", "AUTH"],
            "model_id": ["modelX", "modelX"],
        }
    )

    result = stratified_class_metrics(consensus_df, key_df, by="cer_band")

    assert set(result["strata"].keys()) == {"low"}


def test_invalid_by_raises():
    consensus_df, key_df = _make_cer_band_fixture()
    with pytest.raises(ValueError, match="by"):
        stratified_class_metrics(consensus_df, key_df, by="not_a_real_stratum")
