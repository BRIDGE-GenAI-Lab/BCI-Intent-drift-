"""Tests for idrift.analysis.interface_substudy (revision Task D5).

TDD: written before `idrift.analysis.interface_substudy` existed. Synthetic
frames throughout -- `drift_condition_model` fits a real statsmodels logit,
kept small enough to converge but with planted structure so its sign/OR
direction is asserted; other fits are checked on structure/keys rather than
exact estimates.

The one invariant every test in this module cares about: a `declined` row
must never enter the {faithful, degraded, drift} denominator and must never
enter a benefit-harm transition.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from idrift.analysis.interface_substudy import (
    assemble_forced,
    benefit_harm_by_condition,
    candidate_recovery,
    drift_condition_model,
    per_condition_rates,
    run,
)


# ---------------------------------------------------------------------------
# per_condition_rates
# ---------------------------------------------------------------------------


def _rates_frame():
    """4 conditions, hand-counted outcomes, 2 corpora, so declined_rate and
    the non-declined faithful/degraded/drift rates can be checked exactly."""
    rows = []
    # forced: 4 non-declined rows (2 faithful, 1 degraded, 1 drift), no declines.
    for corpus, outcome in [("AUTH", "faithful"), ("AUTH", "faithful"), ("CRIT", "degraded"), ("CRIT", "drift")]:
        rows.append({"condition": "forced", "corpus": corpus, "outcome": outcome})
    # abstain_enabled: 2 declined + 2 non-declined (1 faithful, 1 drift).
    for corpus, outcome in [("AUTH", "declined"), ("AUTH", "declined"), ("CRIT", "faithful"), ("CRIT", "drift")]:
        rows.append({"condition": "abstain_enabled", "corpus": corpus, "outcome": outcome})
    return pd.DataFrame(rows)


def test_per_condition_rates_excludes_declined_from_drift_denominator():
    res = per_condition_rates(_rates_frame())
    abstain = res["by_condition"]["abstain_enabled"]["overall"]

    assert abstain["n_total"] == 4
    assert abstain["n_declined"] == 2
    assert abstain["declined_rate"] == pytest.approx(0.5)
    # non-declined denominator is 2, NOT 4 -- declined never folds into it.
    assert abstain["n_non_declined"] == 2
    assert abstain["n_faithful"] == 1
    assert abstain["n_drift"] == 1
    assert abstain["rate_faithful"] == pytest.approx(0.5)
    assert abstain["rate_drift"] == pytest.approx(0.5)


def test_per_condition_rates_sum_to_one_and_reports_by_corpus():
    res = per_condition_rates(_rates_frame())
    forced = res["by_condition"]["forced"]["overall"]

    assert forced["n_declined"] == 0
    assert forced["declined_rate"] == pytest.approx(0.0)
    total_rate = forced["rate_faithful"] + forced["rate_degraded"] + forced["rate_drift"]
    assert total_rate == pytest.approx(1.0)

    by_corpus = res["by_condition"]["forced"]["by_corpus"]
    assert set(by_corpus) == {"AUTH", "CRIT"}
    assert by_corpus["AUTH"]["n_faithful"] == 2
    assert by_corpus["CRIT"]["n_degraded"] == 1
    assert by_corpus["CRIT"]["n_drift"] == 1

    assert res["conditions_present"] == ["forced", "abstain_enabled"]


# ---------------------------------------------------------------------------
# assemble_forced
# ---------------------------------------------------------------------------


def _v3_auth_labeled():
    return pd.DataFrame(
        [
            {"message_id": "auth_1", "corpus": "AUTH", "replicate_idx": 0, "label": "faithful"},
            {"message_id": "auth_1", "corpus": "AUTH", "replicate_idx": 9, "label": "drift"},
            # replicate_idx >= MAX_REPLICATE (10): must be excluded.
            {"message_id": "auth_1", "corpus": "AUTH", "replicate_idx": 10, "label": "faithful"},
            # not in subset_msgids: must be excluded.
            {"message_id": "auth_OUTSIDE", "corpus": "AUTH", "replicate_idx": 0, "label": "faithful"},
        ]
    )


def _v2_full_labeled():
    return pd.DataFrame(
        [
            {"message_id": "probe_1", "corpus": "CRIT", "replicate_idx": 0, "label": "degraded"},
            {"message_id": "costello_2", "corpus": "CTRL", "replicate_idx": 3, "label": "faithful"},
            # CRIT/CTRL row but replicate_idx too high: excluded.
            {"message_id": "probe_1", "corpus": "CRIT", "replicate_idx": 15, "label": "faithful"},
            # this v2 frame's OWN (main-run) AUTH row, same message_id as the
            # substudy subset: must be excluded even though the id matches,
            # because forced/AUTH comes from v3, never v2.
            {"message_id": "auth_1", "corpus": "AUTH", "replicate_idx": 0, "label": "drift"},
        ]
    )


def test_assemble_forced_filters_msgids_replicate_and_tags_condition():
    subset_msgids = ["auth_1", "probe_1", "costello_2"]
    out = assemble_forced(subset_msgids, _v3_auth_labeled(), _v2_full_labeled())

    assert len(out) == 4  # 2 AUTH kept (idx 0, 9) + 1 CRIT (idx 0) + 1 CTRL (idx 3)
    assert set(out["condition"]) == {"forced"}
    assert (out["abstained"] == False).all()  # noqa: E712
    assert out["any_candidate_faithful"].isna().all()

    # label -> outcome mapping preserved, no re-derivation.
    by_key = {(r.message_id, r.corpus, r.replicate_idx): r.outcome for r in out.itertuples(index=False)}
    assert by_key[("auth_1", "AUTH", 0)] == "faithful"
    assert by_key[("auth_1", "AUTH", 9)] == "drift"
    assert ("auth_1", "AUTH", 10) not in by_key  # replicate_idx >= 10 excluded
    assert ("auth_1", "AUTH", 0) in by_key and by_key[("auth_1", "AUTH", 0)] != "drift"  # v3's row won, not v2's
    assert by_key[("probe_1", "CRIT", 0)] == "degraded"
    assert ("probe_1", "CRIT", 15) not in by_key
    assert by_key[("costello_2", "CTRL", 3)] == "faithful"


def test_assemble_forced_pair_match_excludes_the_auth_twin_of_a_crit_id():
    """The probe set is drawn FROM the message bank, so a CRIT item and an
    AUTH item share a message_id. Matching id and corpus independently
    admits the AUTH twin of every CRIT id and inflates the forced arm; the
    pair match must keep it out."""
    v3 = pd.DataFrame(
        [
            {"message_id": "costello_7", "corpus": "AUTH", "replicate_idx": 0, "label": "drift"},
            {"message_id": "auth_1", "corpus": "AUTH", "replicate_idx": 0, "label": "faithful"},
        ]
    )
    v2 = pd.DataFrame(
        [{"message_id": "costello_7", "corpus": "CRIT", "replicate_idx": 0, "label": "degraded"}]
    )
    # the subset carries costello_7 ONLY as CRIT; its AUTH twin was never run.
    subset = pd.DataFrame(
        [
            {"message_id": "auth_1", "corpus": "AUTH"},
            {"message_id": "costello_7", "corpus": "CRIT"},
        ]
    )

    out = assemble_forced(subset, v3, v2)

    keys = set(zip(out["message_id"], out["corpus"]))
    assert ("costello_7", "AUTH") not in keys
    assert keys == {("auth_1", "AUTH"), ("costello_7", "CRIT")}

    # tuples work the same as a frame
    out_pairs = assemble_forced([("auth_1", "AUTH"), ("costello_7", "CRIT")], v3, v2)
    assert set(zip(out_pairs["message_id"], out_pairs["corpus"])) == keys

    # bare ids are the legacy path and DO pull the twin -- that is the bug
    # this pair matching exists to prevent.
    legacy = assemble_forced(["auth_1", "costello_7"], v3, v2)
    assert ("costello_7", "AUTH") in set(zip(legacy["message_id"], legacy["corpus"]))


def test_assemble_forced_excludes_ids_outside_subset():
    out = assemble_forced(["auth_1"], _v3_auth_labeled(), _v2_full_labeled())
    assert "auth_OUTSIDE" not in set(out["message_id"])
    assert set(out["corpus"]) == {"AUTH"}


# ---------------------------------------------------------------------------
# drift_condition_model
# ---------------------------------------------------------------------------


def _drift_model_frame(n_per_condition=300, seed=0):
    """forced vs a strictly-lower-drift condition, plus varying cer_target
    so the condition x cer_target interaction has something to fit."""
    rng = np.random.default_rng(seed)
    drift_p = {"forced": 0.65, "abstain_enabled": 0.15, "minimal_edit": 0.45}
    rows = []
    counter = 0
    for cond, p in drift_p.items():
        for _ in range(n_per_condition):
            counter += 1
            message_id = f"m_{counter % 60}"
            cer = float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.4]))
            outcome = "drift" if rng.uniform() < p else rng.choice(["faithful", "degraded"])
            rows.append({"message_id": message_id, "condition": cond, "cer_target": cer, "outcome": outcome})
    return pd.DataFrame(rows)


def test_drift_condition_model_returns_or_ci_and_convergence_per_condition():
    df = _drift_model_frame()
    res = drift_condition_model(df)

    assert res["reference_condition"] == "forced"
    assert res["formula_used"] in {"interaction", "main_effects_only"}
    assert isinstance(res["interaction_converged"], bool)
    assert isinstance(res["converged"], bool)
    assert res["n_declined_excluded"] == 0
    assert res["n_obs"] == len(df)

    ors = res["condition_odds_ratios"]
    assert set(ors) == {"abstain_enabled", "minimal_edit"}
    for cond, rec in ors.items():
        assert set(rec) == {"odds_ratio", "ci_lower", "ci_upper", "p"}
        assert rec["ci_lower"] <= rec["odds_ratio"] <= rec["ci_upper"]


def test_drift_condition_model_lower_drift_condition_has_or_below_one():
    df = _drift_model_frame()
    res = drift_condition_model(df)
    assert res["condition_odds_ratios"]["abstain_enabled"]["odds_ratio"] < 1.0


def test_drift_condition_model_reports_main_effects_reference_alongside_interaction():
    """The CER-averaged fit must be present even when the interaction model
    converged and supplied the headline ORs. Without it the only condition
    effect on offer is the interaction term, which is the OR at cer_target 0
    -- a different quantity that can point the other way."""
    df = _drift_model_frame()
    res = drift_condition_model(df)

    ref = res["main_effects_reference"]
    assert ref["converged"] is True
    assert set(ref["condition_odds_ratios"]) == {"abstain_enabled", "minimal_edit"}
    for rec in ref["condition_odds_ratios"].values():
        assert set(rec) == {"odds_ratio", "ci_lower", "ci_upper", "p"}
        assert rec["ci_lower"] <= rec["odds_ratio"] <= rec["ci_upper"]
    # planted structure: abstain_enabled drifts less than forced at every CER
    assert ref["condition_odds_ratios"]["abstain_enabled"]["odds_ratio"] < 1.0


def test_drift_condition_model_main_effects_reference_ignores_declined_rows():
    """The averaged fit shares the interaction fit's declined-row exclusion;
    a batch of declined rows must not move its odds ratios."""
    df = _drift_model_frame(n_per_condition=150)
    declined = pd.DataFrame(
        [
            {"message_id": f"m_declined_{i}", "condition": "abstain_enabled", "cer_target": 0.2, "outcome": "declined"}
            for i in range(200)
        ]
    )

    plain = drift_condition_model(df)["main_effects_reference"]
    with_declined = drift_condition_model(pd.concat([df, declined], ignore_index=True))["main_effects_reference"]

    for cond, rec in plain["condition_odds_ratios"].items():
        assert with_declined["condition_odds_ratios"][cond]["odds_ratio"] == pytest.approx(rec["odds_ratio"])


def test_drift_condition_model_drops_declined_rows():
    df = _drift_model_frame(n_per_condition=150)
    # graft on a batch of declined rows that would (if wrongly counted as
    # drift==0) dilute the abstain_enabled effect; they must be dropped
    # entirely, not counted as non-drift.
    declined = pd.DataFrame(
        [
            {"message_id": f"m_declined_{i}", "condition": "abstain_enabled", "cer_target": 0.2, "outcome": "declined"}
            for i in range(200)
        ]
    )
    df_with_declined = pd.concat([df, declined], ignore_index=True)

    res_plain = drift_condition_model(df)
    res_with_declined = drift_condition_model(df_with_declined)

    assert res_with_declined["n_declined_excluded"] == 200
    assert res_with_declined["n_obs"] == res_plain["n_obs"]  # declined rows never entered the fit


# ---------------------------------------------------------------------------
# candidate_recovery
# ---------------------------------------------------------------------------


def _candidate_frame():
    return pd.DataFrame(
        [
            {"condition": "candidate_list", "cer_target": 0.1, "outcome": "faithful", "any_candidate_faithful": True},
            {"condition": "candidate_list", "cer_target": 0.1, "outcome": "drift", "any_candidate_faithful": True},
            {"condition": "candidate_list", "cer_target": 0.1, "outcome": "drift", "any_candidate_faithful": False},
            {"condition": "candidate_list", "cer_target": 0.2, "outcome": "degraded", "any_candidate_faithful": False},
            # a non-candidate_list row: must be excluded entirely.
            {"condition": "forced", "cer_target": 0.1, "outcome": "faithful", "any_candidate_faithful": pd.NA},
        ]
    )


def test_candidate_recovery_any_rate_ge_primary_and_lift_matches_difference():
    res = candidate_recovery(_candidate_frame())

    assert res["n_total"] == 4  # forced row excluded
    assert res["primary_faithful_rate"] == pytest.approx(1 / 4)
    assert res["any_candidate_faithful_rate"] == pytest.approx(2 / 4)
    assert res["any_candidate_faithful_rate"] >= res["primary_faithful_rate"]
    assert res["recovery_lift"] == pytest.approx(
        res["any_candidate_faithful_rate"] - res["primary_faithful_rate"]
    )


def test_candidate_recovery_by_cer_target():
    res = candidate_recovery(_candidate_frame())
    by_cer = res["by_cer_target"]
    assert set(by_cer) == {0.1, 0.2}
    assert by_cer[0.1]["n"] == 3
    assert by_cer[0.1]["primary_faithful_rate"] == pytest.approx(1 / 3)
    assert by_cer[0.1]["any_candidate_faithful_rate"] == pytest.approx(2 / 3)
    assert by_cer[0.2]["n"] == 1
    assert by_cer[0.2]["primary_faithful_rate"] == pytest.approx(0.0)
    assert by_cer[0.2]["any_candidate_faithful_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# benefit_harm_by_condition
# ---------------------------------------------------------------------------


def _benefit_harm_frames():
    attempts = pd.DataFrame(
        [
            # forced: A faithful->faithful (no_benefit), B degraded->drift (silent_failure)
            {"message_id": "A", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "condition": "forced", "outcome": "faithful"},
            {"message_id": "B", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "condition": "forced", "outcome": "drift"},
            # abstain_enabled: A declined (must be EXCLUDED before the cross-tab),
            # B degraded->faithful (rescue)
            {"message_id": "A", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "condition": "abstain_enabled", "outcome": "declined"},
            {"message_id": "B", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "condition": "abstain_enabled", "outcome": "faithful"},
        ]
    )
    raw = pd.DataFrame(
        [
            {"message_id": "A", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "raw_label": "faithful"},
            {"message_id": "B", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "raw_label": "degraded"},
        ]
    )
    return attempts, raw


def test_benefit_harm_by_condition_excludes_declined_and_ranks_conditions():
    attempts, raw = _benefit_harm_frames()
    res = benefit_harm_by_condition(attempts, raw)

    assert res["skipped"] is False
    assert res["n_declined_excluded"] == 1

    pooled_by_cond = {r["condition"]: r for r in res["by_condition_pooled"]["by_group"]}
    # forced: n_total==2 (A declined never entered abstain_enabled's forced
    # rows anyway; this checks the declined row from abstain_enabled did not
    # leak into forced's count either).
    assert pooled_by_cond["forced"]["n_total"] == 2
    # abstain_enabled: only B survived (A was declined and excluded).
    assert pooled_by_cond["abstain_enabled"]["n_total"] == 1
    assert pooled_by_cond["abstain_enabled"]["n_rescue"] == 1

    # forced has a silent_failure with no rescue -> net metric finite (0.0);
    # abstain_enabled has a rescue with nothing in the denominator -> inf.
    assert pooled_by_cond["forced"]["faithful_gained_per_drift_introduced"] == pytest.approx(0.0)
    assert math.isinf(pooled_by_cond["abstain_enabled"]["faithful_gained_per_drift_introduced"])

    ranked = res["ranked_by_net_metric"]
    assert [r["condition"] for r in ranked][0] == "abstain_enabled"  # inf ranks first


def test_benefit_harm_by_condition_none_raw_decode_is_skipped_not_crashed():
    attempts, _raw = _benefit_harm_frames()
    res = benefit_harm_by_condition(attempts, None)
    assert res["skipped"] is True
    assert "reason" in res


# ---------------------------------------------------------------------------
# run -- end to end, writes the digest
# ---------------------------------------------------------------------------


def test_run_writes_digest_with_all_four_sections(tmp_path):
    substudy = pd.DataFrame(
        [
            {
                "message_id": "auth_1", "corpus": "AUTH", "condition": "minimal_edit", "prompt_id": "P1",
                "cer_target": 0.2, "replicate_idx": 0, "realized_cer": 0.2, "intended_text": "hi",
                "noisy_text": "h1", "output_message": "hi", "abstained": False, "candidates": None,
                "outcome": "faithful", "any_candidate_faithful": pd.NA,
            },
            {
                "message_id": "auth_1", "corpus": "AUTH", "condition": "abstain_enabled", "prompt_id": "P3",
                "cer_target": 0.2, "replicate_idx": 1, "realized_cer": 0.2, "intended_text": "hi",
                "noisy_text": "h1", "output_message": None, "abstained": True, "candidates": None,
                "outcome": "declined", "any_candidate_faithful": pd.NA,
            },
            {
                "message_id": "probe_1", "corpus": "CRIT", "condition": "candidate_list", "prompt_id": "P4",
                "cer_target": 0.3, "replicate_idx": 0, "realized_cer": 0.3, "intended_text": "go now",
                "noisy_text": "g0 now", "output_message": "go now", "abstained": False, "candidates": ["go now"],
                "outcome": "faithful", "any_candidate_faithful": True,
            },
        ]
    )
    v3 = pd.DataFrame(
        [{"message_id": "auth_1", "corpus": "AUTH", "replicate_idx": 0, "label": "faithful"}]
    )
    v2 = pd.DataFrame(
        [{"message_id": "probe_1", "corpus": "CRIT", "replicate_idx": 0, "label": "drift"}]
    )
    subset = pd.DataFrame(
        [{"message_id": "auth_1", "corpus": "AUTH"}, {"message_id": "probe_1", "corpus": "CRIT"}]
    )

    out_path = tmp_path / "interface_substudy_digest.json"
    result = run(substudy, v3, v2, subset, raw_decode_path=None, out_path=str(out_path))

    assert out_path.exists()
    written = json.loads(out_path.read_text(), parse_constant=lambda c: c)  # tolerate NaN/Infinity tokens
    digest_keys = {"per_condition_rates", "drift_condition_model", "candidate_recovery", "benefit_harm_by_condition"}
    assert digest_keys <= set(result)
    assert digest_keys <= set(written)

    assert result["benefit_harm_by_condition"]["skipped"] is True
    assert result["n_rows_substudy"] == 3
    assert result["n_rows_forced_assembled"] == 2  # 1 AUTH (v3) + 1 CRIT (v2)
    assert result["n_rows_total"] == 5

    # the written file's non-NaN scalars agree with the returned dict.
    assert written["n_rows_total"] == result["n_rows_total"]
    assert written["n_rows_substudy"] == result["n_rows_substudy"]
    assert written["n_rows_forced_assembled"] == result["n_rows_forced_assembled"]
    assert written["benefit_harm_by_condition"]["skipped"] is True
