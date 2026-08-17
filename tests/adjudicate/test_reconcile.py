import pandas as pd
from idrift.adjudicate.reconcile import reconcile, cohen_kappa, assign_final


def test_reconcile_priority():
    assert reconcile("faithful", "drift", "degraded") == ("degraded", "human")
    assert reconcile("faithful", "drift", None) == ("drift", "judge")
    assert reconcile("faithful", None, None) == ("faithful", "nli")


def test_kappa_perfect():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_assign_final_flags_critical():
    df = pd.DataFrame(
        [{"category": "message_critical", "nli_label": "drift", "judge_label": "drift", "human_label": "drift"}]
    )
    out = assign_final(df)
    assert out.loc[0, "final_label"] == "drift" and bool(out.loc[0, "critical"]) is True


# --- Regression: NaN (from a left-merge) must NOT be treated as a present --
# rating. Task 13 review: `if human is not None` / `if judge is not None`
# wrongly picked up pandas NaN (since `NaN is not None` is True), reporting
# label_source="human"/"judge" and a NaN final_label for rows nobody rated --
# silently zeroing out the load-bearing `critical` flag for real drift rows.

def test_assign_final_is_nan_safe_after_left_merge():
    attempts = pd.DataFrame(
        [
            {"row_id": 1, "category": "other", "nli_label": "faithful"},
            {"row_id": 2, "category": "message_critical", "nli_label": "faithful"},
            {"row_id": 3, "category": "message_critical", "nli_label": "faithful"},
        ]
    )
    # Only row 3 has real ratings; rows 1 and 2 get NaN human/judge labels
    # from the left-merge, exactly as in the real Step-5 path.
    ratings = pd.DataFrame(
        [{"row_id": 3, "judge_label": "drift", "human_label": "drift"}]
    )
    df = attempts.merge(ratings, on="row_id", how="left")

    out = assign_final(df)

    # (a) a row with only nli (NaN human/judge) gets final_label=nli value
    #     and label_source=="nli", NOT "human".
    row1 = out.loc[out["row_id"] == 1].iloc[0]
    assert row1["final_label"] == "faithful"
    assert row1["label_source"] == "nli"

    # (b) row 2 is message_critical with judge NaN too (never rated) -- must
    #     also fall through to nli/faithful, not be marked critical.
    row2 = out.loc[out["row_id"] == 2].iloc[0]
    assert row2["label_source"] == "nli"
    assert bool(row2["critical"]) is False

    # (b') row 3: message_critical, judge label "drift", human label "drift"
    #      present -- human wins (highest priority) and critical fires.
    row3 = out.loc[out["row_id"] == 3].iloc[0]
    assert row3["final_label"] == "drift"
    assert row3["label_source"] == "human"
    assert bool(row3["critical"]) is True

    # (c) no row with NaN human is ever labeled source=="human".
    nan_human_rows = out[out["human_label"].isna()]
    assert not (nan_human_rows["label_source"] == "human").any()


def test_assign_final_judge_present_human_nan_uses_judge():
    # A message_critical row whose judge label is "drift" and human is NaN
    # (never reached a human rater) must land on judge, not fall through to
    # nli or get wrongly stamped "human".
    df = pd.DataFrame(
        [
            {
                "category": "message_critical",
                "nli_label": "faithful",
                "judge_label": "drift",
                "human_label": float("nan"),
            }
        ]
    )
    out = assign_final(df)
    assert out.loc[0, "final_label"] == "drift"
    assert out.loc[0, "label_source"] == "judge"
    assert bool(out.loc[0, "critical"]) is True
