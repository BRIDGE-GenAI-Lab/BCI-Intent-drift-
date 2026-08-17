import pandas as pd
import pytest

from idrift.adjudicate.build_rating_sheet import (
    BLINDED_COLUMNS,
    RATING_LABELS,
    build_rating_sheet,
    select_items,
    size_stratified_draw_for_power_target,
)
from idrift.adjudicate.reconcile import validation_frame


CER_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
MODELS = [("modelA", "local"), ("modelB", "local")]
LARGE_MODEL_CLASSES = ["local", "cloud", "hybrid", "other"]


def _make_synthetic_attempts() -> pd.DataFrame:
    """~40 rows across 2 model_ids x 5 CER levels, with ~6 message_critical rows.

    2 models x 5 CER levels x 4 replicate messages = 40 rows. One replicate
    per CER level for modelA is flagged message_critical (5 rows) plus one
    extra modelB row at CER 0.0 (1 row) = 6 message_critical rows total,
    scattered across CER levels and both models (not all in one stratum).
    """
    rows = []
    msg_i = 0
    for model_id, model_class in MODELS:
        for cer in CER_LEVELS:
            for rep in range(4):
                msg_i += 1
                is_critical = (model_id == "modelA" and rep == 0) or (
                    model_id == "modelB" and rep == 1 and cer == 0.0
                )
                category = "message_critical" if is_critical else "other"
                rows.append(
                    {
                        "message_id": f"msg_{msg_i:04d}",
                        "corpus": "costello",
                        "category": category,
                        "intended_text": f"intended text {msg_i}",
                        "cer_target": cer,
                        "seed": 0,
                        "source_subject": "F_01",
                        "noisy_text": f"noisy {msg_i}",
                        "actual_cer": cer,
                        "model_id": model_id,
                        "model_class": model_class,
                        "depth": "postedit",
                        "temperature": 0.0,
                        "prompt_id": "postedit_v1",
                        "output_text": f"output text {msg_i}",
                        "verbalized_conf": 80.0,
                        "final_label": "drift" if is_critical else "faithful",
                    }
                )
    df = pd.DataFrame(rows)
    assert len(df) == 40
    assert (df["category"] == "message_critical").sum() == 6
    return df


def _make_large_synthetic_attempts(reps_per_stratum: int = 30) -> pd.DataFrame:
    """~600 rows across 4 model_classes x 5 CER levels (20 strata, 30
    replicate rows per stratum before removing critical items), with 5
    message-critical rows (one per CER level, all in the "local" class).

    Sized so per-stratum quota granularity is small relative to the
    power-target totals exercised in the disjointness/tolerance tests below
    (many strata, ample non-critical rows per stratum), unlike the small
    ~40-row fixture above (5 strata, shared by the two-model_class fixture)
    where quota flooring dominates the total.
    """
    rows = []
    msg_i = 0
    for model_class in LARGE_MODEL_CLASSES:
        for cer in CER_LEVELS:
            for rep in range(reps_per_stratum):
                msg_i += 1
                is_critical = model_class == "local" and rep == 0
                category = "message_critical" if is_critical else "other"
                rows.append(
                    {
                        "message_id": f"lmsg_{msg_i:05d}",
                        "corpus": "costello",
                        "category": category,
                        "intended_text": f"intended text {msg_i}",
                        "cer_target": cer,
                        "seed": 0,
                        "source_subject": "F_01",
                        "noisy_text": f"noisy {msg_i}",
                        "actual_cer": cer,
                        "model_id": f"model_{model_class}",
                        "model_class": model_class,
                        "depth": "postedit",
                        "temperature": 0.0,
                        "prompt_id": "postedit_v1",
                        "output_text": f"output text {msg_i}",
                        "verbalized_conf": 80.0,
                        "final_label": "drift" if is_critical else "faithful",
                    }
                )
    df = pd.DataFrame(rows)
    n_strata = len(LARGE_MODEL_CLASSES) * len(CER_LEVELS)
    assert len(df) == n_strata * reps_per_stratum
    assert (df["category"] == "message_critical").sum() == len(CER_LEVELS)
    return df


@pytest.fixture
def attempts_parquet(tmp_path):
    df = _make_synthetic_attempts()
    path = tmp_path / "attempts_labeled.parquet"
    df.to_parquet(path, index=False)
    return path, df


def test_every_message_critical_row_appears_in_the_sheet(attempts_parquet, tmp_path):
    path, df = attempts_parquet
    outdir = tmp_path / "out"
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=outdir)

    key = pd.read_csv(outdir / "key.csv")
    critical_orig_ids = set(
        df.loc[df["category"] == "message_critical"].apply(
            lambda r: f"{r.message_id}::{r.model_id}::{r.cer_target}", axis=1
        )
    )
    assert critical_orig_ids.issubset(set(key["orig_attempt_id"]))
    assert (key["category"] == "message_critical").sum() == 6


def test_blinded_sheet_has_no_model_or_cer_or_label_columns(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    outdir = tmp_path / "out"
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=outdir)

    sheet_a = pd.read_csv(outdir / "rating_sheet_raterA.csv")
    assert list(sheet_a.columns) == BLINDED_COLUMNS
    leaked = {"model_id", "model_class", "cer_target", "cer", "final_label", "automated_final_label", "category"}
    assert leaked.isdisjoint(sheet_a.columns)


def test_raterA_and_raterB_sheets_identical_order_and_blank_ratings(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    outdir = tmp_path / "out"
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=outdir)

    sheet_a = pd.read_csv(outdir / "rating_sheet_raterA.csv", keep_default_na=False)
    sheet_b = pd.read_csv(outdir / "rating_sheet_raterB.csv", keep_default_na=False)

    assert sheet_a["item_id"].tolist() == sheet_b["item_id"].tolist()
    assert (sheet_a["rating"] == "").all()
    assert (sheet_a["notes"] == "").all()
    assert (sheet_b["rating"] == "").all()
    assert (sheet_b["notes"] == "").all()


def test_key_round_trips_item_id_to_model_and_cer(attempts_parquet, tmp_path):
    path, df = attempts_parquet
    outdir = tmp_path / "out"
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=outdir)

    key = pd.read_csv(outdir / "key.csv")
    lookup = df.set_index(
        df.apply(lambda r: f"{r.message_id}::{r.model_id}::{r.cer_target}", axis=1)
    )

    assert len(key) > 0
    for _, row in key.iterrows():
        orig = lookup.loc[row["orig_attempt_id"]]
        assert orig["model_id"] == row["model_id"]
        assert float(orig["cer_target"]) == pytest.approx(float(row["cer_target"]))


def test_determinism_same_seed_same_item_id_assignment(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=out1)
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=out2)

    key1 = pd.read_csv(out1 / "key.csv").sort_values("orig_attempt_id").reset_index(drop=True)
    key2 = pd.read_csv(out2 / "key.csv").sort_values("orig_attempt_id").reset_index(drop=True)

    assert key1["item_id"].tolist() == key2["item_id"].tolist()
    assert key1["orig_attempt_id"].tolist() == key2["orig_attempt_id"].tolist()


def test_rating_form_html_written_and_contains_the_four_labels(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    outdir = tmp_path / "out"
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=outdir)

    html_path = outdir / "rating_form.html"
    assert html_path.exists()
    html = html_path.read_text()
    for label in RATING_LABELS:
        assert label in html


# --- Task 4.1: --all-critical, --power-target, and the structural no- -----
# exclusion validation join (guards against the reviewer-identified 395- vs -
# -378 discrepancy: the human panel rated 395 items, but a prior ad hoc
# comparison ran automated-vs-human agreement on only 378 of them -- exactly
# the 17 items that needed third-adjudicator resolution after an initial
# rater disagreement were silently dropped, inflating agreement).

def test_all_critical_flag_includes_every_critical_item_even_with_a_lower_cap(attempts_parquet, tmp_path):
    path, df = attempts_parquet
    outdir = tmp_path / "out"
    # n_critical=2 would normally CAP the message-critical draw; all_critical
    # must override that cap entirely, not just raise it.
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=outdir, n_critical=2, all_critical=True)

    key = pd.read_csv(outdir / "key.csv")
    critical_ids_in_input = set(
        df.loc[df["category"] == "message_critical"].apply(
            lambda r: f"{r.message_id}::{r.model_id}::{r.cer_target}", axis=1
        )
    )
    critical_ids_in_sheet = set(key.loc[key["category"] == "message_critical", "orig_attempt_id"])

    assert critical_ids_in_sheet == critical_ids_in_input
    assert len(critical_ids_in_sheet) == 6


def test_blinded_sheet_still_leaks_nothing_with_all_critical_and_power_target(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    outdir = tmp_path / "out"
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=outdir, all_critical=True, power_target=20)

    sheet_a = pd.read_csv(outdir / "rating_sheet_raterA.csv")
    assert list(sheet_a.columns) == BLINDED_COLUMNS
    leaked = {"model_id", "model_class", "cer_target", "cer", "final_label", "automated_final_label", "category"}
    assert leaked.isdisjoint(sheet_a.columns)


def test_size_stratified_draw_for_power_target_arithmetic():
    # n_stratified = max(0, power_target - n_critical_selected): the
    # non-critical stratified draw is sized so the GRAND TOTAL (critical +
    # stratified) reaches the prespecified power target, floored at 0 when
    # the critical draw alone already meets or exceeds it.
    assert size_stratified_draw_for_power_target(n_critical_selected=6, power_target=20) == 14
    assert size_stratified_draw_for_power_target(n_critical_selected=25, power_target=20) == 0
    assert size_stratified_draw_for_power_target(n_critical_selected=0, power_target=20) == 20
    assert size_stratified_draw_for_power_target(n_critical_selected=20, power_target=20) == 0


def test_power_target_overrides_n_stratified_and_floors_at_zero_when_critical_meets_target(
    attempts_parquet, tmp_path
):
    path, _ = attempts_parquet
    outdir = tmp_path / "out"
    # power_target == the fixture's 6 critical items -> the non-critical
    # stratified draw must be sized to 0, regardless of the (huge) explicit
    # n_stratified value: power_target must actually override it, not just
    # add to it.
    build_rating_sheet(
        path, n_stratified=1000, seed=0, outdir=outdir,
        all_critical=True, power_target=6,
    )
    key = pd.read_csv(outdir / "key.csv")
    assert len(key) == 6
    assert (key["category"] == "message_critical").all()


def test_power_target_larger_value_yields_a_larger_total_sample(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    out_small = tmp_path / "small"
    out_large = tmp_path / "large"
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=out_small, all_critical=True, power_target=6)
    build_rating_sheet(path, n_stratified=10, seed=0, outdir=out_large, all_critical=True, power_target=30)

    key_small = pd.read_csv(out_small / "key.csv")
    key_large = pd.read_csv(out_large / "key.csv")

    assert len(key_small) == 6
    assert len(key_large) > len(key_small)

    # Tightened (rev Task 4.1 review fix): on a fixture with enough strata
    # and rows per stratum, a larger power_target must also yield a
    # proportionally larger total that lands close to the target -- not just
    # "larger than the other run". Reuses the large, many-strata fixture so
    # per-stratum quota flooring is small relative to both targets.
    df_large = _make_large_synthetic_attempts()
    n_strata = len(LARGE_MODEL_CLASSES) * len(CER_LEVELS)
    n_critical = int((df_large["category"] == "message_critical").sum())

    selected_small_target = select_items(df_large, n_stratified=10, seed=0, all_critical=True, power_target=100)
    selected_large_target = select_items(df_large, n_stratified=10, seed=0, all_critical=True, power_target=300)

    assert len(selected_large_target) > len(selected_small_target)
    for power_target, selected in ((100, selected_small_target), (300, selected_large_target)):
        # Documented tolerance (rev Task 4.1 review fix): the realized total
        # is an upper-target approximation, never an exact hit, but it can
        # never fall short by a whole extra stratum's worth of rows.
        assert 0 <= power_target - len(selected) < n_strata
        assert len(selected) == n_critical + size_stratified_draw_for_power_target(n_critical, power_target) - (
            size_stratified_draw_for_power_target(n_critical, power_target) % n_strata
        )


def test_stratified_and_critical_selections_are_disjoint(attempts_parquet):
    # Rev Task 4.1 review fix: the stratified draw must come from the
    # non-critical rows only, so it can never re-select a message-critical
    # row. Before the fix, `_selection_source == "both"` was non-empty
    # (stratified draw overlapping the critical pool), which is exactly why
    # the union undershot n_critical + n_stratified.
    _, df = attempts_parquet
    selected = select_items(df, n_stratified=10, seed=0, all_critical=True, power_target=40)

    assert (selected["_selection_source"] == "both").sum() == 0
    stratified_rows = selected[selected["_selection_source"] == "stratified"]
    assert (stratified_rows["category"] != "message_critical").all()
    critical_rows = selected[selected["_selection_source"] == "message_critical"]
    assert (critical_rows["category"] == "message_critical").all()


def test_power_target_realized_total_matches_critical_plus_quota_capped_stratified_draw():
    # Rev Task 4.1 review fix: on a fixture with enough strata/rows that
    # per-stratum quotas are well below each stratum's non-critical size,
    # the realized total should be EXACTLY n_critical + the quota-capped
    # stratified size (never just "close" in a fuzzy sense), and the gap to
    # the requested power_target should be small and bounded by the
    # documented per-stratum quota granularity (< n_strata).
    df = _make_large_synthetic_attempts()
    n_strata = len(LARGE_MODEL_CLASSES) * len(CER_LEVELS)
    n_critical = int((df["category"] == "message_critical").sum())
    assert n_critical == 5

    power_target = 213  # deliberately does not divide evenly by n_strata=20
    selected = select_items(df, n_stratified=10, seed=0, all_critical=True, power_target=power_target)

    target_stratified = size_stratified_draw_for_power_target(n_critical, power_target)
    quota = max(1, target_stratified // n_strata)
    expected_stratified = quota * n_strata  # every stratum's non-critical pool is >> quota here
    expected_total = n_critical + expected_stratified

    assert (selected["_selection_source"] == "both").sum() == 0
    assert len(selected) == expected_total
    assert 0 <= power_target - len(selected) < n_strata


def test_validation_frame_retains_every_human_labeled_item_including_third_adjudicator_resolved():
    human_df = pd.DataFrame(
        {
            "item_id": [f"R{i:04d}" for i in range(1, 21)],
            "human_label": ["Faithful"] * 17 + ["Drift", "Degraded", "Faithful"],
            # The last 3 items needed a third-adjudicator resolution after
            # rater A / rater B initially disagreed -- exactly the class of
            # item the old buggy comparison silently dropped.
            "resolved_by_third_adjudicator": [False] * 17 + [True, True, True],
        }
    )
    auto_df = pd.DataFrame(
        {
            "item_id": [f"R{i:04d}" for i in range(1, 21)],
            "auto_label": ["faithful"] * 20,
        }
    )

    result = validation_frame(human_df, auto_df)

    assert len(result) == len(human_df) == 20
    assert set(result["item_id"]) == set(human_df["item_id"])
    resolved_ids = set(human_df.loc[human_df["resolved_by_third_adjudicator"], "item_id"])
    assert resolved_ids <= set(result.loc[result["auto_label"].notna(), "item_id"])


def test_validation_frame_395_vs_378_bug_is_structurally_impossible():
    # Reproduce the exact shape of the bug: 395 human-rated items where 17
    # needed third-adjudicator resolution. A naive join that first filters to
    # rows where the raters agreed (before joining to automated labels) would
    # silently shrink 395 -> 378. validation_frame must never do that: the
    # row count always equals len(human_df), independent of any disagreement
    # history.
    n_agree, n_disagree = 378, 17
    item_ids = [f"R{i:04d}" for i in range(1, n_agree + n_disagree + 1)]
    human_df = pd.DataFrame(
        {
            "item_id": item_ids,
            "human_label": ["Faithful"] * n_agree + ["Drift"] * n_disagree,
            "resolved_by_third_adjudicator": [False] * n_agree + [True] * n_disagree,
        }
    )
    auto_df = pd.DataFrame({"item_id": item_ids, "auto_label": ["faithful"] * len(item_ids)})

    # The bug this guards against: naively computing agreement only on rows
    # where the panel already agreed (dropping the 17 before the join).
    naive_buggy_join = human_df.loc[~human_df["resolved_by_third_adjudicator"]].merge(
        auto_df, on="item_id", how="inner"
    )
    assert len(naive_buggy_join) == n_agree  # reproduces the historical 378

    result = validation_frame(human_df, auto_df)
    assert len(result) == n_agree + n_disagree == 395
    assert len(result) != len(naive_buggy_join)


def test_validation_frame_raises_loudly_instead_of_silently_dropping_a_missing_automated_label():
    human_df = pd.DataFrame(
        {
            "item_id": ["R0001", "R0002", "R0003"],
            "human_label": ["Faithful", "Drift", "Degraded"],
            "resolved_by_third_adjudicator": [False, True, False],
        }
    )
    # R0002 (the disagreement-resolved item) has no automated counterpart --
    # simulating exactly the gap that must never be silently swallowed.
    auto_df = pd.DataFrame({"item_id": ["R0001", "R0003"], "auto_label": ["faithful", "degraded"]})

    with pytest.raises(ValueError, match="R0002"):
        validation_frame(human_df, auto_df)

    marked = validation_frame(human_df, auto_df, on_missing="mark")
    assert len(marked) == len(human_df) == 3
    assert set(marked["item_id"]) == {"R0001", "R0002", "R0003"}
    assert marked.loc[marked["item_id"] == "R0002", "has_automated_label"].item() == False
