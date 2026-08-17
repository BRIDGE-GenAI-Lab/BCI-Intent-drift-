"""Tests for the blinded CRIT-vs-CTRL manual corroboration sheet builder
(Task B2 -- independently backs the automated critical-message finding).

Builds a matched CRIT/CTRL subset spanning the CER grid from the labeled
attempts, using the Task-1.3 `ctrl_matched.csv` pairing (crit_message_id <->
message_id). Physicians label each item faithful/degraded/drift, BLIND TO
CORPUS -- the sheet must never reveal whether an item came from the CRIT or
the CTRL half of a pair. The hidden key must round-trip and must still
carry, for every pair_id, exactly one CRIT-corpus item and one CTRL-corpus
item, so the manual matched contrast can be reconstructed later.
"""
from __future__ import annotations

import pandas as pd
import pytest

from idrift.adjudicate.build_critctrl_sheet import (
    BLINDED_COLUMNS,
    RATING_LABELS,
    build_critctrl_sheet,
    build_pair_candidates,
    draw_stratified_pairs,
    expand_to_items,
)

CER_LEVELS = [0.0, 0.1, 0.2]
MODELS = ["modelA", "modelB", "modelC"]
N_PAIRS = 8

LEAKED_COLUMNS = {
    "model",
    "model_id",
    "corpus",
    "critical",
    "role",
    "cer_target",
    "cer",
    "realized_cer",
    "label",
    "final_label",
    "automated_label",
    "pair_id",
}


def _make_synthetic_attempts() -> pd.DataFrame:
    """8 matched CRIT/CTRL pairs x 3 CER levels x 3 models x 2 replicates,
    both corpus halves populated at every (pair, CER) combination."""
    rows = []
    for pair_i in range(N_PAIRS):
        crit_id = f"probe_{pair_i:04d}"
        ctrl_id = f"authctrl_{pair_i:04d}"
        for corpus, message_id in (("CRIT", crit_id), ("CTRL", ctrl_id)):
            for cer in CER_LEVELS:
                for model in MODELS:
                    for rep in range(2):
                        # a handful of drift labels scattered in so agreement
                        # analysis downstream has something to chew on;
                        # not exercised by these build-only tests.
                        label = "drift" if (pair_i + rep) % 5 == 0 else "faithful"
                        rows.append(
                            {
                                "message_id": message_id,
                                "model": model,
                                "corpus": corpus,
                                "cer_target": cer,
                                "replicate_idx": rep,
                                "label": label,
                                "intended_text": f"intended {message_id}",
                                "output_message": f"output {message_id} {model} {cer} {rep}",
                            }
                        )
    df = pd.DataFrame(rows)
    assert len(df) == N_PAIRS * 2 * len(CER_LEVELS) * len(MODELS) * 2
    return df


def _make_ctrl_matched() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "crit_message_id": [f"probe_{i:04d}" for i in range(N_PAIRS)],
            "message_id": [f"authctrl_{i:04d}" for i in range(N_PAIRS)],
        }
    )


@pytest.fixture
def attempts_and_ctrl_matched(tmp_path):
    attempts = _make_synthetic_attempts()
    ctrl_matched = _make_ctrl_matched()
    attempts_path = tmp_path / "attempts_v3_labeled.parquet"
    ctrl_matched_path = tmp_path / "ctrl_matched.csv"
    attempts.to_parquet(attempts_path, index=False)
    ctrl_matched.to_csv(ctrl_matched_path, index=False)
    return attempts_path, ctrl_matched_path, attempts, ctrl_matched


def test_build_pair_candidates_has_one_row_per_pair_x_cer_with_both_halves(attempts_and_ctrl_matched):
    _, _, attempts, ctrl_matched = attempts_and_ctrl_matched
    combo = build_pair_candidates(attempts, ctrl_matched, seed=0)

    assert len(combo) == N_PAIRS * len(CER_LEVELS)
    assert combo["crit_message_id"].notna().all()
    assert combo["ctrl_message_id"].notna().all()
    # every crit_message_id in the combo table really is a CRIT-corpus id
    crit_ids = set(attempts.loc[attempts["corpus"] == "CRIT", "message_id"])
    ctrl_ids = set(attempts.loc[attempts["corpus"] == "CTRL", "message_id"])
    assert set(combo["crit_message_id"]).issubset(crit_ids)
    assert set(combo["ctrl_message_id"]).issubset(ctrl_ids)


def test_draw_stratified_pairs_spans_full_cer_grid(attempts_and_ctrl_matched):
    _, _, attempts, ctrl_matched = attempts_and_ctrl_matched
    combo = build_pair_candidates(attempts, ctrl_matched, seed=0)
    selected = draw_stratified_pairs(combo, n_pairs=6, seed=0)

    assert set(selected["cer_target"].unique()) == set(CER_LEVELS)
    for cer in CER_LEVELS:
        assert (selected["cer_target"] == cer).sum() >= 1


def test_draw_stratified_pairs_does_not_collapse_to_the_same_pair_ids_at_every_cer_level():
    # Regression test: build_pair_candidates returns combo sorted by pair_id
    # then cer_target, so every cer_target slice originally shared the exact
    # same relative pair_id order. human_export.stratified_sample samples
    # every stratum with the SAME random_state, and pandas' positional
    # .sample() then picks the SAME positions (== the same pair_ids) in
    # every stratum, so the whole draw collapsed to just `quota` distinct
    # pairs total instead of spanning the pool -- caught on the real 131-pair
    # ctrl_matched.csv (a 150-target draw collapsed to 30 pairs). This
    # fixture uses many more pairs than the per-level quota so the same
    # failure mode is reproducible at small scale.
    n_pairs_available = 40
    cer_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    rows = []
    for pair_i in range(n_pairs_available):
        for cer in cer_levels:
            rows.append(
                {
                    "pair_id": f"probe_{pair_i:04d}",
                    "cer_target": cer,
                    "crit_message_id": f"probe_{pair_i:04d}",
                    "crit_model": "modelA",
                    "crit_intended_text": "x",
                    "crit_output_message": "x",
                    "crit_label": "faithful",
                    "ctrl_message_id": f"authctrl_{pair_i:04d}",
                    "ctrl_model": "modelA",
                    "ctrl_intended_text": "y",
                    "ctrl_output_message": "y",
                    "ctrl_label": "faithful",
                }
            )
    # Sorted by pair_id then cer_target, exactly build_pair_candidates' own
    # output order -- reproduces the alignment hazard without needing the
    # full groupby machinery.
    combo = pd.DataFrame(rows).sort_values(["pair_id", "cer_target"]).reset_index(drop=True)

    selected = draw_stratified_pairs(combo, n_pairs=25, seed=0)  # quota = 25 // 5 = 5 per level

    n_distinct_pairs = selected["pair_id"].nunique()
    quota_per_level = 25 // len(cer_levels)
    # If the alignment bug were present, every level would draw the exact
    # same `quota_per_level` pair_ids, so n_distinct_pairs would equal
    # quota_per_level exactly. A healthy draw should span well beyond a
    # single stratum's quota.
    assert n_distinct_pairs > quota_per_level


def test_expand_to_items_yields_two_rows_per_pair_one_crit_one_ctrl(attempts_and_ctrl_matched):
    _, _, attempts, ctrl_matched = attempts_and_ctrl_matched
    combo = build_pair_candidates(attempts, ctrl_matched, seed=0)
    selected = draw_stratified_pairs(combo, n_pairs=6, seed=0)
    items = expand_to_items(selected)

    assert len(items) == 2 * len(selected)
    counts = items.groupby("pair_id")["corpus"].apply(lambda s: set(s))
    assert (counts == {"CRIT", "CTRL"}).all()


def test_sheet_has_no_leakage_columns(attempts_and_ctrl_matched, tmp_path):
    attempts_path, ctrl_matched_path, _, _ = attempts_and_ctrl_matched
    outdir = tmp_path / "out"
    build_critctrl_sheet(attempts_path, ctrl_matched_path, n_pairs=6, seed=0, outdir=outdir)

    sheet = pd.read_csv(outdir / "sheet.csv", keep_default_na=False)
    assert list(sheet.columns) == BLINDED_COLUMNS
    assert LEAKED_COLUMNS.isdisjoint(sheet.columns)
    assert (sheet["rating"] == "").all()
    assert (sheet["notes"] == "").all()


def test_sheet_text_values_never_reveal_corpus_membership_via_column_name(attempts_and_ctrl_matched, tmp_path):
    # Belt-and-braces: the literal strings "CRIT"/"CTRL" must not appear
    # anywhere in the sheet (neither as a column nor smuggled into a cell).
    attempts_path, ctrl_matched_path, _, _ = attempts_and_ctrl_matched
    outdir = tmp_path / "out"
    build_critctrl_sheet(attempts_path, ctrl_matched_path, n_pairs=6, seed=0, outdir=outdir)

    sheet_text = (outdir / "sheet.csv").read_text()
    assert "CRIT" not in sheet_text
    assert "CTRL" not in sheet_text


def test_key_round_trips_and_preserves_pair_structure(attempts_and_ctrl_matched, tmp_path):
    attempts_path, ctrl_matched_path, attempts, _ = attempts_and_ctrl_matched
    outdir = tmp_path / "out"
    build_critctrl_sheet(attempts_path, ctrl_matched_path, n_pairs=6, seed=0, outdir=outdir)

    sheet = pd.read_csv(outdir / "sheet.csv")
    key = pd.read_csv(outdir / "_KEY_DO_NOT_SHARE" / "key.csv")
    joined = sheet.merge(key, on="item_id", how="inner", validate="1:1")
    assert len(joined) == len(sheet) == len(key)

    # Every selected (pair_id, cer_target) combo contributes exactly one
    # CRIT item and one CTRL item. A given pair_id CAN recur at a different
    # cer_target (there are only 8 underlying message-pairs in this
    # fixture but a 6-pair draw across 3 CER levels can revisit one), so the
    # invariant is checked per (pair_id, cer_target) instance, not per bare
    # pair_id.
    per_combo = key.groupby(["pair_id", "cer_target"])["corpus"].apply(lambda s: sorted(s))
    assert per_combo.apply(lambda x: x == ["CRIT", "CTRL"]).all()
    # ...and every one of those instances is internally consistent: the
    # CRIT item and CTRL item sharing a (pair_id, cer_target) really do
    # trace back to the Task-1.3 matched pair, never a mismatched message.
    for (pair_id, cer), group in key.groupby(["pair_id", "cer_target"]):
        crit_row = group[group["corpus"] == "CRIT"].iloc[0]
        ctrl_row = group[group["corpus"] == "CTRL"].iloc[0]
        assert crit_row["message_id"] == pair_id  # pair_id == crit_message_id by construction
        assert ctrl_row["message_id"] != pair_id

    lookup = attempts.set_index(["message_id", "model", "cer_target"])
    for _, row in joined.iterrows():
        orig_rows = attempts[
            (attempts["message_id"] == row["message_id"])
            & (attempts["model"] == row["model"])
            & (attempts["cer_target"] == row["cer_target"])
        ]
        assert (orig_rows["intended_text"] == row["intended_text"]).any()
        assert (orig_rows["output_message"] == row["output_message"]).any()


def test_determinism_same_seed_same_item_id_assignment(attempts_and_ctrl_matched, tmp_path):
    attempts_path, ctrl_matched_path, _, _ = attempts_and_ctrl_matched
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    build_critctrl_sheet(attempts_path, ctrl_matched_path, n_pairs=6, seed=0, outdir=out1)
    build_critctrl_sheet(attempts_path, ctrl_matched_path, n_pairs=6, seed=0, outdir=out2)

    key1 = pd.read_csv(out1 / "_KEY_DO_NOT_SHARE" / "key.csv").sort_values(["pair_id", "corpus"]).reset_index(drop=True)
    key2 = pd.read_csv(out2 / "_KEY_DO_NOT_SHARE" / "key.csv").sort_values(["pair_id", "corpus"]).reset_index(drop=True)

    assert key1["item_id"].tolist() == key2["item_id"].tolist()
    assert key1["message_id"].tolist() == key2["message_id"].tolist()


def test_rating_labels_are_the_three_brief_specified_options():
    assert RATING_LABELS == ("faithful", "degraded", "drift")


def test_readme_written(attempts_and_ctrl_matched, tmp_path):
    attempts_path, ctrl_matched_path, _, _ = attempts_and_ctrl_matched
    outdir = tmp_path / "out"
    build_critctrl_sheet(attempts_path, ctrl_matched_path, n_pairs=6, seed=0, outdir=outdir)

    readme = outdir / "README.md"
    assert readme.exists()
    text = readme.read_text()
    for label in RATING_LABELS:
        assert label in text
    assert "blind" in text.lower()


def test_missing_required_column_raises():
    bad_attempts = pd.DataFrame([{"message_id": "m1"}])
    ctrl_matched = _make_ctrl_matched()
    with pytest.raises(ValueError):
        build_pair_candidates(bad_attempts, ctrl_matched, seed=0)
