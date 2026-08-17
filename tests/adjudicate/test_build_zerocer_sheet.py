"""Tests for the blinded zero-CER drift cause-adjudication sheet builder
(Task B1 -- answers the reviewer's Figure 2b / zero-corruption critique).

The physician sheet must show only intended_text/noisy_text/output_message
for the realized_cer==0 & label=="drift" subset, stratified across the five
rule-based cause categories (critical_substitution/formatting/paraphrase/
overgenerative/other) so all are represented in the sample, SHUFFLED, and
BLIND to the automated cause (and to model/corpus/CER/label). The hidden key
must round-trip: joining it back to the sheet on item_id recovers every
hidden field.
"""
from __future__ import annotations

import pandas as pd
import pytest

from idrift.adjudicate.build_zerocer_sheet import (
    BLINDED_COLUMNS,
    CAUSE_OPTIONS,
    _CAUSE_TO_PHYSICIAN_OPTION,
    build_zerocer_sheet,
    select_zero_cer_drift,
)
from idrift.adjudicate.nli_metric import COS_HI

LEAKED_COLUMNS = {
    "model",
    "model_id",
    "model_class",
    "corpus",
    "cer_target",
    "cer",
    "realized_cer",
    "label",
    "final_label",
    "automated_label",
    "automated_cause",
    "_cause",
    "category",
}


def _row(
    message_id,
    intended_text,
    output_message,
    *,
    noisy_text=None,
    model="gemma4:12b",
    corpus="AUTH",
    cer_target=0.0,
    realized_cer=0.0,
    label="drift",
    cos_mpnet=0.5,
    cos_minilm=0.5,
    nli_deberta_fwd="neutral",
    nli_deberta_bwd="neutral",
    nli_roberta_fwd="neutral",
    nli_roberta_bwd="neutral",
    crit_negation_flip=False,
    crit_numeral_change=False,
    crit_recipient_change=False,
    crit_urgency_change=False,
    crit_actionable_omission=False,
    fluency_raw=1.0,
):
    return dict(
        message_id=message_id,
        model=model,
        corpus=corpus,
        cer_target=cer_target,
        realized_cer=realized_cer,
        label=label,
        intended_text=intended_text,
        noisy_text=intended_text if noisy_text is None else noisy_text,
        output_message=output_message,
        cos_mpnet=cos_mpnet,
        cos_minilm=cos_minilm,
        nli_deberta_fwd=nli_deberta_fwd,
        nli_deberta_bwd=nli_deberta_bwd,
        nli_roberta_fwd=nli_roberta_fwd,
        nli_roberta_bwd=nli_roberta_bwd,
        crit_negation_flip=crit_negation_flip,
        crit_numeral_change=crit_numeral_change,
        crit_recipient_change=crit_recipient_change,
        crit_urgency_change=crit_urgency_change,
        crit_actionable_omission=crit_actionable_omission,
        fluency_raw=fluency_raw,
    )


def _critical_substitution_rows(n, offset=0):
    return [
        _row(
            f"crit_{offset + i:03d}",
            "I do not want the ventilator",
            "I want the ventilator",
            cos_mpnet=0.95,
            cos_minilm=0.93,
            nli_deberta_fwd="contradict",
            nli_deberta_bwd="contradict",
            crit_negation_flip=True,
            model=f"model_{i % 3}",
            corpus="CRIT" if i % 2 == 0 else "AUTH",
        )
        for i in range(n)
    ]


def _formatting_rows(n, offset=0):
    return [
        _row(f"fmt_{offset + i:03d}", "hello there friend", "hello there friend.", model=f"model_{i % 3}")
        for i in range(n)
    ]


def _paraphrase_rows(n, offset=0):
    return [
        _row(
            f"para_{offset + i:03d}",
            "I would like some water please",
            "Could I please have some water",
            cos_mpnet=COS_HI + 0.05,
            cos_minilm=COS_HI + 0.03,
            nli_deberta_fwd="entail",
            nli_deberta_bwd="entail",
            model=f"model_{i % 3}",
        )
        for i in range(n)
    ]


def _overgenerative_rows(n, offset=0):
    intended = "Turn off the light"
    output = (
        "Turn off the light in the room and also make sure the door is "
        "locked and check if the window is closed too so everything is "
        "secure for the night"
    )
    return [
        _row(
            f"over_{offset + i:03d}",
            intended,
            output,
            cos_mpnet=0.4,
            cos_minilm=0.4,
            model=f"model_{i % 3}",
            cer_target=0.1 if i % 2 == 0 else 0.0,  # nonzero-target arm can still land realized_cer==0
        )
        for i in range(n)
    ]


def _other_rows(n, offset=0):
    return [
        _row(
            f"other_{offset + i:03d}",
            "Please give me some water",
            "Turn up the volume please",
            cos_mpnet=0.3,
            cos_minilm=0.25,
            model=f"model_{i % 3}",
        )
        for i in range(n)
    ]


def _non_qualifying_rows(n):
    """Rows that must NEVER appear in the sheet: either not realized_cer==0
    or not label=='drift'."""
    rows = []
    for i in range(n):
        if i % 2 == 0:
            rows.append(_row(f"nq_cer_{i:03d}", "a faithful message", "a faithful message", realized_cer=0.2))
        else:
            rows.append(_row(f"nq_label_{i:03d}", "a faithful message", "a faithful message", label="faithful"))
    return rows


def _make_synthetic_attempts(n_per_cause=10) -> pd.DataFrame:
    rows = (
        _critical_substitution_rows(n_per_cause)
        + _formatting_rows(n_per_cause)
        + _paraphrase_rows(n_per_cause)
        + _overgenerative_rows(n_per_cause)
        + _other_rows(n_per_cause)
        + _non_qualifying_rows(10)
    )
    return pd.DataFrame(rows)


@pytest.fixture
def attempts_parquet(tmp_path):
    df = _make_synthetic_attempts()
    path = tmp_path / "attempts_v3_labeled.parquet"
    df.to_parquet(path, index=False)
    return path, df


def test_select_zero_cer_drift_filters_to_realized_cer_zero_and_drift_label():
    df = _make_synthetic_attempts()
    subset = select_zero_cer_drift(df)

    assert (subset["realized_cer"] == 0).all()
    assert (subset["label"] == "drift").all()
    assert len(subset) == 50  # 5 causes x 10 rows; the 10 non-qualifying rows are excluded
    assert set(subset["_cause"].unique()) == set(_CAUSE_TO_PHYSICIAN_OPTION.keys())


def test_build_zerocer_sheet_stratifies_across_all_five_causes(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    outdir = tmp_path / "out"
    build_zerocer_sheet(path, n=25, seed=0, outdir=outdir)

    key = pd.read_csv(outdir / "_KEY_DO_NOT_SHARE" / "key.csv")
    assert set(key["automated_cause"].unique()) == set(_CAUSE_TO_PHYSICIAN_OPTION.keys())
    # every cause has at least one representative in the drawn sample
    for cause in _CAUSE_TO_PHYSICIAN_OPTION:
        assert (key["automated_cause"] == cause).sum() >= 1


def test_sheet_has_no_leakage_columns(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    outdir = tmp_path / "out"
    build_zerocer_sheet(path, n=25, seed=0, outdir=outdir)

    sheet = pd.read_csv(outdir / "sheet.csv", keep_default_na=False)
    assert list(sheet.columns) == BLINDED_COLUMNS
    assert LEAKED_COLUMNS.isdisjoint(sheet.columns)

    # belt-and-braces: no cell value in the sheet equals a physician-facing
    # cause option or an internal cause name (the label column itself must
    # start blank, not pre-filled with the automated call).
    assert (sheet["cause"] == "").all()
    for option in CAUSE_OPTIONS:
        assert option not in sheet.columns


def test_sheet_only_contains_zero_cer_drift_rows(attempts_parquet, tmp_path):
    path, df = attempts_parquet
    outdir = tmp_path / "out"
    build_zerocer_sheet(path, n=25, seed=0, outdir=outdir)

    key = pd.read_csv(outdir / "_KEY_DO_NOT_SHARE" / "key.csv")
    non_qualifying_message_ids = set(df.loc[df["message_id"].str.startswith("nq_"), "message_id"])
    assert non_qualifying_message_ids.isdisjoint(set(key["message_id"]))


def test_key_round_trips_item_id_to_message_and_cause(attempts_parquet, tmp_path):
    path, df = attempts_parquet
    outdir = tmp_path / "out"
    build_zerocer_sheet(path, n=25, seed=0, outdir=outdir)

    sheet = pd.read_csv(outdir / "sheet.csv")
    key = pd.read_csv(outdir / "_KEY_DO_NOT_SHARE" / "key.csv")
    joined = sheet.merge(key, on="item_id", how="inner", validate="1:1")
    assert len(joined) == len(sheet) == len(key)

    lookup = df.set_index("message_id")
    for _, row in joined.iterrows():
        orig = lookup.loc[row["message_id"]]
        assert orig["intended_text"] == row["intended_text"]
        assert orig["output_message"] == row["output_message"]
        assert row["automated_cause"] in _CAUSE_TO_PHYSICIAN_OPTION


def test_determinism_same_seed_same_item_id_assignment(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    build_zerocer_sheet(path, n=25, seed=0, outdir=out1)
    build_zerocer_sheet(path, n=25, seed=0, outdir=out2)

    key1 = pd.read_csv(out1 / "_KEY_DO_NOT_SHARE" / "key.csv").sort_values("message_id").reset_index(drop=True)
    key2 = pd.read_csv(out2 / "_KEY_DO_NOT_SHARE" / "key.csv").sort_values("message_id").reset_index(drop=True)

    assert key1["item_id"].tolist() == key2["item_id"].tolist()
    assert key1["message_id"].tolist() == key2["message_id"].tolist()


def test_readme_written(attempts_parquet, tmp_path):
    path, _ = attempts_parquet
    outdir = tmp_path / "out"
    build_zerocer_sheet(path, n=25, seed=0, outdir=outdir)

    readme = outdir / "README.md"
    assert readme.exists()
    text = readme.read_text()
    for option in CAUSE_OPTIONS:
        assert option in text


def test_missing_required_column_raises():
    df = pd.DataFrame([{"message_id": "m1", "realized_cer": 0.0, "label": "drift"}])
    with pytest.raises(ValueError):
        select_zero_cer_drift(df)
