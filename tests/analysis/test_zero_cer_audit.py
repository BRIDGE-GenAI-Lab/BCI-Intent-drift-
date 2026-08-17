"""Tests for the zero-CER drift forensic categorizer (revision Task 5.4).

Answers Rev1 #4 / Rev2 #18: why does the automated ensemble call "drift" on
an input that had zero realized decoder error? This module bins every such
case into a documented cause taxonomy using only the already-computed signal
columns (no models, no re-inference).

All tests run on hand-built rows constructed here, never on the real labeled
parquet (a labeling job was running remotely at the time this task was
implemented -- see the task brief's environment note).
"""
import pandas as pd
import pytest

from idrift.adjudicate.nli_metric import COS_HI
from idrift.analysis.zero_cer_audit import (
    categorize_zero_cer_drift,
    unresolved_breakdown,
    zero_cer_breakdown,
)

# ---------------------------------------------------------------------------
# Shared row builder. Every field of the input contract must be present:
# realized_cer, label, intended_text, output_message, cos_mpnet, cos_minilm,
# nli_deberta_fwd/bwd, nli_roberta_fwd/bwd, the five crit_* bool cols,
# fluency_raw, message_id.
# ---------------------------------------------------------------------------


def _row(
    message_id,
    intended_text,
    output_message,
    *,
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
        realized_cer=realized_cer,
        label=label,
        intended_text=intended_text,
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


_REQUIRED_COLUMNS = list(_row("x", "a", "b").keys())


# ---------------------------------------------------------------------------
# 1. critical_substitution -- a genuine negation flip, still CER 0.
# ---------------------------------------------------------------------------


def test_negation_flip_crit_row_categorized_as_critical_substitution():
    df = pd.DataFrame(
        [
            _row(
                "m1",
                "I do not want the ventilator",
                "I want the ventilator",
                cos_mpnet=0.95,
                cos_minilm=0.93,
                nli_deberta_fwd="contradict",
                nli_deberta_bwd="contradict",
                nli_roberta_fwd="contradict",
                nli_roberta_bwd="contradict",
                crit_negation_flip=True,
            )
        ]
    )
    result = categorize_zero_cer_drift(df)

    assert result["total_zero_cer_drift"] == 1
    assert result["by_cause"]["critical_substitution"] == 1
    assert result["examples"]["critical_substitution"] == ["m1"]


# ---------------------------------------------------------------------------
# 2. formatting -- cosmetic-only difference.
# ---------------------------------------------------------------------------


def test_punctuation_only_difference_categorized_as_formatting():
    df = pd.DataFrame([_row("m2", "hello", "hello.")])
    result = categorize_zero_cer_drift(df)

    assert result["by_cause"]["formatting"] == 1
    assert result["examples"]["formatting"] == ["m2"]


# ---------------------------------------------------------------------------
# 3. paraphrase -- high cosine both embedders + bidirectional entailment on
#    at least one NLI model + no crit flag.
# ---------------------------------------------------------------------------


def test_semantically_equivalent_reword_categorized_as_paraphrase():
    df = pd.DataFrame(
        [
            _row(
                "m3",
                "I would like some water please",
                "Could I please have some water",
                cos_mpnet=COS_HI + 0.05,
                cos_minilm=COS_HI + 0.03,
                nli_deberta_fwd="entail",
                nli_deberta_bwd="entail",
                nli_roberta_fwd="neutral",
                nli_roberta_bwd="neutral",
            )
        ]
    )
    result = categorize_zero_cer_drift(df)

    assert result["by_cause"]["paraphrase"] == 1
    assert result["examples"]["paraphrase"] == ["m3"]


def test_paraphrase_requires_cos_hi_not_hardcoded_threshold():
    # Cosine sits just BELOW COS_HI on one embedder -- must NOT qualify as
    # paraphrase even though it would under a hardcoded 0.75 if COS_HI were
    # ever raised above that. Also has no crit flag, isn't a formatting
    # match, and isn't overgenerative -- so it must fall to "other".
    df = pd.DataFrame(
        [
            _row(
                "m3b",
                "I would like some water please",
                "Could I please have some water",
                cos_mpnet=COS_HI - 0.01,
                cos_minilm=COS_HI - 0.01,
                nli_deberta_fwd="entail",
                nli_deberta_bwd="entail",
                nli_roberta_fwd="entail",
                nli_roberta_bwd="entail",
            )
        ]
    )
    result = categorize_zero_cer_drift(df)

    assert result["by_cause"]["paraphrase"] == 0
    assert result["by_cause"]["other"] == 1


# ---------------------------------------------------------------------------
# 4. overgenerative -- output length far exceeds intended, no crit flag.
# ---------------------------------------------------------------------------


def test_five_times_longer_output_categorized_as_overgenerative():
    intended = "Turn off the light"
    output = (
        "Turn off the light in the room and also make sure the door is "
        "locked and check if the window is closed too so everything is "
        "secure for the night"
    )
    assert len(output) >= 5 * len(intended)

    df = pd.DataFrame(
        [
            _row(
                "m4",
                intended,
                output,
                cos_mpnet=0.4,
                cos_minilm=0.4,
                nli_deberta_fwd="neutral",
                nli_deberta_bwd="neutral",
                nli_roberta_fwd="neutral",
                nli_roberta_bwd="neutral",
            )
        ]
    )
    result = categorize_zero_cer_drift(df)

    assert result["by_cause"]["overgenerative"] == 1
    assert result["examples"]["overgenerative"] == ["m4"]


# ---------------------------------------------------------------------------
# 5. other -- residual: not crit, not formatting, not paraphrase, not
#    overgenerative.
# ---------------------------------------------------------------------------


def test_unrelated_similar_length_output_categorized_as_other():
    df = pd.DataFrame(
        [
            _row(
                "m5",
                "Please give me some water",
                "Turn up the volume please",
                cos_mpnet=0.3,
                cos_minilm=0.25,
                nli_deberta_fwd="neutral",
                nli_deberta_bwd="neutral",
                nli_roberta_fwd="neutral",
                nli_roberta_bwd="neutral",
            )
        ]
    )
    result = categorize_zero_cer_drift(df)

    assert result["by_cause"]["other"] == 1
    assert result["examples"]["other"] == ["m5"]


# ---------------------------------------------------------------------------
# 6. Precedence -- a row that is BOTH crit AND would otherwise qualify as
#    paraphrase must resolve to critical_substitution (this is the
#    right-reason first failing test the workflow started from).
# ---------------------------------------------------------------------------


def test_precedence_crit_and_paraphrase_resolves_to_critical_substitution():
    df = pd.DataFrame(
        [
            _row(
                "m6",
                "I do not want water",
                "I want water",
                cos_mpnet=0.95,
                cos_minilm=0.93,
                nli_deberta_fwd="entail",
                nli_deberta_bwd="entail",
                nli_roberta_fwd="entail",
                nli_roberta_bwd="entail",
                crit_negation_flip=True,
            )
        ]
    )
    result = categorize_zero_cer_drift(df)

    assert result["by_cause"]["critical_substitution"] == 1
    assert result["by_cause"]["paraphrase"] == 0
    assert result["examples"]["critical_substitution"] == ["m6"]


# ---------------------------------------------------------------------------
# 7. Filtering -- only realized_cer == 0 AND label == "drift" rows count.
# ---------------------------------------------------------------------------


def test_only_zero_cer_drift_rows_are_counted():
    df = pd.DataFrame(
        [
            _row("keep", "hello", "hello.", realized_cer=0.0, label="drift"),
            _row("skip_cer", "hello", "hello.", realized_cer=0.05, label="drift"),
            _row("skip_label", "hello", "hello.", realized_cer=0.0, label="faithful"),
            _row("skip_both", "hello", "hello.", realized_cer=0.1, label="degraded"),
        ]
    )
    result = categorize_zero_cer_drift(df)

    assert result["total_zero_cer_drift"] == 1
    assert result["by_cause"]["formatting"] == 1
    assert result["examples"]["formatting"] == ["keep"]


# ---------------------------------------------------------------------------
# 8. Empty input -> zeros, all causes present as keys.
# ---------------------------------------------------------------------------


def test_empty_input_returns_zeros_for_every_cause():
    df = pd.DataFrame(columns=_REQUIRED_COLUMNS)
    result = categorize_zero_cer_drift(df)

    assert result["total_zero_cer_drift"] == 0
    assert result["by_cause"] == {
        "critical_substitution": 0,
        "formatting": 0,
        "paraphrase": 0,
        "overgenerative": 0,
        "other": 0,
    }
    assert result["examples"] == {
        "critical_substitution": [],
        "formatting": [],
        "paraphrase": [],
        "overgenerative": [],
        "other": [],
    }


# ---------------------------------------------------------------------------
# 9. Examples capped at <= 10 per cause, even when many more rows qualify.
# ---------------------------------------------------------------------------


def test_examples_capped_at_ten_per_cause():
    rows = [
        _row(f"m{i}", "hello", "hello.", label="drift", realized_cer=0.0)
        for i in range(15)
    ]
    df = pd.DataFrame(rows)
    result = categorize_zero_cer_drift(df)

    assert result["by_cause"]["formatting"] == 15
    assert len(result["examples"]["formatting"]) == 10


# ---------------------------------------------------------------------------
# 10. Determinism -- same input, same output, every time.
# ---------------------------------------------------------------------------


def test_determinism_same_df_reproduces_identical_result():
    df = pd.DataFrame(
        [
            _row("m1", "I do not want the ventilator", "I want the ventilator", crit_negation_flip=True),
            _row("m2", "hello", "hello."),
            _row(
                "m3",
                "I would like some water please",
                "Could I please have some water",
                cos_mpnet=COS_HI + 0.05,
                cos_minilm=COS_HI + 0.03,
                nli_deberta_fwd="entail",
                nli_deberta_bwd="entail",
            ),
        ]
    )
    result_a = categorize_zero_cer_drift(df)
    result_b = categorize_zero_cer_drift(df)

    assert result_a == result_b


# ---------------------------------------------------------------------------
# 11. Input-contract validation -- missing required column raises.
# ---------------------------------------------------------------------------


def test_missing_required_column_raises():
    df = pd.DataFrame([_row("m1", "hello", "hello.")]).drop(columns=["cos_mpnet"])
    with pytest.raises(ValueError, match="cos_mpnet"):
        categorize_zero_cer_drift(df)


def test_notes_documents_precedence_and_cos_hi():
    df = pd.DataFrame(columns=_REQUIRED_COLUMNS)
    result = categorize_zero_cer_drift(df)

    assert "critical_substitution > formatting > paraphrase > overgenerative > other" in result["notes"]
    assert str(COS_HI) in result["notes"]


# ---------------------------------------------------------------------------
# Task B8 -- zero_cer_breakdown (per-model / per-corpus tabulation, with the
# residual "other" cause relabeled "unresolved_or_mixed") and
# unresolved_breakdown (exploratory diagnostic split of that residual
# bucket). Same hand-built-fixture convention as above: no real parquet.
# ---------------------------------------------------------------------------


def _row_mc(message_id, intended_text, output_message, *, model, corpus, **kwargs):
    """`_row` plus the two grouping columns `zero_cer_breakdown` requires."""
    d = _row(message_id, intended_text, output_message, **kwargs)
    d["model"] = model
    d["corpus"] = corpus
    return d


def _breakdown_fixture_df() -> pd.DataFrame:
    """8 qualifying rows with a known, hand-checked model x corpus x cause
    composition, plus 1 non-qualifying row (nonzero realized_cer) that must
    be excluded from every level of the breakdown.

    modelA/AUTH: critical_substitution x1 (m1), formatting x1 (m2)
    modelA/CRIT: paraphrase x1 (m3)
    modelB/AUTH: overgenerative x1 (m4)
    modelB/CRIT: residual/"other" x4 -- m5a (NLI-contradiction-driven),
        m5b (cosine-driven, plain), m_border (cosine-driven + borderline
        cosine near COS_HI), m_empty (cosine-driven + near-empty output).

    Totals: modelA=3, modelB=5 | AUTH=3, CRIT=5 | overall=8.
    """
    rows = [
        _row_mc(
            "m1", "I do not want the ventilator", "I want the ventilator",
            model="modelA", corpus="AUTH",
            cos_mpnet=0.95, cos_minilm=0.93,
            nli_deberta_fwd="contradict", nli_deberta_bwd="contradict",
            nli_roberta_fwd="contradict", nli_roberta_bwd="contradict",
            crit_negation_flip=True,
        ),
        _row_mc("m2", "hello", "hello.", model="modelA", corpus="AUTH"),
        # Same text as m2, but nonzero realized_cer -- must be filtered out
        # at every level (overall, by_model, by_corpus).
        _row_mc(
            "skip_cer", "hello", "hello.", model="modelA", corpus="AUTH",
            realized_cer=0.2,
        ),
        _row_mc(
            "m3", "I would like some water please", "Could I please have some water",
            model="modelA", corpus="CRIT",
            cos_mpnet=COS_HI + 0.05, cos_minilm=COS_HI + 0.03,
            nli_deberta_fwd="entail", nli_deberta_bwd="entail",
            nli_roberta_fwd="neutral", nli_roberta_bwd="neutral",
        ),
        _row_mc(
            "m4", "Turn off the light",
            "Turn off the light in the room and also make sure the door is "
            "locked and check if the window is closed too so everything is "
            "secure for the night",
            model="modelB", corpus="AUTH",
            cos_mpnet=0.4, cos_minilm=0.4,
        ),
        _row_mc(
            "m5a", "Please give me some water", "Turn up the volume please",
            model="modelB", corpus="CRIT",
            cos_mpnet=0.3, cos_minilm=0.25,
            nli_deberta_fwd="contradict",
        ),
        _row_mc(
            "m5b", "Please give me some water", "Can you close the door now",
            model="modelB", corpus="CRIT",
            cos_mpnet=0.3, cos_minilm=0.25,
        ),
        _row_mc(
            "m_border", "Please give me some water", "Hand me a cup of water now",
            model="modelB", corpus="CRIT",
            cos_mpnet=COS_HI - 0.01, cos_minilm=0.3,
        ),
        _row_mc(
            "m_empty", "Please give me some water", "hm",
            model="modelB", corpus="CRIT",
            cos_mpnet=0.2, cos_minilm=0.15,
        ),
    ]
    return pd.DataFrame(rows)


def test_zero_cer_breakdown_overall_matches_categorize_with_renamed_residual():
    df = _breakdown_fixture_df()
    result = zero_cer_breakdown(df)

    assert result["total_zero_cer_drift"] == 8
    assert result["by_cause"] == {
        "critical_substitution": 1,
        "formatting": 1,
        "paraphrase": 1,
        "overgenerative": 1,
        "unresolved_or_mixed": 4,
    }
    assert "other" not in result["by_cause"]


def test_zero_cer_breakdown_by_model():
    df = _breakdown_fixture_df()
    result = zero_cer_breakdown(df)

    assert result["by_model"]["modelA"]["total_zero_cer_drift"] == 3
    assert result["by_model"]["modelA"]["by_cause"] == {
        "critical_substitution": 1,
        "formatting": 1,
        "paraphrase": 1,
        "overgenerative": 0,
        "unresolved_or_mixed": 0,
    }
    assert result["by_model"]["modelB"]["total_zero_cer_drift"] == 5
    assert result["by_model"]["modelB"]["by_cause"] == {
        "critical_substitution": 0,
        "formatting": 0,
        "paraphrase": 0,
        "overgenerative": 1,
        "unresolved_or_mixed": 4,
    }


def test_zero_cer_breakdown_by_corpus():
    df = _breakdown_fixture_df()
    result = zero_cer_breakdown(df)

    assert result["by_corpus"]["AUTH"]["total_zero_cer_drift"] == 3
    assert result["by_corpus"]["AUTH"]["by_cause"] == {
        "critical_substitution": 1,
        "formatting": 1,
        "paraphrase": 0,
        "overgenerative": 1,
        "unresolved_or_mixed": 0,
    }
    assert result["by_corpus"]["CRIT"]["total_zero_cer_drift"] == 5
    assert result["by_corpus"]["CRIT"]["by_cause"] == {
        "critical_substitution": 0,
        "formatting": 0,
        "paraphrase": 1,
        "overgenerative": 0,
        "unresolved_or_mixed": 4,
    }


def test_zero_cer_breakdown_reports_automated_rule_based_method():
    df = _breakdown_fixture_df()
    result = zero_cer_breakdown(df)
    assert result["categorization_method"] == "automated_rule_based"


def test_zero_cer_breakdown_missing_model_or_corpus_column_raises():
    df_no_model = _breakdown_fixture_df().drop(columns=["model"])
    with pytest.raises(ValueError, match="model"):
        zero_cer_breakdown(df_no_model)

    df_no_corpus = _breakdown_fixture_df().drop(columns=["corpus"])
    with pytest.raises(ValueError, match="corpus"):
        zero_cer_breakdown(df_no_corpus)


# ---------------------------------------------------------------------------
# unresolved_breakdown
# ---------------------------------------------------------------------------


def test_unresolved_breakdown_total_and_diagnostic_split():
    df = _breakdown_fixture_df()
    result = unresolved_breakdown(df)

    assert result["total_unresolved"] == 4
    assert result["diagnostic_split"] == {
        "nli_contradiction_driven": 1,
        "cosine_driven_low_similarity": 3,
        "entailment_gate_failed_high_cosine": 0,
    }
    assert sum(result["diagnostic_split"].values()) == result["total_unresolved"]


def test_unresolved_breakdown_overlay_flags_are_independent_counts():
    df = _breakdown_fixture_df()
    result = unresolved_breakdown(df)

    assert result["overlay_flags"]["borderline_cosine_near_threshold"] == 1
    assert result["overlay_flags"]["empty_or_near_empty_output"] == 1


def test_unresolved_breakdown_examples_only_from_residual_rows():
    df = _breakdown_fixture_df()
    result = unresolved_breakdown(df)

    assert result["examples"]["nli_contradiction_driven"] == ["m5a"]
    assert set(result["examples"]["cosine_driven_low_similarity"]) == {"m5b", "m_border", "m_empty"}

    # Rows resolved to a positive cause (m1/m2/m3/m4) must never leak in.
    all_examples = [mid for lst in result["examples"].values() for mid in lst]
    assert "m1" not in all_examples
    assert "m2" not in all_examples
    assert "m3" not in all_examples
    assert "m4" not in all_examples


def test_unresolved_breakdown_is_exploratory_and_automated():
    df = _breakdown_fixture_df()
    result = unresolved_breakdown(df)

    assert result["exploratory"] is True
    assert result["categorization_method"] == "automated_rule_based"
    assert "EXPLORATORY" in result["notes"]


def test_unresolved_breakdown_missing_required_column_raises():
    df = _breakdown_fixture_df().drop(columns=["cos_mpnet"])
    with pytest.raises(ValueError, match="cos_mpnet"):
        unresolved_breakdown(df)


def test_unresolved_breakdown_empty_input_returns_zeros():
    df = pd.DataFrame(columns=_REQUIRED_COLUMNS)
    result = unresolved_breakdown(df)

    assert result["total_unresolved"] == 0
    assert result["diagnostic_split"] == {
        "nli_contradiction_driven": 0,
        "cosine_driven_low_similarity": 0,
        "entailment_gate_failed_high_cosine": 0,
    }
    assert result["overlay_flags"] == {
        "borderline_cosine_near_threshold": 0,
        "empty_or_near_empty_output": 0,
    }
