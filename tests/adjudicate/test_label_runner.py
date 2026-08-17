"""Tests for the batched full-cohort labeling runner (revision Task 5.0).

Tests 1-6 are FAST: they build hand-crafted signal rows directly (or, for
test 6, exercise `taxonomy.label`'s `crit=` pass-through directly) and never
touch a model, so `derive_labels` -- and the Step A taxonomy extension it
depends on -- are proven correct without spaCy/torch/transformers.

Test 7 is the load-bearing SLOW equivalence gate: it runs the real batched
`compute_signals` + `derive_labels` over a fixed real slice of
`output/intermediate/o2_v2_outputs.parquet` and asserts every row matches
the independent per-item `ensemble.ensemble_label(..., default_evaluators(),
...)` path exactly. It is marked `@pytest.mark.slow` AND gated behind an
env var (`IDRIFT_RUN_SLOW_TESTS=1`) plus the input parquet's existence, so
the fast suite (`uv run pytest tests/adjudicate/ -q`, no `-m` filter) never
accidentally loads a model.
"""
import os
from pathlib import Path

import pandas as pd
import pytest

from idrift.adjudicate.critical_rules import detect
from idrift.adjudicate.label_runner import derive_labels
from idrift.adjudicate.taxonomy import label as taxonomy_label

_O2_PARQUET = Path("output/intermediate/o2_v2_outputs.parquet")
_RUN_SLOW = os.environ.get("IDRIFT_RUN_SLOW_TESTS") == "1"


def _clean_venv_sidecars() -> None:
    """Mirror `label_runner._clean_venv_sidecars` at the top of the one test
    here that loads real models: exFAT AppleDouble `._*` sidecar files crash
    transformers/sentence-transformers imports on this workspace's volume."""
    venv = Path(".venv")
    if venv.exists():
        for path in venv.rglob("._*"):
            try:
                path.unlink()
            except OSError:
                pass


def _make_signal_row(**overrides) -> dict:
    """A hand-built row with all columns `derive_labels` expects:
    intended_text/output_message plus every raw signal column
    `compute_signals` would have produced. Defaults describe a clean,
    unambiguous, fluent, bidirectional-entailment, no-critical-flag pair so
    each test only needs to override the handful of fields it cares about."""
    row = {
        "intended_text": "do not move",
        "output_message": "move",
        "nli_deberta_fwd": "entail",
        "nli_deberta_bwd": "entail",
        "nli_roberta_fwd": "entail",
        "nli_roberta_bwd": "entail",
        "cos_mpnet": 0.95,
        "cos_minilm": 0.95,
        "fluency_raw": 0.9,
        "crit_negation_flip": False,
        "crit_numeral_change": False,
        "crit_recipient_change": False,
        "crit_urgency_change": False,
        "crit_actionable_omission": False,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Test 1: crit flag + fluency gate -> drift vs degraded.
# ---------------------------------------------------------------------------


def test_1_crit_flag_is_drift_when_fluent_degraded_when_not():
    fluent_row = _make_signal_row(crit_negation_flip=True, fluency_raw=0.9)
    disfluent_row = _make_signal_row(crit_negation_flip=True, fluency_raw=0.1)
    signals = pd.DataFrame([fluent_row, disfluent_row])

    result = derive_labels(signals, tau=0.5, tie_break="primary")

    assert result.loc[0, "label"] == "drift"
    assert result.loc[1, "label"] == "degraded"
    # Every evaluator agrees (crit and fluency are shared across all 4).
    assert result.loc[0, "agreement"] == pytest.approx(1.0)
    assert result.loc[1, "agreement"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 2: clean high-cosine bidirectional-entail no-crit fluent row -> faithful.
# ---------------------------------------------------------------------------


def test_2_clean_bidirectional_entail_no_crit_fluent_is_faithful():
    row = _make_signal_row(intended_text="I am in pain", output_message="I'm suffering")
    signals = pd.DataFrame([row])

    result = derive_labels(signals)

    assert result.loc[0, "label"] == "faithful"
    assert result.loc[0, "agreement"] == pytest.approx(1.0)
    assert bool(result.loc[0, "tie"]) is False


# ---------------------------------------------------------------------------
# Test 3: tau sensitivity flips drift <-> degraded around the row's fluency_raw.
# ---------------------------------------------------------------------------


def test_3_tau_sensitivity_flips_drift_and_degraded():
    row = _make_signal_row(crit_negation_flip=True, fluency_raw=0.5)
    signals = pd.DataFrame([row])

    below = derive_labels(signals, tau=0.4)  # 0.5 >= 0.4 -> fluent -> drift
    above = derive_labels(signals, tau=0.6)  # 0.5 >= 0.6 is False -> degraded

    assert below.loc[0, "label"] == "drift"
    assert above.loc[0, "label"] == "degraded"


# ---------------------------------------------------------------------------
# Test 4: tie_break knob on a hand-built 2-2 vote split.
# ---------------------------------------------------------------------------


def test_4_tie_break_knob_changes_label_on_2_2_split():
    # Both deberta evaluators (deberta+mpnet, deberta+minilm) see
    # bidirectional entailment -> "faithful"; both roberta evaluators see a
    # contradiction -> "drift". cos/crit/fluency are shared across all 4
    # evaluators (real signals are per-row, not per-evaluator), so only the
    # NLI columns need to differ to produce a clean 2-2 split.
    row = _make_signal_row(
        nli_deberta_fwd="entail",
        nli_deberta_bwd="entail",
        nli_roberta_fwd="contradict",
        nli_roberta_bwd="contradict",
    )
    signals = pd.DataFrame([row])

    primary = derive_labels(signals, tie_break="primary")
    drift_averse = derive_labels(signals, tie_break="drift_averse")
    severity = derive_labels(signals, tie_break="severity")

    assert bool(primary.loc[0, "tie"]) is True
    assert bool(drift_averse.loc[0, "tie"]) is True
    assert bool(severity.loc[0, "tie"]) is True

    # "primary" defers to the first evaluator in `_EVALUATOR_ORDER`
    # (deberta+mpnet), whose own vote is "faithful".
    assert primary.loc[0, "label"] == "faithful"
    # "drift_averse" picks the lowest-severity tied label.
    assert drift_averse.loc[0, "label"] == "faithful"
    # "severity" (the old drift-favoring default) picks the highest.
    assert severity.loc[0, "label"] == "drift"


# ---------------------------------------------------------------------------
# Test 5: determinism.
# ---------------------------------------------------------------------------


def test_5_derive_labels_is_deterministic():
    signals = pd.DataFrame(
        [
            _make_signal_row(),
            _make_signal_row(
                intended_text="I am in pain and need help",
                output_message="I am in pain",
                crit_actionable_omission=True,
            ),
        ]
    )

    first = derive_labels(signals)
    second = derive_labels(signals)

    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Test 6: taxonomy.label crit= injection equivalence (Step A).
# ---------------------------------------------------------------------------


def test_6_taxonomy_label_crit_injection_is_equivalent_to_internal_detect():
    def _stub(nli_result="neutral", cosine=0.9, fluent=0.9):
        return (lambda a, b: cosine), (lambda p, h: nli_result), (lambda text: fluent)

    cases = [
        ("do not move", "move", _stub(nli_result="neutral", cosine=0.9, fluent=0.9)),
        ("I am in pain", "I'm suffering", _stub(nli_result="entail", cosine=0.95, fluent=0.9)),
        ("I am in pain and need help", "I am in pain", _stub(nli_result="neutral", cosine=0.95, fluent=0.9)),
    ]
    for intended, output, (cos_fn, nli_fn, fluency_fn) in cases:
        without_crit = taxonomy_label(intended, output, cos_fn, nli_fn, fluency_fn)
        injected = taxonomy_label(intended, output, cos_fn, nli_fn, fluency_fn, crit=detect(intended, output))
        assert injected == without_crit


# ---------------------------------------------------------------------------
# Test 8 (added for Task A2, harden/2026-07): `use_critical_rules=False`
# ignores the five crit_* flags -- pins both branches on the same row.
# ---------------------------------------------------------------------------


def test_8_use_critical_rules_false_ignores_crit_flags():
    # A row where the ONLY meaning-difference evidence is a fired crit flag:
    # bidirectional entailment and cosine at/above threshold, so absent the
    # crit override the meaning channel alone would call this "no
    # difference" (faithful, since fluent).
    row = _make_signal_row(
        nli_deberta_fwd="entail", nli_deberta_bwd="entail",
        nli_roberta_fwd="entail", nli_roberta_bwd="entail",
        cos_mpnet=0.95, cos_minilm=0.95, fluency_raw=0.9,
        crit_actionable_omission=True,
    )
    signals = pd.DataFrame([row])

    rules_in = derive_labels(signals, use_critical_rules=True)
    rules_out = derive_labels(signals, use_critical_rules=False)

    # Default (True) is unchanged prior behavior: the crit flag overrides
    # the entailment/cosine verdict -> drift (fluent + meaning differs).
    assert rules_in.loc[0, "label"] == "drift"
    # False: the crit flag is ignored entirely; NLI/cosine alone say the
    # meaning is preserved and the row is fluent -> faithful.
    assert rules_out.loc[0, "label"] == "faithful"


def test_8b_use_critical_rules_false_still_relies_on_nli_and_cosine():
    # With no crit flags fired AND use_critical_rules=False, behavior must
    # be identical to the True branch (nothing to ignore) -- proves the
    # flag only ever *removes* the crit override, never adds a new path.
    row = _make_signal_row()  # default: bidirectional entail, all crit False
    signals = pd.DataFrame([row])

    rules_in = derive_labels(signals, use_critical_rules=True)
    rules_out = derive_labels(signals, use_critical_rules=False)

    assert rules_in.loc[0, "label"] == rules_out.loc[0, "label"] == "faithful"


# ---------------------------------------------------------------------------
# Test 7: the load-bearing real-model equivalence gate (slow).
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_SLOW or not _O2_PARQUET.exists(),
    reason="slow real-model equivalence test; set IDRIFT_RUN_SLOW_TESTS=1 and ensure o2_v2_outputs.parquet exists",
)
def test_7_batched_pipeline_matches_per_item_ensemble_label_on_a_real_slice():
    from idrift.adjudicate import ensemble
    from idrift.adjudicate.label_runner import compute_signals

    _clean_venv_sidecars()

    df_full = pd.read_parquet(_O2_PARQUET)
    # Deterministic ~80-row slice: CER0 (mostly faithful) + CER0.4 (more
    # likely to hit critical-rule/contradiction paths).
    cer0 = df_full[df_full["cer_target"] == 0.0].sample(n=40, random_state=42)
    cer04 = df_full[df_full["cer_target"] == 0.4].sample(n=40, random_state=42)
    sample = pd.concat([cer0, cer04]).reset_index(drop=True)

    signals = compute_signals(sample, device="mps", batch_size=32)
    merged = pd.concat([sample[["intended_text", "output_message"]], signals], axis=1)
    labels = derive_labels(merged, tau=0.5, tie_break="primary")

    evaluators = ensemble.default_evaluators()

    for i in range(len(sample)):
        intended = sample.loc[i, "intended_text"]
        output = sample.loc[i, "output_message"]
        expected = ensemble.ensemble_label(intended, output, evaluators, tie_break="primary")

        assert labels.loc[i, "label"] == expected["label"], f"row {i}: label mismatch"
        for name, vote in expected["votes"].items():
            assert labels.loc[i, f"vote__{name}"] == vote, f"row {i}: vote mismatch for {name}"
        assert labels.loc[i, "agreement"] == pytest.approx(expected["agreement"]), f"row {i}: agreement mismatch"
        assert bool(labels.loc[i, "tie"]) == expected["tie"], f"row {i}: tie mismatch"
