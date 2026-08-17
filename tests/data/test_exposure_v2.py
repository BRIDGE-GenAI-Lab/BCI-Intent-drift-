"""Tests for the replicate-aware exposure generator (rev Task 1.2).

`_edit_cer` is a small dependency-free (pure-Python) Levenshtein-based CER
computed independently of the production code path (which uses the already
-declared `python-Levenshtein` dependency via `idrift.lib.cer.cer`), so the
`realized_cer == measured edit distance / len` assertion is a genuine check
against a second implementation, not a tautology against the same helper the
implementation calls.
"""
import numpy as np

from idrift.data.exposure_v2 import build_exposure_v2
from idrift.data.grid import GRID_ALPHABET


def _msg(text, message_id="m1", corpus="AUTH"):
    return {"message_id": message_id, "corpus": corpus, "text": text}


def _pooled():
    """Synthetic uniform 36x36 confusion matrix over the grid alphabet.

    Mirrors the synthetic-CONF pattern already used in test_noise_model.py
    (uniform confusion) rather than loading the real materialized
    confusion_overall.npy, so this test suite stays fast and independent of
    build artifacts.
    """
    n = len(GRID_ALPHABET)
    return np.full((n, n), 1 / n)


def _edit_cer(reference: str, hypothesis: str) -> float:
    """Pure-Python Wagner-Fischer edit distance / len(reference)."""
    if len(reference) == 0:
        return 0.0
    a, b = reference, hypothesis
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1] / len(a)


# --- Step 1: the brief's failing test, verbatim ------------------------------

def test_replicates_are_distinct_and_metadata_consistent():
    df = build_exposure_v2([_msg("I AM IN PAIN AND NEED HELP")], [0.2], n_rep=20,
                            confusion=_pooled(), seed=7)
    assert df.replicate_idx.nunique() == 20
    assert df.noisy_text.nunique() >= 2                 # not one arbitrary draw
    for r in df.itertuples():
        assert abs(_edit_cer(r.text, r.noisy_text) - r.realized_cer) < 1e-9
        assert r.n_errors == r.sub_count + r.ins_count + r.del_count
    d2 = build_exposure_v2([_msg("I AM IN PAIN AND NEED HELP")], [0.2], n_rep=20,
                            confusion=_pooled(), seed=7)
    assert list(df.noisy_text) == list(d2.noisy_text)   # deterministic


# --- Supporting structural checks -------------------------------------------

def test_error_positions_within_range_and_expected_columns():
    df = build_exposure_v2([_msg("I AM IN PAIN AND NEED HELP")], [0.3], n_rep=5,
                            confusion=_pooled(), seed=11)
    expected_cols = {
        "message_id", "corpus", "text", "cer_target", "replicate_idx", "seed",
        "noisy_text", "realized_cer", "n_errors", "error_positions",
        "sub_count", "ins_count", "del_count", "corrupted_content_words",
        "corrupted_negation", "corrupted_numeral", "n_passthrough_chars",
    }
    assert expected_cols.issubset(df.columns)
    for r in df.itertuples():
        assert isinstance(r.error_positions, list)
        assert all(0 <= p < len(r.text) for p in r.error_positions)


def test_corpus_and_message_id_pass_through():
    df = build_exposure_v2(
        [_msg("CALL MY DAUGHTER", message_id="crit_01", corpus="CRIT")],
        [0.1], n_rep=2, confusion=_pooled(), seed=1,
    )
    assert (df.message_id == "crit_01").all()
    assert (df.corpus == "CRIT").all()


def test_different_replicate_idx_gives_different_stable_seed():
    df = build_exposure_v2([_msg("I AM IN PAIN AND NEED HELP")], [0.2], n_rep=20,
                            confusion=_pooled(), seed=7)
    assert df.seed.nunique() == 20  # each replicate derives its own sub-seed


# --- Metadata: negation / numeral / content-word flags -----------------------

def test_full_corruption_sets_negation_and_numeral_flags():
    # cer_target=1.0 with the default ins_del_rate=0.1 saturates the whole
    # [0, 1) draw range for every in-grid character (indel_budget=0.1 +
    # sub_budget=0.9 == 1.0), so EVERY character position in this all-in-grid
    # message (letters, digit 5, spaces) is touched by a sub/ins/del on every
    # replicate, deterministically covering the "NOT" and "5" tokens.
    text = "I DO NOT WANT 5 PILLS"
    df = build_exposure_v2([_msg(text, message_id="neg1")], [1.0], n_rep=1,
                            confusion=_pooled(), seed=3)
    row = df.iloc[0]
    # `.iloc[0]` boxes column values into a mixed-dtype row Series, which
    # turns a plain Python bool into numpy.bool_ (== True but not `is True`);
    # assert truthiness, not identity.
    assert row.corrupted_negation
    assert row.corrupted_numeral
    assert row.corrupted_content_words >= 2   # at least WANT and PILLS


def test_zero_cer_target_sets_no_corruption_flags():
    text = "I DO NOT WANT 5 PILLS"
    df = build_exposure_v2([_msg(text, message_id="neg0")], [0.0], n_rep=1,
                            confusion=_pooled(), seed=3)
    row = df.iloc[0]
    assert not row.corrupted_negation
    assert not row.corrupted_numeral
    assert row.corrupted_content_words == 0
    assert row.error_positions == []
    assert row.n_errors == 0


def test_out_of_grid_char_passthrough_is_counted():
    # A comma is not on the 36-cell grid alphabet; at cer_target=0 nothing is
    # corrupted at all (identity path), so the comma passes straight through
    # -- and must be counted, quantifying the reviewer-flagged silent
    # pass-through limitation rather than hiding it.
    text = "HELLO, WORLD"
    df = build_exposure_v2([_msg(text, message_id="oog1")], [0.0], n_rep=1,
                            confusion=_pooled(), seed=5)
    row = df.iloc[0]
    assert row.noisy_text == text
    assert row.n_passthrough_chars == 1  # just the comma
