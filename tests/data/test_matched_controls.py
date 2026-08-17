"""Tests for the matched non-critical control set (CTRL; rev Task 1.3).

Reviewers rejected the original "critical vs overall" comparison as
confounded by length/negation/numerals: message-critical (CRIT) probe items
are systematically shorter/more negation-and-numeral-heavy than the
authentic corpus at large, so any CRIT-vs-AUTH difference could just be
those nuisance features, not criticality. This module instead builds one
matched non-critical CONTROL per CRIT item -- drawn from the authentic pool,
nearest-neighbor matched (without replacement) on standardized
[char_len, word_count, has_numeral, has_negation, mean_word_freq] -- so the
CRIT-vs-CTRL comparison holds those nuisance features approximately constant.
"""
import pandas as pd

from idrift.data.matched_controls import build_controls

# --- Step 1: the brief's failing test, verbatim (5 CRIT + 50-message pool) --

# Five CRIT items spanning the four combinations of has_numeral/has_negation
# (two share the "neither" combination), each with a deliberately-planted
# "good match" pool candidate below (near-identical char_len/word_count and
# an EXACT has_numeral/has_negation match), so the nearest-neighbor
# assignment is unambiguous and the test does not depend on tie-breaking.
_CRIT_ROWS = [
    {"message_id": "probe_0000", "text": "I need help right now"},        # no numeral, no negation
    {"message_id": "probe_0001", "text": "Call my daughter please"},      # no numeral, no negation
    {"message_id": "probe_0002", "text": "I do not want 5 pills"},        # numeral + negation
    {"message_id": "probe_0003", "text": "Turn on channel 7 now"},        # numeral, no negation
    {"message_id": "probe_0004", "text": "Never leave me alone here"},    # no numeral, negation
]

# One deliberately close match per CRIT item above (same order).
_GOOD_MATCHES = [
    {"message_id": "pool_good_0", "text": "Please help me out now"},
    {"message_id": "pool_good_1", "text": "Call my brother please"},
    {"message_id": "pool_good_2", "text": "I do not need 3 shots"},
    {"message_id": "pool_good_3", "text": "Switch to channel 9 now"},
    {"message_id": "pool_good_4", "text": "Never call me again today"},
]


def _filler_rows(n_short: int = 20, n_long: int = 25) -> list[dict]:
    """Filler pool candidates deliberately far outside the CRIT items'
    char_len range (~21-26 chars) in either direction -- a 2-char message and
    an 80+-char message -- so they never out-compete the planted
    `_GOOD_MATCHES` on standardized char_len/word_count distance."""
    rows = []
    for i in range(n_short):
        rows.append({"message_id": f"pool_short_{i}", "text": "Hi"})  # 2 chars, 1 word
    for i in range(n_long):
        # ~85+ chars, 15+ words: far above every CRIT item's length/word count
        text = ("This is a very long filler sentence padded out with extra words "
                f"so it is not a close match number {i}")
        rows.append({"message_id": f"pool_long_{i}", "text": text})
    return rows


def _crit_df() -> pd.DataFrame:
    return pd.DataFrame(_CRIT_ROWS)


def _pool_df() -> pd.DataFrame:
    rows = list(_GOOD_MATCHES) + _filler_rows()
    assert len(rows) == 50
    return pd.DataFrame(rows)


def test_five_crit_and_fifty_pool_gives_five_distinct_within_tolerance():
    crit_df = _crit_df()
    pool_df = _pool_df()

    out = build_controls(crit_df, pool_df, seed=0)

    assert len(out) == 5
    assert out["message_id"].is_unique  # controls all distinct
    assert set(out["crit_message_id"]) == set(crit_df["message_id"])

    for row in out.itertuples():
        crit_len = len(crit_df.loc[crit_df.message_id == row.crit_message_id, "text"].iloc[0])
        # within +-20% char length of its CRIT item
        assert abs(len(row.text) - crit_len) <= 0.20 * crit_len
        # agree on has_numeral/has_negation where a match exists (it does,
        # for every CRIT item in this fixture, by construction)
        assert row.has_numeral == row.crit_has_numeral
        assert row.has_negation == row.crit_has_negation

    # deterministic by seed
    out_again = build_controls(crit_df, pool_df, seed=0)
    pd.testing.assert_frame_equal(out, out_again)


def test_controls_are_the_planted_near_matches():
    """With the fixture's near-exact planted matches, greedy nearest-
    neighbor matching should select them over the far-off fillers."""
    out = build_controls(_crit_df(), _pool_df(), seed=0)
    got = dict(zip(out["crit_message_id"], out["message_id"]))
    want = {c["message_id"]: g["message_id"] for c, g in zip(_CRIT_ROWS, _GOOD_MATCHES)}
    assert got == want


def test_without_replacement_no_pool_message_used_twice():
    out = build_controls(_crit_df(), _pool_df(), seed=0)
    assert out["message_id"].nunique() == len(out)


def test_intended_text_column_is_also_supported():
    """crit_df/auth_pool built from the real corpus (probe_set.json,
    corpus_costello.parquet) use `intended_text`, not `text`."""
    crit_df = _crit_df().rename(columns={"text": "intended_text"})
    pool_df = _pool_df().rename(columns={"text": "intended_text"})
    out = build_controls(crit_df, pool_df, seed=0)
    assert len(out) == 5
    assert "text" in out.columns and "crit_text" in out.columns
