import numpy as np
import pandas as pd

from idrift.data.corpus_sample import power_for_slope, sample_authentic


def _synth_corpus(n_rows: int, n_categories: int = 5) -> pd.DataFrame:
    """Build a synthetic corpus spanning `n_categories` categories with
    uneven (not equal) sizes, so proportional (not equal) stratified
    allocation is actually exercised. Each row's text length cycles through
    three deterministic tiers (short/medium/long) so the corpus spans all
    three length terciles once `sample_authentic` computes them, and every
    text is unique so no duplicate-text guard is trivially satisfied."""
    categories = [f"cat{i}" for i in range(n_categories)]
    weights = np.array([3, 2, 2, 1, 1][:n_categories], dtype=float)
    weights = weights / weights.sum()
    counts = np.round(weights * n_rows).astype(int)
    counts[-1] += n_rows - counts.sum()  # fix rounding drift to hit exactly n_rows

    # short / medium / long character-length ranges; varied within each tier
    # (not a single repeated value) so the terciles computed by
    # `sample_authentic` land on three genuinely distinct bins instead of
    # colliding at tied quantile edges.
    tier_ranges = [(5, 15), (25, 45), (60, 100)]
    rows = []
    idx = 0
    for cat, cnt in zip(categories, counts):
        for i in range(cnt):
            lo, hi = tier_ranges[i % 3]
            tier_len = lo + (idx % (hi - lo))
            text = f"{cat}_{i:03d}_" + ("x" * tier_len)
            rows.append({"message_id": f"msg_{idx:04d}", "text": text, "category": cat})
            idx += 1
    return pd.DataFrame(rows)


def test_stratified_sample_is_proportional_and_deterministic():
    df = _synth_corpus(200)  # helper in the test
    s1 = sample_authentic(df, n=60, seed=0)
    s2 = sample_authentic(df, n=60, seed=0)
    assert len(s1) == 60 and s1.message_id.is_unique
    assert list(s1.message_id) == list(s2.message_id)
    # proportional across categories within +-1
    for cat, g in df.groupby("category"):
        exp = round(60 * len(g) / len(df))
        assert abs((s1.category == cat).sum() - exp) <= 1


def test_sample_has_expected_columns_and_length_bins():
    df = _synth_corpus(200)
    s = sample_authentic(df, n=60, seed=0)
    assert {"message_id", "text", "category", "length_bin"}.issubset(s.columns)
    assert set(s.length_bin.unique()).issubset({"short", "medium", "long"})
    # every one of the three tiers should be represented at n=60/200
    assert set(s.length_bin.unique()) == {"short", "medium", "long"}


def test_no_duplicate_texts_even_when_corpus_has_cross_category_duplicates():
    df = _synth_corpus(200)
    # inject a cross-category duplicate text, as the real Costello corpus has
    # (the same phrase legitimately filed under two categories)
    dup_row = df.iloc[0].copy()
    dup_row["message_id"] = "msg_dup"
    dup_row["category"] = df.iloc[-1]["category"]
    df = pd.concat([df, pd.DataFrame([dup_row])], ignore_index=True)

    s = sample_authentic(df, n=60, seed=0)
    assert s.text.is_unique


def test_different_seeds_can_give_different_samples():
    df = _synth_corpus(200)
    s1 = sample_authentic(df, n=60, seed=0)
    s2 = sample_authentic(df, n=60, seed=1)
    assert list(s1.message_id) != list(s2.message_id)


def test_power_monotone_increasing_with_n_msg():
    # holding effect/n_rep/alpha fixed, more messages (more independent
    # clusters) should never yield lower power than fewer messages.
    small = power_for_slope(effect=1.6, n_msg=20, n_rep=10)
    large = power_for_slope(effect=1.6, n_msg=200, n_rep=10)
    assert large >= small


def test_power_is_a_probability():
    p = power_for_slope(effect=1.6, n_msg=50, n_rep=10)
    assert 0.0 <= p <= 1.0
