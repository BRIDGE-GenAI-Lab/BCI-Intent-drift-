"""Matched non-critical control set (CTRL; revision Task 1.3).

Reviewers rejected the original "message-critical vs overall" comparison as
confounded: the message-critical (CRIT) probe set is systematically shorter
and more negation-/numeral-heavy than the authentic corpus at large, so any
CRIT-vs-AUTH difference in drift could just be those nuisance features, not
criticality itself (message-critical is an input attribute of the message,
not an outcome class -- see docs/preregistration.md). This module builds one
matched non-critical CONTROL per CRIT item, drawn from the authentic pool via
greedy nearest-neighbor matching (without replacement) on standardized
[char_len, word_count, has_numeral, has_negation, mean_word_freq], so the
CRIT-vs-CTRL contrast holds those nuisance features approximately constant
and the resulting balance table (`ctrl_balance.csv`) can show it worked.

`has_numeral`/`has_negation` are defined identically to
`idrift.data.exposure_v2`'s post-hoc corruption flags (same token regex, same
negation word list) so a message's criticality-matching features and its
corruption-flag features never silently disagree on what counts as a numeral
or a negation.

Word frequency: `mean_word_freq` is the mean `wordfreq.zipf_frequency` (base
-10 log-scaled frequency per billion words, the library's standard unit)
across a message's tokens, using the `wordfreq` package's "best" (small+large
combined) English wordlist -- not a fabricated or hand-rolled frequency
table. A token absent from the wordlist gets `zipf_frequency` 0.0 (the
library's documented behavior for out-of-vocabulary words), which is a
legitimate low-frequency value for this purpose, not an error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import wordfreq

from idrift.data.corpus_sample import _text_column
from idrift.data.exposure_v2 import _NEGATION_WORDS, _TOKEN_RE

# The five standardized matching features, in the order the brief specifies.
_FEATURE_COLUMNS = ["char_len", "word_count", "has_numeral", "has_negation", "mean_word_freq"]


def _mean_word_freq(tokens: list[str]) -> float:
    """Mean `wordfreq.zipf_frequency` across `tokens` (English, "best"
    wordlist). A message with no tokens (empty/punctuation-only text) has no
    words to average, so it gets 0.0 -- the same value an all-out-of-
    vocabulary message would get, which is the correct floor for "no
    common-English-word content" either way."""
    if not tokens:
        return 0.0
    return float(np.mean([wordfreq.zipf_frequency(tok.lower(), "en") for tok in tokens]))


def _has_numeral(tokens: list[str]) -> bool:
    """True if any token contains a digit character. Mirrors
    `idrift.data.exposure_v2._corruption_flags`'s numeral test exactly
    (`any(c.isdigit() for c in token)`), applied to the whole message instead
    of just the corrupted positions."""
    return any(any(c.isdigit() for c in tok) for tok in tokens)


def _has_negation(tokens: list[str]) -> bool:
    """True if any token is a known negation word or an "n't" contraction.
    Mirrors `idrift.data.exposure_v2._corruption_flags`'s negation test
    exactly (same `_NEGATION_WORDS` set, same "n't"-substring fallback for
    novel contractions), applied to the whole message."""
    for tok in tokens:
        low = tok.lower()
        if low in _NEGATION_WORDS or "n't" in low:
            return True
    return False


def _features_for_text(text: str) -> dict:
    s = str(text)
    tokens = _TOKEN_RE.findall(s)
    return {
        "char_len": float(len(s)),
        "word_count": float(len(tokens)),
        "has_numeral": _has_numeral(tokens),
        "has_negation": _has_negation(tokens),
        "mean_word_freq": _mean_word_freq(tokens),
    }


def compute_features(texts: pd.Series) -> pd.DataFrame:
    """Compute the five matching features for every text in `texts`,
    preserving `texts`'s index."""
    return pd.DataFrame([_features_for_text(t) for t in texts], index=texts.index)


def build_controls(crit_df: pd.DataFrame, auth_pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Greedy nearest-neighbor match one non-critical control per CRIT item.

    For each row of `crit_df` (in the order given), selects the still-unused
    row of `auth_pool` with the smallest standardized Euclidean distance on
    `_FEATURE_COLUMNS` ([char_len, word_count, has_numeral, has_negation,
    mean_word_freq]), then removes that pool row from further consideration
    ("without replacement" -- a pool message matched to one CRIT item can
    never also be matched to another).

    Standardization: features are z-scored using the mean/SD of the POOLED
    CRIT + auth_pool candidate set (not the pool alone), the convention used
    in the matching literature for a caliper/distance shared by both groups
    being matched (Austin, 2011, "Optimal caliper widths..."). A feature
    that is constant across the pooled set (SD == 0, e.g. a tiny test
    fixture where every candidate shares one flag) is left un-standardized
    rather than divided by zero -- since it is constant it already
    contributes nothing to distance after mean-centering, so this changes no
    match.

    Ties (identical standardized distance to more than one still-unused pool
    row) are broken by a seeded random permutation of the pool's row order,
    computed once from `seed` -- so the same (crit_df, auth_pool, seed)
    always resolves ties the same way, and only ties are seed-sensitive.

    Args:
        crit_df: one row per CRIT item, with `message_id` and a `text` (or
            `intended_text`) column.
        auth_pool: candidate control messages, with `message_id` and a
            `text` (or `intended_text`) column. Must have at least as many
            rows as `crit_df` (one control per CRIT item, without
            replacement). Callers are responsible for pre-excluding any
            message that should never be eligible as a control (e.g. a
            message already drawn into the AUTH sample, or a CRIT item's own
            text) -- this function treats every `auth_pool` row as an
            eligible candidate.
        seed: seed for the tie-breaking random generator.

    Returns:
        DataFrame, one row per CRIT item, columns: `crit_message_id`,
        `crit_text`, `crit_char_len`, `crit_word_count`, `crit_has_numeral`,
        `crit_has_negation`, `crit_mean_word_freq`, `message_id` (the
        matched control's own id), `text` (the matched control's text),
        `char_len`, `word_count`, `has_numeral`, `has_negation`,
        `mean_word_freq` (the matched control's own feature values), and
        `match_distance` (the standardized Euclidean distance actually
        realized for that match).
    """
    crit_df = crit_df.reset_index(drop=True)
    auth_pool = auth_pool.reset_index(drop=True)

    if len(auth_pool) < len(crit_df):
        raise ValueError(
            f"auth_pool has only {len(auth_pool)} candidates, fewer than "
            f"{len(crit_df)} CRIT items to match -- one control per CRIT item "
            "without replacement is not possible"
        )

    crit_text_col = _text_column(crit_df)
    pool_text_col = _text_column(auth_pool)

    crit_feat = compute_features(crit_df[crit_text_col])
    pool_feat = compute_features(auth_pool[pool_text_col])

    combined = pd.concat(
        [crit_feat[_FEATURE_COLUMNS], pool_feat[_FEATURE_COLUMNS]], ignore_index=True
    ).astype(float)
    means = combined.mean(axis=0)
    stds = combined.std(axis=0, ddof=0).replace(0.0, 1.0)

    crit_z = ((crit_feat[_FEATURE_COLUMNS].astype(float) - means) / stds).to_numpy()
    pool_z = ((pool_feat[_FEATURE_COLUMNS].astype(float) - means) / stds).to_numpy()

    rng = np.random.default_rng(seed)
    tiebreak = rng.permutation(len(auth_pool))

    used = np.zeros(len(auth_pool), dtype=bool)
    records = []
    for i in range(len(crit_df)):
        diffs = pool_z - crit_z[i]
        dist_sq = np.einsum("ij,ij->i", diffs, diffs)
        dist_masked = np.where(used, np.inf, dist_sq)
        # Primary key = distance (ascending); secondary key = the seeded
        # tiebreak permutation, so exact ties are resolved deterministically
        # by seed rather than by array/insertion order.
        order = np.lexsort((tiebreak, dist_masked))
        best_idx = int(order[0])
        if not np.isfinite(dist_masked[best_idx]):
            raise ValueError(
                "auth_pool exhausted before matching all CRIT items "
                "(this should be unreachable given the length check above)"
            )
        used[best_idx] = True

        crit_row = crit_df.iloc[i]
        pool_row = auth_pool.iloc[best_idx]
        cf = crit_feat.iloc[i]
        pf = pool_feat.iloc[best_idx]

        records.append({
            "crit_message_id": crit_row["message_id"],
            "crit_text": crit_row[crit_text_col],
            "crit_char_len": cf["char_len"],
            "crit_word_count": cf["word_count"],
            "crit_has_numeral": bool(cf["has_numeral"]),
            "crit_has_negation": bool(cf["has_negation"]),
            "crit_mean_word_freq": cf["mean_word_freq"],
            "message_id": pool_row["message_id"],
            "text": pool_row[pool_text_col],
            "char_len": pf["char_len"],
            "word_count": pf["word_count"],
            "has_numeral": bool(pf["has_numeral"]),
            "has_negation": bool(pf["has_negation"]),
            "mean_word_freq": pf["mean_word_freq"],
            "match_distance": float(np.sqrt(dist_masked[best_idx])),
        })

    return pd.DataFrame(records)


def _smd(crit_values: pd.Series, ctrl_values: pd.Series) -> float:
    """Standardized mean difference (CRIT - CTRL) / pooled SD, the standard
    matching-balance statistic (Austin, 2009): pooled SD is
    sqrt((var_crit + var_ctrl) / 2), not a shared/combined-sample SD, so
    the SMD reflects the two matched groups' own spread. A feature constant
    in both groups (pooled SD == 0) has SMD defined as 0.0 (no imbalance
    possible on a value that never varies)."""
    crit_values = crit_values.astype(float)
    ctrl_values = ctrl_values.astype(float)
    pooled_sd = float(np.sqrt((crit_values.var(ddof=0) + ctrl_values.var(ddof=0)) / 2))
    if pooled_sd == 0.0:
        return 0.0
    return float((crit_values.mean() - ctrl_values.mean()) / pooled_sd)


def balance_table(matched: pd.DataFrame) -> pd.DataFrame:
    """Build the CRIT-vs-CTRL standardized-mean-difference balance table
    from a `build_controls` output, one row per matching feature.

    Columns: `feature`, `crit_mean`, `ctrl_mean`, `smd`. SMDs near 0 indicate
    the matching succeeded in equating CRIT and CTRL on that feature; by
    convention |SMD| < 0.1 is considered good balance and |SMD| < 0.25
    acceptable (Austin, 2009; Normand et al., 2001).
    """
    rows = []
    for feat in _FEATURE_COLUMNS:
        crit_col = matched[f"crit_{feat}"]
        ctrl_col = matched[feat]
        rows.append({
            "feature": feat,
            "crit_mean": float(crit_col.astype(float).mean()),
            "ctrl_mean": float(ctrl_col.astype(float).mean()),
            "smd": _smd(crit_col, ctrl_col),
        })
    return pd.DataFrame(rows)


# --- Step 5 runner -----------------------------------------------------------

def _load_crit_df() -> pd.DataFrame:
    """Load the frozen message-critical probe set
    (`output/intermediate/probe_set.json`) as a `message_id`/`text` frame."""
    import json

    from idrift.lib.io_utils import _dir

    items = json.loads((_dir() / "probe_set.json").read_text())
    return pd.DataFrame(
        [{"message_id": it["message_id"], "text": it["intended_text"]} for it in items]
    )


def _load_ctrl_candidate_pool(crit_df: pd.DataFrame) -> pd.DataFrame:
    """Build the real-data CTRL candidate pool: every
    `output/intermediate/corpus_costello.parquet` phrase EXCEPT those already
    drawn into the Task 1.1 AUTH sample (`auth_sample.csv`) or that share text
    with a CRIT item.

    Judgment call: the brief's `build_controls(crit_df, auth_pool, seed)`
    interface does not itself enforce disjointness from AUTH or CRIT (a
    generic candidate pool is not this function's job -- see its
    docstring), so this loader does it for the real Step 5 run. Excluding
    already-AUTH-sampled phrases keeps AUTH and CTRL as disjoint message
    arms in the downstream exposure pipeline (a message is never
    simultaneously the "authentic corpus" arm and the "matched control"
    arm); excluding CRIT-text phrases prevents a CTRL row from being
    text-identical to the very CRIT item it is meant to contrast against.
    """
    from idrift.lib.io_utils import _dir

    corpus = pd.read_parquet(_dir() / "corpus_costello.parquet")
    auth_sample = pd.read_csv(_dir() / "auth_sample.csv")

    excluded_text = set(auth_sample["text"]) | set(crit_df["text"])
    pool = corpus[~corpus["intended_text"].isin(excluded_text)].reset_index(drop=True)
    return pool


def main(seed: int = 0) -> pd.DataFrame:
    """Build `output/intermediate/ctrl_matched.csv` (one matched non-critical
    control per real CRIT item) and `output/intermediate/ctrl_balance.csv`
    (the CRIT-vs-CTRL standardized-mean-difference balance table), printing
    the balance table."""
    from idrift.lib.io_utils import _dir, log_provenance

    crit_df = _load_crit_df()
    pool = _load_ctrl_candidate_pool(crit_df)

    matched = build_controls(crit_df, pool, seed=seed)

    out_dir = _dir()
    ctrl_path = out_dir / "ctrl_matched.csv"
    matched.to_csv(ctrl_path, index=False)

    balance = balance_table(matched)
    balance_path = out_dir / "ctrl_balance.csv"
    balance.to_csv(balance_path, index=False)

    log_provenance({
        "ctrl_matched": {
            "rows": int(len(matched)),
            "n_crit": int(len(crit_df)),
            "candidate_pool_size": int(len(pool)),
            "seed": seed,
            "max_abs_smd": float(balance["smd"].abs().max()),
            "balance": balance.set_index("feature")["smd"].round(4).to_dict(),
        }
    })

    print(f"wrote {ctrl_path} rows={len(matched)}")
    print(f"wrote {balance_path}")
    print(balance.to_string(index=False))
    return matched


if __name__ == "__main__":
    main()
