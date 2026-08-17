"""CRIT/CTRL matching balance table: SMD before vs. after matching
(revision Task B4).

Reviewers flagged that "matched controls" was asserted -- `idrift.data.
matched_controls.build_controls` greedily nearest-neighbor matches one
non-critical CONTROL (CTRL) per message-critical (CRIT) probe item on
[char_len, word_count, has_numeral, has_negation, mean_word_freq] -- but no
balance diagnostic was shown proving the match actually equated CRIT and
CTRL on those covariates, only after the fact. `matched_controls.
balance_table` already reports an AFTER-matching-only table (one row per
covariate, `crit_mean`/`ctrl_mean`/`smd`) from its own `build_controls`
output. This module adds the BEFORE side reviewers actually asked for --
what the imbalance looks like in the raw candidate pools with no matching
applied -- next to that same after-matching number, in one tidy,
Love-plot-ready table, so both are visible together.

Standardized mean difference (SMD)
------------------------------------
For a covariate x, SMD = (mean(x_crit) - mean(x_ctrl)) / pooled_sd, pooled_sd
= sqrt((var(x_crit) + var(x_ctrl)) / 2) -- population variance (ddof=0),
the same convention `idrift.data.matched_controls._smd` already uses
(Austin, 2009). A covariate with pooled_sd == 0 (constant in both groups,
e.g. a fixture where every candidate shares one flag) has SMD defined as
0.0 -- a value that never varies cannot be imbalanced -- rather than
dividing by zero or propagating NaN. By convention |SMD| < 0.1 is
considered good balance and |SMD| < 0.25 acceptable (Austin, 2009;
Normand et al., 2001).

BEFORE vs. AFTER
------------------
- BEFORE: `crit_frame` (every CRIT item's own covariate values) vs.
  `ctrl_frame` (every candidate in the CTRL pool BEFORE any matching was
  applied -- e.g. `idrift.data.matched_controls._load_ctrl_candidate_pool`'s
  output for the real Step-5 run). This is the imbalance that would exist
  if CRIT were simply contrasted against the authentic corpus at large,
  with no matching -- the reviewer-flagged confound.
- AFTER: `matched_pairs` (one row per REALIZED CRIT-CTRL match, in the wide
  shape `idrift.data.matched_controls.build_controls` returns: `crit_<cov>`
  for the CRIT item's own value, bare `<cov>` for its matched CTRL
  partner's own value). This is the actual matched design the downstream
  CRIT-vs-CTRL contrast (`idrift.analysis.matched_compare`) runs on.

Do not fabricate: `ctrl_frame` and `matched_pairs` are each optional
(`ctrl_frame` may be `None` or empty; `matched_pairs` defaults to `None`).
Whichever side is unavailable is reported as `NaN` -- never guessed at,
never silently backfilled from the other side -- with a `note` explaining
why. Availability of one side never blanks the other: a caller who only
has the realized matches (no recoverable before-pool) still gets a
complete AFTER column, with BEFORE honestly null.

`n_unmatched` ("if determinable"): the number of `ctrl_frame` candidates
NOT used as anyone's matched partner, i.e. how many of the raw candidate
pool were left over. Determinable only when both `ctrl_frame` and
`matched_pairs` carry a shared `message_id` column (the convention
`matched_controls.build_controls` already uses for the matched CTRL
partner's own id) -- otherwise `None`, not a guess.

Deterministic; American spelling; no em dashes (double hyphens for asides,
per house style).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# The five Task 1.3 matching covariates, in the order
# `idrift.data.matched_controls._FEATURE_COLUMNS` uses them.
DEFAULT_COVARIATES = ("char_len", "word_count", "has_numeral", "has_negation", "mean_word_freq")

# Columns that identify a row rather than describe a covariate -- never
# auto-inferred as a matching covariate.
_ID_LIKE_COLUMNS = {
    "message_id", "crit_message_id", "id", "pair_id",
    "text", "crit_text", "match_distance",
}


def _smd(a: pd.Series, b: pd.Series) -> float:
    """Standardized mean difference (mean_a - mean_b) / pooled_sd, pooled_sd
    = sqrt((var_a + var_b) / 2) (population variance, ddof=0). 0.0 when
    pooled_sd is 0 (guarded, not divided-by-zero or NaN-propagated)."""
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    pooled_sd = float(np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2))
    if pooled_sd == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled_sd)


def _resolve_covariates(crit_frame, ctrl_frame, matched_pairs, covariates):
    if covariates is not None:
        return list(covariates)

    preferred = [c for c in DEFAULT_COVARIATES if c in crit_frame.columns]
    if preferred:
        return preferred

    # Fall back to any crit_frame column that also appears somewhere else
    # (ctrl_frame directly, or matched_pairs under its own bare name) --
    # only relevant for non-standard covariate names.
    fallback = []
    for c in crit_frame.columns:
        if c in _ID_LIKE_COLUMNS:
            continue
        in_ctrl = ctrl_frame is not None and c in ctrl_frame.columns
        in_matched = matched_pairs is not None and c in matched_pairs.columns
        if in_ctrl or in_matched:
            fallback.append(c)
    return fallback


def _n_unmatched(ctrl_frame, matched_pairs):
    """Number of `ctrl_frame` candidates not used as a matched partner, or
    `None` if not determinable (no shared `message_id` id column)."""
    if ctrl_frame is None or len(ctrl_frame) == 0 or matched_pairs is None or len(matched_pairs) == 0:
        return None
    if "message_id" not in ctrl_frame.columns or "message_id" not in matched_pairs.columns:
        return None
    ctrl_ids = set(ctrl_frame["message_id"])
    matched_ids = set(matched_pairs["message_id"])
    return int(len(ctrl_ids - matched_ids))


def balance_table(
    crit_frame: pd.DataFrame,
    ctrl_frame: pd.DataFrame | None,
    matched_pairs: pd.DataFrame | None = None,
    covariates=None,
) -> pd.DataFrame:
    """Build the CRIT/CTRL SMD balance table, before and after matching.

    Args:
        crit_frame: one row per CRIT item, with a column per matching
            covariate (e.g. `char_len`, `word_count`, ...).
        ctrl_frame: one row per candidate in the CTRL pool BEFORE matching
            (the full pool `idrift.data.matched_controls.build_controls`
            drew its matches from), with the same covariate columns.
            `None` or empty if that pool is not recoverable -- `smd_before`
            is then left null (`NaN`), never fabricated.
        matched_pairs: one row per REALIZED CRIT-CTRL match, in the wide
            shape `idrift.data.matched_controls.build_controls` returns:
            `crit_<covariate>` for the CRIT item's own value, bare
            `<covariate>` for its matched CTRL partner's own value.
            Defaults to `None` (before-only table) -- `smd_after` is then
            left null, never fabricated.
        covariates: which covariate names to report. Defaults to whichever
            of `DEFAULT_COVARIATES` are present in `crit_frame`.

    Returns:
        DataFrame, one row per covariate, columns: `covariate`,
        `smd_before`, `smd_after`, `note` (explains any null value; empty
        string when both sides were computed), `max_abs_smd_after` (the
        largest |SMD| across all covariates' after-matching values, `NaN`
        if none were computable; broadcast to every row), `n_unmatched`
        (see `_n_unmatched`; broadcast to every row, `None` if not
        determinable). Love-plot ready (one covariate per row, before/after
        SMD as the two series to plot).

    Raises:
        ValueError: if `covariates` is not given and no covariate column
            could be resolved from `crit_frame`/`ctrl_frame`/`matched_pairs`.
    """
    covariates = _resolve_covariates(crit_frame, ctrl_frame, matched_pairs, covariates)
    if not covariates:
        raise ValueError(
            "balance_table: no covariate columns could be resolved from crit_frame/"
            "ctrl_frame/matched_pairs; pass covariates= explicitly"
        )

    before_frame_available = ctrl_frame is not None and len(ctrl_frame) > 0
    before_frame_reason = None if before_frame_available else (
        "ctrl_frame (before-matching CTRL candidate pool) not supplied or empty; "
        "smd_before left null, not fabricated"
    )

    after_frame_available = matched_pairs is not None and len(matched_pairs) > 0
    after_frame_reason = None if after_frame_available else (
        "matched_pairs not supplied or empty; smd_after left null, not fabricated"
    )

    rows = []
    for cov in covariates:
        note_parts = []

        if before_frame_reason is not None:
            smd_before = float("nan")
            note_parts.append(before_frame_reason)
        elif cov not in crit_frame.columns or cov not in ctrl_frame.columns:
            smd_before = float("nan")
            note_parts.append(
                f"covariate '{cov}' missing from crit_frame or ctrl_frame; smd_before left null"
            )
        else:
            smd_before = _smd(crit_frame[cov], ctrl_frame[cov])

        crit_col = f"crit_{cov}"
        if after_frame_reason is not None:
            smd_after = float("nan")
            note_parts.append(after_frame_reason)
        elif crit_col not in matched_pairs.columns or cov not in matched_pairs.columns:
            smd_after = float("nan")
            note_parts.append(
                f"matched_pairs is missing '{crit_col}' or '{cov}'; smd_after left null"
            )
        else:
            smd_after = _smd(matched_pairs[crit_col], matched_pairs[cov])

        rows.append({
            "covariate": cov,
            "smd_before": smd_before,
            "smd_after": smd_after,
            "note": "; ".join(note_parts),
        })

    after_values = [r["smd_after"] for r in rows]
    finite_after = [abs(v) for v in after_values if not math.isnan(v)]
    max_abs_smd_after = float(max(finite_after)) if finite_after else float("nan")

    n_unmatched = _n_unmatched(ctrl_frame, matched_pairs)

    for r in rows:
        r["max_abs_smd_after"] = max_abs_smd_after
        r["n_unmatched"] = n_unmatched

    return pd.DataFrame(
        rows,
        columns=["covariate", "smd_before", "smd_after", "note", "max_abs_smd_after", "n_unmatched"],
    )


# ---------------------------------------------------------------------------
# Real-data runner: writes output/balance_digest.json.
# ---------------------------------------------------------------------------


def _load_real_frames():
    """Load the real Step-5 artifacts `balance_table` needs: the full CRIT
    probe set, the full CTRL candidate pool as it stood BEFORE matching, and
    the realized matched pairs -- reusing `idrift.data.matched_controls`'s
    own loaders/feature computation rather than re-deriving them, so this
    digest can never silently disagree with what `matched_controls.main()`
    actually matched on.
    """
    from idrift.data.corpus_sample import _text_column
    from idrift.data.matched_controls import (
        _load_crit_df,
        _load_ctrl_candidate_pool,
        compute_features,
    )
    from idrift.lib.io_utils import _dir

    crit_df = _load_crit_df()
    before_pool = _load_ctrl_candidate_pool(crit_df)
    matched_pairs = pd.read_csv(_dir() / "ctrl_matched.csv")

    crit_frame = compute_features(crit_df[_text_column(crit_df)])
    crit_frame["message_id"] = crit_df["message_id"].to_numpy()

    ctrl_frame = compute_features(before_pool[_text_column(before_pool)])
    ctrl_frame["message_id"] = before_pool["message_id"].to_numpy()

    return crit_frame, ctrl_frame, matched_pairs


def main() -> pd.DataFrame:
    """Build `output/balance_digest.json`: the CRIT/CTRL SMD balance table
    (before vs. after matching) on the real Step-5 artifacts, printing the
    after-matching SMDs.
    """
    import json

    from idrift.analysis.digests import _default, _out_dir

    crit_frame, ctrl_frame, matched_pairs = _load_real_frames()
    table = balance_table(crit_frame, ctrl_frame, matched_pairs)

    max_abs_smd_after = float(table["max_abs_smd_after"].iloc[0])
    n_unmatched = table["n_unmatched"].iloc[0]
    n_unmatched = None if n_unmatched is None or (isinstance(n_unmatched, float) and math.isnan(n_unmatched)) else int(n_unmatched)

    digest = {
        "balance_table": table.to_dict(orient="records"),
        "max_abs_smd_after": max_abs_smd_after,
        "n_unmatched": n_unmatched,
        "n_crit": int(len(crit_frame)),
        "n_ctrl_candidate_pool_before_matching": int(len(ctrl_frame)),
        "n_matched_pairs": int(len(matched_pairs)),
        "notes": (
            "Before-matching pool IS recoverable for this real run: "
            "idrift.data.matched_controls._load_ctrl_candidate_pool(crit_df) "
            "reconstructs the exact CTRL candidate pool build_controls matched "
            "from (output/intermediate/corpus_costello.parquet, minus the Task "
            "1.1 auth_sample.csv texts and the CRIT items' own texts), so "
            "smd_before reflects the real, non-fabricated pre-matching pool -- "
            "not a null placeholder. smd_after uses the realized matched pairs "
            "in output/intermediate/ctrl_matched.csv. |SMD| < 0.1 = good "
            "balance, < 0.25 = acceptable (Austin, 2009)."
        ),
    }

    out_path = _out_dir() / "balance_digest.json"
    out_path.write_text(json.dumps(digest, indent=2, default=_default, sort_keys=True))

    print(f"wrote {out_path}")
    print(table[["covariate", "smd_before", "smd_after"]].to_string(index=False))
    print(f"max|SMD| after = {max_abs_smd_after:.4f}; n_unmatched = {n_unmatched}")

    return table


if __name__ == "__main__":
    main()
