"""Message-deduplicated sensitivity for the current 20-model target-CER
primary slope (reviewer major #8).

Why this module exists
-----------------------
Reviewer major #8 asked for a message-deduplication sensitivity check on
the CURRENT primary specification. eTable 8 already reports this check for
the ORIGINAL 7-model panel, but against the OLD primary exposure
(realized CER, fit by `scratchpad/harden/a1_dedup.py` via
`idrift.analysis.mixed_models.fit_multinomial`) -- a superseded analysis.
Reviewer major #3 has since made target CER (the experimentally assigned
corruption level) the primary exposure
(`idrift.analysis.target_cer_primary.fit_ordinal_model`), fit on the
current 20-model panel. This module reruns eTable 8's exact
deduplication method against that current primary model instead.

Reuse, not reimplementation
----------------------------
Both pieces are reused verbatim from already reviewer-accepted code,
matching `a1_dedup.py`'s approach of refitting the SAME model function on a
deduplicated frame rather than reimplementing the regression:

- `idrift.analysis.target_cer_primary.fit_ordinal_model` is the exact
  primary-slope fit (`idrift/analysis/target_cer_primary.py`'s `run()` is
  what produced `output/target_cer_primary_v3_digest.json`'s
  `or_per_10pp` of 2.296).
- The deduplication rule is `a1_dedup.py`'s: as established there (and
  reconfirmed here by `idrift.analysis.corpus_overlap`), CTRL's 131 message
  ids are a full SUBSET of AUTH's 2,168 (CTRL rows re-use AUTH message
  identities), while CRIT's 131 ids never appear in AUTH or CTRL. Dropping
  all CTRL-corpus rows therefore removes the double-counted message
  identities, leaving 2,299 unique messages (2,168 AUTH + 131 CRIT).
  Unlike the old 7-model panel -- where every model ran the full grid (5
  cer_target levels x 20 replicates) and dropping CTRL landed on an EXACT
  700-rows-per-message balance -- the current 20-model panel includes 3
  reduced-replicate reasoning models (gpt-oss:20b, gpt-oss:120b,
  deepseek-r1:32b run far fewer generations than the other 17 full-grid
  models), so the deduplicated panel is only NEAR-balanced: most messages
  (99.6%) land at 1,751 or 1,748 rows, with a small remainder between
  1,732 and 1,739 rows. `dedup_is_count_balanced` is reported honestly as
  `False` for this reason, with the actual row-count distribution attached
  so the manuscript claim is accurate about the current panel rather than
  reusing the old panel's exact-balance language.

Deterministic (no random draws: both fits are plain Newton-Raphson MLE on a
fixed design, identical to `target_cer_primary.fit_ordinal_model`).
American spelling; no em dashes (double hyphens for asides), per house
style.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from idrift.analysis.target_cer_primary import fit_ordinal_model

_Z95 = 1.959963984540054  # two-sided 95% normal quantile (matches target_cer_primary/target_cer_cluster)
_PP10 = 0.1


def _or_ci(beta: float, se: float) -> list[float]:
    """OR-per-10pp-scale 95% CI from a log-odds beta/se, reused verbatim
    from `idrift.analysis.target_cer_cluster._or_ci`: rescale by 0.1 (one
    10pp step) before exponentiating, valid because exp is monotonic."""
    lo = (beta - _Z95 * se) * _PP10
    hi = (beta + _Z95 * se) * _PP10
    return [float(np.exp(lo)), float(np.exp(hi))]

DEFAULT_PARQUET = "output/intermediate/attempts_v3plus_labeled.parquet"
DEFAULT_OUT = "output/target_cer_dedup_v3_digest.json"


def run(parquet_path, out_path: str) -> dict:
    """Refit the current target-CER primary ordinal slope on a message-
    deduplicated, count-balanced panel (CTRL-corpus rows dropped, since
    CTRL's message ids fully duplicate a subset of AUTH's), and write the
    result to `out_path` as JSON.

    Args:
        parquet_path: path to the labeled attempts parquet (columns
            `cer_target`, `label`, `message_id`, `model`, `corpus`), or an
            already-loaded DataFrame with those columns.
        out_path: path to write the JSON digest to.

    Returns:
        dict: `{or_per_10pp_full, or_per_10pp_deduplicated,
        beta_deduplicated, ci_deduplicated, n_obs_full,
        n_obs_deduplicated, n_unique_messages, rows_per_message_dedup,
        dedup_is_count_balanced, or_delta, notes}`.
    """
    df = parquet_path if isinstance(parquet_path, pd.DataFrame) else pd.read_parquet(parquet_path)

    full_fit = fit_ordinal_model(df)

    dedup_df = df[df["corpus"] != "CTRL"].copy()
    n_unique_messages = int(dedup_df["message_id"].nunique())
    rows_per_message = dedup_df.groupby("message_id").size()
    dedup_is_count_balanced = bool((rows_per_message == rows_per_message.iloc[0]).all())
    rows_per_message_distribution = {
        int(k): int(v) for k, v in rows_per_message.value_counts().sort_index(ascending=False).items()
    }

    dedup_fit = fit_ordinal_model(dedup_df)

    digest = {
        "or_per_10pp_full": full_fit["or_per_10pp"],
        "or_per_10pp_deduplicated": dedup_fit["or_per_10pp"],
        "or_ci_full": _or_ci(full_fit["beta"], full_fit["se"]),
        "or_ci_deduplicated": _or_ci(dedup_fit["beta"], dedup_fit["se"]),
        "beta_full": full_fit["beta"],
        "beta_deduplicated": dedup_fit["beta"],
        "ci_full": full_fit["ci"],
        "ci_deduplicated": dedup_fit["ci"],
        "n_obs_full": full_fit["n_obs"],
        "n_obs_deduplicated": dedup_fit["n_obs"],
        "n_unique_messages": n_unique_messages,
        "rows_per_message_dedup": int(rows_per_message.iloc[0]) if dedup_is_count_balanced else None,
        "dedup_is_count_balanced": dedup_is_count_balanced,
        "rows_per_message_distribution": rows_per_message_distribution,
        "or_delta": dedup_fit["or_per_10pp"] - full_fit["or_per_10pp"],
        "notes": (
            "Deduplication drops all CTRL-corpus rows (131 message ids, a full subset of "
            "AUTH's 2,168) from the current 20-model target-CER primary panel "
            "(idrift.analysis.target_cer_primary.fit_ordinal_model), leaving the 2,299 "
            "unique AUTH+CRIT messages. Mirrors eTable 8's method (a1_dedup.py), applied "
            "to the current primary exposure (target CER) and current 20-model panel, not "
            "the old realized-CER 7-model analysis eTable 8 Panel A already reports. Unlike "
            "the old 7-model panel, the current panel is only NEAR count-balanced (not "
            "exactly, see rows_per_message_distribution) because 3 of the 20 models are "
            "reduced-replicate reasoning models with far fewer generations per message than "
            "the other 17 full-grid models."
        ),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(digest, indent=2))
    return digest


if __name__ == "__main__":
    print(run(DEFAULT_PARQUET, DEFAULT_OUT))
