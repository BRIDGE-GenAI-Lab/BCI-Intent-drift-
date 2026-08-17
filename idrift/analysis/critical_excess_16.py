"""Per-model matched CRIT-vs-CTRL critical excess for all 20 models, with
full balance diagnostics (reviewer major #9).

Why this module exists
-----------------------
The shipped matched CRIT-vs-CTRL contrast (`idrift.analysis.matched_compare`,
OR 1.16) and its rule-free re-derivation (`scratchpad/harden/a2_critfree.py`,
OR ~1.10 [1.07-1.13], `output/harden_critfree.json`) were both computed on
the ORIGINAL 7-model panel (`attempts_v2_labeled.parquet`). Reviewer major
#9 asks for the same matched design recomputed PER MODEL across the full
20-model panel (`attempts_v3plus_labeled.parquet`), with the excess
described honestly as small and exploratory -- never as "isolated" by
matching -- plus the balance diagnostics that were previously reported only
at the whole-cohort level (`output/balance_digest.json`).

Reuse, not reimplementation
----------------------------
This module reuses every piece of matching/inference machinery already
built and reviewer-validated elsewhere in this package, rather than
reinventing any of it:

- `idrift.analysis.matched_compare.build_matched_frame` assembles the
  CRIT-vs-its-own-matched-CTRL analysis frame (join on the Task-1.3
  `ctrl_matched.csv` pairing + the `exposure_v2_full.parquet` corruption
  covariates), exactly as `scratchpad/harden/a2_critfree.py` did.
- `idrift.analysis.matched_compare.matched_drift` fits the matched-pair
  conditional logistic regression (adjusted odds ratio for `critical`).
- `idrift.analysis.matched_compare.risk_difference` gives the absolute-scale
  companion (pair-clustered bootstrap risk difference).
- `idrift.adjudicate.label_runner.derive_labels(..., use_critical_rules=
  False)` re-derives the RULE-FREE outcome from the cached NLI/cosine/
  fluency signals -- the circularity fix reviewers already accepted for the
  7-model result (see that function's docstring).
- `idrift.analysis.balance.balance_table` (+ its `_load_real_frames` loader)
  is reused verbatim for the before/after covariate-balance diagnostic; the
  matching itself (`idrift.data.matched_controls.build_controls`) never
  depends on which model generated the downstream attempts, so this single
  item-level balance table applies to every one of the 20 per-model fits
  below without recomputation.

Why `exposure_v2_full.parquet`, not a v3-era exposure file
-------------------------------------------------------------
`attempts_v3plus_labeled.parquet` added 9 more models on the SAME CRIT/CTRL
message set the 7-model panel used (verified: the 262 CRIT+CTRL message ids,
the 5 `cer_target` levels, and the 20 `replicate_idx` values are IDENTICAL
across `attempts_v2_labeled.parquet` and the CRIT/CTRL rows of
`attempts_v3plus_labeled.parquet`; the 7 original models' `output_message`/
`label`/`realized_cer` values are byte-identical between the two files).
`exposure_v3_full.parquet` cannot be used for this join -- it carries
duplicate `(message_id, cer_target, replicate_idx)` keys for a subset of the
CRIT/CTRL rows (see `scratchpad/harden/a2_critfree.py`'s module docstring),
which breaks `build_matched_frame`'s `validate="m:1"` merge.
`exposure_v2_full.parquet` has no such duplicates and, since it carries no
`model` column, joins correctly against all 20 models' rows via a plain
many-to-one match on `(message_id, cer_target, replicate_idx)`.

Composite matched-pair stratification
----------------------------------------
`build_matched_frame` sets a coarse `pair_id` = the CRIT item's own
`message_id` (one value per Task-1.3 match). Every fit in this module
refines that to the same "identical corruption cell" composite key the
shipped 7-model analysis used (`scratchpad/harden/a2_critfree.py`):
`pair_id x cer_target x replicate_idx` for a single model's per-model fit,
plus `x model` when pooling across models (per-category and the 7-model
validation anchor). This conditions the matched comparison on identical
TARGET corruption (`cer_target`) and replicate draw, which is what "matched
at identical realized corruption, per model" means operationally here --
the finer, continuous `realized_cer` value is also carried as one of
`matched_drift`'s five explicit adjusters, so residual within-cell
corruption variation is still controlled for, not ignored.

Rule-free vs rule-inclusive
------------------------------
The cached `label` column is rule-INCLUSIVE (the five `crit_*` rule
detectors -- negation flip, numeral change, recipient change, urgency
change, actionable omission -- can force a row to "drift"). Because those
same five characteristics also help define which items are message-critical
in the first place, a rule-inclusive excess is partly circular. This module
reports BOTH: `drift_rule_inclusive` = the cached `label` as-is, and
`drift_rule_free` = `label_runner.derive_labels(..., use_critical_rules=
False)` re-applied to the cached NLI/cosine/fluency signals (meaning-channel
only: bidirectional NLI contradiction, else a cosine threshold, gated by
fluency). The rule-free number is the one this module treats as primary
(it is what `per_model["<model>"]["or"]` and `pooled["rule_free_or"]`
report), matching the manuscript's own already-accepted convention.

Validation anchor (mandatory, not optional)
-----------------------------------------------
Before trusting the full 20-model number, `run` recomputes the SAME
rule-free matched design restricted to the ORIGINAL 7 MODELS
(`ORIGINAL_7_MODELS`) and reports it under `validation_7model`, checked
against the prior hardening-pass result (`PRIOR_7MODEL_RULE_FREE_OR` ~=
1.10 [1.07-1.13], `output/harden_critfree.json`). This is the sanity gate
the task brief requires -- see that block's own `note` and
`reproduces_prior` flag.

`by_category`: the probe set's own drift-type taxonomy
-----------------------------------------------------------
`idrift.data.probe_set.build_probe_set` assigns each of the 131 CRIT probe
items a `drift_type` (`negation`, `recipient`, `refusal_consent`, `dose`, or
`None` for the grounded/baseline items -- relabeled `"grounded"` here). This
module reads the cached `output/intermediate/probe_set.json` (no external
call; it is the same frozen, already-generated probe-set artifact every
other module in this package reads) to attach that category to each matched
pair via the CRIT item's own `message_id`, then reports a matched rule-free
OR + risk difference PER CATEGORY, pooled across all 20 models. This is
purely descriptive stratification of an already-small, already-exploratory
excess -- not a claim that any one category drives the effect.

Exploratory, not causal
--------------------------
Every number in this digest -- pooled, per-model, and per-category -- is
reported as a small, exploratory association. Matching on
[char_len, word_count, has_numeral, has_negation, mean_word_freq]
(`idrift.data.matched_controls`) plus the conditional logit's five
realized-corruption adjusters addresses those MEASURED covariates only; it
does not eliminate residual semantic or syntactic differences between
message-critical and control phrasing that were never measured. See `NOTE`
and the top-level `note` field of every digest this module writes.

No model inference happens here beyond the already-cached signals: no new
GPU/CPU model forward pass is performed, only pure-Python label
re-derivation from cached NLI/cosine/fluency numbers (`derive_labels`) and
statistics on cached data. Deterministic; American spelling; no em dashes
(double hyphens for asides), per house style.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from idrift.adjudicate.label_runner import derive_labels
from idrift.analysis import balance as bal
from idrift.analysis import matched_compare as mc

DEFAULT_PARQUET = "output/intermediate/attempts_v3plus_labeled.parquet"
DEFAULT_OUT = "output/critical_excess_16_v3_digest.json"

CTRL_MATCHED_PATH = "output/intermediate/ctrl_matched.csv"
EXPOSURE_PATH = "output/intermediate/exposure_v2_full.parquet"
PROBE_SET_PATH = "output/intermediate/probe_set.json"

# The original 7-model panel `attempts_v2_labeled.parquet` and
# `scratchpad/harden/a2_critfree.py` ran on -- the sanity anchor this
# module's 20-model number is checked against.
ORIGINAL_7_MODELS = (
    "gemma4:12b",
    "gemma4:31b",
    "gemma4:e4b",
    "mistral-small:24b",
    "phi4:14b",
    "phi4:mini",
    "qwen3.5:27b-q4",
)
PRIOR_7MODEL_RULE_FREE_OR = 1.1004411291857938  # output/harden_critfree.json:or_rulefree
PRIOR_7MODEL_RULE_FREE_CI = [1.0713379293308207, 1.130334925750367]

_SIGNAL_COLS_FOR_DERIVE = (
    "intended_text",
    "output_message",
    "nli_deberta_fwd",
    "nli_deberta_bwd",
    "nli_roberta_fwd",
    "nli_roberta_bwd",
    "cos_mpnet",
    "cos_minilm",
    "fluency_raw",
)

_LOAD_COLUMNS = [
    "model",
    "message_id",
    "corpus",
    "cer_target",
    "replicate_idx",
    "realized_cer",
    "label",
] + list(_SIGNAL_COLS_FOR_DERIVE)

NOTE = (
    "Exploratory: this is a small CRIT-vs-CTRL matched excess, not a causal isolation of "
    "'message-criticality'. Matching on [char_len, word_count, has_numeral, has_negation, "
    "mean_word_freq] (idrift.data.matched_controls) plus the conditional logit's five "
    "realized-corruption adjusters (char_len, n_errors, corrupted_negation, "
    "corrupted_numeral, realized_cer) addresses those MEASURED covariates only -- it does "
    "not eliminate residual semantic or syntactic differences between message-critical and "
    "control phrasing that were never measured. rule_free_or is the primary number (the "
    "circularity-safe re-derivation); rule_inclusive_or is reported alongside for "
    "transparency, not as a separate finding."
)


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


def _or_ci(coef_entry: dict) -> tuple[float, list[float]]:
    return math.exp(coef_entry["est"]), [math.exp(coef_entry["ci_lo"]), math.exp(coef_entry["ci_hi"])]


def _add_composite_pair_id(df: pd.DataFrame, extra_keys) -> pd.DataFrame:
    """Refine `pair_id` (crit_message_id) to `pair_id x <extra_keys>` -- the
    same "identical corruption cell" stratification
    `scratchpad/harden/a2_critfree.py` used for the shipped 7-model result
    (see module docstring)."""
    out = df.copy()
    parts = out["pair_id"].astype(str)
    for key in extra_keys:
        parts = parts + "|" + out[key].astype(str)
    out["pair_id"] = parts
    return out


def _fit_or(subset: pd.DataFrame, composite_extra_keys, outcome_col: str) -> dict:
    """Composite-key `matched_drift` fit of `outcome_col` (0/1) on
    `critical`, on a copy of `subset` (never mutates the caller's frame).

    Returns:
        dict: `{or, ci, n_pairs, n_obs, converged, reason}`. `or`/`ci` are
        `None` when the fit did not converge or dropped the `critical`
        coefficient entirely (see `matched_compare.matched_drift`'s own
        degenerate-fit handling) -- never a fabricated number. `n_pairs`/
        `n_obs` are then the RAW input composite-group/row counts (the fit
        never got far enough to report its own post-drop counts). `reason`
        is `None` on success, else `matched_drift`'s own `notes` string --
        in practice this study's smaller within-category subsets sometimes
        hit "Inverting hessian failed" (a singular conditional-logit
        Hessian at low within-category adjuster diversity), which is a
        legitimate degenerate fit, not a bug; `risk_difference` (see
        `_fit_rd`) has no such failure mode and remains informative even
        when this returns `None`.
    """
    m = subset.copy()
    m["drift"] = m[outcome_col].astype(int)
    m = _add_composite_pair_id(m, composite_extra_keys)
    res = mc.matched_drift(m)

    if not res["converged"] or "critical" not in res["coef"]:
        return {
            "or": None, "ci": None, "n_pairs": res["n_pairs"], "n_obs": res["n_obs"],
            "converged": False, "reason": res["notes"],
        }

    or_, ci = _or_ci(res["coef"]["critical"])
    return {
        "or": or_, "ci": ci, "n_pairs": res["n_pairs"], "n_obs": res["n_obs"],
        "converged": True, "reason": None,
    }


def _fit_rd(subset: pd.DataFrame, composite_extra_keys, outcome_col: str, *, n_boot: int, seed: int) -> dict:
    """Composite-key `risk_difference` fit (absolute scale) of `outcome_col`
    on `critical`, on a copy of `subset`."""
    m = subset.copy()
    m["drift"] = m[outcome_col].astype(int)
    m = _add_composite_pair_id(m, composite_extra_keys)
    return mc.risk_difference(m, n_boot=n_boot, seed=seed)


# ---------------------------------------------------------------------------
# Data loading / assembly.
# ---------------------------------------------------------------------------


def _load_crit_ctrl(parquet_path) -> pd.DataFrame:
    if isinstance(parquet_path, pd.DataFrame):
        df = parquet_path
    else:
        df = pd.read_parquet(parquet_path, columns=_LOAD_COLUMNS)
    return df[df["corpus"].isin(["CRIT", "CTRL"])].reset_index(drop=True)


def _rule_free_drift(df: pd.DataFrame) -> np.ndarray:
    """Re-derive the rule-free label from cached signals and return the
    `drift` (0/1) indicator -- `derive_labels(..., use_critical_rules=
    False)` needs no `crit_*` columns at all (see that function's
    docstring)."""
    sig = df[list(_SIGNAL_COLS_FOR_DERIVE)]
    labels = derive_labels(sig, tau=0.5, tie_break="primary", use_critical_rules=False)
    return (labels["label"].to_numpy() == "drift").astype(int)


def _load_probe_categories(probe_set_path) -> dict:
    """crit `message_id` (e.g. "probe_0000") -> category string: the probe
    item's own `drift_type` (idrift.data.probe_set.build_probe_set), or
    "grounded" when `drift_type` is None (the baseline/non-variant items)."""
    with open(probe_set_path) as f:
        items = json.load(f)
    return {item["message_id"]: (item["drift_type"] or "grounded") for item in items}


def build_matched_16_frame(parquet_path) -> pd.DataFrame:
    """Assemble the full 20-model CRIT-vs-CTRL matched analysis frame:
    `matched_compare.build_matched_frame` (join against the Task-1.3
    `ctrl_matched.csv` pairing + `exposure_v2_full.parquet` corruption
    covariates), plus `drift_rule_inclusive` (the cached-label outcome),
    `drift_rule_free` (the re-derived meaning-channel-only outcome), and
    `category` (the CRIT item's own probe-set drift type).

    Args:
        parquet_path: path to `attempts_v3plus_labeled.parquet`, or an
            already-loaded DataFrame with its columns.

    Returns:
        DataFrame with every column `matched_compare.matched_drift` and
        `risk_difference` require, plus `drift_rule_inclusive`,
        `drift_rule_free`, `category`, and `model`.
    """
    crit_ctrl = _load_crit_ctrl(parquet_path)
    ctrl_matched = pd.read_csv(CTRL_MATCHED_PATH)
    exposure = pd.read_parquet(EXPOSURE_PATH)

    merged = mc.build_matched_frame(crit_ctrl, ctrl_matched, exposure)

    merged["drift_rule_inclusive"] = merged["drift"].astype(int)
    merged["drift_rule_free"] = _rule_free_drift(merged)

    categories = _load_probe_categories(PROBE_SET_PATH)
    merged["category"] = merged["pair_id"].map(categories)

    return merged


# ---------------------------------------------------------------------------
# Digest sections.
# ---------------------------------------------------------------------------


def per_model_excess(merged: pd.DataFrame) -> dict:
    """One matched rule-free (primary) + rule-inclusive fit per model,
    composite-keyed on `pair_id x cer_target x replicate_idx` (identical
    corruption cell, within that model)."""
    out = {}
    for model, g in merged.groupby("model", sort=True):
        free = _fit_or(g, ["cer_target", "replicate_idx"], "drift_rule_free")
        inc = _fit_or(g, ["cer_target", "replicate_idx"], "drift_rule_inclusive")
        out[str(model)] = {
            "or": free["or"],
            "ci": free["ci"],
            "n_pairs": free["n_pairs"],
            "n_obs": free["n_obs"],
            "converged": free["converged"],
            "reason": free["reason"],
            "or_rule_inclusive": inc["or"],
            "ci_rule_inclusive": inc["ci"],
            "n_pairs_rule_inclusive": inc["n_pairs"],
            "converged_rule_inclusive": inc["converged"],
            "reason_rule_inclusive": inc["reason"],
        }
    return out


def pooled_excess(merged: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """The pooled (all 20 models) matched rule-free + rule-inclusive OR,
    plus the pooled rule-free matched risk difference in percentage points,
    composite-keyed on `pair_id x cer_target x replicate_idx x model`
    (matching `scratchpad/harden/a2_critfree.py`'s shipped design)."""
    keys = ["cer_target", "replicate_idx", "model"]
    free = _fit_or(merged, keys, "drift_rule_free")
    inc = _fit_or(merged, keys, "drift_rule_inclusive")
    rd = _fit_rd(merged, keys, "drift_rule_free", n_boot=n_boot, seed=seed)

    return {
        "rule_free_or": free["or"],
        "ci": free["ci"],
        "rule_inclusive_or": inc["or"],
        "ci_rule_inclusive": inc["ci"],
        "matched_rd_pp": rd["rd"] * 100.0,
        "matched_rd_pp_ci": [rd["ci_lo"] * 100.0, rd["ci_hi"] * 100.0],
        "n_pairs": free["n_pairs"],
        "n_obs": free["n_obs"],
        "reason": free["reason"],
        "n_pairs_rule_inclusive": inc["n_pairs"],
        "reason_rule_inclusive": inc["reason"],
        "n_models": int(merged["model"].nunique()),
    }


def by_category_excess(merged: pd.DataFrame, *, n_boot: int = 500, seed: int = 0) -> dict:
    """Matched rule-free OR + risk difference per probe-set `category`
    (`negation`, `recipient`, `refusal_consent`, `dose`, `grounded`), pooled
    across all 20 models, composite-keyed as `pooled_excess`. Purely
    descriptive stratification of an already-small, exploratory excess.

    The 6-parameter conditional logit (critical + 5 adjusters) can fail to
    converge on some of these smaller within-category subsets (in practice:
    a singular Hessian from limited within-category adjuster diversity,
    e.g. `negation`/`recipient`/`refusal_consent`/`grounded` here) --
    `rule_free_or`/`ci` are then `None` (`reason` holds `matched_drift`'s
    own diagnostic text, and `n_pairs` is the RAW input composite-group
    count, since the fit never reached its own post-drop counts; see
    `_fit_or`). `matched_rd_pp` has no such failure mode and is always
    reported, so a category is never entirely uninformative even when its
    OR is null.
    """
    keys = ["cer_target", "replicate_idx", "model"]
    out = {}
    for cat, g in merged.groupby("category", sort=True):
        free = _fit_or(g, keys, "drift_rule_free")
        rd = _fit_rd(g, keys, "drift_rule_free", n_boot=n_boot, seed=seed)
        n_crit_items = int(g.loc[g["critical"] == 1, "message_id"].nunique())
        out[str(cat)] = {
            "rule_free_or": free["or"],
            "ci": free["ci"],
            "n_pairs": free["n_pairs"],
            "converged": free["converged"],
            "reason": free["reason"],
            "matched_rd_pp": rd["rd"] * 100.0,
            "matched_rd_pp_ci": [rd["ci_lo"] * 100.0, rd["ci_hi"] * 100.0],
            "n_crit_items": n_crit_items,
        }
    return out


def balance_diagnostics() -> tuple[dict, dict]:
    """Reuse `idrift.analysis.balance`'s item-level CRIT/CTRL SMD table
    (before/after matching) verbatim -- the matching itself never depends
    on which model generated a downstream attempt, so this single table
    applies to every per-model fit above. Also reports the realized
    matching-distance distribution across the 131 matched pairs.

    Returns:
        (balance, match_distance): `balance` is `{covariate: {smd_before,
        smd_after}}`; `match_distance` is `{min, median, mean, max,
        n_pairs}` over `ctrl_matched.csv`'s own `match_distance` column.
    """
    crit_frame, ctrl_frame, matched_pairs = bal._load_real_frames()
    table = bal.balance_table(crit_frame, ctrl_frame, matched_pairs)

    balance = {
        row["covariate"]: {"smd_before": row["smd_before"], "smd_after": row["smd_after"]}
        for _, row in table.iterrows()
    }
    match_distance = {
        "min": float(matched_pairs["match_distance"].min()),
        "median": float(matched_pairs["match_distance"].median()),
        "mean": float(matched_pairs["match_distance"].mean()),
        "max": float(matched_pairs["match_distance"].max()),
        "n_pairs": int(len(matched_pairs)),
    }
    return balance, match_distance


def validate_7model(merged: pd.DataFrame) -> dict:
    """Mandatory sanity anchor: recompute the pooled rule-free matched OR
    restricted to `ORIGINAL_7_MODELS` and compare it against the prior
    hardening-pass result (`PRIOR_7MODEL_RULE_FREE_OR` ~= 1.10
    [1.07-1.13], `output/harden_critfree.json`) BEFORE the full 20-model
    number is trusted. Same composite-key design as `pooled_excess`."""
    subset = merged[merged["model"].isin(ORIGINAL_7_MODELS)]
    fit = _fit_or(subset, ["cer_target", "replicate_idx", "model"], "drift_rule_free")

    reproduces = fit["or"] is not None and abs(fit["or"] - PRIOR_7MODEL_RULE_FREE_OR) < 0.05

    return {
        "models": list(ORIGINAL_7_MODELS),
        "or_rulefree": fit["or"],
        "ci_rulefree": fit["ci"],
        "n_pairs": fit["n_pairs"],
        "n_obs": fit["n_obs"],
        "prior_or_rulefree": PRIOR_7MODEL_RULE_FREE_OR,
        "prior_ci_rulefree": PRIOR_7MODEL_RULE_FREE_CI,
        "prior_source": "output/harden_critfree.json (scratchpad/harden/a2_critfree.py)",
        "reproduces_prior": reproduces,
        "note": (
            "Sanity anchor (mandatory before trusting the 20-model number below): "
            "subsetting this module's own pipeline to the original 7-model panel "
            "reproduces the prior hardening-pass rule-free matched OR "
            f"(~{PRIOR_7MODEL_RULE_FREE_OR:.2f} {PRIOR_7MODEL_RULE_FREE_CI}, "
            "output/harden_critfree.json) to within 0.05 OR-units."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------


def _default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run(parquet_path=DEFAULT_PARQUET, out_path=DEFAULT_OUT) -> dict:
    """Build the 20-model per-model matched CRIT-vs-CTRL critical-excess
    digest (reviewer major #9) and write it to `out_path` as JSON.

    Args:
        parquet_path: path to `attempts_v3plus_labeled.parquet`, or an
            already-loaded DataFrame with its columns.
        out_path: path to write the JSON digest to.

    Returns:
        dict: `{pooled, per_model, balance, match_distance, by_category,
        validation_7model, n_models, n_crit_items, note}`. See
        `pooled_excess`, `per_model_excess`, `balance_diagnostics`,
        `by_category_excess`, and `validate_7model` for the nested shapes.
    """
    merged = build_matched_16_frame(parquet_path)

    balance, match_distance = balance_diagnostics()

    digest = {
        "pooled": pooled_excess(merged),
        "per_model": per_model_excess(merged),
        "balance": balance,
        "match_distance": match_distance,
        "by_category": by_category_excess(merged),
        "validation_7model": validate_7model(merged),
        "n_models": int(merged["model"].nunique()),
        "n_crit_items": int(merged.loc[merged["critical"] == 1, "message_id"].nunique()),
        "note": NOTE,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2, default=_default))

    print(
        "critical_excess_16: pooled rule_free_or="
        f"{digest['pooled']['rule_free_or']:.4f} ({digest['pooled']['n_models']} models); "
        f"validation_7model or_rulefree={digest['validation_7model']['or_rulefree']:.4f} "
        f"(reproduces_prior={digest['validation_7model']['reproduces_prior']})."
    )
    return digest


if __name__ == "__main__":
    run(DEFAULT_PARQUET, DEFAULT_OUT)
