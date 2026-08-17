"""Drift-specific confidence calibration for all 16 models (reviewer major #8).

Why this module exists
-----------------------
`scratchpad/harden/a4_calibration.py` (Task A4, reviewer #9) built a
drift-focused companion to the manuscript's headline calibration metric
(`idrift.analysis.calibration_v2`, AUROC/ECE of confidence -> faithful):
it reports `auroc_drift_vs_nondrift`, the AUROC of stated confidence
discriminating `label == "drift"` specifically, rather than faithful vs.
{degraded OR drift} pooled. That module ran on the 7-model
`attempts_v3_labeled.parquet`. Reviewer major #8 asks for the SAME
drift-specific question -- AUROC, calibration, sensitivity/specificity/FNR
at an operating threshold, and the missing-confidence pattern -- extended
to the full 16-model panel (`attempts_v3plus_labeled.parquet`,
3,888,000 rows), plus new by-CER and by-critical-rule breakdowns this
module adds.

This module reuses A4's metric conventions rather than inventing new ones:

- **Drift-risk score**: `score_for_drift = 1 - confidence` (confidence
  rescaled to [0, 1] first via the study's usual scale detection, see
  `idrift.analysis.confidence_analysis._detect_scale`). A well-calibrated
  model is expected to state LOWER confidence exactly when it is about to
  drift, so `drift_auroc = roc_auc_score(label == "drift", score_for_drift)
  > 0.5` means stated confidence really does carry information about
  drift specifically (matching A4's `auroc_drift_vs_nondrift`, same
  direction, same positive class).
- **ece_drift**: the same reliability-diagram ECE used throughout this
  package (`idrift.analysis.calibration_v2.ece`), applied to
  `score_for_drift` as the "predicted probability of drift" against the
  observed `label == "drift"` indicator -- the drift-specific analogue of
  A4's `ece_faithful_vs_rest` (which applies raw confidence against
  `label == "faithful"`).
- **Threshold metrics at 0.9**: A4 already reports
  `drift_fnr_at_threshold[t] = P(confidence >= t | label == "drift")` for
  t in {0.7, 0.9} -- the false-negative rate of a "confidence < t flags
  this output for review" rule (native, RAW confidence, not
  `score_for_drift`). This module reports that same quantity at the single
  candidate threshold t = 0.9 (`fnr_at_0.9`), plus its natural
  complements: `sens_at_0.9 = P(confidence < t | drift)` (the fraction of
  true drift the rule correctly flags) and
  `spec_at_0.9 = P(confidence >= t | not drift)` (the fraction of
  non-drift the rule correctly leaves unflagged).

Missing / unparsable confidence
---------------------------------
`confidence` may be NaN (the model emitted no parseable verbalized
confidence at all; unparsable strings were already converted to NaN
upstream at labeling time, so `isna()` captures both "missing" and
"unparsable" -- there is no separate unparsable sentinel in this parquet).
NaN-confidence rows are EXCLUDED from every confidence-based statistic
below (drift_auroc, ece_drift, the threshold metrics) and NEVER imputed;
`n_missing_conf` reports the excluded count per group so a reader can judge
how much of that group's rows the confidence-based metrics are silent on.
Rows with NaN confidence still count in the group's raw `n` and
`n_drift_all` denominators reported alongside (see "n_drift_all vs
n_drift_scored" below), just not in any confidence-conditioned statistic.

Missingness is emphatically NOT random with respect to drift: pooled
across the whole panel, missing-confidence rows are drift ~94% of the
time, versus a ~31% baseline drift rate across all rows (`missingness_
audit`). Concretely, a naive `n_drift_scored / n_scored` or
`n_drift_scored / n` computed only on confidence-parseable rows silently
undercounts a group's TRUE drift count -- for phi3:mini alone, 4,368
missing-confidence rows are ~96% drift, so its scored-only count omits
roughly 4,200 drift rows the scored subset never sees. Every block below
therefore reports BOTH `n_drift_scored` (the drift count among the
n_scored, confidence-parseable rows -- the denominator every AUROC/ECE/
threshold statistic above actually uses) and `n_drift_all` (the drift
count over the FULL group, including missing-confidence rows -- the one
to use for a group's overall drift rate, `n_drift_all / n`). A top-level
`missingness_audit` block reports the missing-vs-baseline drift-rate
enrichment overall and per model.

Degenerate groups
------------------
A group (a model, a CER level, a critical-rule stratum) can have too little
information for `drift_auroc` to be defined: fewer than two distinct
`label == "drift"` classes among its scoreable rows, zero scoreable rows,
or a constant `score_for_drift` (confidence never varies, so nothing to
discriminate with). Each such case is guarded explicitly -- `drift_auroc`
is reported as `None` (never raised, never silently coerced to 0.5) with
`degenerate: true` and a human-readable `reason`; `ece_drift` and the
threshold metrics fall back to `None` only when there are zero scoreable
rows (a constant confidence value can still yield a defined ECE and
threshold metrics even when AUROC cannot be computed from it).

by_cer and by_critical: NOT the same "critical" as CRIT/CTRL
----------------------------------------------------------------
`by_critical` splits rows on whether ANY of the five rule-based
critical-content detectors fired on that generation
(`idrift.analysis.zero_cer_audit.CRIT_COLUMNS`: `crit_negation_flip`,
`crit_numeral_change`, `crit_recipient_change`, `crit_urgency_change`,
`crit_actionable_omission`). This is a PER-GENERATION mechanism signal that
can fire on any corpus's row (AUTH, CRIT, or CTRL) -- it is a different
construct from `corpus == "CRIT"` (message-critical CHALLENGE-SET
membership, a fixed property of the input assigned before corruption; see
the manuscript Methods and this project's standing convention, e.g.
`idrift.analysis.matched_compare`'s `critical` design-arm indicator). To
avoid conflating the two, this module's `by_critical` keys are named
`rule_fired` / `no_rule_fired`, not `critical`/`noncritical`. Every group by
construction where `rule_fired` is true has zero faithful rows (a fired
critical-content detector routes the taxonomy away from "faithful" by
definition; see `idrift.adjudicate.taxonomy.label`), so its `drift_auroc`
is drift-vs-degraded discrimination specifically, not drift-vs-everything;
this is noted in the block's own `notes` field, not just here.

Both `by_cer` and `by_critical` pool rows ACROSS all 16 models within each
stratum -- like A4's explicitly-labeled `pooled` block, this is descriptive
context, not a claim that confidence is on one shared scale across models
(see `pooling_note`). `per_model` is the primary, comparable-within-model
breakdown; `by_cer` and `by_critical` answer "does the drift-signal in
confidence vary by corruption level / by critical-rule firing", pooled for
readability.

No model inference happens here: every column consumed
(`confidence`, `label`, `model`, `cer_target`, the five `crit_*` flags) is
already present in the cached, already-labeled parquet. Deterministic; no
bootstrap or resampling. American spelling; no em dashes (double hyphens
for asides), per house style.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from idrift.analysis.calibration_v2 import ece as _ece
from idrift.analysis.zero_cer_audit import CRIT_COLUMNS

DEFAULT_PARQUET = "output/intermediate/attempts_v3plus_labeled.parquet"
DEFAULT_OUT = "output/calibration_16_v3_digest.json"

CONF_THRESHOLD = 0.9  # the candidate operating threshold this task asks for

_REQUIRED_COLUMNS = ("model", "label", "confidence", "cer_target") + CRIT_COLUMNS

POOLING_NOTE = (
    "Verbalized model confidence is not on a common scale across models -- an "
    "'80% confident' statement from one model is not the same probability claim as "
    "from another (see idrift.analysis.calibration_v2 and "
    "scratchpad/harden/a4_calibration.py module docstrings for the same caveat "
    "applied to this study's other calibration analyses). per_model is the primary, "
    "within-model-comparable breakdown. by_cer and by_critical pool rows across all "
    "20 models within each stratum for descriptive context only; those pooled values "
    "are NOT a claim that models share one confidence scale, and should not be read "
    "as a single calibration number for 'the panel'."
)


def _require_columns(df: pd.DataFrame, fn_name: str) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{fn_name}: df is missing required column(s): {missing}")


def _detect_scale(confidence: pd.Series) -> str:
    """Detect whether `confidence` is on a 0-100 or 0-1 scale (same idiom as
    `idrift.analysis.confidence_analysis._detect_scale` and
    `scratchpad/harden/a4_calibration.py._detect_scale`)."""
    if len(confidence) == 0:
        return "0-1"
    return "0-100" if confidence.max() > 1 else "0-1"


def _metrics_for_group(group: pd.DataFrame, scale_multiplier: float) -> dict:
    """Compute every drift-specific calibration statistic for one group (a
    single model's rows, one CER level's rows, or one critical-rule
    stratum's rows), pooling across whatever the caller already sliced.

    Args:
        group: DataFrame with `label` and `confidence` columns (a subset of
            the full labeled frame).
        scale_multiplier: 100.0 if `confidence` is on a 0-100 scale, else
            1.0 (see `_detect_scale`); used to rescale `confidence` to
            [0, 1] and the 0.9 threshold to the matching native scale.

    Returns:
        dict: `{n, n_drift_all, n_drift_scored, n_scored, n_missing_conf,
        drift_auroc, ece_drift, fnr_at_0.9, sens_at_0.9, spec_at_0.9,
        degenerate, reason, notes}`. `drift_auroc` is `None` (with
        `degenerate: True` and a `reason`) when undefined; `ece_drift` and
        the threshold metrics are `None` only when there are zero
        scoreable (non-missing-confidence) rows.

        `n_drift_all` vs `n_drift_scored`: missing/unparsable confidence is
        NOT missing at random with respect to drift -- pooled across the
        whole panel, missing-confidence rows are ~94% drift vs a ~31%
        baseline drift rate (see `missingness_audit`). `n_drift_scored` is
        the drift count over `n_scored` rows only (the denominator every
        confidence-based statistic above uses); `n_drift_all` is the drift
        count over the FULL group (`n` rows, including missing-confidence
        ones). A consumer computing a group's overall drift RATE must use
        `n_drift_all / n`, never `n_drift_scored / n` or
        `n_drift_scored / n_scored` as a stand-in for the full-group rate
        -- for models with a lot of missing confidence (e.g. phi3:mini),
        the scored-only count silently undercounts drift by thousands of
        rows.
    """
    n = int(len(group))
    n_missing_conf = int(group["confidence"].isna().sum())
    scored = group[group["confidence"].notna()]
    n_scored = int(len(scored))

    n_drift_all = int((group["label"] == "drift").sum())
    is_drift = (scored["label"] == "drift").astype(int)
    n_drift_scored = int(is_drift.sum())

    degenerate = False
    reason = None

    if n_scored == 0:
        drift_auroc = None
        ece_drift = None
        fnr = sens = spec = None
        degenerate = True
        reason = "no rows with a parseable (non-missing) confidence value"
    else:
        conf_norm = scored["confidence"].astype(float) / scale_multiplier
        score_for_drift = (1.0 - conf_norm).to_numpy()
        is_drift_arr = is_drift.to_numpy()

        if is_drift.nunique() < 2:
            drift_auroc = None
            degenerate = True
            reason = (
                "fewer than two distinct drift/non-drift classes among "
                "scoreable rows -- AUROC undefined"
            )
        elif pd.Series(score_for_drift).nunique() < 2:
            drift_auroc = None
            degenerate = True
            reason = (
                "confidence is constant among scoreable rows -- nothing "
                "for AUROC to discriminate with"
            )
        else:
            drift_auroc = float(roc_auc_score(is_drift_arr, score_for_drift))

        ece_drift = float(_ece(score_for_drift, is_drift_arr))

        eff_threshold = CONF_THRESHOLD * scale_multiplier
        # "flagged" = the operational rule "confidence < t triggers review".
        flagged = (scored["confidence"].astype(float) < eff_threshold).to_numpy()
        drift_mask = is_drift_arr.astype(bool)

        tp = int((flagged & drift_mask).sum())
        fn = int((~flagged & drift_mask).sum())
        tn = int((~flagged & ~drift_mask).sum())
        fp = int((flagged & ~drift_mask).sum())

        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else None
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else None
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else None

    return {
        "n": n,
        "n_drift_all": n_drift_all,
        "n_drift_scored": n_drift_scored,
        "n_scored": n_scored,
        "n_missing_conf": n_missing_conf,
        "drift_auroc": drift_auroc,
        "ece_drift": ece_drift,
        "fnr_at_0.9": fnr,
        "sens_at_0.9": sens,
        "spec_at_0.9": spec,
        "degenerate": degenerate,
        "reason": reason,
        "notes": (
            "drift_auroc: AUROC of score_for_drift = (1 - confidence) predicting "
            "label == 'drift' against all non-drift rows in this group (matches "
            "scratchpad/harden/a4_calibration.py's auroc_drift_vs_nondrift). "
            "ece_drift: idrift.analysis.calibration_v2.ece(score_for_drift, "
            "label == 'drift'). fnr_at_0.9 = P(confidence >= 0.9 | drift): the "
            "fraction of true drift a 'confidence < 0.9 flags for review' rule "
            "would MISS; sens_at_0.9 = 1 - fnr_at_0.9 = P(confidence < 0.9 | "
            "drift); spec_at_0.9 = P(confidence >= 0.9 | not drift). All "
            "confidence-based fields (drift_auroc, ece_drift, the threshold "
            "metrics) are computed on n_scored rows only (n_missing_conf rows "
            "excluded, never imputed). IMPORTANT DENOMINATOR NOTE: n_drift_scored "
            "is the drift count over n_scored rows only; n_drift_all is the drift "
            "count over ALL n rows in this group, including missing-confidence "
            "ones. Missingness is NOT random with respect to drift (missing-"
            "confidence rows are drift far more often than the panel baseline; "
            "see the top-level missingness_audit block), so a group's overall "
            "drift rate must be computed as n_drift_all / n, never "
            "n_drift_scored / n or n_drift_scored / n_scored."
        ),
    }


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


def per_model_calibration(df: pd.DataFrame, scale_multiplier: float) -> dict:
    """`_metrics_for_group`, one block per distinct `model`."""
    return {
        str(model): _metrics_for_group(group, scale_multiplier)
        for model, group in df.groupby("model")
    }


def missingness_audit(df: pd.DataFrame) -> dict:
    """Audit the missing-confidence PATTERN itself: is missing confidence
    missing at random with respect to drift, or enriched for it?

    Reports, overall and per model, the drift rate AMONG missing-confidence
    rows against the group's own overall (all-rows) baseline drift rate.
    This is the finding that motivates `n_drift_all` vs `n_drift_scored` in
    every other block of this digest (see module docstring): if missing
    confidence were missing at random, its drift rate would roughly equal
    the baseline and the scored-only denominator wouldn't matter much; it
    is not.

    Args:
        df: the full labeled frame (any subset -- callers use this at the
            whole-panel level in `run`).

    Returns:
        dict: `{overall_baseline_drift_rate, overall_drift_rate_among_missing,
        overall_enrichment, n_missing_total, per_model: {model: {n_missing,
        baseline_drift_rate, drift_rate_among_missing, enrichment}}, note}`.
        `drift_rate_among_missing` and `enrichment` are `None` for a group
        with zero missing-confidence rows (nothing to compute a rate over).
        `enrichment` is `None` if the group's baseline drift rate is itself
        zero (would divide by zero).
    """
    n_missing_total = int(df["confidence"].isna().sum())
    missing = df[df["confidence"].isna()]

    overall_baseline = float((df["label"] == "drift").mean()) if len(df) else float("nan")
    overall_missing_rate = (
        float((missing["label"] == "drift").mean()) if len(missing) else None
    )
    overall_enrichment = (
        overall_missing_rate / overall_baseline
        if (overall_missing_rate is not None and overall_baseline > 0)
        else None
    )

    per_model = {}
    for model, group in df.groupby("model"):
        group_missing = group[group["confidence"].isna()]
        n_missing = int(len(group_missing))
        baseline = float((group["label"] == "drift").mean()) if len(group) else float("nan")
        rate_missing = (
            float((group_missing["label"] == "drift").mean()) if n_missing else None
        )
        enrichment = (
            rate_missing / baseline if (rate_missing is not None and baseline > 0) else None
        )
        per_model[str(model)] = {
            "n_missing": n_missing,
            "baseline_drift_rate": baseline,
            "drift_rate_among_missing": rate_missing,
            "enrichment": enrichment,
        }

    return {
        "overall_baseline_drift_rate": overall_baseline,
        "overall_drift_rate_among_missing": overall_missing_rate,
        "overall_enrichment": overall_enrichment,
        "n_missing_total": n_missing_total,
        "per_model": per_model,
        "note": (
            "Missing/unparsable confidence is NOT missing at random with respect to "
            "drift. baseline_drift_rate = drift rate over ALL rows in the group; "
            "drift_rate_among_missing = drift rate over just that group's "
            "missing-confidence rows; enrichment = the ratio of the two. Overall, "
            f"{n_missing_total} of {len(df)} rows have missing confidence, of which "
            f"{'NA' if overall_missing_rate is None else f'{overall_missing_rate:.1%}'} "
            f"are drift, against a {overall_baseline:.1%} baseline -- roughly "
            f"{'NA' if overall_enrichment is None else f'{overall_enrichment:.1f}x'} "
            "enrichment. This is why every per_model/by_cer/by_critical block reports "
            "both n_drift_scored and n_drift_all rather than only the scored-subset "
            "count."
        ),
    }


def by_cer_calibration(df: pd.DataFrame, scale_multiplier: float) -> dict:
    """`_metrics_for_group`, one block per distinct `cer_target` level,
    pooling rows across all models at that level (see module docstring)."""
    out = {}
    for level, group in df.groupby("cer_target"):
        block = _metrics_for_group(group, scale_multiplier)
        block["pooling_note"] = POOLING_NOTE
        out[f"{float(level):.1f}"] = block
    return out


def by_critical_calibration(df: pd.DataFrame, scale_multiplier: float) -> dict:
    """`_metrics_for_group`, split on whether ANY of the five rule-based
    critical-content detectors fired (`rule_fired`) or none did
    (`no_rule_fired`), pooling rows across all models within each stratum.

    This is NOT the CRIT/CTRL challenge-set design arm -- see module
    docstring "by_cer and by_critical" section.
    """
    rule_fired_mask = df[list(CRIT_COLUMNS)].any(axis=1)
    rule_fired_block = _metrics_for_group(df[rule_fired_mask], scale_multiplier)
    no_rule_block = _metrics_for_group(df[~rule_fired_mask], scale_multiplier)
    caveat = (
        "rule_fired = at least one of idrift.analysis.zero_cer_audit.CRIT_COLUMNS "
        "(crit_negation_flip, crit_numeral_change, crit_recipient_change, "
        "crit_urgency_change, crit_actionable_omission) is True for this generation. "
        "This is a per-generation MECHANISM signal, distinct from corpus == 'CRIT' "
        "(the message-critical CHALLENGE-SET design arm, a fixed property of the "
        "input assigned before corruption -- see the manuscript Methods and "
        "idrift.analysis.matched_compare's 'critical' indicator). Any row where a "
        "critical-content rule fired is, by the taxonomy's own construction, never "
        "labeled faithful (a fired detector routes the label away from faithful), so "
        "rule_fired's drift_auroc is drift-vs-degraded discrimination, not "
        "drift-vs-everything."
    )
    rule_fired_block["pooling_note"] = POOLING_NOTE
    rule_fired_block["caveat"] = caveat
    no_rule_block["pooling_note"] = POOLING_NOTE
    no_rule_block["caveat"] = caveat
    return {"rule_fired": rule_fired_block, "no_rule_fired": no_rule_block}


def run(parquet_path, out_path) -> dict:
    """Build the 16-model drift-specific calibration digest and write it to
    `out_path` as JSON (reviewer major #8).

    Args:
        parquet_path: path to the labeled attempts parquet (columns
            `model`, `label`, `confidence`, `cer_target`, the five
            `crit_*` flags), or an already-loaded DataFrame with those
            columns.
        out_path: path to write the JSON digest to.

    Returns:
        dict: `{per_model, by_cer, by_critical, missingness_audit,
        conf_threshold, scale_detected, n_total_rows, n_missing_conf_total,
        pooling_note, notes}`. See `_metrics_for_group` for the per-block
        shape, `missingness_audit` for the missing-confidence-pattern
        block, and the module docstring for the `by_cer` / `by_critical`
        pooling semantics.

    Raises:
        ValueError: if the input is missing a required column.
    """
    df = parquet_path if isinstance(parquet_path, pd.DataFrame) else pd.read_parquet(parquet_path)
    _require_columns(df, "calibration_16.run")

    n_missing_total = int(df["confidence"].isna().sum())
    scale_detected = _detect_scale(df.loc[df["confidence"].notna(), "confidence"])
    scale_multiplier = 100.0 if scale_detected == "0-100" else 1.0

    digest = {
        "per_model": per_model_calibration(df, scale_multiplier),
        "by_cer": by_cer_calibration(df, scale_multiplier),
        "by_critical": by_critical_calibration(df, scale_multiplier),
        "missingness_audit": missingness_audit(df),
        "conf_threshold": CONF_THRESHOLD,
        "scale_detected": scale_detected,
        "n_total_rows": int(len(df)),
        "n_missing_conf_total": n_missing_total,
        "pooling_note": POOLING_NOTE,
        "notes": (
            "Drift-specific calibration for all 16 models (reviewer major #8), "
            "extending scratchpad/harden/a4_calibration.py's 7-model "
            "auroc_drift_vs_nondrift analysis. per_model is primary; by_cer / "
            "by_critical are pooled-across-models descriptive breakdowns (see "
            "pooling_note and each block's own notes). missingness_audit reports "
            "the drift-rate enrichment among missing-confidence rows -- see "
            "n_drift_all vs n_drift_scored in every other block."
        ),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2, default=_default))

    return digest


if __name__ == "__main__":
    run(DEFAULT_PARQUET, DEFAULT_OUT)
