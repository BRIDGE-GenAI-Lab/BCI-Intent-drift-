"""Confidently-wrong rate, missing-confidence audit, and predicted-drift
probability grid (revision Task B3).

The manuscript DEFINES "confidently wrong" (a drift output emitted at high
verbalized confidence) but never actually REPORTS a rate for it. This module
closes that gap with three small, independent functions:

- `confidently_wrong_rate`: among DRIFT outputs, what fraction were emitted
  at high verbalized confidence; and, the flip side, among all
  high-confidence outputs, what fraction were NOT faithful. Reported BOTH
  pooled and per model -- verbalized-confidence scales differ noticeably
  across models in this study (see `idrift.analysis.calibration`'s per-model
  ECE/AUROC), so a single pooled number would blur a well-calibrated model
  into a badly-calibrated one. NaN-confidence rows are excluded from every
  denominator (never imputed), with the excluded count reported so a reader
  can judge how much of the corpus that discards.
- `missing_confidence_audit`: where verbalized confidence is simply absent,
  tabulated by model x cer_target so a reader can see whether missingness
  concentrates in one model/CER combination (e.g. a model that occasionally
  fails to emit the confidence field at higher corruption) or is spread
  thin and structureless.
- `predicted_drift_grid`: a deliberately simple, interpretable companion to
  the fuller mixed-effects models elsewhere in this package
  (`idrift.analysis.mixed_models`) -- a message-clustered logistic
  regression of drift on `realized_cer` ALONE, read off at CER = 0.0, 0.1,
  0.2, 0.3, 0.4. This is not meant to replace the covariate-adjusted models;
  it is the number the manuscript can quote in one sentence ("predicted
  drift probability rises from X% at CER 0 to Y% at CER 0.4").

Confidence-scale handling: verbalized confidence in this study's parquet is
on a 0-100 scale (max observed value is 100 for every model), but the
functions below detect the scale from the data (`max > 1` => 0-100) rather
than assuming it, and rescale `conf_threshold` (given on a 0-1 scale, e.g.
the default 0.9) to match -- so a caller who somehow gets 0-1-scaled
confidence in still gets the right comparison. `scale_detected` and
`effective_threshold` are always reported in the output so this is never a
silent conversion.

Deterministic; American spelling; no em dashes (double hyphens for asides,
per house style).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

# The five CER grid points the manuscript reads predicted drift probability
# off of. Matches the study's cer_target grid (0.0, 0.1, 0.2, 0.3, 0.4).
CER_GRID = (0.0, 0.1, 0.2, 0.3, 0.4)

_CONFIDENTLY_WRONG_REQUIRED_COLUMNS = ("label", "confidence", "model")
_MISSING_AUDIT_REQUIRED_COLUMNS = ("model", "cer_target", "confidence")
_PREDICTED_GRID_REQUIRED_COLUMNS = ("label", "realized_cer", "message_id")


def _require_columns(df: pd.DataFrame, required: tuple[str, ...], fn_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{fn_name}: df is missing required column(s): {missing}")


def _detect_scale(confidence: pd.Series) -> str:
    """Detect whether `confidence` is on a 0-100 or 0-1 scale.

    Args:
        confidence: non-null confidence values.

    Returns:
        str: "0-100" if the max observed value exceeds 1, else "0-1". An
            empty series (no non-null confidence at all) defaults to "0-1"
            -- arbitrary but harmless, since every downstream count is zero
            in that case regardless of which scale is reported.
    """
    if len(confidence) == 0:
        return "0-1"
    return "0-100" if confidence.max() > 1 else "0-1"


def _confidently_wrong_stats(group: pd.DataFrame, effective_threshold: float) -> dict:
    """Compute the two headline confidently-wrong statistics for one group
    (a model's rows, or the whole pooled frame)."""
    n = int(len(group))
    is_drift = group["label"] == "drift"
    is_high_conf = group["confidence"] >= effective_threshold

    n_drift = int(is_drift.sum())
    n_drift_at_high_confidence = int((is_drift & is_high_conf).sum())
    frac_drift_at_high_confidence = (
        n_drift_at_high_confidence / n_drift if n_drift > 0 else float("nan")
    )

    n_high_confidence = int(is_high_conf.sum())
    n_high_confidence_not_faithful = int((is_high_conf & (group["label"] != "faithful")).sum())
    confidently_wrong = (
        n_high_confidence_not_faithful / n_high_confidence if n_high_confidence > 0 else float("nan")
    )

    return {
        "n": n,
        "n_drift": n_drift,
        "n_drift_at_high_confidence": n_drift_at_high_confidence,
        "frac_drift_at_high_confidence": frac_drift_at_high_confidence,
        "n_high_confidence": n_high_confidence,
        "n_high_confidence_not_faithful": n_high_confidence_not_faithful,
        "confidently_wrong_rate": confidently_wrong,
    }


def confidently_wrong_rate(df: pd.DataFrame, conf_threshold: float = 0.9) -> dict:
    """Quantify "confidently wrong": drift-at-high-confidence and
    high-confidence-but-wrong rates, pooled and per model.

    Two related but distinct statistics are reported for each group (pooled
    and per model):

    - `frac_drift_at_high_confidence`: among rows labeled "drift", the
      fraction emitted at confidence >= `conf_threshold`. Answers "when the
      model drifts, is it usually confident while doing so?"
    - `confidently_wrong_rate`: among ALL rows at confidence >=
      `conf_threshold` (any label), the fraction that were NOT faithful
      (i.e. drift or degraded). Answers "when the model is confident, how
      often is it actually wrong?" -- this is the number the manuscript's
      "confidently wrong" phrase should cite.

    Args:
        df: DataFrame with `label`, `confidence`, `model` columns.
            `confidence` may contain NaN; NaN rows are excluded from every
            denominator (never imputed).
        conf_threshold: confidence threshold on a 0-1 scale (default 0.9).
            Rescaled internally to match `df["confidence"]`'s detected
            scale (see `_detect_scale`); the effective, scale-matched
            threshold actually used is always returned as
            `effective_threshold`.

    Returns:
        dict: `{conf_threshold, scale_detected, effective_threshold,
        n_excluded_nan, pooled: {...}, per_model: {model: {...}}}`, where
        each `{...}` block has keys `n, n_drift, n_drift_at_high_confidence,
        frac_drift_at_high_confidence, n_high_confidence,
        n_high_confidence_not_faithful, confidently_wrong_rate`. Both rate
        fields are NaN (not raised) when their denominator is zero.

    Raises:
        ValueError: if `df` is missing `label`, `confidence`, or `model`.
    """
    _require_columns(df, _CONFIDENTLY_WRONG_REQUIRED_COLUMNS, "confidently_wrong_rate")

    n_excluded_nan = int(df["confidence"].isna().sum())
    non_na = df[df["confidence"].notna()]

    scale_detected = _detect_scale(non_na["confidence"])
    effective_threshold = conf_threshold * 100 if scale_detected == "0-100" else conf_threshold

    pooled = _confidently_wrong_stats(non_na, effective_threshold)
    per_model = {
        model: _confidently_wrong_stats(group, effective_threshold)
        for model, group in non_na.groupby("model")
    }

    return {
        "conf_threshold": float(conf_threshold),
        "scale_detected": scale_detected,
        "effective_threshold": float(effective_threshold),
        "n_excluded_nan": n_excluded_nan,
        "pooled": pooled,
        "per_model": per_model,
        "notes": (
            f"confidence scale detected as {scale_detected} from df['confidence'].max() "
            f"on {len(non_na)} non-NaN row(s); conf_threshold={conf_threshold} rescaled to "
            f"effective_threshold={effective_threshold} for comparison (>= is inclusive). "
            f"{n_excluded_nan} NaN-confidence row(s) excluded from every denominator below. "
            "frac_drift_at_high_confidence = P(confidence >= threshold | label == 'drift'). "
            "confidently_wrong_rate = P(label != 'faithful' | confidence >= threshold). "
            "Report per-model, not just pooled -- confidence scales differ across models."
        ),
    }


def missing_confidence_audit(df: pd.DataFrame) -> dict:
    """Tabulate NaN-confidence rows by model x cer_target.

    Args:
        df: DataFrame with `model`, `cer_target`, `confidence` columns.

    Returns:
        dict: `{total_missing, table, notes}`. `table` is a list of
        `{model, cer_target, n_missing}` records, one per (model,
        cer_target) combination observed in `df` -- including combinations
        with zero missing rows, so the audit also shows where missingness
        is NOT a problem. `total_missing` always equals
        `df["confidence"].isna().sum()`.

    Raises:
        ValueError: if `df` is missing `model`, `cer_target`, or
            `confidence`.
    """
    _require_columns(df, _MISSING_AUDIT_REQUIRED_COLUMNS, "missing_confidence_audit")

    n_missing_total = int(df["confidence"].isna().sum())
    missing_df = df[df["confidence"].isna()]
    cell_counts = missing_df.groupby(["model", "cer_target"]).size()

    models = sorted(df["model"].dropna().unique())
    cer_targets = sorted(df["cer_target"].dropna().unique())

    table = []
    running_total = 0
    for model in models:
        for cer_target in cer_targets:
            n = int(cell_counts.get((model, cer_target), 0))
            running_total += n
            table.append({"model": model, "cer_target": float(cer_target), "n_missing": n})

    # Defensive invariant: the per-cell grid must reconstruct the same total
    # as a direct isna().sum() (would only diverge if a NaN-confidence row
    # carried a model/cer_target value absent from the full df, which
    # should not happen).
    if running_total != n_missing_total:
        raise AssertionError(
            f"missing_confidence_audit: table total {running_total} != "
            f"df['confidence'].isna().sum() {n_missing_total} -- some NaN row(s) fell "
            "outside the observed model x cer_target grid."
        )

    return {
        "total_missing": n_missing_total,
        "table": table,
        "notes": (
            f"{n_missing_total} of {len(df)} row(s) have NaN confidence, tabulated across "
            f"{len(models)} model(s) x {len(cer_targets)} cer_target level(s). Cells with "
            "zero missing rows are included explicitly."
        ),
    }


def predicted_drift_grid(df: pd.DataFrame, *, seed: int = 0) -> dict:
    """Message-clustered logistic fit of drift on realized_cer, read off at
    a fixed CER grid.

    Deliberately simple: `realized_cer` is the ONLY predictor (no
    covariates, no random effects) so the result is a single, easily quoted
    number per CER level -- an interpretable companion to the fuller
    covariate-adjusted models in `idrift.analysis.mixed_models`, not a
    replacement for them.

    Engine: `statsmodels.api.Logit` (`label == "drift"` vs. an intercept +
    `realized_cer`), fit with `cov_type="cluster"` clustered on
    `message_id` (the same clustering unit used throughout this package,
    since replicate attempts on the same message are not independent).

    Args:
        df: DataFrame with `label`, `realized_cer`, `message_id` columns.
            Rows with a NaN in any of those three columns are dropped
            before fitting.
        seed: accepted for interface consistency with other stochastic
            fits in this package (e.g. `idrift.analysis.mixed_models`'s
            `fit_vb`); NOT consumed here. `Logit.fit`'s Newton-Raphson MLE
            on a fixed design is already deterministic, so no randomness
            needs seeding for this simple fit.

    Returns:
        dict: `{cer_grid, predicted_drift_probability: {"0.0": float, ...,
        "0.4": float}, coefficients: {name: {est, se}}, n_obs, converged,
        seed, notes}`.

    Raises:
        ValueError: if `df` is missing `label`, `realized_cer`, or
            `message_id`.
    """
    _require_columns(df, _PREDICTED_GRID_REQUIRED_COLUMNS, "predicted_drift_grid")

    data = df.dropna(subset=list(_PREDICTED_GRID_REQUIRED_COLUMNS)).copy()
    y = (data["label"] == "drift").astype(int)
    exog = sm.add_constant(data[["realized_cer"]].astype(float), has_constant="add")

    model = sm.Logit(y, exog)
    result = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": data["message_id"].to_numpy()})

    grid_exog = pd.DataFrame({"const": 1.0, "realized_cer": list(CER_GRID)})
    predicted_probs = result.predict(grid_exog)
    predicted_drift_probability = {
        f"{cer:.1f}": float(p) for cer, p in zip(CER_GRID, predicted_probs)
    }

    coefficients = {
        name: {"est": float(result.params[name]), "se": float(result.bse[name])}
        for name in result.params.index
    }

    converged = bool(result.mle_retvals.get("converged", True))

    return {
        "cer_grid": list(CER_GRID),
        "predicted_drift_probability": predicted_drift_probability,
        "coefficients": coefficients,
        "n_obs": int(result.nobs),
        "converged": converged,
        "seed": seed,
        "notes": (
            "engine=statsmodels.api.Logit (label=='drift' ~ 1 + realized_cer), "
            "cov_type='cluster', cov_kwds={'groups': message_id}. Deliberately univariate "
            "(realized_cer only, no covariates or random effects) for a single quotable "
            "number per CER grid point; see idrift.analysis.mixed_models for the "
            "covariate-adjusted models. `seed` is accepted for interface consistency but "
            "unused -- this fit's Newton-Raphson MLE is already deterministic."
        ),
    }
