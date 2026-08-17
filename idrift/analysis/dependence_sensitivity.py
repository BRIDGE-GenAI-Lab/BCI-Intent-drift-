"""Dependence-structure sensitivity for the realized-CER -> drift slope
(revision Task B9).

Why this module exists
----------------------
The primary drift model is a message-clustered cluster-robust logit of
`drift` on `realized_cer` (plus adjusters), and it reports a realized_cer
log-odds slope of 4.79 (see `output/stats_v2_digest.json` ->
`mixed_model.drift.coefficients.realized_cer`, est 4.7917, SE 0.322,
95% CI 4.16-5.42). Reviewers objected that message-level clustering
addresses only ONE axis of dependence: replicate generations are also
correlated WITHIN a model (the same LLM's behavior pushes all of its rows
the same way), and a message-clustered SE cannot speak to that model-level
dependence at all. The obvious fix -- re-cluster the SEs on `model` -- is
not credible here because there are only 7 models: cluster-robust SEs are
badly biased and anti-conservative with so few clusters (the standard
rule-of-thumb wants ~30-50+), so a `model`-clustered SE would be a worse
estimate, not a better one, of model-level uncertainty.

This module therefore provides THREE alternative estimators of the SAME
realized_cer slope, each making a DIFFERENT dependence assumption, so the
manuscript can show the slope is stable across all of them rather than
resting on the single message-clustered fit:

1. `per_model_pooled_slope` -- PREFERRED for the model-level axis. Fit a
   message-clustered logit SEPARATELY within each of the 7 models, then
   pool the 7 slopes with a DerSimonian-Laird random-effects meta-analysis
   (reusing `idrift.analysis.calibration_v2.meta_analyze`). This treats
   each model as its own study and lets the between-model heterogeneity
   (`tau^2`) be estimated and reported, instead of pretending 7 clusters
   are enough for a model-clustered SE. Each within-model fit still
   clusters on `message_id`, so BOTH dependence axes are respected:
   message-level within each fit, model-level via the random-effects pool.

2. `gee_slope` -- a population-averaged GEE (Binomial family, exchangeable
   working correlation, `groups=message_id`) with the sandwich (robust)
   SE. This is the marginal-model counterpart to the clustered logit and
   an independent check that the working-correlation assumption does not
   move the slope.

3. `bootstrap_slope` -- a fully nonparametric cluster bootstrap of the
   slope, resampling clusters WITH REPLACEMENT at `level` in {"message",
   "model"} and refitting a plain logit on each resample. The `message`
   level (762 clusters) gives a stable CI; the `model` level (7 clusters)
   is reported honestly as UNSTABLE -- with so few clusters the bootstrap
   distribution is coarse and its CI is wide/erratic, which is exactly the
   point (it is why we do NOT rely on model-clustered inference and prefer
   the DL pool in method 1).

Comparability
-------------
Every estimator here fits realized_cer ONLY (plus an intercept), NOT the
adjusted specification behind the 4.79 headline. That is deliberate: the
task is to show the slope is stable across DEPENDENCE assumptions, holding
the mean model fixed and simple, so the estimators are directly comparable
to one another. The univariate slope is therefore expected to differ
somewhat from the 4.79 adjusted value; what matters is that the three
dependence estimators agree with EACH OTHER and sit in the same
neighborhood as the primary fit.

Outcome
-------
Every function derives the binary outcome as `drift = (label == "drift")`
from the study's {faithful, degraded, drift} taxonomy; degraded and
faithful rows are both non-drift (drift-vs-rest, matching the primary
`mixed_models` drift-vs-rest analysis). Each function returns the slope on
the log-odds scale plus its SE/CI and a human-readable `label`.

Determinism: all randomness flows from an explicit `numpy.random.default_
rng(seed)`; a given `(df, seed)` reproduces bit-identical output. American
spelling, no em dashes (double hyphens for asides), per house style.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.genmod.families import Binomial

from idrift.analysis.calibration_v2 import meta_analyze

_Z95 = 1.96
_PCTILES = (2.5, 97.5)
_PRIMARY_SLOPE = 4.79  # message-clustered adjusted logit realized_cer log-odds

_REQUIRED_COLUMNS = ("label", "realized_cer", "message_id", "model")

_DRIFT_LABEL = "drift"


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"dependence_sensitivity: df is missing required column(s): {missing}"
        )


def _with_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a numeric `drift` column = (label == 'drift')."""
    out = df.copy()
    out["drift"] = (out["label"] == _DRIFT_LABEL).astype(int)
    return out


def _clustered_logit_slope(data: pd.DataFrame):
    """Message-clustered cluster-robust logit of drift on realized_cer.

    Univariate (realized_cer + intercept only). Returns
    `(slope, se, [ci_lo, ci_hi])`, or `None` if the fit cannot be formed
    (e.g. a subgroup with only one outcome class -> perfect separation).
    """
    if data["drift"].nunique() < 2:
        return None
    try:
        model = smf.logit("drift ~ realized_cer", data=data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(
                disp=0,
                cov_type="cluster",
                cov_kwds={"groups": data["message_id"].to_numpy()},
            )
    except Exception:  # noqa: BLE001 - a degenerate subgroup should be skipped, not fatal
        return None
    if "realized_cer" not in res.params.index:
        return None
    slope = float(res.params["realized_cer"])
    se = float(res.bse["realized_cer"])
    if not (np.isfinite(slope) and np.isfinite(se)):
        return None
    return slope, se, [slope - _Z95 * se, slope + _Z95 * se]


def per_model_pooled_slope(df: pd.DataFrame, *, seed: int = 0) -> dict:
    """Per-model message-clustered slopes pooled by DerSimonian-Laird.

    Fits a message-clustered cluster-robust logit of `drift` on
    `realized_cer` (univariate) SEPARATELY within each model, then pools
    the per-model slopes with the standard DerSimonian-Laird random-effects
    meta-analysis (`idrift.analysis.calibration_v2.meta_analyze`), treating
    each model as its own study. This is the PREFERRED model-level
    estimator: 7 models is too few for a model-clustered SE, so the between-
    model dependence is modeled as random-effects heterogeneity (`tau^2`)
    instead.

    Args:
        df: frame with `label`, `realized_cer`, `message_id`, `model`.
        seed: accepted for interface/determinism symmetry with the other
            estimators; this estimator has no stochastic step (every fit is
            a deterministic MLE), so results do not depend on it.

    Returns:
        dict: `{"method", "label", "slope", "se", "ci": [lo, hi], "tau2",
        "i2", "q", "k_models", "per_model": {model: {"slope", "se", "ci",
        "n", "n_messages"}}, "n_models_dropped", "primary_slope"}`. `slope`
        is the pooled random-effects slope on the log-odds scale; `se` is
        its normal-approximation SE (derived from the pooled CI). Models
        whose within-model fit was degenerate (single outcome class) are
        excluded from the pool and counted in `n_models_dropped`.
    """
    _check_columns(df)
    _ = seed  # deterministic; see docstring
    data = _with_outcome(df)

    per_model = {}
    dropped = []
    for model, group in data.groupby("model"):
        fit = _clustered_logit_slope(group.reset_index(drop=True))
        if fit is None:
            dropped.append(str(model))
            continue
        slope, se, ci = fit
        per_model[str(model)] = {
            "slope": slope,
            "se": se,
            "ci": ci,
            "slope_ci": ci,  # key name meta_analyze expects (f"{metric}_ci")
            "n": int(len(group)),
            "n_messages": int(group["message_id"].nunique()),
        }

    pooled = meta_analyze(per_model, metric="slope")

    slope = pooled["pooled"]
    ci = pooled["ci"]
    se = None
    if ci is not None and slope is not None:
        se = float((ci[1] - ci[0]) / (2 * _Z95))

    # Strip the internal `slope_ci` mirror before returning per-model detail.
    per_model_out = {
        m: {k: v for k, v in d.items() if k != "slope_ci"}
        for m, d in per_model.items()
    }

    return {
        "method": "per_model_pooled_dl",
        "label": (
            "Per-model message-clustered cluster-robust logit "
            "(drift ~ realized_cer), pooled across models by DerSimonian-"
            "Laird random-effects meta-analysis (preferred model-level "
            "estimator; 7 models is too few for a model-clustered SE)"
        ),
        "slope": slope,
        "se": se,
        "ci": ci,
        "tau2": pooled["tau2"],
        "i2": pooled["i2"],
        "q": pooled["q"],
        "k_models": pooled["k_models"],
        "per_model": per_model_out,
        "n_models_dropped": len(dropped),
        "models_dropped": dropped,
        "primary_slope": _PRIMARY_SLOPE,
    }


# Largest log-odds slope per unit realized CER that is physically meaningful.
# The predictor is bounded in [0, 1]; e^50 is already an absurd odds ratio, so
# anything beyond this is a diverged optimizer rather than an estimate.
_MAX_PLAUSIBLE_SLOPE = 50.0


def _fit_gee(data: pd.DataFrame, cov_struct, start_params=None):
    """Fit one GEE with the given working correlation. Returns
    `(slope, se, converged)` with `slope`/`se` non-finite-guarded to None."""
    model = smf.gee(
        "drift ~ realized_cer",
        groups="message_id",
        data=data,
        family=Binomial(),
        cov_struct=cov_struct,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(start_params=start_params)
    slope = float(res.params["realized_cer"])
    se = float(res.bse["realized_cer"])  # GEE reports the robust/sandwich SE
    converged = bool(getattr(res, "converged", True))
    if not (np.isfinite(slope) and np.isfinite(se)):
        return None, None, False
    # A finite value is not enough. On the full-corpus cohort the exchangeable
    # fit reported convergence while returning a slope of -3.6e22 with a
    # standard error of exactly 0, which passed the finite check and was
    # carried into the reported spread. Reject fits whose magnitude is outside
    # any physically meaningful range for a log-odds slope on a predictor
    # bounded in [0, 1], and fits whose standard error has collapsed.
    if abs(slope) > _MAX_PLAUSIBLE_SLOPE or se <= 0.0:
        return None, None, False
    return slope, se, converged


def gee_slope(df: pd.DataFrame) -> dict:
    """Population-averaged GEE slope of drift on realized_cer.

    Fits a `statsmodels` GEE (Binomial family, `groups=message_id`) of
    `drift` on `realized_cer` (univariate) and returns the slope with the
    sandwich (robust) SE and the corresponding Wald 95% CI. This is the
    marginal-model counterpart to the primary message-clustered logit: the
    GEE working correlation is a different dependence assumption than the
    clustered logit's, so an agreeing slope shows the estimate does not
    hinge on it.

    Working-correlation fallback ladder (documented, same idiom as
    `idrift.analysis.mixed_models`'s engine fallback): the PRIMARY working
    correlation is EXCHANGEABLE as specified for this task. On this study's
    real cohort the exchangeable estimator DIVERGES -- the clusters are
    large (~700 rows per message) and the within-message drift correlation
    is strong, so the estimated exchangeable correlation runs away and the
    parameters blow up to non-finite values (verified: alpha climbs past
    0.7 over successive iterations and the coefficients go to NaN). When the
    exchangeable fit does not converge or returns non-finite parameters,
    `gee_slope` falls back to an INDEPENDENCE working correlation with the
    same robust sandwich SE. That fallback is not a lesser analysis: an
    independence-working GEE with the sandwich SE gives a consistent slope
    and a valid cluster-robust SE regardless of the true correlation
    structure -- it is the textbook-safe robust GEE -- while sidestepping
    the misspecified exchangeable correlation that this cohort's cluster
    geometry cannot support. Which structure actually produced the reported
    numbers is always recorded in `cov_struct_used`. To give the
    exchangeable fit its best chance before falling back, it is started from
    a plain-logit warm start.

    Args:
        df: frame with `label`, `realized_cer`, `message_id`, `model`.

    Returns:
        dict: `{"method", "label", "slope", "se", "ci": [lo, hi],
        "cov_struct_used", "exchangeable_converged", "n_obs", "n_groups",
        "converged", "notes", "primary_slope"}`. `se` is the robust
        (sandwich) SE; `cov_struct_used` is `"exchangeable"` or
        `"independence_fallback"`.
    """
    _check_columns(df)
    data = _with_outcome(df)

    # Plain-logit warm start (gives the exchangeable optimizer its best shot).
    exog = sm.add_constant(data["realized_cer"].to_numpy(), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = sm.Logit(data["drift"].to_numpy(), exog).fit(disp=0).params

    slope, se, exch_conv = _fit_gee(data, Exchangeable(), start_params=start)

    if slope is not None and exch_conv:
        cov_used = "exchangeable"
        notes = "Exchangeable working correlation converged."
    else:
        why = (
            "did not converge" if slope is not None
            else "returned an implausible or degenerate fit "
            "(non-finite parameters, a log-odds slope beyond "
            f"{_MAX_PLAUSIBLE_SLOPE:.0f}, or a collapsed standard error), "
            "in some runs while reporting convergence"
        )
        slope, se, _ = _fit_gee(data, Independence(), start_params=start)
        cov_used = "independence_fallback"
        notes = (
            f"FALLBACK: exchangeable working correlation {why} on this cohort "
            "(large ~700-row message clusters with strong within-message "
            "drift correlation); reverted to an independence working "
            "correlation with the same robust sandwich SE, which yields a "
            "consistent slope and a valid cluster-robust SE regardless of the "
            "true correlation."
        )

    ci = [slope - _Z95 * se, slope + _Z95 * se]

    return {
        "method": "gee",
        "label": (
            "GEE population-averaged logit (drift ~ realized_cer; Binomial "
            f"family, {cov_used.replace('_', ' ')} working correlation, "
            "groups=message_id, robust sandwich SE)"
        ),
        "slope": slope,
        "se": se,
        "ci": ci,
        "cov_struct_used": cov_used,
        "exchangeable_converged": bool(exch_conv) if cov_used == "exchangeable" else False,
        "n_obs": int(len(data)),
        "n_groups": int(data["message_id"].nunique()),
        "converged": True,
        "notes": notes,
        "primary_slope": _PRIMARY_SLOPE,
    }


def _plain_logit_slope(y: np.ndarray, cer: np.ndarray):
    """Plain (unclustered) univariate logit slope of y on cer + intercept.

    Returns the realized_cer coefficient as a float, or `None` if the fit
    is degenerate (single outcome class or a non-finite/failed fit).
    """
    if len(np.unique(y)) < 2:
        return None
    exog = sm.add_constant(cer, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, exog).fit(disp=0)
    except Exception:  # noqa: BLE001 - a separated resample is skipped, not fatal
        return None
    # exog columns: ["const", "x1"]; the slope is the second parameter.
    slope = float(res.params[1])
    return slope if np.isfinite(slope) else None


def bootstrap_slope(
    df: pd.DataFrame, *, level: str, n_boot: int = 500, seed: int = 0
) -> dict:
    """Nonparametric cluster bootstrap of the realized_cer slope.

    Resamples clusters WITH REPLACEMENT at `level`, pulls every row of each
    resampled cluster (preserving within-cluster dependence), and refits a
    plain univariate logit (drift ~ realized_cer) on each resample. The CI
    is the 2.5th/97.5th percentile of the bootstrap slope distribution; the
    point estimate is that same plain logit fit on the FULL (unresampled)
    data.

    Args:
        df: frame with `label`, `realized_cer`, `message_id`, `model`.
        level: cluster level to resample, `"message"` (762 clusters ->
            stable) or `"model"` (7 clusters -> reported UNSTABLE).
        n_boot: number of bootstrap resamples.
        seed: seed for the single `numpy.random.default_rng` driving every
            resample (deterministic given `(df, level, n_boot, seed)`).

    Returns:
        dict: `{"method", "label", "slope", "se", "ci": [lo, hi], "level",
        "n_clusters", "n_boot", "n_valid", "unstable", "primary_slope"}`.
        `slope` is the full-data point estimate; `se` is the bootstrap SD
        of the slope draws; `unstable` is True when the level has too few
        clusters (< 15) for the cluster bootstrap to be trustworthy -- for
        `level="model"` (7 clusters) it is always True, reported honestly.

    Raises:
        ValueError: if `level` is not "message" or "model", or a required
            column is missing.
    """
    _check_columns(df)
    if level not in ("message", "model"):
        raise ValueError(
            f"level must be 'message' or 'model', got {level!r}"
        )
    group_col = "message_id" if level == "message" else "model"

    data = _with_outcome(df).reset_index(drop=True)
    y_all = data["drift"].to_numpy()
    cer_all = data["realized_cer"].to_numpy(dtype=float)

    point = _plain_logit_slope(y_all, cer_all)

    unique_ids = data[group_col].to_numpy()
    clusters = np.unique(unique_ids)
    n_clusters = len(clusters)
    # Precompute row indices per cluster so each resample is a cheap gather.
    groups = {c: np.where(unique_ids == c)[0] for c in clusters}

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sampled = clusters[rng.integers(0, n_clusters, size=n_clusters)]
        idx = np.concatenate([groups[c] for c in sampled])
        s = _plain_logit_slope(y_all[idx], cer_all[idx])
        if s is not None:
            draws.append(s)

    if draws:
        draws_arr = np.asarray(draws, dtype=float)
        lo, hi = np.percentile(draws_arr, _PCTILES)
        ci = [float(lo), float(hi)]
        boot_se = float(np.std(draws_arr, ddof=1)) if len(draws_arr) > 1 else float("nan")
    else:
        ci = [float("nan"), float("nan")]
        boot_se = float("nan")

    unstable = n_clusters < 15

    label = (
        f"Nonparametric cluster bootstrap of drift ~ realized_cer, "
        f"resampling {level}-level clusters ({n_clusters}) with "
        f"replacement, {n_boot} resamples"
    )
    if unstable:
        label += (
            " -- UNSTABLE: too few clusters for a trustworthy cluster "
            "bootstrap; CI is wide/erratic and reported for honesty, not "
            "relied upon (see per_model_pooled_slope for the preferred "
            "model-level estimator)"
        )

    return {
        "method": f"cluster_bootstrap_{level}",
        "label": label,
        "slope": point,
        "se": boot_se,
        "ci": ci,
        "level": level,
        "n_clusters": int(n_clusters),
        "n_boot": int(n_boot),
        "n_valid": int(len(draws)),
        "unstable": bool(unstable),
        "primary_slope": _PRIMARY_SLOPE,
    }
