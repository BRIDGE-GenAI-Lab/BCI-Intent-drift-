"""Target CER as the primary experimental exposure (reviewer major #3).

Why this module exists
-----------------------
The study's existing primary specification (see the manuscript Methods and
`idrift.analysis.confidence_analysis.predicted_drift_grid`) is a
message-clustered cluster-robust logistic regression of drift on
`realized_cer`: the corruption rate actually measured in a given message's
generated output, a POST-generation quantity. A reviewer (major #3) pointed
out that `realized_cer` is not what the experiment assigned -- `cer_target`
(the intended corruption level drawn for that exposure, one of
{0.0, 0.1, 0.2, 0.3, 0.4}) is the experimentally assigned exposure, and
should therefore be the PRIMARY dose-response, with realized CER reported
as a secondary, descriptive quantity (see a later task; not recomputed
here).

This module recomputes the primary dose-response with `cer_target` as the
sole exposure, on the same cached, already-labeled panel used throughout
this package (`output/intermediate/attempts_v3plus_labeled.parquet`,
3,888,000 rows, 16 models). No model inference is performed here -- this is
pure analysis of already-generated, already-labeled data.

Three views of the same dose-response are reported:

- `drift_rate_by_corpus_and_level` -- the non-parametric drift rate at each
  target-CER grid point, within each corpus (AUTH/CRIT/CTRL). This is the
  descriptive analogue of `idrift.analysis.drift_curve.drift_rate_by_cer`,
  just keyed by corpus explicitly.
- `fit_ordinal_model` -- target CER treated as a continuous, linear-in-target
  predictor (the "dose" reading), reporting the log-odds slope with its 95%
  CI and a derived odds ratio per 10-percentage-point increase in target
  CER. A CI on the log-odds slope already implies one on any monotonic
  transform of it (the derived OR included here), so a second CI is not
  separately computed for `or_per_10pp` -- matching this study's existing
  realized-CER reporting convention (see the manuscript Results: "log-odds
  4.77 [95% CI, 4.43-5.11] per unit realized CER, an odds ratio of 1.61 per
  10-percentage-point increase").
- `fit_categorical_model` -- target CER treated as a categorical factor
  (reference level 0.0), reporting each non-reference level's own odds
  ratio and 95% CI -- the flexible, non-monotonicity-assuming companion to
  the ordinal fit, matching `idrift.analysis.multinomial`'s convention of
  reporting odds ratios with their own CI per categorical term.

Both model fits use the full pooled panel (all three corpora together),
mirroring how the realized-CER primary specification is fit
(`predicted_drift_grid` is called on the full labeled frame, not restricted
to one corpus -- see `idrift.analysis.run_v3.build_confidence_digest`).

Clustering
----------
Every response for one BCI message shares that message's assigned
corruption draw, so rows are not independent within `message_id`. Both fits
therefore use message-clustered cluster-robust standard errors
(`cov_type="cluster", cov_kwds={"groups": message_id}`), the same clustering
unit used throughout this package.

Realized CER is NOT an exposure anywhere in this module; it is reported
separately (secondary dose-response) by a later task.

Deterministic (no random draws: both fits are plain Newton-Raphson MLE on a
fixed design). American spelling; no em dashes (double hyphens for asides),
per house style.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# The study's cer_target grid (matches idrift.analysis.confidence_analysis.CER_GRID).
CER_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.4)
REFERENCE_LEVEL = 0.0

_REQUIRED_COLUMNS = ("cer_target", "label", "message_id", "model", "corpus")

_Z95 = 1.959963984540054  # two-sided 95% normal quantile (matches idrift.analysis.multinomial)

DEFAULT_PARQUET = "output/intermediate/attempts_v3plus_labeled.parquet"
DEFAULT_OUT = "output/target_cer_primary_v3_digest.json"

SECONDARY_NOTE = (
    "Realized CER (the corruption rate actually realized in a message's generated output, a "
    "post-generation quantity) is reported separately as the secondary dose-response; this "
    "module treats only the experimentally assigned target CER as the primary exposure "
    "(reviewer major #3)."
)


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"target_cer_primary: df is missing required column(s): {missing}")


def drift_rate_by_corpus_and_level(df: pd.DataFrame) -> dict:
    """Non-parametric drift rate at each target-CER grid level, within each
    corpus.

    Args:
        df: DataFrame with `corpus`, `cer_target`, `label` columns.

    Returns:
        dict: {corpus (str): {cer_target level (float): drift rate
        (float)}}, one entry per (corpus, level) combination observed in
        `df`.
    """
    d = df.assign(is_drift=(df["label"] == "drift").astype(int))
    by_corpus: dict[str, dict[float, float]] = {}
    for corpus, group in d.groupby("corpus"):
        rates = group.groupby("cer_target")["is_drift"].mean().sort_index()
        by_corpus[str(corpus)] = {float(level): float(rate) for level, rate in rates.items()}
    return by_corpus


def fit_ordinal_model(df: pd.DataFrame) -> dict:
    """Message-clustered logistic regression of drift on target CER,
    treated as continuous (linear-in-target / ordinal dose-response).

    Engine: `statsmodels.api.Logit` (`label == "drift"` ~ 1 + `cer_target`),
    cluster-robust on `message_id`, fit on the full pooled panel (all
    corpora together).

    Args:
        df: DataFrame with `label`, `cer_target`, `message_id` columns.

    Returns:
        dict: `{beta, se, ci, or_per_10pp, n_obs, n_clusters, converged,
        notes}`. `beta`/`ci` are the log-odds slope for `cer_target` and its
        95% CI; `or_per_10pp` = `exp(beta * 0.1)` is a derived point
        estimate (see module docstring for why it has no separate CI).
    """
    y = (df["label"] == "drift").astype(int)
    exog = sm.add_constant(df[["cer_target"]].astype(float), has_constant="add")
    groups = df["message_id"].to_numpy()

    model = sm.Logit(y, exog)
    result = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})

    beta = float(result.params["cer_target"])
    se = float(result.bse["cer_target"])
    ci = [beta - _Z95 * se, beta + _Z95 * se]
    or_per_10pp = float(np.exp(beta * 0.1))
    converged = bool(result.mle_retvals.get("converged", True))

    return {
        "beta": beta,
        "se": se,
        "ci": ci,
        "or_per_10pp": or_per_10pp,
        "n_obs": int(result.nobs),
        "n_clusters": int(pd.Series(groups).nunique()),
        "converged": converged,
        "notes": (
            "engine=statsmodels.api.Logit (label=='drift' ~ 1 + cer_target), "
            "cov_type='cluster', cov_kwds={'groups': message_id}, fit on the full pooled "
            "panel (all corpora). beta/ci are on the log-odds scale; or_per_10pp = "
            "exp(beta * 0.1) is a derived point estimate, not separately re-fit."
        ),
    }


def fit_categorical_model(df: pd.DataFrame) -> dict:
    """Message-clustered logistic regression of drift on target CER,
    treated as a categorical factor (reference level 0.0), reporting each
    non-reference level's own odds ratio and 95% CI.

    Engine: `statsmodels.formula.api.logit`
    (`drift ~ C(cer_target, Treatment(reference=0.0))`), cluster-robust on
    `message_id`, fit on the full pooled panel (all corpora together).

    Args:
        df: DataFrame with `label`, `cer_target`, `message_id` columns.

    Returns:
        dict keyed by each non-reference level as a "0.1"-style string ->
        `{or, ci, p}` (odds ratio vs. the 0.0 reference, its 95% CI on the
        OR scale, and the Wald p-value), plus sibling keys
        `reference_level`, `n_obs`, `n_clusters`, `converged`, `notes`.
    """
    data = df.copy()
    data["drift"] = (data["label"] == "drift").astype(int)
    groups = data["message_id"].to_numpy()

    formula = "drift ~ C(cer_target, Treatment(reference=0.0))"
    model = smf.logit(formula, data=data)
    result = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})

    out: dict = {}
    for level in CER_LEVELS:
        if level == REFERENCE_LEVEL:
            continue
        term = f"C(cer_target, Treatment(reference=0.0))[T.{level}]"
        coef = float(result.params[term])
        se = float(result.bse[term])
        out[f"{level:.1f}"] = {
            "or": float(np.exp(coef)),
            "ci": [float(np.exp(coef - _Z95 * se)), float(np.exp(coef + _Z95 * se))],
            "p": float(result.pvalues[term]),
        }

    out["reference_level"] = REFERENCE_LEVEL
    out["n_obs"] = int(result.nobs)
    out["n_clusters"] = int(pd.Series(groups).nunique())
    out["converged"] = bool(result.mle_retvals.get("converged", True))
    out["notes"] = (
        "engine=statsmodels.formula.api.logit "
        "(drift ~ C(cer_target, Treatment(reference=0.0))), cov_type='cluster', "
        "cov_kwds={'groups': message_id}, fit on the full pooled panel (all corpora). "
        "or/ci are on the odds-ratio scale, each non-reference level vs. the 0.0 reference."
    )
    return out


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


def run(parquet_path, out_path) -> dict:
    """Recompute the primary drift dose-response using target CER (the
    experimentally assigned corruption level) as the primary exposure,
    per reviewer major #3, and write the result to `out_path` as JSON.

    Args:
        parquet_path: path to the labeled attempts parquet (columns
            `cer_target`, `label`, `message_id`, `model`, `corpus`), or an
            already-loaded DataFrame with those columns.
        out_path: path to write the JSON digest to.

    Returns:
        dict: `{by_corpus, ordinal_model, categorical_model, n_rows,
        notes}`. See `drift_rate_by_corpus_and_level`, `fit_ordinal_model`,
        and `fit_categorical_model` for the nested shapes. `notes` restates
        that realized CER is a separate, secondary dose-response (not
        computed by this module).

    Raises:
        ValueError: if the input is missing a required column.
    """
    df = parquet_path if isinstance(parquet_path, pd.DataFrame) else pd.read_parquet(parquet_path)
    _check_columns(df)

    digest = {
        "by_corpus": drift_rate_by_corpus_and_level(df),
        "ordinal_model": fit_ordinal_model(df),
        "categorical_model": fit_categorical_model(df),
        "n_rows": int(len(df)),
        "notes": SECONDARY_NOTE,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2, default=_default))

    print("target_cer_primary: realized CER is reported separately as the secondary dose-response (T2), not computed here.")

    return digest


if __name__ == "__main__":
    run(DEFAULT_PARQUET, DEFAULT_OUT)
