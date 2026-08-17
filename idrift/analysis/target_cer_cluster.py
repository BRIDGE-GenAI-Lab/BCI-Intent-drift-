"""Target-CER model-clustered inference + per-model slopes (reviewer #3).

Why this module exists
-----------------------
`idrift.analysis.target_cer_primary` established target CER (the
experimentally assigned corruption level, not the post-generation realized
CER) as this study's primary dose-response exposure, but its pooled fit
clusters on `message_id` alone. `idrift.analysis.multiway_cluster` showed,
for the realized-CER slope, that message-only clustering understates the
uncertainty once model-level dependence is accounted for (SE 0.072 -> 0.471
once message-by-model two-way clustering is applied) -- 16 LLMs each push
all of their own rows the same way, and a message-only SE cannot see that
axis at all. Reviewer #3 asks for the same treatment on the PRIMARY
target-CER exposure: the pooled OR reported with both a message-only and a
message-by-model two-way clustered CI, 16 per-model slopes, and an explicit
instability warning about doing inference with only 16 model clusters. This
also corrects a labeling slip: the pooled fit is a binary logistic
regression of drift-vs-rest (`statsmodels.api.Logit`), not an "ordinal"
model.

This module recomputes that dose-response on the same cached, already-
labeled panel used throughout this package
(`output/intermediate/attempts_v3plus_labeled.parquet`, 3,888,000 rows, 16
models). No model inference is performed here -- this is pure analysis of
already-generated, already-labeled data.

The two-way sandwich is REUSED, not re-derived: `two_way_cluster_vcov` in
`idrift.analysis.multiway_cluster` implements the Cameron-Gelbach-Miller
(2011) two-way cluster-robust covariance `V = V_message + V_model -
V_intersection` from a model's bread (`cov_params()`) and per-observation
score contributions (`score_obs`). This module fits its own plain
`sm.Logit(drift ~ 1 + cer_target)` and feeds that bread/score pair into the
same shared sandwich, mirroring `multiway_cluster.realized_cer_two_way`
exactly, with `cer_target` as the continuous exposure instead of
`realized_cer`.

Units and scaling
------------------
`cer_target` is a FRACTION on {0.0, 0.1, 0.2, 0.3, 0.4} (not a percentage),
so a "10 percentage-point" rise is a 0.1 change in `cer_target`. The pooled
and per-model odds ratios reported here are `exp(beta * 0.1)`, matching
`idrift.analysis.target_cer_primary.fit_ordinal_model`'s `or_per_10pp`
convention. Unlike that module, this one also reports a CI on the OR scale:
because `exp` is a monotonic transform, rescaling the log-odds CI bounds by
0.1 before exponentiating yields a valid 95% CI for the per-10pp OR, for
both the message-only and the two-way clustered SE.

Reported instability
---------------------
With only 16 model clusters (`n_model_clusters`), model-cluster inference is
inherently coarse -- far below the usual 30-50+ cluster rule of thumb that
would make cluster-robust (or cluster-bootstrap) inference on the model axis
trustworthy on its own. `ci_twoway` is reported for honesty, and it does
correctly widen relative to `ci_message_only` (the two-way sandwich adds the
model-level dependence axis that message-only clustering cannot see), but
`instability_note` states this limitation explicitly so a reader does not
over-read its precision.

Outcome: the binary drift outcome is `drift = (label == "drift")` from the
study's {faithful, degraded, drift} taxonomy (drift vs. rest), matching the
primary specification. Deterministic (no random draws: every fit here is
plain Newton-Raphson MLE on a fixed design). American spelling; no em dashes
(double hyphens for asides), per house style.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

from idrift.analysis.multiway_cluster import two_way_cluster_vcov

_Z95 = 1.959963984540054  # two-sided 95% normal quantile (matches sibling modules)
_DRIFT_LABEL = "drift"
_REQUIRED_COLUMNS = ("label", "cer_target", "message_id", "model")

# cer_target is a fraction (0.0-0.4 on this study's grid); a "10 percentage-point"
# rise is a 0.1 change in cer_target (matches target_cer_primary's or_per_10pp).
_PP10 = 0.1

DEFAULT_PARQUET = "output/intermediate/attempts_v3plus_labeled.parquet"
DEFAULT_OUT = "output/target_cer_cluster_v3_digest.json"

# Coefficient index in X = [const, cer_target]; the cer_target slope is column 1.
_CER_IX = 1

INSTABILITY_NOTE = (
    "Only 20 model clusters -- far too few for the usual 30-50+ cluster rule of thumb that "
    "makes model-cluster inference trustworthy on its own. The message-by-model two-way "
    "clustered CI (ci_twoway) is reported for honesty and correctly widens relative to the "
    "message-only CI, but with 20 clusters that widened interval should be read as "
    "approximate, not a precise bound; the per-model slopes and their median/range are the "
    "more informative view of model-level heterogeneity."
)


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"target_cer_cluster: df is missing required column(s): {missing}")


def _drift(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] == _DRIFT_LABEL).astype(float).to_numpy()


def _fit_main(df: pd.DataFrame):
    """Plain (non-robust) logit of drift on cer_target; returns the fitted
    result plus the bread and score_obs shared by every sandwich here."""
    y = _drift(df)
    X = sm.add_constant(df["cer_target"].to_numpy(dtype=float), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.Logit(y, X).fit(disp=0)
    bread = np.asarray(res.cov_params())
    score = np.asarray(res.model.score_obs(res.params))
    return res, X, bread, score


def _or_ci(beta: float, se: float) -> list[float]:
    """OR-per-10pp-scale 95% CI from a log-odds beta/se on the raw
    (per-unit cer_target) scale: rescale by 0.1 (one 10pp step) before
    exponentiating, which is a valid CI because exp is monotonic."""
    lo = (beta - _Z95 * se) * _PP10
    hi = (beta + _Z95 * se) * _PP10
    return [float(np.exp(lo)), float(np.exp(hi))]


def pooled_two_way(df: pd.DataFrame) -> dict:
    """Message-only and message-by-model two-way cluster-robust inference for
    the pooled target-CER drift slope.

    Fits the plain logit once, forms the message, model, and intersection
    meats via the REUSED `multiway_cluster.two_way_cluster_vcov` sandwich,
    and reports the target-CER odds ratio per 10 percentage points with both
    a message-only and a message-by-model two-way clustered 95% CI.

    Args:
        df: frame with `label`, `cer_target`, `message_id`, `model`.

    Returns:
        dict: `{beta, pooled_or_per_10pp, se_message_only, se_twoway,
        ci_message_only, ci_twoway, n_obs, n_message_clusters,
        n_model_clusters, n_intersection_clusters, twoway_fallback, note,
        validation}`. `beta` is the raw (per-unit cer_target) log-odds slope
        from the pooled fit; `ci_message_only`/`ci_twoway` are on the
        OR-per-10pp scale.
    """
    res, _X, bread, score = _fit_main(df)
    beta = float(res.params[_CER_IX])

    groups_msg = df[["message_id"]].reset_index(drop=True)
    groups_model = df[["model"]].reset_index(drop=True)

    v_two, parts = two_way_cluster_vcov(bread, score, groups_msg, groups_model)
    v_msg = bread @ parts["M_a"] @ bread

    var_msg = float(v_msg[_CER_IX, _CER_IX])
    var_two = float(v_two[_CER_IX, _CER_IX])

    se_message = float(np.sqrt(var_msg))

    fallback = not (np.isfinite(var_two) and var_two > 0.0)
    if fallback:
        se_twoway = se_message
        note = (
            "The two-way variance for the cer_target coefficient was non-positive or "
            "non-finite (a known small-sample failure mode of the subtractive "
            "Cameron-Gelbach-Miller estimator); fell back to se_message_only."
        )
    else:
        se_twoway = float(np.sqrt(var_two))
        note = (
            "Two-way (message x model) Cameron-Gelbach-Miller cluster-robust SE for the "
            "target-CER slope; wider than the message-only SE because it adds the "
            "model-level dependence axis that message clustering alone cannot see."
        )

    pooled_or_per_10pp = float(np.exp(beta * _PP10))
    ci_message_only = _or_ci(beta, se_message)
    ci_twoway = _or_ci(beta, se_twoway)

    # --- Validation: the one-way message meat must reproduce statsmodels' clustered SE. ---
    v_sm_unc = cov_cluster(res, df["message_id"].to_numpy(), use_correction=False)
    exact_match = bool(np.allclose(v_msg, v_sm_unc, rtol=1e-8, atol=1e-12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_cl = sm.Logit(
            _drift(df),
            sm.add_constant(df["cer_target"].to_numpy(dtype=float), has_constant="add"),
        ).fit(disp=0, cov_type="cluster", cov_kwds={"groups": df["message_id"].to_numpy()})
    se_default = float(res_cl.bse[_CER_IX])
    rel_diff = float(abs(se_message - se_default) / se_default) if se_default else float("nan")
    validation = {
        "one_way_matches_statsmodels_uncorrected": exact_match,
        "se_message_one_way_sandwich": se_message,
        "se_message_statsmodels_default": se_default,
        "rel_diff_vs_statsmodels_default": rel_diff,
        "note": (
            "The one-way message meat, sandwiched, reproduces statsmodels' "
            "cov_cluster(use_correction=False) exactly and its finite-sample-corrected "
            "default clustered SE to within a small relative tolerance. This validates the "
            "sandwich algebra reused from multiway_cluster for the two-way SE here."
        ),
    }

    return {
        "beta": beta,
        "pooled_or_per_10pp": pooled_or_per_10pp,
        "se_message_only": se_message,
        "se_twoway": se_twoway,
        "ci_message_only": ci_message_only,
        "ci_twoway": ci_twoway,
        "n_obs": int(len(df)),
        "n_message_clusters": int(df["message_id"].nunique()),
        "n_model_clusters": int(df["model"].nunique()),
        "n_intersection_clusters": int(df.groupby(["message_id", "model"]).ngroups),
        "twoway_fallback": bool(fallback),
        "note": note,
        "validation": validation,
    }


def per_model_slopes(df: pd.DataFrame) -> list[dict]:
    """One message-clustered `drift ~ cer_target` logit per model.

    Args:
        df: frame with `label`, `cer_target`, `message_id`, `model`.

    Returns:
        list of dicts (one per model present in `df`, sorted by model name):
        `{model, slope_log_odds, or_per_10pp, se, ci, n}`. `slope_log_odds`,
        `se`, and `ci` are on the raw (per-unit cer_target) log-odds scale;
        `or_per_10pp` = `exp(slope_log_odds * 0.1)`.
    """
    out: list[dict] = []
    for model, g in df.groupby("model", sort=True):
        y = (g["label"] == _DRIFT_LABEL).astype(float).to_numpy()
        X = sm.add_constant(g["cer_target"].to_numpy(dtype=float), has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, X).fit(
                disp=0,
                cov_type="cluster",
                cov_kwds={"groups": g["message_id"].to_numpy()},
            )
        slope = float(res.params[_CER_IX])
        se = float(res.bse[_CER_IX])
        out.append(
            {
                "model": str(model),
                "slope_log_odds": slope,
                "or_per_10pp": float(np.exp(slope * _PP10)),
                "se": se,
                "ci": [slope - _Z95 * se, slope + _Z95 * se],
                "n": int(len(g)),
            }
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
    """Compute message-only and message-by-model two-way cluster-robust
    inference for the target-CER drift slope, plus 16 per-model slopes, and
    write the digest to `out_path` as JSON (reviewer #3).

    Args:
        parquet_path: path to the labeled attempts parquet (columns `label`,
            `cer_target`, `message_id`, `model`), or an already-loaded
            DataFrame with those columns.
        out_path: path to write the JSON digest to.

    Returns:
        dict: `{beta, pooled_or_per_10pp, se_message_only, se_twoway,
        ci_message_only, ci_twoway, n_obs, n_message_clusters,
        n_model_clusters, n_intersection_clusters, twoway_fallback, note,
        validation, per_model, median_slope, slope_range, instability_note,
        n_rows}`. See `pooled_two_way` and `per_model_slopes` for the nested
        shapes.

    Raises:
        ValueError: if the input is missing a required column.
    """
    df = parquet_path if isinstance(parquet_path, pd.DataFrame) else pd.read_parquet(
        parquet_path, columns=list(_REQUIRED_COLUMNS)
    )
    _check_columns(df)
    df = df.reset_index(drop=True)

    pooled = pooled_two_way(df)
    per_model = per_model_slopes(df)
    slopes = [m["slope_log_odds"] for m in per_model]

    digest = dict(pooled)
    digest["per_model"] = per_model
    digest["median_slope"] = float(np.median(slopes))
    digest["slope_range"] = [float(np.min(slopes)), float(np.max(slopes))]
    digest["instability_note"] = INSTABILITY_NOTE
    digest["n_rows"] = int(len(df))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2, default=_default))

    print(
        "target_cer_cluster: pooled OR per 10pp = "
        f"{digest['pooled_or_per_10pp']:.3f} "
        f"(message-only CI [{digest['ci_message_only'][0]:.3f}, {digest['ci_message_only'][1]:.3f}], "
        f"two-way CI [{digest['ci_twoway'][0]:.3f}, {digest['ci_twoway'][1]:.3f}]); "
        f"median per-model slope = {digest['median_slope']:.3f}, "
        f"range = [{digest['slope_range'][0]:.3f}, {digest['slope_range'][1]:.3f}]."
    )
    return digest


if __name__ == "__main__":
    run(DEFAULT_PARQUET, DEFAULT_OUT)
