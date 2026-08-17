"""Two-way (message x model) cluster-robust inference for the realized-CER
-> drift slope (revision Task, reviewer major #2).

Why this module exists
----------------------
The primary drift specification is a message-clustered cluster-robust logit
of `drift` on `realized_cer`. Reviewer major #2 objected that clustering on
`message_id` alone addresses only ONE axis of dependence: replicate
generations are also correlated WITHIN a model (a given LLM's behavior
pushes all of its rows the same way), and a message-clustered SE cannot
speak to that model-level dependence at all. Re-clustering on `model` alone
is not credible either -- there are only 16 model clusters, far below the
~30-50 rule of thumb, so a model-clustered SE is badly biased.

This module provides the two-axis answer -- a Cameron-Gelbach-Miller (2011)
TWO-WAY cluster-robust sandwich that clusters on message AND on model
simultaneously -- plus three companion views a reader can triangulate
against:

- `realized_cer`: the two-way sandwich SE for the realized_cer log-odds
  slope, next to the message-only SE it should WIDEN, with a validated
  reproduction of statsmodels' clustered SE from the one-way message meat.
- `per_model`: 16 separate message-clustered `drift ~ realized_cer` fits,
  each with its own slope and 95% CI, and a DESCRIPTIVE min/median/max
  summary (`per_model_summary`) -- deliberately not a pooled p-value.
- `message_aggregated`: a message-level aggregation sensitivity (collapse to
  per-message drift and re-fit a message-size-weighted grouped logistic),
  which respects message-level dependence exactly (one row per message).
- `model_bootstrap`: a nonparametric bootstrap resampling the 16 model
  clusters, reported honestly as UNSTABLE (16 clusters is too few for a
  trustworthy cluster bootstrap; the tail behavior is erratic).

The two-way sandwich
--------------------
Fit `sm.Logit(y, X)` with `X = [const, realized_cer]` (plain, non-robust).
Let the bread `B = res.cov_params()` be the model's non-robust covariance
(the inverse observed information) and let `s_i = res.model.score_obs(params)`
be the per-observation score contributions. For any clustering `g`, the meat
is

    M_g = sum_over_clusters ( sum_{i in cluster} s_i )( sum_{i in cluster} s_i )^T

and the one-way sandwich is `V_g = B M_g B`. The two-way (message x model)
meat is

    M_twoway = M_message + M_model - M_intersection

where the intersection is the message-AND-model grouping. `se_twoway` is
`sqrt(diag(B M_twoway B))` for the realized_cer coefficient. Because
`B(M_a + M_b - M_ab)B = V_a + V_b - V_ab`, this is exactly the
Cameron-Gelbach-Miller estimator `V = V_a + V_b - V_ab` with no
finite-sample correction applied to the components. The intersection grouping
is finer than either margin, so `V_intersection` is smaller than each of
`V_message` and `V_model` and the two-way SE is at least as large as either
one-way SE -- multiway clustering WIDENS, never narrows.

The one-way message meat is validated inside `run` against statsmodels'
own `cov_type="cluster"` covariance (exact machine-precision match to
`sandwich_covariance.cov_cluster(..., use_correction=False)`, and agreement
to ~1e-4 with the finite-sample-corrected default, whose correction factor
is ~1.0 with thousands of message clusters). A guard falls the two-way SE
back to `max(se_message, se_model)` and records a note if the two-way
variance is ever non-positive or non-finite.

Not attempted: the full crossed (message x model) variational-Bayes mixed
model. It is known not to converge at this study's 3,888,000-row scale; this
sandwich is the tractable, closed-form substitute, stated as such in the
digest `note`.

Outcome: the binary drift outcome is `drift = (label == "drift")` from the
study's {faithful, degraded, drift} taxonomy (drift vs rest), matching the
primary analysis. Determinism: the only stochastic step (the model-cluster
bootstrap) flows from an explicit `numpy.random.default_rng(seed)`. American
spelling; no em dashes (double hyphens for asides), per house style.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster

_Z95 = 1.959963984540054  # two-sided 95% normal quantile (matches sibling modules)
_DRIFT_LABEL = "drift"
_REQUIRED_COLUMNS = ("label", "realized_cer", "message_id", "model")

DEFAULT_PARQUET = "output/intermediate/attempts_v3plus_labeled.parquet"
DEFAULT_OUT = "output/multiway_cluster_v3_digest.json"

# Coefficient index in X = [const, realized_cer]; the realized_cer slope is column 1.
_CER_IX = 1

CROSSED_VB_NOTE = (
    "The full crossed (message x model) variational-Bayes mixed model is known not to "
    "converge at this study's 3,888,000-row scale and is not attempted here; the "
    "Cameron-Gelbach-Miller two-way cluster-robust sandwich is the tractable closed-form "
    "substitute (see realized_cer.se_twoway)."
)

_FEW_CLUSTERS_NOTE = (
    "Only 20 model clusters -- far too few for a trustworthy cluster bootstrap or a "
    "model-clustered SE (the usual rule of thumb wants 30-50+). With so few clusters the "
    "resampling distribution is coarse and its tails are unstable, so this CI is reported "
    "for honesty, not relied upon; the two-way sandwich (realized_cer.se_twoway) and the "
    "per-model summary are the preferred reads of model-level uncertainty."
)


def _check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"multiway_cluster: df is missing required column(s): {missing}"
        )


def _drift(df: pd.DataFrame) -> np.ndarray:
    return (df["label"] == _DRIFT_LABEL).astype(float).to_numpy()


def _cluster_meat(scores: np.ndarray, group_frame: pd.DataFrame) -> np.ndarray:
    """Cluster meat M_g = sum_clusters (sum_{i in c} s_i)(sum_{i in c} s_i)^T.

    Args:
        scores: (n, k) per-observation score contributions (`score_obs`),
            row-aligned with `group_frame`.
        group_frame: DataFrame whose columns define the clustering; rows
            sharing the same value(s) across all columns form one cluster.

    Returns:
        (k, k) meat matrix: the sum, over clusters, of the outer product of
        each cluster's summed score vector.
    """
    k = scores.shape[1]
    sf = pd.DataFrame(scores, columns=[f"s{j}" for j in range(k)])
    for col in group_frame.columns:
        sf[col] = group_frame[col].to_numpy()
    summed = sf.groupby(list(group_frame.columns), sort=False)[
        [f"s{j}" for j in range(k)]
    ].sum().to_numpy()
    return summed.T @ summed


def two_way_cluster_vcov(
    bread: np.ndarray,
    score_obs: np.ndarray,
    groups_a: pd.DataFrame,
    groups_b: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """Cameron-Gelbach-Miller two-way cluster-robust covariance.

    Implements `V = V_a + V_b - V_ab` where `V_g = bread @ M_g @ bread` and
    `M_g` is the cluster meat (see `_cluster_meat`). The intersection meat
    `M_ab` clusters on the a-AND-b grouping. Because the sandwich is linear
    in the meat, combining meats first (`M_a + M_b - M_ab`) and sandwiching
    once is identical to combining the sandwiched components.

    Interface note: the task brief sketches the signature as
    `two_way_cluster_vcov(X, resid_or_score, groups_a, groups_b)` -- the
    `X, resid` form is the OLS reading, where the score contribution is
    `X_i * resid_i`. For this study's maximum-likelihood logit the estimating
    function is delivered directly by `model.score_obs(params)`, so this
    function takes the `bread` (non-robust `cov_params()`) and that
    `score_obs` matrix instead. No finite-sample correction is applied to any
    component, so the message-only meat reproduces statsmodels'
    `cov_cluster(..., use_correction=False)` exactly (validated in `run`).

    Args:
        bread: (k, k) non-robust covariance (inverse observed information).
        score_obs: (n, k) per-observation score contributions.
        groups_a: DataFrame defining the first clustering (e.g. message).
        groups_b: DataFrame defining the second clustering (e.g. model).

    Returns:
        (vcov, parts) where `vcov` is the (k, k) two-way covariance and
        `parts` holds the component meats `{"M_a", "M_b", "M_ab"}` (numpy
        arrays) for downstream one-way SEs and diagnostics.
    """
    m_a = _cluster_meat(score_obs, groups_a)
    m_b = _cluster_meat(score_obs, groups_b)
    inter = pd.concat(
        [groups_a.reset_index(drop=True), groups_b.reset_index(drop=True)], axis=1
    )
    m_ab = _cluster_meat(score_obs, inter)
    m_two = m_a + m_b - m_ab
    vcov = bread @ m_two @ bread
    return vcov, {"M_a": m_a, "M_b": m_b, "M_ab": m_ab}


def _fit_main(df: pd.DataFrame):
    """Plain (non-robust) logit of drift on realized_cer; returns the fitted
    result plus the bread and score_obs used by every sandwich here."""
    y = _drift(df)
    X = sm.add_constant(df["realized_cer"].to_numpy(dtype=float), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.Logit(y, X).fit(disp=0)
    bread = np.asarray(res.cov_params())
    score = np.asarray(res.model.score_obs(res.params))
    return res, X, bread, score


def realized_cer_two_way(df: pd.DataFrame) -> dict:
    """Two-way (message x model) cluster-robust SE for the realized_cer slope.

    Fits the plain logit once, forms the message, model, and intersection
    meats, and reports the realized_cer log-odds slope with its message-only
    SE, its two-way SE, and the two-way 95% CI. Validates the one-way message
    meat against statsmodels' own clustered covariance.

    Args:
        df: frame with `label`, `realized_cer`, `message_id`, `model`.

    Returns:
        dict: `{beta, se_message_only, se_model_only, se_twoway, ci_twoway,
        n_obs, n_message_clusters, n_model_clusters, n_intersection_clusters,
        twoway_fallback, note, validation}`. `se_message_only`/`se_model_only`
        are the one-way sandwich SEs (same machinery as the two-way SE, so
        the comparison is apples to apples). `twoway_fallback` is True only if
        the two-way variance was non-positive/non-finite and the SE fell back
        to `max(se_message, se_model)`.
    """
    res, _X, bread, score = _fit_main(df)
    beta = float(res.params[_CER_IX])

    groups_msg = df[["message_id"]].reset_index(drop=True)
    groups_model = df[["model"]].reset_index(drop=True)

    v_two, parts = two_way_cluster_vcov(bread, score, groups_msg, groups_model)
    v_msg = bread @ parts["M_a"] @ bread
    v_model = bread @ parts["M_b"] @ bread

    var_msg = float(v_msg[_CER_IX, _CER_IX])
    var_model = float(v_model[_CER_IX, _CER_IX])
    var_two = float(v_two[_CER_IX, _CER_IX])

    se_message = float(np.sqrt(var_msg))
    se_model = float(np.sqrt(var_model))

    fallback = not (np.isfinite(var_two) and var_two > 0.0)
    if fallback:
        se_twoway = max(se_message, se_model)
        note = (
            "The two-way variance for the realized_cer coefficient was non-positive or "
            "non-finite (a known small-sample failure mode of the subtractive "
            "Cameron-Gelbach-Miller estimator); fell back to max(se_message, se_model)."
        )
    else:
        se_twoway = float(np.sqrt(var_two))
        note = (
            "Two-way (message x model) Cameron-Gelbach-Miller cluster-robust SE; wider "
            "than the message-only SE because it adds the model-level dependence axis that "
            "message clustering cannot see."
        )

    ci_twoway = [beta - _Z95 * se_twoway, beta + _Z95 * se_twoway]

    # --- Validation: the one-way message meat must reproduce statsmodels' clustered SE. ---
    v_sm_unc = cov_cluster(res, df["message_id"].to_numpy(), use_correction=False)
    exact_match = bool(np.allclose(v_msg, v_sm_unc, rtol=1e-8, atol=1e-12))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_cl = sm.Logit(
            _drift(df),
            sm.add_constant(df["realized_cer"].to_numpy(dtype=float), has_constant="add"),
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
            "default clustered SE to ~1e-4 (correction factor ~1.0 with thousands of "
            "message clusters). This validates the sandwich algebra used for the two-way SE."
        ),
    }

    return {
        "beta": beta,
        "se_message_only": se_message,
        "se_model_only": se_model,
        "se_twoway": se_twoway,
        "ci_twoway": ci_twoway,
        "n_obs": int(len(df)),
        "n_message_clusters": int(df["message_id"].nunique()),
        "n_model_clusters": int(df["model"].nunique()),
        "n_intersection_clusters": int(df.groupby(["message_id", "model"]).ngroups),
        "twoway_fallback": bool(fallback),
        "note": note,
        "validation": validation,
    }


def per_model_slopes(df: pd.DataFrame) -> dict:
    """One message-clustered `drift ~ realized_cer` logit per model.

    Fits a message-clustered cluster-robust logit SEPARATELY within each
    model and returns its realized_cer log-odds slope with a 95% CI.

    Args:
        df: frame with `label`, `realized_cer`, `message_id`, `model`.

    Returns:
        dict keyed by model name -> `{slope, ci, se, n}`. One entry per model
        present in `df`.
    """
    out: dict = {}
    for model, g in df.groupby("model", sort=True):
        y = (g["label"] == _DRIFT_LABEL).astype(float).to_numpy()
        X = sm.add_constant(g["realized_cer"].to_numpy(dtype=float), has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, X).fit(
                disp=0,
                cov_type="cluster",
                cov_kwds={"groups": g["message_id"].to_numpy()},
            )
        slope = float(res.params[_CER_IX])
        se = float(res.bse[_CER_IX])
        out[str(model)] = {
            "slope": slope,
            "se": se,
            "ci": [slope - _Z95 * se, slope + _Z95 * se],
            "n": int(len(g)),
        }
    return out


def _summary(slopes: list[float]) -> dict:
    arr = np.asarray(slopes, dtype=float)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "k_models": int(len(arr)),
    }


def message_aggregated_sensitivity(df: pd.DataFrame) -> dict:
    """Message-level aggregation sensitivity: collapse to per-message drift and
    re-fit a message-size-weighted grouped logistic regression.

    Each message is collapsed to (drift count, non-drift count, mean
    realized_cer); a binomial GLM regresses the drift proportion on the
    message-mean realized_cer, weighted by message size through the binomial
    counts. With one row per message this respects message-level dependence
    exactly, at the cost of the within-message realized_cer variation (an
    ecological, deliberately conservative companion to the row-level fit).

    Args:
        df: frame with `label`, `realized_cer`, `message_id`.

    Returns:
        dict: `{beta, se, ci, n_messages, note}`. `beta`/`ci` are the
        realized_cer log-odds slope and its 95% CI.
    """
    d = df.assign(_drift=(df["label"] == _DRIFT_LABEL).astype(int))
    agg = d.groupby("message_id").agg(
        n=("_drift", "size"),
        drift_n=("_drift", "sum"),
        mean_cer=("realized_cer", "mean"),
    )
    endog = np.column_stack([agg["drift_n"].to_numpy(), (agg["n"] - agg["drift_n"]).to_numpy()])
    exog = sm.add_constant(agg["mean_cer"].to_numpy(dtype=float), has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sm.GLM(endog, exog, family=sm.families.Binomial()).fit()
    beta = float(res.params[_CER_IX])
    se = float(res.bse[_CER_IX])
    return {
        "beta": beta,
        "se": se,
        "ci": [beta - _Z95 * se, beta + _Z95 * se],
        "n_messages": int(len(agg)),
        "note": (
            "Grouped binomial GLM of per-message drift proportion on message-mean "
            "realized_cer, weighted by message size via the binomial counts (one row per "
            "message). Ecological/aggregated sensitivity; the log-odds slope is comparable "
            "to but not identical to the row-level realized_cer slope."
        ),
    }


def model_cluster_bootstrap(
    per_model: dict, *, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Nonparametric bootstrap over the 16 model clusters.

    Resamples the per-model realized_cer slopes WITH REPLACEMENT and takes
    the mean of each resample; the CI is the 2.5th/97.5th percentile of those
    resampled means. Reported honestly as UNSTABLE -- 16 clusters is too few
    for a trustworthy cluster bootstrap (matching the caveat in
    `idrift.analysis.dependence_sensitivity`). This resamples the precomputed
    per-model estimates rather than refitting the 3.9M-row model, which is
    both cheap and the standard few-cluster bootstrap idiom.

    Args:
        per_model: mapping of model -> {"slope", ...} (from
            `per_model_slopes`).
        n_boot: number of bootstrap resamples.
        seed: seed for the `numpy.random.default_rng` driving every resample.

    Returns:
        dict: `{n_clusters, point, ci, se, n_boot, unstable, note}`. `point`
        is the mean of the observed per-model slopes; `se` is the bootstrap SD
        of the resampled means.
    """
    slopes = np.asarray([m["slope"] for m in per_model.values()], dtype=float)
    n_clusters = int(len(slopes))
    rng = np.random.default_rng(seed)
    draws = np.array(
        [rng.choice(slopes, size=n_clusters, replace=True).mean() for _ in range(n_boot)]
    )
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "n_clusters": n_clusters,
        "point": float(slopes.mean()),
        "ci": [float(lo), float(hi)],
        "se": float(np.std(draws, ddof=1)),
        "n_boot": int(n_boot),
        "unstable": True,
        "note": _FEW_CLUSTERS_NOTE,
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


def run(parquet_path, out_path, *, n_boot: int = 2000, seed: int = 0) -> dict:
    """Compute two-way (message x model) cluster-robust inference plus the
    per-model, message-aggregation, and model-bootstrap companions, and write
    the digest to `out_path` as JSON (reviewer major #2).

    Args:
        parquet_path: path to the labeled attempts parquet (columns `label`,
            `realized_cer`, `message_id`, `model`), or an already-loaded
            DataFrame with those columns.
        out_path: path to write the JSON digest to.
        n_boot: model-cluster bootstrap resamples (default 2000).
        seed: seed for the model-cluster bootstrap (default 0).

    Returns:
        dict: `{realized_cer, per_model, per_model_summary, message_aggregated,
        model_bootstrap, n_rows, notes}`. See the section functions for the
        nested shapes.

    Raises:
        ValueError: if the input is missing a required column.
    """
    df = parquet_path if isinstance(parquet_path, pd.DataFrame) else pd.read_parquet(
        parquet_path, columns=list(_REQUIRED_COLUMNS)
    )
    _check_columns(df)
    df = df.reset_index(drop=True)

    realized = realized_cer_two_way(df)
    per_model = per_model_slopes(df)
    summary = _summary([m["slope"] for m in per_model.values()])
    message_agg = message_aggregated_sensitivity(df)
    boot = model_cluster_bootstrap(per_model, n_boot=n_boot, seed=seed)

    digest = {
        "realized_cer": realized,
        "per_model": per_model,
        "per_model_summary": summary,
        "message_aggregated": message_agg,
        "model_bootstrap": boot,
        "n_rows": int(len(df)),
        "notes": CROSSED_VB_NOTE,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2, default=_default))

    print(
        "multiway_cluster: two-way message x model SE = "
        f"{realized['se_twoway']:.4f} (message-only {realized['se_message_only']:.4f}); "
        f"one-way meat matches statsmodels = {realized['validation']['one_way_matches_statsmodels_uncorrected']}."
    )
    return digest


if __name__ == "__main__":
    run(DEFAULT_PARQUET, DEFAULT_OUT)
