"""Calibration v2: per-model ECE/AUROC with message-clustered bootstrap
CIs, and DerSimonian-Laird random-effects meta-analysis across models
(revision Task 5.2).

Why v2, not a change to `idrift.analysis.calibration`
------------------------------------------------------
The Task 8 calibration module (`idrift.analysis.calibration`) reports a
single pooled ECE/AUROC over every row, with an iid (unclustered) bootstrap
implied by its callers. Reviewers flagged two problems with that as the
headline calibration analysis:

1. Every model verbalizes its own confidence on its own internal scale --
   "80% confident" from one model family is not the same probability
   statement as "80% confident" from another. Pooling all rows together
   into one ECE silently assumes those scales are comparable; they are
   not. Confidence must be reported PER MODEL, and if pooled across models
   at all, only via a random-effects meta-analysis that explicitly models
   (and reports) the resulting between-model heterogeneity -- never a
   naive fixed pool that pretends every model's confidence sits on one
   scale.
2. Generation replicates within the same source message are correlated
   (the same underlying decode noise or LLM behavior pushes every
   replicate of one message the same way), so a bootstrap that resamples
   individual ROWS treats every one of those correlated replicates as an
   independent unit and understates uncertainty. The CI must resample
   distinct `message_id`s (a cluster bootstrap), the same idiom already
   used in `idrift.adjudicate.misclassification`.

This module is therefore new, not a patch to `calibration.py`: the input
contract, bootstrap idiom, and pooling strategy are all different, and the
old module's single-scale pooled ECE/AUROC remains available for whatever
(if anything) still calls it.

Input contract
---------------
`per_model_calibration` consumes a tidy per-generation DataFrame with
columns `confidence` (float, already rescaled to [0, 1] -- see
`build_calibration_frame` below), `correct` (0/1 or bool), `model`, and
`message_id`. It does NOT derive `correct` or rescale `confidence` itself;
that derivation is a separate, explicitly-named step (`build_calibration_
frame`) so a caller can never accidentally feed in an un-rescaled 0-100
verbalized confidence and get a silently-wrong ECE -- `ece` raises
`ValueError` instead if any confidence value falls outside [0, 1].

ECE (`ece`)
------------
Standard binned Expected Calibration Error:

    ECE = sum over bins b of (n_b / N) * |acc_b - mean_conf_b|

where `acc_b` is the empirical fraction correct in bin `b` and
`mean_conf_b` is the mean stated confidence in bin `b`. Two binning
schemes are supported, both spanning [0, 1]:

- `"equal_width"`: `n_bins` bins of equal width (`numpy.linspace(0, 1,
  n_bins + 1)` edges).
- `"equal_frequency"`: `n_bins` quantile bins of the observed confidence
  distribution (`numpy.quantile` edges), so each bin holds (as close to)
  an equal share of the data regardless of how skewed the confidence
  distribution is. Duplicate edges (e.g. many tied confidence values
  collapsing several quantile cuts to the same point) are deduplicated,
  which can yield fewer than `n_bins` effective bins; this is the standard
  quantile-binning behavior and is documented, not silently masked.

Every bin's right edge is exclusive except the last bin's, which is
inclusive, so a confidence of exactly 1.0 lands in the final bin instead
of being silently dropped from every bin's count. An empty bin contributes
0 to the sum (its `n_b / N` weight is 0).

Message-clustered bootstrap (`per_model_calibration`)
------------------------------------------------------
Per model, `per_model_calibration` reports the ECE and AUROC point
estimates on that model's full (unresampled) data, plus a 95% CI for each
built with the SAME message-clustered bootstrap idiom as
`idrift.adjudicate.misclassification`: each of `n_boot` iterations
independently resamples that model's distinct `message_id`s WITH
REPLACEMENT, pulls every row belonging to each resampled message (so
within-message correlation in confidence/correctness is preserved), and
recomputes ECE/AUROC on the resulting resampled frame. The CI is the
2.5th/97.5th percentile of the bootstrap distribution. All randomness
comes from a single `numpy.random.default_rng(seed)` constructed once per
call (never the global `numpy.random` state), consumed in a fixed model
iteration order (`pandas.DataFrame.groupby` sorts group keys by default),
so the same `(df, seed)` always reproduces bit-identical output.

AUROC (`sklearn.metrics.roc_auc_score` of confidence vs. `correct`) is
undefined when a model's sample (or a bootstrap resample of it) has only
one distinct `correct` class -- e.g. every generation from that model
happened to be labeled correct in the validation window used. This is
guarded explicitly: `per_model_calibration` reports `auroc: None` with a
reason folded into `notes` instead of letting `roc_auc_score` raise, and
any one-class bootstrap draw is dropped from the AUROC CI's percentile
computation (rather than crashing the whole run over one degenerate draw).

Confidence-comparability caveat
----------------------------------
Every per-model result's `notes` field states, verbatim, that verbalized
confidence is not calibrated or comparable across models -- the reviewer-
facing caveat that justifies why cross-model pooling (see `meta_analyze`)
uses random effects rather than a naive fixed pool.

DerSimonian-Laird random-effects meta-analysis (`meta_analyze`)
-------------------------------------------------------------------
Given the per-model dict `per_model_calibration` returns, `meta_analyze`
pools one metric (`"ece"` or `"auroc"`) across models via the standard
DerSimonian-Laird (1986) random-effects estimator:

    y_i, v_i   : each model's point estimate and its bootstrap-CI-derived
                 variance, v_i = se_i^2, se_i ~= (ci_hi_i - ci_lo_i) / (2 *
                 1.96) (the CI half-width implies the SE under a normal
                 approximation -- exactly the reverse of how the CI was
                 built in `per_model_calibration`).
    w_i        = 1 / v_i                          (fixed-effect weights)
    mu_FE      = sum(w_i * y_i) / sum(w_i)         (fixed-effect pooled mean)
    Q          = sum(w_i * (y_i - mu_FE)^2)        (Cochran's Q)
    df         = k - 1                              (k = number of models)
    C          = sum(w_i) - sum(w_i^2) / sum(w_i)
    tau2       = max(0, (Q - df) / C)               (between-model variance)
    w_i*       = 1 / (v_i + tau2)                   (random-effects weights)
    pooled     = sum(w_i* * y_i) / sum(w_i*)
    se_pooled  = sqrt(1 / sum(w_i*))
    ci         = pooled +/- 1.96 * se_pooled
    I2         = max(0, (Q - df) / Q) * 100         (% variance from
                                                       heterogeneity, 0 if
                                                       Q == 0)

Models missing the requested metric (e.g. `auroc: None` from the one-class
guard) are excluded from the pool and do not count toward `k_models`. With
zero poolable models, `pooled`/`ci`/`tau2`/`q`/`i2` are all `None` and
`k_models` is 0. With exactly one poolable model, there is no between-model
variance to estimate (`tau2 = 0`, `q = 0`, `i2 = 0`) and the "pooled"
estimate/CI is just that one model's own normal-approximation CI. The
returned dict always carries the same comparability `caveat` as the
per-model notes, describing the pooled estimate as descriptive rather than
a claim that confidence is calibrated on a shared scale across models.

House style: American spelling, no em dashes (double hyphens for asides).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_PCTILES = (2.5, 97.5)
_Z95 = 1.96

_COMPARABILITY_CAVEAT = (
    "Verbalized model confidence is not calibrated on a common scale "
    "across models -- an '80% confident' statement from one model is not "
    "the same probability claim as from another. Calibration is reported "
    "per model; any cross-model pooling uses random-effects meta-analysis "
    "(DerSimonian-Laird), and the pooled estimate is descriptive, not a "
    "claim of one shared calibration scale."
)

_REQUIRED_CALIBRATION_COLUMNS = ("confidence", "correct", "model", "message_id")


def _check_conf_range(conf: np.ndarray) -> None:
    if not conf.size:
        return
    n_nan = int(np.isnan(conf).sum())
    if n_nan:
        raise ValueError(
            f"conf contains {n_nan} NaN value(s) -- build_calibration_frame "
            "should already have dropped rows with missing confidence before "
            "this point; a NaN reaching the metric layer means the caller "
            "bypassed that drop."
        )
    if np.nanmin(conf) < 0.0 or np.nanmax(conf) > 1.0:
        raise ValueError(
            "conf must be in [0, 1] (got values outside that range) -- "
            "rescale a 0-100 verbalized confidence with build_calibration_frame "
            "or an explicit /100 before calling ece()."
        )


def _bin_edges(conf: np.ndarray, n_bins: int, scheme: str) -> np.ndarray:
    if scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif scheme == "equal_frequency":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(conf, quantiles)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges = np.unique(edges)
    else:
        raise ValueError(
            f"scheme must be 'equal_width' or 'equal_frequency', got {scheme!r}"
        )
    return edges


def ece(conf, correct, n_bins: int = 10, scheme: str = "equal_width") -> float:
    """Expected Calibration Error: bin-weighted |accuracy - confidence| gap.

    Args:
        conf: array-like confidence values, MUST already be in [0, 1]
            (raises `ValueError` otherwise -- see module docstring).
        correct: array-like binary correctness indicator, same length as
            `conf`.
        n_bins: number of bins.
        scheme: `"equal_width"` (equal-width bins spanning [0, 1]) or
            `"equal_frequency"` (quantile bins of the observed `conf`
            distribution).

    Returns:
        float: the expected calibration error, in [0, 1]. An empty bin
            contributes 0.
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    _check_conf_range(conf)

    n = len(conf)
    if n == 0:
        return 0.0

    edges = _bin_edges(conf, n_bins, scheme)

    total = 0.0
    n_edges = len(edges)
    for i in range(n_edges - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == n_edges - 2:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc_bin = correct[mask].mean()
        conf_bin = conf[mask].mean()
        total += (cnt / n) * abs(acc_bin - conf_bin)
    return float(total)


def _safe_auroc(conf: np.ndarray, correct: np.ndarray):
    """AUROC of confidence predicting correctness, guarding the degenerate
    one-class case (undefined) instead of letting `roc_auc_score` raise.

    Returns:
        (auroc, reason): `auroc` is `float` or `None`; `reason` is `None`
        or a human-readable string naming why AUROC is undefined.
    """
    y = np.asarray(correct)
    if len(np.unique(y)) < 2:
        return None, (
            "AUROC undefined: fewer than two distinct correctness classes "
            "in this sample."
        )
    return float(roc_auc_score(y, conf)), None


def _bootstrap_ece_auroc_message_clustered(
    conf: np.ndarray,
    correct: np.ndarray,
    message_ids: np.ndarray,
    n_bins: int,
    scheme: str,
    n_boot: int,
    rng: np.random.Generator,
):
    """Message-clustered bootstrap CIs for ECE and AUROC.

    Each iteration resamples the distinct `message_id`s (with replacement)
    and pulls every row of each resampled message, preserving within-
    message correlation (see module docstring). Returns
    `(ece_ci, auroc_ci)`, each a `[lo, hi]` list; `auroc_ci` is `None` if
    every bootstrap draw was one-class (AUROC undefined throughout).
    """
    unique_ids = np.unique(message_ids)
    n_ids = len(unique_ids)
    groups = {mid: np.where(message_ids == mid)[0] for mid in unique_ids}

    ece_draws = np.empty(n_boot)
    auroc_draws = []
    for b in range(n_boot):
        sampled_ids = unique_ids[rng.integers(0, n_ids, size=n_ids)]
        idx = np.concatenate([groups[mid] for mid in sampled_ids])
        c_b, y_b = conf[idx], correct[idx]

        ece_draws[b] = ece(c_b, y_b, n_bins=n_bins, scheme=scheme)
        auroc_b, _reason = _safe_auroc(c_b, y_b)
        if auroc_b is not None:
            auroc_draws.append(auroc_b)

    ece_lo, ece_hi = np.percentile(ece_draws, _PCTILES)
    ece_ci = [float(ece_lo), float(ece_hi)]

    if auroc_draws:
        auroc_lo, auroc_hi = np.percentile(np.array(auroc_draws), _PCTILES)
        auroc_ci = [float(auroc_lo), float(auroc_hi)]
    else:
        auroc_ci = None

    return ece_ci, auroc_ci


def per_model_calibration(
    df: pd.DataFrame,
    *,
    n_bins: int = 10,
    scheme: str = "equal_width",
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Per-model ECE + AUROC with message-clustered bootstrap CIs.

    Args:
        df: tidy per-generation frame with columns `confidence` (float,
            already in [0, 1]), `correct` (0/1 or bool), `model`,
            `message_id`. See module docstring for the input contract;
            use `build_calibration_frame` to derive this from the raw
            labeled parquet.
        n_bins: passed through to `ece`.
        scheme: passed through to `ece`.
        n_boot: number of message-clustered bootstrap iterations.
        seed: seed for the single `numpy.random.default_rng` driving every
            bootstrap draw across every model in this call (models are
            visited in `groupby`'s sorted key order, so a given `(df,
            seed)` always reproduces bit-identical output).

    Returns:
        dict: `{model_name: {"ece": float, "ece_ci": [lo, hi], "auroc":
        float | None, "auroc_ci": [lo, hi] | None, "n": int,
        "n_messages": int, "notes": str}}`. `notes` always states the
        confidence-comparability caveat (see module docstring), with the
        AUROC one-class reason appended when it applies.

    Raises:
        ValueError: if `df` is missing any required column, or (via
            `ece`) if any `confidence` value falls outside [0, 1].
    """
    missing = [c for c in _REQUIRED_CALIBRATION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required column(s): {missing}")

    rng = np.random.default_rng(seed)
    results = {}
    for model, group in df.groupby("model"):
        group = group.reset_index(drop=True)
        conf = group["confidence"].to_numpy(dtype=float)
        correct = group["correct"].to_numpy(dtype=float)
        message_ids = group["message_id"].to_numpy()

        ece_point = ece(conf, correct, n_bins=n_bins, scheme=scheme)
        auroc_point, auroc_reason = _safe_auroc(conf, correct)

        ece_ci, auroc_ci = _bootstrap_ece_auroc_message_clustered(
            conf, correct, message_ids, n_bins, scheme, n_boot, rng
        )
        if auroc_point is None:
            auroc_ci = None

        notes = _COMPARABILITY_CAVEAT
        if auroc_reason:
            notes = f"{notes} {auroc_reason}"

        results[model] = {
            "ece": ece_point,
            "ece_ci": ece_ci,
            "auroc": auroc_point,
            "auroc_ci": auroc_ci,
            "n": int(len(group)),
            "n_messages": int(pd.unique(message_ids).size),
            "notes": notes,
        }
    return results


def meta_analyze(per_model_stats: dict, metric: str = "ece") -> dict:
    """DerSimonian-Laird random-effects pooling of one metric across models.

    Args:
        per_model_stats: the dict `per_model_calibration` returns (or any
            dict with the same per-model `{metric: float | None,
            f"{metric}_ci": [lo, hi] | None}` shape).
        metric: `"ece"` or `"auroc"` (or any key present per-model with a
            matching `f"{metric}_ci"` key).

    Returns:
        dict: `{"pooled": float | None, "ci": [lo, hi] | None, "tau2":
        float | None, "q": float | None, "i2": float | None, "k_models":
        int, "caveat": str}`. Models with a `None` point estimate or CI
        for `metric` (e.g. the AUROC one-class guard) are excluded and do
        not count toward `k_models`. If no model is poolable, every
        numeric field is `None` and `k_models` is 0. See module docstring
        for the exact DerSimonian-Laird formulas used.
    """
    ci_key = f"{metric}_ci"
    y_list, v_list = [], []
    for stats in per_model_stats.values():
        val = stats.get(metric)
        ci = stats.get(ci_key)
        if val is None or ci is None:
            continue
        lo, hi = ci
        se = (hi - lo) / (2 * _Z95)
        var = max(se * se, 1e-12)
        y_list.append(val)
        v_list.append(var)

    k = len(y_list)
    if k == 0:
        return {
            "pooled": None,
            "ci": None,
            "tau2": None,
            "q": None,
            "i2": None,
            "k_models": 0,
            "caveat": _COMPARABILITY_CAVEAT,
        }

    yi = np.asarray(y_list, dtype=float)
    vi = np.asarray(v_list, dtype=float)

    if k == 1:
        se = float(np.sqrt(vi[0]))
        return {
            "pooled": float(yi[0]),
            "ci": [float(yi[0] - _Z95 * se), float(yi[0] + _Z95 * se)],
            "tau2": 0.0,
            "q": 0.0,
            "i2": 0.0,
            "k_models": 1,
            "caveat": _COMPARABILITY_CAVEAT,
        }

    w_fe = 1.0 / vi
    sum_w_fe = w_fe.sum()
    mu_fe = float((w_fe * yi).sum() / sum_w_fe)
    q = float((w_fe * (yi - mu_fe) ** 2).sum())
    dof = k - 1
    c = float(sum_w_fe - (w_fe**2).sum() / sum_w_fe)
    tau2 = max(0.0, (q - dof) / c) if c > 0 else 0.0

    w_re = 1.0 / (vi + tau2)
    sum_w_re = w_re.sum()
    pooled = float((w_re * yi).sum() / sum_w_re)
    se_pooled = float(np.sqrt(1.0 / sum_w_re))
    ci = [pooled - _Z95 * se_pooled, pooled + _Z95 * se_pooled]
    i2 = max(0.0, (q - dof) / q) * 100.0 if q > 0 else 0.0

    return {
        "pooled": pooled,
        "ci": ci,
        "tau2": float(tau2),
        "q": q,
        "i2": float(i2),
        "k_models": k,
        "caveat": _COMPARABILITY_CAVEAT,
    }


def build_calibration_frame(labeled_path) -> pd.DataFrame:
    """Derive the `per_model_calibration` input frame from a raw labeled
    parquet: `correct = (label == "faithful")`, `confidence` rescaled from
    its raw 0-100 verbalized scale to [0, 1] by `/100`.

    Args:
        labeled_path: path to a parquet file with (at least) columns
            `label` (one of the study's {faithful, degraded, drift}
            taxonomy) and `confidence` (raw, 0-100 scale).

    Returns:
        DataFrame: a copy of the input frame with `correct` added and
            `confidence` overwritten in place with the rescaled [0, 1]
            value. Every other column (including `model`/`message_id`) is
            passed through untouched. Rows where the raw `confidence` was
            missing (NaN -- a model that emitted no verbalized confidence)
            are DROPPED, since they cannot contribute to calibration; the
            count of dropped rows is both printed as a one-line notice and
            attached as `out.attrs["dropped_nan_confidence"]`.

    Raises:
        ValueError: if `label` or `confidence` is missing.

    Note: unit-tested on a tiny hand-built frame only -- do NOT call this
    on the real labeled parquet from this task (a background labeling job
    owns `output/intermediate/`; see the task brief). The real labeled
    cohort is known to carry a small number of rows (119 of 533,400) with
    a missing verbalized confidence; those are dropped here rather than
    raised on, so the Step-5 calibration run does not crash over rows that
    legitimately have no confidence to calibrate.
    """
    df = pd.read_parquet(labeled_path)
    missing = [c for c in ("label", "confidence") if c not in df.columns]
    if missing:
        raise ValueError(f"labeled frame is missing required column(s): {missing}")

    out = df.copy()
    out["correct"] = out["label"] == "faithful"
    out["confidence"] = out["confidence"].astype(float) / 100.0

    nan_mask = out["confidence"].isna()
    n_dropped = int(nan_mask.sum())
    out = out.loc[~nan_mask].reset_index(drop=True)
    print(f"build_calibration_frame: dropped {n_dropped} rows with missing confidence")
    out.attrs["dropped_nan_confidence"] = n_dropped
    return out
