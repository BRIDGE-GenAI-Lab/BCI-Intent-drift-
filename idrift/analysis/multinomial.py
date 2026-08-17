"""Multinomial outcome model + drift-vs-faithful / drift-vs-degraded
contrasts (revision Task B2).

Why this module exists
----------------------
The study's primary drift model collapses the three-way label
{faithful, degraded, drift} into a binary drift-vs-rest outcome. A reviewer
objected that this reference class is incoherent for a SAFETY argument:
`degraded` is a VISIBLE failure (the decode is obviously mangled, so a human
does not act on it) while `faithful` is a success -- lumping them into one
"not drift" baseline hides that the two have OPPOSITE safety meaning, and a
positive drift-vs-rest coefficient could in principle be driven by either.
This module fits models that keep the three classes separate:

  * `fit_multinomial` -- a single `statsmodels` `MNLogit` of `label` with
    base category = "faithful", so it reports two equations at once:
    degraded|faithful and drift|faithful (each on the odds-ratio scale, with
    a 95% CI and p). This is the honest multi-class replacement for the
    collapsed binary model.

  * `fit_binary_contrast(df, positive, negative)` -- a two-class,
    message-clustered logit restricted to the two named labels. Called with
    ("drift", "degraded") it delivers the drift-vs-DEGRADED contrast the
    reviewer specifically wanted (does realized corruption push the model
    toward a fluent-but-wrong drift over an obviously-degraded output?);
    called with ("drift", "faithful") it gives the drift-vs-SUCCESS contrast.

Predictors (both models, matching the study's message-level corruption
covariates)::

    realized_cer + corrupted_numeral + corrupted_negation + char_len
        + C(model_family)

`realized_cer` is the primary predictor. `model_family` is derived from the
Ollama model tag by splitting on ":" (e.g. "gemma4:31b" -> "gemma4"); it is
entered as a fixed effect `C(model_family)`. `char_len` is supplied directly
if present, otherwise derived as `intended_text.str.len()`.

Clustering
----------
Every response for one BCI message shares that message's realized corruption,
so rows are NOT independent within `message_id`. Both fits therefore use
message-clustered cluster-robust standard errors
(`cov_type="cluster", cov_kwds={"groups": df["message_id"]}`), the same
cluster unit the rest of the analysis uses.

Convergence guard
------------------
Both fits try `method="newton"` first and fall back to `method="lbfgs"` if
Newton raises or reports non-convergence; the method that actually produced
the reported result is recorded in the returned structure's `method_used`
field (and `converged`), so a caller never has to assume which optimizer won.

Return shape (JSON-serializable, tidy)
--------------------------------------
`fit_multinomial` returns::

    {
      "model": "MNLogit",
      "base_category": "faithful",
      "non_base_classes": ["degraded", "drift"],
      "primary_predictor": "realized_cer",
      "cov_type": "cluster",
      "method_used": "newton" | "lbfgs",
      "converged": bool,
      "n_obs": int,
      "n_clusters": int,
      "terms": [ ...predictor names... ],
      "equations": {
          "degraded": [ {term, coef, se, odds_ratio, ci_lower, ci_upper, p}, ... ],
          "drift":    [ ... ],
      },
    }

`fit_binary_contrast` returns::

    {
      "model": "Logit",
      "positive": <label>, "negative": <label>,
      "outcome": "P(label == <positive>)",
      "cov_type": "cluster",
      "method_used": ..., "converged": ...,
      "n_obs": int, "n_clusters": int,
      "terms": [ {term, coef, se, odds_ratio, ci_lower, ci_upper, p}, ... ],
    }
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import patsy
from statsmodels.discrete.discrete_model import Logit, MNLogit

logger = logging.getLogger(__name__)

# Fixed class ordering used to code the MNLogit endog so that "faithful" is
# the reference (code 0) and the two non-base equations come out in a stable
# [degraded, drift] order regardless of the data's row order.
CLASS_ORDER = ("faithful", "degraded", "drift")
BASE_CATEGORY = "faithful"
NON_BASE_CLASSES = ("degraded", "drift")
PRIMARY_PREDICTOR = "realized_cer"

_RHS = "realized_cer + corrupted_numeral + corrupted_negation + char_len + C(model_family)"

# Columns the fits actually consume (before deriving model_family / char_len).
_BASE_REQUIRED = ("label", "realized_cer", "corrupted_numeral", "corrupted_negation", "message_id")

_Z = 1.959963984540054  # two-sided 95% normal quantile


# ---------------------------------------------------------------------------
# Frame preparation
# ---------------------------------------------------------------------------
def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `df` guaranteed to have `model_family` and `char_len`,
    with the fit-consumed columns free of missing values.

    `model_family` is derived from `model` (split on ":") when absent;
    `char_len` from `intended_text.str.len()` when absent. Rows with a missing
    value in any consumed predictor are dropped (patsy would otherwise drop
    them silently while a naive n_obs still counted them).
    """
    missing = [c for c in _BASE_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"multinomial: df is missing required column(s): {missing}")

    out = df.copy()

    if "model_family" not in out.columns:
        if "model" not in out.columns:
            raise ValueError(
                "multinomial: need either a 'model_family' column or a 'model' column "
                "to derive it from (split on ':')"
            )
        out["model_family"] = out["model"].astype(str).str.split(":").str[0]

    if "char_len" not in out.columns:
        if "intended_text" not in out.columns:
            raise ValueError(
                "multinomial: need either a 'char_len' column or an 'intended_text' "
                "column to derive it from (str.len())"
            )
        out["char_len"] = out["intended_text"].astype(str).str.len()

    consumed = list(_BASE_REQUIRED) + ["model_family", "char_len"]
    before = len(out)
    out = out.dropna(subset=consumed)
    dropped = before - len(out)
    if dropped:
        logger.warning("multinomial._prepare dropped %d row(s) with missing predictors", dropped)

    return out


def _records(params, bse, pvalues) -> list[dict]:
    """Build tidy per-term records (OR / 95% CI on the OR scale / p) from three
    aligned pandas Series (coef, se, p) indexed by term name."""
    records = []
    for term in params.index:
        coef = float(params[term])
        se = float(bse[term])
        records.append(
            {
                "term": term,
                "coef": coef,
                "se": se,
                "odds_ratio": float(np.exp(coef)),
                "ci_lower": float(np.exp(coef - _Z * se)),
                "ci_upper": float(np.exp(coef + _Z * se)),
                "p": float(pvalues[term]),
            }
        )
    return records


def _fit_with_fallback(model, *, seed: int, cluster_groups):
    """Fit `model` (MNLogit or Logit) with cluster-robust cov, trying Newton
    then L-BFGS. Returns (result, method_used, converged).

    The seed is applied to the legacy global RNG before each fit for
    reproducibility; neither optimizer here draws random start values by
    default, but honoring the seed argument keeps the call deterministic if a
    future statsmodels version ever did.
    """
    cov_kwds = {"groups": cluster_groups}
    last_err = None
    last_res = None
    last_method = None
    for method in ("newton", "lbfgs"):
        try:
            np.random.seed(seed)
            res = model.fit(
                method=method,
                disp=0,
                maxiter=200,
                cov_type="cluster",
                cov_kwds=cov_kwds,
            )
        except Exception as exc:  # noqa: BLE001 -- fall through to next method
            last_err = exc
            logger.warning("multinomial: method=%s raised (%s); trying next", method, exc)
            continue

        converged = bool(res.mle_retvals.get("converged", True)) if hasattr(res, "mle_retvals") else True
        last_res, last_method = res, method
        if converged:
            logger.info("multinomial: converged with method=%s", method)
            return res, method, True
        logger.warning("multinomial: method=%s did not converge; trying next", method)
        last_err = RuntimeError(f"method={method} reported non-convergence")

    # Both methods exhausted: surface the last (unconverged) result if any fit
    # produced one at all; otherwise every attempt raised, so re-raise.
    if last_res is not None:
        logger.warning("multinomial: no method converged; returning last result (%s) unconverged", last_method)
        return last_res, last_method, False
    raise RuntimeError(f"multinomial: model failed to fit under newton and lbfgs ({last_err})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fit_multinomial(df: pd.DataFrame, *, seed: int = 0) -> dict:
    """Fit an `MNLogit` of `label` (base = "faithful") on the study's
    message-level corruption predictors, with message-clustered robust SEs.

    Returns a tidy dict with a `degraded|faithful` and a `drift|faithful`
    equation, each reporting odds ratios, 95% CIs, and p per term. See the
    module docstring for the exact shape.
    """
    prepared = _prepare(df)

    bad = sorted(set(prepared["label"].unique()) - set(CLASS_ORDER))
    if bad:
        raise ValueError(f"fit_multinomial: df['label'] has value(s) outside {CLASS_ORDER}: {bad}")

    present = [c for c in CLASS_ORDER if c in set(prepared["label"].unique())]
    if BASE_CATEGORY not in present:
        raise ValueError("fit_multinomial: base category 'faithful' absent from df['label']")
    non_base_present = [c for c in NON_BASE_CLASSES if c in present]
    if len(non_base_present) < 1:
        raise ValueError("fit_multinomial: need at least one of {degraded, drift} present")

    # Code endog so faithful=0 (reference); MNLogit orders its output columns
    # by the sorted non-base codes, i.e. [degraded(1), drift(2)] here.
    cat = pd.Categorical(prepared["label"], categories=list(CLASS_ORDER))
    codes = np.asarray(cat.codes)

    exog = patsy.dmatrix(_RHS, prepared, return_type="dataframe")
    groups = prepared["message_id"].to_numpy()

    model = MNLogit(codes, exog)
    res, method_used, converged = _fit_with_fallback(model, seed=seed, cluster_groups=groups)

    # res.params / bse / pvalues are DataFrames: index = exog terms,
    # columns = positional 0..J-2 for the non-base categories in code order.
    non_base_in_code_order = [c for c in CLASS_ORDER if c != BASE_CATEGORY and c in present]
    equations: dict[str, list[dict]] = {}
    for pos, cls in enumerate(non_base_in_code_order):
        params_col = res.params.iloc[:, pos]
        bse_col = res.bse.iloc[:, pos]
        pval_col = res.pvalues.iloc[:, pos]
        equations[cls] = _records(params_col, bse_col, pval_col)

    return {
        "model": "MNLogit",
        "base_category": BASE_CATEGORY,
        "non_base_classes": non_base_in_code_order,
        "primary_predictor": PRIMARY_PREDICTOR,
        "cov_type": "cluster",
        "cluster_unit": "message_id",
        "method_used": method_used,
        "converged": converged,
        "n_obs": int(len(prepared)),
        "n_clusters": int(pd.Series(groups).nunique()),
        "terms": list(exog.columns),
        "equations": equations,
    }


def fit_binary_contrast(df: pd.DataFrame, positive: str, negative: str, *, seed: int = 0) -> dict:
    """Fit a message-clustered `Logit` of (label == `positive`) on the study's
    predictors, restricted to rows whose label is `positive` or `negative`.

    `fit_binary_contrast(df, "drift", "degraded")` is the drift-vs-DEGRADED
    contrast (fluent-wrong vs visibly-mangled) the reviewer asked for;
    `fit_binary_contrast(df, "drift", "faithful")` is the drift-vs-SUCCESS
    contrast. Returns a tidy dict (odds ratios / 95% CIs / p per term).
    """
    for name in (positive, negative):
        if name not in CLASS_ORDER:
            raise ValueError(f"fit_binary_contrast: unknown label {name!r}; expected one of {CLASS_ORDER}")
    if positive == negative:
        raise ValueError("fit_binary_contrast: positive and negative must differ")

    prepared = _prepare(df)
    subset = prepared[prepared["label"].isin([positive, negative])].copy()
    if subset.empty:
        raise ValueError(f"fit_binary_contrast: no rows with label in {{{positive!r}, {negative!r}}}")
    n_pos = int((subset["label"] == positive).sum())
    n_neg = int((subset["label"] == negative).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"fit_binary_contrast: one class is empty after subsetting "
            f"({positive}={n_pos}, {negative}={n_neg}); cannot fit a contrast"
        )

    endog = (subset["label"] == positive).astype(int).to_numpy()
    exog = patsy.dmatrix(_RHS, subset, return_type="dataframe")
    groups = subset["message_id"].to_numpy()

    model = Logit(endog, exog)
    res, method_used, converged = _fit_with_fallback(model, seed=seed, cluster_groups=groups)

    terms = _records(res.params, res.bse, res.pvalues)

    return {
        "model": "Logit",
        "positive": positive,
        "negative": negative,
        "outcome": f"P(label == {positive!r})",
        "cov_type": "cluster",
        "cluster_unit": "message_id",
        "method_used": method_used,
        "converged": converged,
        "n_obs": int(len(subset)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_clusters": int(pd.Series(groups).nunique()),
        "terms": terms,
    }


# ---------------------------------------------------------------------------
# Real-data assembly helper (attach message-level corruption covariates)
# ---------------------------------------------------------------------------
# `corpus` is a join key so message_ids shared across corpora (CTRL controls
# reuse AUTH's costello_* ids, with a different corruption draw) attach their
# own corpus's corruption metadata. Backward-compatible with v2, where the
# 3-key (message_id, cer_target, replicate_idx) was already unique.
_JOIN_KEYS = ("message_id", "corpus", "cer_target", "replicate_idx")
_ATTACH = ("corrupted_negation", "corrupted_numeral", "n_errors")


def attach_message_covariates(labeled, exposure) -> pd.DataFrame:
    """LEFT-join `corrupted_negation`/`corrupted_numeral`/`n_errors` from the
    exposure frame onto the labeled attempts frame on
    (message_id, cer_target, replicate_idx).

    Both arguments may be a path or an already-loaded DataFrame. `model` is
    left intact (so `_prepare` can derive `model_family` from it) and
    `char_len` is derived downstream from `intended_text`. Raises if the join
    fans out (validate="m:1") or leaves any labeled row unmatched.
    """
    labeled = labeled if isinstance(labeled, pd.DataFrame) else pd.read_parquet(labeled)
    exposure = exposure if isinstance(exposure, pd.DataFrame) else pd.read_parquet(exposure)

    for frame, name in ((labeled, "labeled"), (exposure, "exposure")):
        miss = [k for k in _JOIN_KEYS if k not in frame.columns]
        if miss:
            raise ValueError(f"attach_message_covariates: {name} frame missing join key(s) {miss}")
    miss_attach = [c for c in _ATTACH if c not in exposure.columns]
    if miss_attach:
        raise ValueError(f"attach_message_covariates: exposure frame missing column(s) {miss_attach}")

    ex = exposure[list(_JOIN_KEYS) + list(_ATTACH)].copy()
    merged = labeled.merge(ex, on=list(_JOIN_KEYS), how="left", validate="m:1")

    unmatched = int(merged[list(_ATTACH)].isna().any(axis=1).sum())
    if unmatched:
        raise ValueError(
            f"attach_message_covariates: {unmatched} labeled row(s) had no exposure match on "
            f"{_JOIN_KEYS}; resolve before fitting"
        )
    return merged
