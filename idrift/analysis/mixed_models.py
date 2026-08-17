"""Cross-classified mixed-effects model for the {faithful, degraded, drift}
outcome on realized message-level corruption (revision Task 5.1).

Deviation from the brief's aspirational design (documented, per the
CONTROLLER RESOLUTION in `.superpowers/sdd/rev-task-5.1-brief.md`): `bambi`
and `pymc` are NOT installed in this environment and will NOT be added
(pymc is a heavy, compilation-fragile dependency; adding it mid-revision is
out of scope). There is also no native `statsmodels` multinomial model with
crossed random effects, so the 3-class outcome is fit as three ONE-VS-REST
binomial GLMMs (drift-vs-rest -- PRIMARY, degraded-vs-rest, faithful-vs-rest)
via `statsmodels` alone.

Primary engine: `BinomialBayesMixedGLM.from_formula` (variational Bayes),
with `vc_formula` giving CROSSED random intercepts for `message_id`,
`model_id`, and `corpus`. `family` is folded in as a FIXED effect
(`C(family)`), not a fourth crossed random intercept: `model_id` already
determines `family` deterministically (each model tag belongs to exactly
one family), so a random intercept for `model_id` already captures all
family-level variation on its own; adding a separate `family` random
intercept nested inside an already-saturating `model_id` random intercept
would be redundant/non-identifiable. A small number of family levels (2-4
in this study) is also exactly the case where a fixed effect, not a random
one, is the standard recommendation (too few levels to estimate a variance
component for that factor sensibly).

Fixed effects (every one-vs-rest fit): `realized_cer` (continuous, PRIMARY
predictor) + `n_errors` + `char_len` + `corrupted_negation` +
`corrupted_numeral` + `C(family)`.

`fit_vb(..., scale_fe=True)` is used deliberately, not the statsmodels
default (`scale_fe=False`): on this study's realized covariate scales (a
0-1 `realized_cer` next to `char_len` in the tens, `n_errors` up to
double digits), the unscaled fixed-effects design matrix is poorly
conditioned for the VB optimizer -- empirically, BFGS terminates with
`success=False` ("precision loss") on unscaled data despite a
near-zero gradient, while centering/scaling the fixed-effects columns
before optimizing (and back-transforming the returned coefficients to the
original scale, which `scale_fe=True` does internally) reliably reaches
`success=True`. This is the numerical-conditioning fix, not a change to
what is estimated: reported coefficients are on the original covariate
scale either way.

VB-init randomness and determinism: `BinomialBayesMixedGLM.fit_vb`'s
default variational-SD starting values are drawn from the LEGACY GLOBAL
`numpy.random` state (`s = -0.5 + 0.1 * np.random.normal(...)` inside
statsmodels, not a local `Generator` -- verified by reading
`statsmodels/genmod/bayes_mixed_glm.py`), not from `numpy.random.default_rng`.
`_fit_vb` below therefore calls `np.random.seed(seed)` immediately before
`fit_vb()` so a given `(df, seed)` reproduces bit-identical output; this is
the ONLY place `fit_multinomial` touches the global RNG, and it is confined
to immediately before the one call that consumes it.

Fallback ladder (documented, per the brief): if `BinomialBayesMixedGLM`'s
VB fit does not converge (`scipy.optimize`'s own `success` flag) or raises
any exception, `_fit_outcome_one_vs_rest` falls back to a message-clustered
cluster-robust logit (`statsmodels.formula.api.logit` with
`cov_type="cluster", cov_kwds={"groups": message_id}`) for the fixed-effect
coefficients and CIs -- exactly the proxy this study's power calculation
already used -- and reports variance components as unavailable, with a
reason recorded in `notes`. Which engine actually fired is always recorded
in the returned `ModelResult`'s `"engine"` field
(`"glmm_vb"` or `"clustered_logit_fallback"`) so a caller (and this
module's own tests) can act on it instead of assuming one engine always
wins. NOT implemented: a wall-clock timeout for "impractically slow" --
enforcing that safely would need a subprocess/signal-based interrupt around
a synchronous `scipy.optimize` call, which is out of scope for this task
and not needed at the simulated-test or expected full-cohort scale (both
fit in well under a second per outcome in this environment); only the
convergence-flag half of the ladder is implemented.

`ModelResult` shape (one per one-vs-rest outcome; see `fit_multinomial`):
    {
        "outcome": "faithful" | "degraded" | "drift",
        "engine": "glmm_vb" | "clustered_logit_fallback",
        "coefficients": {name: {"est": float, "se": float,
                                 "ci_lo": float, "ci_hi": float}},
        "variance_components": {name: float},  # posterior-median random-
            # intercept SD (exp(posterior mean log-SD)), `glmm_vb` only;
            # {} (unavailable) for the fallback engine.
        "n_obs": int,
        "n_groups": {"message_id": int, "model_id": int, "corpus": int,
                     "family": int},  # unique level counts
        "converged": bool,
        "notes": str,
    }

Note on `CLASS_ORDER`: the brief points at
`idrift.adjudicate.taxonomy.CLASS_ORDER`, but that tuple is actually
defined in `idrift.adjudicate.validation_metrics` (`taxonomy.py` has no
`CLASS_ORDER` of its own) -- imported from there below instead of
duplicating the literal tuple, per this project's existing single-source-
of-truth convention (`idrift.adjudicate.misclassification` does the same).

Deterministic; American spelling; no em dashes (double hyphens for
asides, per house style).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

from idrift.adjudicate.validation_metrics import CLASS_ORDER

# Fixed effects shared by every one-vs-rest fit. `family` is a fixed effect
# (see module docstring for why); the other five are the brief's exact
# prespecified predictor list, `realized_cer` first as the PRIMARY predictor.
_FIXED_FORMULA = "realized_cer + n_errors + char_len + corrupted_negation + corrupted_numeral + C(family)"

# Crossed random intercepts: message_id, model_id, corpus.
_VC_FORMULA = {
    "message_id": "0 + C(message_id)",
    "model_id": "0 + C(model_id)",
    "corpus": "0 + C(corpus)",
}

_REQUIRED_COLUMNS = (
    "label", "realized_cer", "n_errors", "char_len", "corrupted_negation",
    "corrupted_numeral", "message_id", "replicate_idx", "model_id", "family", "corpus",
)

_GROUP_COLUMNS = ("message_id", "model_id", "corpus", "family")

# The drift-vs-rest fit is the PRIMARY analysis (see module + brief).
PRIMARY_OUTCOME = "drift"

# Columns `fit_multinomial` actually consumes (formula fixed effects, crossed
# random-intercept groups, and the fallback's cluster groups): every
# `_REQUIRED_COLUMNS` entry except `replicate_idx`, which is a join key only
# and never enters the fit itself (revision Task 5.1 review fix). patsy's
# default `missing='drop'` silently drops any row with a NaN in one of these
# columns inside both `_fit_vb` and `_fit_clustered_fallback`, which would
# otherwise (a) leave `n_obs` (captured from the full, pre-drop frame)
# overstated relative to what was actually fit, and (b) crash the fallback,
# whose `cov_kwds={"groups": data["message_id"]}` is built at full length
# against a silently-shrunk design matrix. `build_analysis_frame` correctly
# LEFT-joins and keeps unmatched rows visible on purpose (do not change
# that); the guard below is what makes the FIT path refuse to silently eat
# them instead.
_FIT_NA_CHECK_COLUMNS = tuple(c for c in _REQUIRED_COLUMNS if c != "replicate_idx")


def _formula(y_col: str) -> str:
    return f"{y_col} ~ {_FIXED_FORMULA}"


def _fit_vb(data: pd.DataFrame, seed: int) -> dict:
    """Primary engine: `BinomialBayesMixedGLM.fit_vb`, deterministic in `seed`.

    Returns a partial `ModelResult` (no `outcome`/`n_obs`/`n_groups` yet --
    `_fit_outcome_one_vs_rest` fills those in).
    """
    model = BinomialBayesMixedGLM.from_formula(_formula("_y"), _VC_FORMULA, data=data)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np.random.seed(seed)  # see module docstring: controls fit_vb's VB-init SD draw
        res = model.fit_vb(scale_fe=True)
    converged = bool(res.optim_retvals.get("success", False))

    coefficients = {
        name: {
            "est": float(est),
            "se": float(se),
            "ci_lo": float(est - 1.96 * se),
            "ci_hi": float(est + 1.96 * se),
        }
        for name, est, se in zip(model.fep_names, res.fe_mean, res.fe_sd)
    }
    # vcp_mean/vcp_sd are posterior mean/SD of the LOG standard deviation of
    # each random-intercept group (statsmodels convention); exp(posterior
    # mean log-SD) is the posterior MEDIAN of the SD (the lognormal case),
    # a standard point-estimate choice, not the (harder to get in closed
    # form) posterior mean of the SD itself.
    variance_components = {name: float(np.exp(vm)) for name, vm in zip(model.vcp_names, res.vcp_mean)}

    opt_message = str(res.optim_retvals.get("message", ""))
    warn_texts = "; ".join(str(w.message) for w in caught) or "none"
    notes = (
        "engine=glmm_vb (BinomialBayesMixedGLM.fit_vb, scale_fe=True). "
        f"scipy optimizer {'converged' if converged else 'did NOT converge'} ({opt_message}). "
        f"warnings: {warn_texts}. variance_components are posterior-median random-intercept "
        "SDs (exp[posterior mean log-SD]), not variances."
    )
    return {
        "engine": "glmm_vb",
        "coefficients": coefficients,
        "variance_components": variance_components,
        "converged": converged,
        "notes": notes,
    }


def _fit_clustered_fallback(data: pd.DataFrame) -> dict:
    """Fallback engine: message-clustered cluster-robust logit.

    No randomness enters this fit (plain Newton-Raphson MLE on a fixed
    design), so it needs no seed to be deterministic.
    """
    model = smf.logit(_formula("_y"), data=data)
    res = model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": data["message_id"]})
    ci = res.conf_int()
    coefficients = {
        name: {
            "est": float(res.params[name]),
            "se": float(res.bse[name]),
            "ci_lo": float(ci.loc[name, 0]),
            "ci_hi": float(ci.loc[name, 1]),
        }
        for name in res.params.index
    }
    return {
        "engine": "clustered_logit_fallback",
        "coefficients": coefficients,
        "variance_components": {},
        "converged": bool(res.mle_retvals.get("converged", True)),
        "notes": (
            "engine=clustered_logit_fallback (smf.logit with cov_type='cluster', "
            "cov_kwds={'groups': message_id}). variance_components are unavailable with "
            "this engine (no random-effects structure is fit); coefficients/CIs are the "
            "message-clustered cluster-robust point estimates instead."
        ),
    }


def _fit_outcome_one_vs_rest(df: pd.DataFrame, outcome: str, seed: int, force_fallback: bool = False) -> dict:
    data = df.copy()
    data["_y"] = (data["label"] == outcome).astype(int)
    n_obs = int(len(data))
    n_groups = {col: int(data[col].nunique()) for col in _GROUP_COLUMNS}

    vb_error = None
    vb_result = None
    if force_fallback:
        # The VB GLMM is intractable at the full v3 scale (>1.7M rows: fit_vb
        # ran >1h without terminating) and did not converge at the smaller v2
        # scale either, so its result is never the reported estimate. Skip it
        # and use the preregistered cluster-robust logit fallback directly.
        vb_error = (
            "VB skipped (force_fallback=True): variational-Bayes GLMM intractable at full scale "
            "(>1.7M rows) and non-convergent at v2 scale; preregistered cluster-robust logit used"
        )
    else:
        try:
            vb_result = _fit_vb(data, seed)
        except Exception as exc:  # noqa: BLE001 - any VB failure triggers the documented fallback
            vb_error = f"{type(exc).__name__}: {exc}"

    if vb_result is not None and vb_result["converged"]:
        vb_result.update(outcome=outcome, n_obs=n_obs, n_groups=n_groups)
        return vb_result

    fallback = _fit_clustered_fallback(data)
    if vb_result is not None:
        reason = f"primary VB fit ran but did not converge ({vb_result['notes']})"
    else:
        reason = f"primary VB fit raised an exception before returning a result ({vb_error})"
    fallback["notes"] = f"FALLBACK LADDER FIRED: {reason}. {fallback['notes']}"
    fallback.update(outcome=outcome, n_obs=n_obs, n_groups=n_groups)
    return fallback


def fit_multinomial(df: pd.DataFrame, *, seed: int = 0, force_fallback: bool = False) -> dict:
    """Fit the {faithful, degraded, drift} outcome as three one-vs-rest
    cross-classified binomial GLMMs on realized message-level corruption.

    Args:
        df: already-assembled long DataFrame (see `build_analysis_frame`)
            with columns `label` (in `CLASS_ORDER`), `realized_cer`,
            `n_errors`, `char_len`, `corrupted_negation`, `corrupted_numeral`,
            `message_id`, `replicate_idx`, `model_id`, `family`, `corpus`.
        seed: seed controlling the primary VB engine's init draw (see module
            docstring); the fallback engine is deterministic regardless.
        force_fallback: skip the primary VB GLMM and use the cluster-robust
            logit fallback directly. Intended for large cohorts where VB is
            intractable (it did not converge at v2 scale and does not
            terminate at the >1.7M-row v3 scale), so the fallback is the
            reported estimate regardless; default False preserves the
            VB-then-fallback ladder.

    Returns:
        dict keyed by outcome (every value in `CLASS_ORDER`), each a
        `ModelResult` dict (see module docstring). The entry keyed
        `PRIMARY_OUTCOME` ("drift") is the study's PRIMARY analysis; the
        other two are secondary one-vs-rest fits over the same taxonomy.

    Raises:
        ValueError: if `df` is missing a required column, if `df["label"]`
            contains a value outside `CLASS_ORDER`, or if any row has a
            missing value in a column the fit consumes (see
            `_FIT_NA_CHECK_COLUMNS`) -- e.g. an unmatched row surviving
            `build_analysis_frame`'s LEFT join -- since patsy would
            otherwise silently drop that row inside the fit while `n_obs`
            still counted it.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"fit_multinomial: df is missing required column(s): {missing}")
    bad_labels = sorted(set(df["label"].unique()) - set(CLASS_ORDER))
    if bad_labels:
        raise ValueError(f"fit_multinomial: df['label'] has value(s) outside CLASS_ORDER {CLASS_ORDER}: {bad_labels}")

    na_rows = df[list(_FIT_NA_CHECK_COLUMNS)].isna().any(axis=1)
    n_missing = int(na_rows.sum())
    if n_missing:
        raise ValueError(
            f"fit_multinomial received {n_missing} row(s) with missing values in model columns "
            f"{list(_FIT_NA_CHECK_COLUMNS)}; these are unmatched labeled rows from the exposure "
            "join and must be resolved before fitting (n_obs would otherwise be overstated and "
            "the clustered fallback would crash)"
        )

    # PRIMARY_OUTCOME first (readability of the returned dict / any printed
    # summary); functionally each one-vs-rest fit is independent of the others.
    ordered = [PRIMARY_OUTCOME] + [c for c in CLASS_ORDER if c != PRIMARY_OUTCOME]
    return {outcome: _fit_outcome_one_vs_rest(df, outcome, seed, force_fallback=force_fallback) for outcome in ordered}


# ---------------------------------------------------------------------------
# Step-5 assembly helper.
# ---------------------------------------------------------------------------

# `corpus` is a join key so that message_ids shared across corpora (the CTRL
# controls reuse the same costello_* ids as AUTH, with a DIFFERENT corruption
# draw) attach their own corpus's corruption metadata, not another corpus's.
# Backward-compatible with v2, where (message_id, cer_target, replicate_idx)
# was already unique so adding corpus changes no match.
_JOIN_KEYS = ("message_id", "corpus", "cer_target", "replicate_idx")
_EXPOSURE_ATTACH_COLUMNS = ("n_errors", "char_len", "corrupted_negation", "corrupted_numeral")


def build_analysis_frame(labeled_path, exposure_path) -> pd.DataFrame:
    """Assemble `fit_multinomial`'s input contract by LEFT-joining the
    labeled attempts frame to the exposure frame.

    Args:
        labeled_path: path to `attempts_v2_labeled.parquet` (columns include
            `label`, `realized_cer`, `message_id`, `model`, `corpus`,
            `cer_target`, `replicate_idx` -- see
            `idrift.models.o2_runner_v2.main`'s output row and
            `idrift.adjudicate.label_runner` for the real schema), OR an
            already-loaded DataFrame with those columns (accepted directly,
            e.g. from a hand-built test fixture, matching the
            DataFrame-or-path convention already used by
            `idrift.adjudicate.build_rating_sheet`).
        exposure_path: path to `exposure_v2_full.parquet` (columns include
            `message_id`, `cer_target`, `replicate_idx`, `n_errors`,
            `corrupted_negation`, `corrupted_numeral`, and `text` -- see
            `idrift.data.exposure_v2.build_exposure_v2`), OR an
            already-loaded DataFrame.

    Join and derivation:
        - LEFT join on `(message_id, cer_target, replicate_idx)`: every
          labeled row is kept; a labeled row with no matching exposure row
          gets NaN in the four attached columns (unmatched handling is
          explicit, not a silent drop).
        - Only `n_errors`, `char_len`, `corrupted_negation`,
          `corrupted_numeral` (plus the join keys) are pulled from the
          exposure side, so an accidental name collision on any other
          overlapping column (e.g. `corpus`, both sides have one) can never
          happen -- the labeled side's own `corpus` is always what
          survives.
        - `validate="m:1"` on the merge enforces "no row multiplication":
          if the exposure side is not unique on the join key (which would
          silently fan out labeled rows), pandas raises `MergeError`
          instead of returning a bloated frame.
        - `char_len`: `idrift.data.exposure_v2.build_exposure_v2`'s current
          real schema does not (yet) materialize a literal `char_len`
          column, only `text` (the corruption reference). If the exposure
          frame already has a `char_len` column, it is used as-is;
          otherwise it is derived here as `text.str.len()`. Documented
          resolution of an ambiguity in the brief, not a silent guess: the
          brief lists `char_len` as something to "attach" from the exposure
          side, which this does either way.
        - `family`: derived from `model_id`, per the brief ("Derive family
          from `model_id` [strip the size/tag suffix]"). The labeled
          frame's real column for this is named `model` (see
          `idrift.models.o2_runner_v2.main`'s output row, literally
          `"model": args.model`, e.g. `"gemma3:27b"`), not `model_id` --
          renamed to `model_id` here (only if a bare `model` column is
          present) because `fit_multinomial`'s input contract names it
          `model_id`. The Ollama tag convention is `{family}:{size}`, so
          `family` is `model_id` split on `":"`, first segment (a tag with
          no colon is its own family, unchanged).

    Returns:
        DataFrame: one row per labeled row (same row count as `labeled`),
        with `n_errors`/`char_len`/`corrupted_negation`/`corrupted_numeral`
        attached, `model` renamed to `model_id` (if present), and `family`
        derived from `model_id`.

    Raises:
        ValueError: if a join key column is missing from either frame.
    """
    labeled = labeled_path if isinstance(labeled_path, pd.DataFrame) else pd.read_parquet(labeled_path)
    exposure = exposure_path if isinstance(exposure_path, pd.DataFrame) else pd.read_parquet(exposure_path)

    missing_labeled = [c for c in _JOIN_KEYS if c not in labeled.columns]
    missing_exposure = [c for c in _JOIN_KEYS if c not in exposure.columns]
    if missing_labeled or missing_exposure:
        raise ValueError(
            "build_analysis_frame: missing join key column(s) -- "
            f"labeled missing {missing_labeled}, exposure missing {missing_exposure}"
        )

    exposure_slim = exposure.copy()
    if "char_len" not in exposure_slim.columns:
        if "text" not in exposure_slim.columns:
            raise ValueError(
                "build_analysis_frame: exposure frame has neither 'char_len' nor 'text' "
                "to derive it from"
            )
        exposure_slim["char_len"] = exposure_slim["text"].astype(str).str.len()

    missing_attach = [c for c in _EXPOSURE_ATTACH_COLUMNS if c not in exposure_slim.columns]
    if missing_attach:
        raise ValueError(f"build_analysis_frame: exposure frame is missing column(s): {missing_attach}")

    exposure_slim = exposure_slim[list(_JOIN_KEYS) + list(_EXPOSURE_ATTACH_COLUMNS)]

    merged = labeled.merge(exposure_slim, on=list(_JOIN_KEYS), how="left", validate="m:1")

    if "model_id" not in merged.columns and "model" in merged.columns:
        merged = merged.rename(columns={"model": "model_id"})

    merged["family"] = merged["model_id"].astype(str).str.split(":").str[0]

    return merged
