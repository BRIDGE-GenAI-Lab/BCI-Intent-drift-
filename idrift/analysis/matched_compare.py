"""CRIT-vs-CTRL matched-pair conditional logistic drift comparison
(revision Task 5.3), replacing the reviewer-flagged critical-vs-overall
comparison.

Reviewers rejected the original "message-critical vs overall" contrast as
confounded: the message-critical (CRIT) probe set is systematically
shorter and more negation-/numeral-heavy than the corpus at large, so any
raw CRIT-vs-rest difference in drift could just be those nuisance features,
not criticality itself. Task 1.3 (`idrift.data.matched_controls`) already
built one matched non-critical CONTROL (CTRL) per CRIT item, balanced on
[char_len, word_count, has_numeral, has_negation, mean_word_freq]. This
module runs the actual CRIT-vs-CTRL contrast on that matched design: a
CONDITIONAL (matched-pair) logistic regression of the binary drift outcome
on `critical` status, adjusted for the realized-attempt-level covariates
[char_len, n_errors, corrupted_negation, corrupted_numeral, realized_cer],
conditioning on the matched pair itself.

Why conditional logistic regression answers the confound
----------------------------------------------------------
`statsmodels.discrete.conditional_models.ConditionalLogit` fits a
stratified logistic model in which every group (here, `pair_id`, the
Task-1.3 CRIT-CTRL match) gets its own implicit intercept, but the
likelihood is CONDITIONAL on the total number of drift events observed
within that group -- the group-specific intercepts are differenced out
analytically rather than estimated (see the class docstring: "Every group
is implicitly given an intercept ... the intercepts are not present [after
conditioning]"). Any confounder that is constant WITHIN a matched pair
(whatever made this particular CRIT item and its matched CTRL item behave
similarly before assignment to `critical`) is automatically controlled for,
with no need to name or measure it. The five explicit adjusters
(char_len/n_errors/corrupted_negation/corrupted_numeral/realized_cer) are a
second, complementary layer: because a single CRIT-CTRL pair spans many
attempt-level rows (different models/CER targets/replicates), those five
covariates DO vary within a pair (unlike a pure item-level confounder), so
they still need to be adjusted for explicitly in the regression, on top of
the pair-conditioning.

`critical` itself is the CONTRAST of interest: 1 for every attempt-row
generated from the CRIT half of a pair, 0 for every attempt-row generated
from its matched CTRL half. Its coefficient is the adjusted log-odds
contrast this module reports.

`pair_id`: the Task-1.3 CRIT-CTRL match id
--------------------------------------------
Per the input contract, `pair_id` is "the CRIT-CTRL match id" -- i.e. one
value per Task-1.3 match, not a finer per-attempt stratum. `build_matched_
frame` (below) sets `pair_id` to the CRIT item's own `message_id`: every
attempt-row generated from EITHER the CRIT item OR its one matched CTRL
item shares that same `pair_id`, so a `pair_id` group mixes many attempt
conditions (models/CER targets/replicates) from both halves of the match.
This is why the five adjusters above are load-bearing: they carry the
attempt-level exposure information the pair-level conditioning cannot.

Concordant-pair dropping (documented, not a bug)
--------------------------------------------------
`ConditionalLogit` silently (with a `UserWarning`) drops any group with zero
within-group variance in the outcome -- e.g. a pair whose CRIT-half and
CTRL-half rows are ALL drift or ALL non-drift contributes no information to
a conditional likelihood (there is nothing to condition on). This is
standard behavior for conditional logistic regression on matched sets, not
a defect in this module. `matched_drift` captures that warning's text into
`notes` and reports `n_pairs`/`n_obs` from the model's OWN post-drop counts
(`res.n_groups`/`res.nobs`), not the raw input row/group counts, so the
returned counts always describe what was actually fit.

`converged`: `ConditionalLogit`'s own `.fit()` (verified by reading
`statsmodels/discrete/conditional_models.py`) discards the underlying
`scipy` optimizer's raw result (and its `mle_retvals` convergence flag) the
moment it builds the returned `ConditionalResults` object -- only `params`
and `cov_params()` survive that wrapping, so there is no `mle_retvals`
(or equivalent) to read back from a normal `.fit()` call. `converged` here
is therefore an honest, directly-checkable stand-in: `True` iff the fit
raised no exception and every reported `est`/`se` is finite; any exception,
or any non-finite `est`/`se`, is treated as `converged=False` and recorded
verbatim in `notes`.

Deterministic; American spelling; no em dashes (double hyphens for asides,
per house style).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

# The five prespecified adjusters, in the CONTROLLER RESOLUTION's order,
# plus `critical` (the contrast of interest) first so it always heads the
# returned `coef` dict.
_ADJUSTERS = ("char_len", "n_errors", "corrupted_negation", "corrupted_numeral", "realized_cer")
_DESIGN_COLUMNS = ("critical",) + _ADJUSTERS

# `matched_drift`'s full input contract (see module + brief "Input contract").
_REQUIRED_COLUMNS = ("drift", "critical") + _ADJUSTERS + ("pair_id", "message_id")


def matched_drift(df: pd.DataFrame) -> dict:
    """Adjusted CRIT-vs-CTRL matched-pair conditional logistic regression of
    `drift` on `critical`, conditioning on `pair_id` (the Task-1.3 CRIT-CTRL
    match) and adjusting for the five realized-corruption covariates.

    Args:
        df: DataFrame with `drift` (0/1), `critical` (0/1), `char_len`,
            `n_errors`, `corrupted_negation` (0/1), `corrupted_numeral`
            (0/1), `realized_cer`, `pair_id`, `message_id` (see
            `build_matched_frame`, which assembles exactly this shape from
            the real Step-5 artifacts).

    Returns:
        dict: `{engine: "conditional_logit", coef: {name: {est, se, ci_lo,
        ci_hi, p}} for name in ("critical", *the five adjusters), n_obs,
        n_pairs, converged, notes}`. `coef["critical"]` is the adjusted
        CRIT-vs-CTRL contrast this task replaces the old critical-vs-overall
        comparison with; the other five entries are the adjusters' own
        coefficients (reported for transparency, not a separate finding).
        `n_obs`/`n_pairs` are the model's OWN post-drop counts (see module
        docstring on concordant-pair dropping), not `len(df)`/
        `df["pair_id"].nunique()`.

    Raises:
        ValueError: if `df` is missing a required column, or if any row has
            a missing value in one of them (`ConditionalLogit` has no
            formula-layer `missing='drop'` to silently absorb a NaN the way
            `statsmodels.formula.api` does -- a NaN here would instead
            surface as an opaque numerical failure deep inside the
            optimizer, so it is caught up front with a clear message
            instead).
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"matched_drift: df is missing required column(s): {missing}")

    na_rows = df[list(_REQUIRED_COLUMNS)].isna().any(axis=1)
    n_missing = int(na_rows.sum())
    if n_missing:
        raise ValueError(
            f"matched_drift received {n_missing} row(s) with missing values in required "
            f"columns {list(_REQUIRED_COLUMNS)}; resolve before fitting"
        )

    endog = df["drift"].astype(int)
    exog = df[list(_DESIGN_COLUMNS)].astype(float)
    groups = df["pair_id"]

    n_pairs_input = int(groups.nunique())
    n_obs_input = int(len(df))

    fit_error = None
    res = None
    warn_texts: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model = ConditionalLogit(endog, exog, groups=groups)
            res = model.fit(disp=0)
        except Exception as exc:  # noqa: BLE001 - any fit failure is reported via `converged`/`notes`, not raised
            fit_error = f"{type(exc).__name__}: {exc}"
        warn_texts = [str(w.message) for w in caught]

    drop_note = "; ".join(warn_texts) if warn_texts else "no groups dropped"

    if res is None:
        notes = (
            "engine=conditional_logit (statsmodels.discrete.conditional_models.ConditionalLogit). "
            f"Fit raised an exception before returning a result: {fit_error}. "
            f"pair_id groups in input: {n_pairs_input}; rows in input: {n_obs_input}. "
            "Conditioning is on the matched pair (pair_id, the Task-1.3 CRIT-CTRL match); "
            f"warnings during the attempted fit: {drop_note}."
        )
        return {
            "engine": "conditional_logit",
            "coef": {},
            "n_obs": n_obs_input,
            "n_pairs": n_pairs_input,
            "converged": False,
            "notes": notes,
        }

    params = res.params
    bse = res.bse
    pvalues = res.pvalues
    ci = res.conf_int()

    finite_ok = bool(np.isfinite(params.to_numpy()).all() and np.isfinite(bse.to_numpy()).all())

    coef = {
        name: {
            "est": float(params[name]),
            "se": float(bse[name]),
            "ci_lo": float(ci.loc[name, 0]),
            "ci_hi": float(ci.loc[name, 1]),
            "p": float(pvalues[name]),
        }
        for name in _DESIGN_COLUMNS
    }

    n_obs = int(res.nobs)
    n_pairs = int(res.n_groups)
    n_pairs_dropped = n_pairs_input - n_pairs

    notes = (
        "engine=conditional_logit (statsmodels.discrete.conditional_models.ConditionalLogit, "
        "endog=drift, exog=[critical + char_len + n_errors + corrupted_negation + "
        "corrupted_numeral + realized_cer], groups=pair_id). Conditioning is on the matched "
        "pair (pair_id, the Task-1.3 CRIT-CTRL match): every pair's group-specific intercept is "
        "differenced out of the conditional likelihood, so any confounder constant within a "
        "matched pair is controlled for without being named. coef['critical'] is the adjusted "
        "CRIT-vs-CTRL contrast; the other five entries are the adjusters' own coefficients. "
        f"{n_pairs_dropped} of {n_pairs_input} input pairs dropped for having no within-group "
        "outcome variance (fully concordant -- same drift/non-drift outcome on every row in that "
        "pair, contributing nothing to a conditional likelihood; standard for conditional "
        f"logistic regression, not a bug). Fit warnings: {drop_note}. "
        f"converged={finite_ok} (fit raised no exception and every reported est/se is finite; "
        "ConditionalLogit's own .fit() does not expose the underlying optimizer's convergence "
        "flag on its returned result, see module docstring)."
    )

    return {
        "engine": "conditional_logit",
        "coef": coef,
        "n_obs": n_obs,
        "n_pairs": n_pairs,
        "converged": finite_ok,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Step-5 assembly helper.
# ---------------------------------------------------------------------------

_JOIN_KEYS = ("message_id", "cer_target", "replicate_idx")
_EXPOSURE_ATTACH_COLUMNS = ("n_errors", "char_len", "corrupted_negation", "corrupted_numeral")


def _pair_mapping(ctrl_matched: pd.DataFrame) -> pd.DataFrame:
    """Expand a Task-1.3 `ctrl_matched` frame (one row per CRIT item, with
    `crit_message_id` and the matched control's own `message_id`) into a
    long `message_id` -> (`critical`, `pair_id`) mapping: one row for the
    CRIT half (`critical=1`) and one row for the CTRL half (`critical=0`)
    of every match, both sharing `pair_id = crit_message_id` (see module
    docstring: `pair_id` is "the CRIT-CTRL match id", and the CRIT item's
    own id is a natural, already-unique choice for it).

    Raises:
        ValueError: if any `message_id` appears more than once across the
            mapping (a CRIT item reused as another item's control, or a
            duplicated `ctrl_matched` row) -- this would silently fan out
            the downstream join, so it is caught here instead.
    """
    crit_half = pd.DataFrame({
        "message_id": ctrl_matched["crit_message_id"],
        "critical": 1,
        "pair_id": ctrl_matched["crit_message_id"],
    })
    ctrl_half = pd.DataFrame({
        "message_id": ctrl_matched["message_id"],
        "critical": 0,
        "pair_id": ctrl_matched["crit_message_id"],
    })
    mapping = pd.concat([crit_half, ctrl_half], ignore_index=True)

    dup_ids = sorted(set(mapping.loc[mapping["message_id"].duplicated(keep=False), "message_id"]))
    if dup_ids:
        raise ValueError(
            "build_matched_frame: ctrl_matched has message_id value(s) appearing more than once "
            f"across the CRIT/CTRL mapping (would fan out the join): {dup_ids}"
        )

    return mapping


def build_matched_frame(labeled_path, ctrl_matched_path, exposure_path) -> pd.DataFrame:
    """Assemble `matched_drift`'s input contract from the real Step-5
    artifacts: labeled drift outcomes + exposure predictors + the Task-1.3
    CRIT-CTRL matched pairs.

    Args:
        labeled_path: path to `attempts_v2_labeled.parquet` (columns
            include `label`, `message_id`, `cer_target`, `replicate_idx`,
            `realized_cer` -- see `idrift.adjudicate.label_runner.run`'s
            output row and `idrift.analysis.mixed_models.build_analysis_
            frame`'s docstring for the real schema), OR an already-loaded
            DataFrame with those columns (accepted directly, matching the
            DataFrame-or-path convention already used elsewhere in this
            package, e.g. `mixed_models.build_analysis_frame`).
        ctrl_matched_path: path to `output/intermediate/ctrl_matched.csv`
            (the Task-1.3 `idrift.data.matched_controls.build_controls`
            output: one row per CRIT item, with `crit_message_id` and the
            matched control's own `message_id`), OR an already-loaded
            DataFrame.
        exposure_path: path to `exposure_v2_full.parquet` (columns include
            `message_id`, `cer_target`, `replicate_idx`, `n_errors`,
            `corrupted_negation`, `corrupted_numeral`, and `text` -- see
            `idrift.data.exposure_v2.build_exposure_v2`), OR an
            already-loaded DataFrame.

    Join and derivation:
        - `critical`/`pair_id` are attached via an INNER join on
          `message_id` against the Task-1.3 mapping (see `_pair_mapping`):
          only labeled rows belonging to a CRIT item or its one matched
          CTRL item survive; every other row (AUTH, or anything not part of
          a Task-1.3 match) is dropped, since it has no `critical`/`pair_id`
          to assign -- this module's whole contrast is CRIT-vs-its-own-
          matched-CTRL, not CRIT/CTRL-vs-AUTH.
        - `n_errors`/`char_len`/`corrupted_negation`/`corrupted_numeral` are
          then LEFT-joined in from the exposure frame on `(message_id,
          cer_target, replicate_idx)`, the same join keys and idiom
          `mixed_models.build_analysis_frame` uses (see that function's
          docstring): `char_len` is used as-is if already a literal column
          on the exposure side, else derived as `text.str.len()`.
          `validate="m:1"` on both merges enforces "no row multiplication"
          (a labeled row can match at most one mapping row and at most one
          exposure row) -- pandas raises `MergeError` instead of silently
          fanning out otherwise.
        - `drift` = `(label == "drift")`, cast to int, per the input
          contract ("`drift` (0/1; = 1 when label=='drift' else 0)").
        - `realized_cer` is assumed already present on the labeled side (it
          is generated once per exposure row and carried through the model
          runner unchanged -- see `idrift.models.o2_runner_v2`'s output
          row), not re-attached from the exposure frame.

    Returns:
        DataFrame with every column `matched_drift` requires (`drift`,
        `critical`, `char_len`, `n_errors`, `corrupted_negation`,
        `corrupted_numeral`, `realized_cer`, `pair_id`, `message_id`), one
        row per surviving labeled row (no fan-out).

    Raises:
        ValueError: if a join key column is missing from either frame, if
            the exposure frame has neither `char_len` nor `text`, if
            `ctrl_matched` maps any `message_id` more than once (see
            `_pair_mapping`), or if the assembled frame is still missing any
            of `matched_drift`'s required columns (e.g. `labeled_path` has
            no `realized_cer`).
    """
    labeled = labeled_path if isinstance(labeled_path, pd.DataFrame) else pd.read_parquet(labeled_path)
    ctrl_matched = ctrl_matched_path if isinstance(ctrl_matched_path, pd.DataFrame) else pd.read_csv(ctrl_matched_path)
    exposure = exposure_path if isinstance(exposure_path, pd.DataFrame) else pd.read_parquet(exposure_path)

    missing_labeled = [c for c in _JOIN_KEYS if c not in labeled.columns]
    missing_exposure = [c for c in _JOIN_KEYS if c not in exposure.columns]
    if missing_labeled or missing_exposure:
        raise ValueError(
            "build_matched_frame: missing join key column(s) -- "
            f"labeled missing {missing_labeled}, exposure missing {missing_exposure}"
        )

    if "label" not in labeled.columns:
        raise ValueError("build_matched_frame: labeled frame is missing a 'label' column to derive drift from")

    exposure_slim = exposure.copy()
    if "char_len" not in exposure_slim.columns:
        if "text" not in exposure_slim.columns:
            raise ValueError(
                "build_matched_frame: exposure frame has neither 'char_len' nor 'text' to derive it from"
            )
        exposure_slim["char_len"] = exposure_slim["text"].astype(str).str.len()

    missing_attach = [c for c in _EXPOSURE_ATTACH_COLUMNS if c not in exposure_slim.columns]
    if missing_attach:
        raise ValueError(f"build_matched_frame: exposure frame is missing column(s): {missing_attach}")

    exposure_slim = exposure_slim[list(_JOIN_KEYS) + list(_EXPOSURE_ATTACH_COLUMNS)]

    mapping = _pair_mapping(ctrl_matched)

    restricted = labeled.merge(mapping, on="message_id", how="inner", validate="m:1")
    merged = restricted.merge(exposure_slim, on=list(_JOIN_KEYS), how="left", validate="m:1")

    merged["drift"] = (merged["label"] == "drift").astype(int)

    missing_output = [c for c in _REQUIRED_COLUMNS if c not in merged.columns]
    if missing_output:
        raise ValueError(
            f"build_matched_frame: assembled frame is still missing column(s) {missing_output} "
            "that matched_drift requires"
        )

    return merged


# ---------------------------------------------------------------------------
# Task B6: absolute-scale effect measures.
#
# The conditional-logit `matched_drift` above reports the CRIT-vs-CTRL
# contrast on the odds-ratio scale (OR ~ 1.16). Reviewers judged that OR
# "statistically strong but clinically modest" and asked for the same
# contrast on the ABSOLUTE scale: the risk difference RD = P(drift|CRIT) -
# P(drift|CTRL), an additional-drifts-per-1000 rescale (RD*1000), and a
# criticality-by-CER interaction. The interaction is reported as
# stratum-specific RDs within each cer_target level -- the clean, always-
# converging choice (an explicit critical x cer product term in the
# conditional-logit design is not added, because on the full cohort the
# coarse pair-level conditioning already fails to converge and needs the
# finer identical-corruption strata, and a stratified RD answers the same
# "does the gap change with corruption?" question without any optimizer
# fragility -- see the digest's `_note`). All CIs are pair-CLUSTERED
# percentile bootstraps: whole matched pairs (pair_id) are the resampling
# unit, so within-pair correlation across the many attempt-level rows a
# single pair spans (models x CER targets x replicates) is preserved rather
# than treated as independent. Deterministic given `seed`.
# ---------------------------------------------------------------------------

_RD_REQUIRED_COLUMNS = ("drift", "critical", "pair_id")


def _pair_arm_stats(df: pd.DataFrame):
    """Per matched-pair (pair_id) drift sum/count split by arm, as four
    numpy arrays aligned by a common 0..P-1 pair index.

    `critical==1` is the CRIT arm, `critical==0` the CTRL arm. Precomputing
    these per-pair sufficient statistics lets the pair-clustered bootstrap
    recompute the CRIT-minus-CTRL risk difference in O(n_pairs) per replicate
    (sum the sampled pairs' sums/counts, divide) without re-touching the
    row-level frame -- so 1000 resamples over a 180k-row frame stay cheap.

    Returns:
        (crit_sum, crit_n, ctrl_sum, ctrl_n): float arrays indexed by pair.
    """
    crit = df["critical"].to_numpy()
    drift = df["drift"].to_numpy().astype(float)
    is_crit = crit == 1
    is_ctrl = crit == 0

    codes, _ = pd.factorize(df["pair_id"], sort=True)
    n_pairs = int(codes.max()) + 1 if len(codes) else 0

    crit_sum = np.zeros(n_pairs)
    crit_n = np.zeros(n_pairs)
    ctrl_sum = np.zeros(n_pairs)
    ctrl_n = np.zeros(n_pairs)
    np.add.at(crit_sum, codes[is_crit], drift[is_crit])
    np.add.at(crit_n, codes[is_crit], 1.0)
    np.add.at(ctrl_sum, codes[is_ctrl], drift[is_ctrl])
    np.add.at(ctrl_n, codes[is_ctrl], 1.0)
    return crit_sum, crit_n, ctrl_sum, ctrl_n


def _clustered_rd(df: pd.DataFrame, *, n_boot: int, seed: int) -> dict:
    """Point RD = P(drift|CRIT) - P(drift|CTRL) and its pair-clustered
    percentile-bootstrap 95% CI, plus the two arm-specific rates and counts.

    The bootstrap resamples whole pairs (clusters) with replacement; a
    replicate whose resample happens to leave either arm with no observations
    is dropped (guarded, effectively never happens once every pair carries
    both arms, as the matched design guarantees). CI bounds are the 2.5th and
    97.5th percentiles of the finite replicates.
    """
    crit_sum, crit_n, ctrl_sum, ctrl_n = _pair_arm_stats(df)
    n_pairs = len(crit_sum)

    tot_crit_n = float(crit_n.sum())
    tot_ctrl_n = float(ctrl_n.sum())
    if tot_crit_n == 0 or tot_ctrl_n == 0:
        raise ValueError(
            "_clustered_rd: one arm has no rows (need both CRIT (critical==1) and CTRL "
            f"(critical==0) observations; got n_crit={int(tot_crit_n)}, n_ctrl={int(tot_ctrl_n)})"
        )

    p_crit = float(crit_sum.sum() / tot_crit_n)
    p_ctrl = float(ctrl_sum.sum() / tot_ctrl_n)
    rd = p_crit - p_ctrl

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_pairs, size=n_pairs)
        cn = crit_n[idx].sum()
        tn = ctrl_n[idx].sum()
        if cn > 0 and tn > 0:
            boots[b] = crit_sum[idx].sum() / cn - ctrl_sum[idx].sum() / tn
        else:
            boots[b] = np.nan

    valid = boots[np.isfinite(boots)]
    ci_lo = float(np.percentile(valid, 2.5)) if valid.size else float("nan")
    ci_hi = float(np.percentile(valid, 97.5)) if valid.size else float("nan")
    boot_se = float(np.std(valid, ddof=1)) if valid.size > 1 else float("nan")

    return {
        "rd": rd,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_crit": p_crit,
        "p_ctrl": p_ctrl,
        "n_crit": int(tot_crit_n),
        "n_ctrl": int(tot_ctrl_n),
        "n_pairs": int(n_pairs),
        "n_boot": int(n_boot),
        "n_boot_used": int(valid.size),
        "boot_se": boot_se,
    }


def _check_rd_columns(df: pd.DataFrame, required) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"risk_difference: df is missing required column(s): {missing}")
    na = df[list(required)].isna().any(axis=1)
    n_missing = int(na.sum())
    if n_missing:
        raise ValueError(
            f"risk_difference received {n_missing} row(s) with missing values in required "
            f"columns {list(required)}; resolve before computing the risk difference"
        )


def risk_difference(df: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """Absolute CRIT-vs-CTRL risk difference RD = P(drift|CRIT) -
    P(drift|CTRL), with a pair-clustered percentile-bootstrap 95% CI, plus
    the same effect expressed as additional drifts per 1000 (RD*1000).

    This is the absolute-scale companion to `matched_drift`'s odds-ratio
    contrast, added for Task B6 because reviewers asked for the clinically
    interpretable magnitude alongside the OR.

    Args:
        df: DataFrame with `drift` (0/1), `critical` (0/1), and `pair_id`
            (the CRIT-CTRL match id; the bootstrap resampling unit). Extra
            columns are ignored, so the frame from `build_matched_frame` can
            be passed directly.
        n_boot: number of pair-clustered bootstrap replicates.
        seed: RNG seed (fixed -> deterministic CI).

    Returns:
        dict: `{estimand, rd, ci_lo, ci_hi, additional_drifts_per_1000:
        {est, ci_lo, ci_hi}, p_crit, p_ctrl, n_crit, n_ctrl, n_pairs,
        n_boot, seed, boot_se, notes}`. `additional_drifts_per_1000` is a
        pure rescale of `rd` and its CI by 1000.

    Raises:
        ValueError: if a required column is missing, if any required-column
            value is missing, or if either arm has no rows.
    """
    _check_rd_columns(df, _RD_REQUIRED_COLUMNS)
    core = _clustered_rd(df, n_boot=n_boot, seed=seed)
    rd = core["rd"]

    notes = (
        "estimand=risk_difference RD = P(drift|CRIT) - P(drift|CTRL) on the matched CRIT-vs-CTRL "
        "design (the absolute-scale companion to matched_drift's odds-ratio contrast). CI is a "
        f"pair-clustered percentile bootstrap ({core['n_boot_used']} of {n_boot} replicates finite): "
        "whole matched pairs (pair_id) are resampled with replacement, preserving within-pair "
        "correlation across the many attempt-level rows a pair spans (models x CER targets x "
        "replicates). additional_drifts_per_1000 = RD*1000 (a pure rescale of RD and its CI). "
        f"n_pairs={core['n_pairs']}, n_crit={core['n_crit']}, n_ctrl={core['n_ctrl']}, seed={seed}."
    )

    return {
        "estimand": "risk_difference_crit_minus_ctrl",
        "rd": rd,
        "ci_lo": core["ci_lo"],
        "ci_hi": core["ci_hi"],
        "additional_drifts_per_1000": {
            "est": rd * 1000.0,
            "ci_lo": core["ci_lo"] * 1000.0,
            "ci_hi": core["ci_hi"] * 1000.0,
        },
        "p_crit": core["p_crit"],
        "p_ctrl": core["p_ctrl"],
        "n_crit": core["n_crit"],
        "n_ctrl": core["n_ctrl"],
        "n_pairs": core["n_pairs"],
        "n_boot": core["n_boot"],
        "seed": int(seed),
        "boot_se": core["boot_se"],
        "notes": notes,
    }


def risk_difference_by_cer(df: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0) -> dict:
    """The criticality-by-CER interaction on the absolute scale: the CRIT-
    vs-CTRL risk difference computed WITHIN each `cer_target` level (one RD
    per corruption band), each with its own pair-clustered bootstrap 95% CI
    and additional-drifts-per-1000 rescale.

    Reported as stratum-specific RDs rather than a single critical x cer
    product term in the conditional-logit model: on the full cohort the
    coarse pair-level conditional logit already fails to converge (see the
    module docstring and the digest `_note`), so a stratified RD is the clean
    way to show whether the absolute gap changes with corruption, with no
    optimizer fragility. Each stratum gets a distinct but deterministic
    bootstrap stream (`seed + i` for the i-th CER level, ascending).

    Args:
        df: as `risk_difference`, plus a `cer_target` column to stratify on.
        n_boot: bootstrap replicates per stratum.
        seed: base RNG seed.

    Returns:
        dict: `{estimand, interaction_form: "stratified", by_cer: [ {cer_
        target, rd, ci_lo, ci_hi, additional_drifts_per_1000, p_crit,
        p_ctrl, n_crit, n_ctrl, n_pairs} per CER level, ascending ], n_boot,
        seed, notes}`.

    Raises:
        ValueError: as `risk_difference`, and if `cer_target` is missing.
    """
    required = _RD_REQUIRED_COLUMNS + ("cer_target",)
    _check_rd_columns(df, required)

    strata = []
    for i, cer in enumerate(sorted(df["cer_target"].unique())):
        sub = df[df["cer_target"] == cer]
        core = _clustered_rd(sub, n_boot=n_boot, seed=seed + i)
        rd = core["rd"]
        strata.append({
            "cer_target": float(cer),
            "rd": rd,
            "ci_lo": core["ci_lo"],
            "ci_hi": core["ci_hi"],
            "additional_drifts_per_1000": rd * 1000.0,
            "p_crit": core["p_crit"],
            "p_ctrl": core["p_ctrl"],
            "n_crit": core["n_crit"],
            "n_ctrl": core["n_ctrl"],
            "n_pairs": core["n_pairs"],
        })

    notes = (
        "criticality-by-CER interaction on the absolute scale: CRIT-vs-CTRL risk difference within "
        "each cer_target level, each with its own pair-clustered percentile-bootstrap 95% CI. "
        "Reported as stratum-specific RDs (interaction_form=stratified), not a critical x cer product "
        "term, because the coarse pair-level conditional logit does not converge on the full cohort "
        "(see module docstring / digest _note); the stratified RD shows whether the absolute gap "
        "changes with corruption without optimizer fragility. Each stratum uses a distinct "
        f"deterministic bootstrap stream (seed+i, base seed={seed}); {n_boot} replicates per stratum."
    )

    return {
        "estimand": "risk_difference_crit_minus_ctrl_by_cer_target",
        "interaction_form": "stratified",
        "by_cer": strata,
        "n_boot": int(n_boot),
        "seed": int(seed),
        "notes": notes,
    }
