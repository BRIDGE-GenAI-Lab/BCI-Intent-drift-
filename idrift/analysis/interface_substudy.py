"""Interface-condition effect on drift, abstention, candidate recovery, and
benefit-harm (revision Task D5).

Headline question this module answers: does allowing abstention
(`abstain_enabled`) or conservative editing (`minimal_edit`,
`copy_when_uncertain`) reduce SILENT drift, and at what cost to faithful
throughput -- versus forced reconstruction (`forced`/P0, the reference
condition) and the two conditions that add rather than withhold
(`candidate_list`, `expansion`)? See `idrift.interfaces.conditions.
CONDITION_ORDER` for the full 6-condition registry.

`forced` is NOT produced by the substudy generation run -- it is the
already-labeled main-run baseline, reused at analysis time (not
re-generated) so the comparison uses byte-identical exposures.
`assemble_forced` reconstructs it from `attempts_v3_labeled.parquet` (AUTH)
and `attempts_v2_labeled.parquet` (CRIT/CTRL), subset to the substudy's 562
message_ids with `replicate_idx < idrift.substudy_sample.MAX_REPLICATE`.

The one safety invariant every function here respects: a `declined`
(abstained) row is NEVER folded into the {faithful, degraded, drift}
denominator, and never enters a benefit-harm transition (an abstention has
no raw-decode transition meaning -- there is no LLM output to compare
against the raw decode). Every function that could accidentally do so
excludes `declined` rows explicitly and reports how many were excluded.

Reused, not reimplemented:
- `idrift.analysis.benefit_harm.transition_matrix` for the five clinical
  benefit-harm categories, called per condition via its `group_cols`.
- The message-clustered cluster-robust logistic idiom from
  `idrift.analysis.multinomial`/`idrift.analysis.mixed_models` (Newton then
  L-BFGS fallback, `cov_type="cluster"` on `message_id`) -- mirrored here as
  `_fit_clustered_logit`, not imported, since the two source modules' fallback
  helpers are private and shaped around their own model objects (MNLogit /
  smf.logit-with-precomputed-`_y`), not a `condition`-contrast formula.

Pure pandas/numpy/statsmodels analysis on already-labeled frames. No models,
no I/O beyond reading the input parquets and writing the one digest JSON.

American spelling; no em dashes (double hyphens for asides, per house
style).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from idrift.analysis.benefit_harm import transition_matrix
from idrift.analysis.digests import _default
from idrift.interfaces.conditions import CONDITION_ORDER
from idrift.substudy_sample import MAX_REPLICATE

_Z = 1.959963984540054  # two-sided 95% normal quantile (matches multinomial._Z)

_OUTCOME_CLASSES = ("faithful", "degraded", "drift")
_DECLINED = "declined"

_CONDITION_TERM_RE = re.compile(
    r"^C\(condition,\s*Treatment\(['\"]forced['\"]\)\)\[T\.(?P<cond>.+)\]$"
)


def _as_df(x) -> pd.DataFrame:
    """Accept an already-loaded DataFrame or a parquet path (the
    DataFrame-or-path convention used throughout this package, e.g.
    `idrift.analysis.benefit_harm._as_df`)."""
    if isinstance(x, pd.DataFrame):
        return x
    return pd.read_parquet(x)


# ---------------------------------------------------------------------------
# 1. per_condition_rates
# ---------------------------------------------------------------------------


def _rate_block(sub: pd.DataFrame) -> dict:
    """Declined-aware rate summary for one slice of rows.

    `declined_rate` is declined/n_total (declined counted, never hidden).
    The faithful/degraded/drift rates are a fraction of NON-declined rows
    only -- declined rows never enter that denominator.
    """
    n_total = int(len(sub))
    n_declined = int((sub["outcome"] == _DECLINED).sum())
    declined_rate = (n_declined / n_total) if n_total else float("nan")

    non_declined = sub[sub["outcome"] != _DECLINED]
    n_non_declined = int(len(non_declined))

    block = {
        "n_total": n_total,
        "n_declined": n_declined,
        "declined_rate": declined_rate,
        "n_non_declined": n_non_declined,
    }
    for cls in _OUTCOME_CLASSES:
        c = int((non_declined["outcome"] == cls).sum())
        block[f"n_{cls}"] = c
        block[f"rate_{cls}"] = (c / n_non_declined) if n_non_declined else float("nan")
    return block


def per_condition_rates(labeled_all: pd.DataFrame) -> dict:
    """Declined/faithful/degraded/drift rates per interface condition,
    overall and broken out by corpus.

    Returns:
        dict: `{"conditions_present": [...CONDITION_ORDER order...],
        "by_condition": {condition: {"overall": <rate block>,
        "by_corpus": {corpus: <rate block>}}}}`. A rate block is
        `_rate_block`'s shape: `n_total`, `n_declined`, `declined_rate`,
        `n_non_declined`, and `n_<cls>`/`rate_<cls>` for each of
        {faithful, degraded, drift} -- the three rates sum to 1 over the
        non-declined rows.
    """
    present = [c for c in CONDITION_ORDER if c in set(labeled_all["condition"])]

    by_condition = {}
    for cond in present:
        sub = labeled_all[labeled_all["condition"] == cond]
        by_corpus = {
            str(corpus): _rate_block(csub) for corpus, csub in sub.groupby("corpus", sort=True)
        }
        by_condition[cond] = {"overall": _rate_block(sub), "by_corpus": by_corpus}

    return {"conditions_present": present, "by_condition": by_condition}


# ---------------------------------------------------------------------------
# 2. assemble_forced -- reconstruct the P0/forced reference condition
# ---------------------------------------------------------------------------

_FORCED_REQUIRED = ("message_id", "corpus", "replicate_idx", "label")


def _subset_pairs(subset):
    """Normalize the subset argument to a set of `(message_id, corpus)`
    pairs, or `None` when only bare message_ids were supplied.

    Accepts a DataFrame carrying `message_id` + `corpus` (the
    `substudy_exposures.parquet` shape), an iterable of 2-tuples, or an
    iterable of bare ids. Matching on the PAIR is what keeps a CRIT item
    from also dragging in its AUTH twin -- see `assemble_forced`.
    """
    if isinstance(subset, pd.DataFrame):
        if "corpus" not in subset.columns:
            return None
        return set(zip(subset["message_id"], subset["corpus"]))

    items = list(subset)
    if items and all(isinstance(x, tuple) and len(x) == 2 for x in items):
        return set(items)
    return None


def _subset_ids(subset) -> set:
    """Bare message_ids for the legacy id-only path. A DataFrame yields its
    `message_id` COLUMN -- iterating a frame would yield its column names."""
    if isinstance(subset, pd.DataFrame):
        return set(subset["message_id"])
    return set(subset)


def _subset_mask(frame: pd.DataFrame, pairs, ids) -> pd.Series:
    """Row mask for the subset: on `(message_id, corpus)` when pairs are
    available, otherwise on `message_id` alone."""
    if pairs is not None:
        keys = pd.MultiIndex.from_arrays([frame["message_id"], frame["corpus"]])
        return pd.Series(keys.isin(pairs), index=frame.index)
    return frame["message_id"].isin(ids)


def assemble_forced(subset, v3_labeled, v2_labeled) -> pd.DataFrame:
    """Reconstruct the `forced` (P0) reference condition for the substudy
    from already-labeled main-run data -- it is reused, never regenerated.

    AUTH forced rows come from `v3_labeled` (subset to `corpus == "AUTH"`);
    CRIT/CTRL forced rows come from `v2_labeled` (subset to
    `corpus.isin(["CRIT", "CTRL"])`). The explicit corpus filter on each
    side matters even though `subset` already narrows the id space:
    `v2_labeled` is the FULL main run and also carries its own AUTH rows
    (a different, larger AUTH draw than the substudy's), which the corpus
    filter keeps out. Both sides are further restricted to the subset and
    to `replicate_idx < idrift.substudy_sample.MAX_REPLICATE` (10).

    PASS THE `(message_id, corpus)` PAIRS, NOT BARE IDS. The probe items
    are drawn FROM the Costello message bank, so a CRIT item and an AUTH
    item routinely share a `message_id`. Filtering `corpus == "AUTH"` and
    `message_id.isin(ids)` independently therefore also admits the AUTH
    twin of every CRIT/CTRL id in the subset -- on the real substudy that
    inflated the forced arm from 300 to 431 AUTH messages (242,550 rows
    instead of 196,700) and silently changed its corpus mix, biasing every
    condition-versus-forced contrast. A correctly assembled forced arm has
    exactly as many rows as each substudy condition.

    Args:
        subset: the substudy's exposure set. Either a DataFrame with
            `message_id` + `corpus` (e.g.
            `output/intermediate/substudy_exposures.parquet`, 562 messages)
            or an iterable of `(message_id, corpus)` tuples; both match on
            the pair. An iterable of bare message_ids is still accepted for
            back-compatibility and matches on id alone, which is only safe
            when no id is shared across corpora.
        v3_labeled: `attempts_v3_labeled.parquet` (path or DataFrame).
        v2_labeled: `attempts_v2_labeled.parquet` (path or DataFrame).

    Returns:
        DataFrame: the AUTH + CRIT/CTRL rows concatenated, with `label`
        copied to a new `outcome` column (the substudy's label taxonomy is
        the same {faithful, degraded, drift}; `forced` never abstains, so
        `outcome` never takes the "declined" value here), `condition` set
        to `"forced"`, `abstained` set to `False`, and `any_candidate_faithful`
        set to pandas `NA` (nullable boolean) -- `forced` never offers
        alternates, so "any candidate faithful" is not a meaningful
        question for it, matching `idrift.substudy_label.assemble_labeled`'s
        convention for every non-`candidate_list` condition.

    Raises:
        ValueError: if either input frame is missing a required column.
    """
    v3 = _as_df(v3_labeled)
    v2 = _as_df(v2_labeled)

    for name, frame in (("v3_labeled", v3), ("v2_labeled", v2)):
        missing = [c for c in _FORCED_REQUIRED if c not in frame.columns]
        if missing:
            raise ValueError(f"assemble_forced: {name} is missing column(s): {missing}")

    pairs = _subset_pairs(subset)
    ids = None if pairs is not None else _subset_ids(subset)

    auth = v3[
        (v3["corpus"] == "AUTH") & _subset_mask(v3, pairs, ids) & (v3["replicate_idx"] < MAX_REPLICATE)
    ].copy()
    crit_ctrl = v2[
        v2["corpus"].isin(["CRIT", "CTRL"]) & _subset_mask(v2, pairs, ids) & (v2["replicate_idx"] < MAX_REPLICATE)
    ].copy()

    combined = pd.concat([auth, crit_ctrl], ignore_index=True, sort=False)
    combined["outcome"] = combined["label"]
    combined["condition"] = "forced"
    combined["abstained"] = False
    combined["any_candidate_faithful"] = pd.array([pd.NA] * len(combined), dtype="boolean")
    return combined


# ---------------------------------------------------------------------------
# 3. drift_condition_model -- message-clustered logit, condition vs forced
# ---------------------------------------------------------------------------


def _fit_clustered_logit(formula: str, data: pd.DataFrame, *, seed: int = 0):
    """Message-clustered logit, Newton then L-BFGS fallback (mirrors the
    `_fit_with_fallback` idiom in `idrift.analysis.multinomial` /
    `idrift.analysis.mixed_models`: try Newton first, fall back to L-BFGS if
    it raises or does not converge; whichever method actually produced the
    returned result is reported).

    Returns:
        tuple: `(result_or_None, method_used_or_None, converged: bool)`. A
        patsy/statsmodels failure (singular design, perfect separation --
        plausible on a small synthetic frame or a degenerate condition
        subgroup) is caught per method rather than raised; if every method
        raises, returns `(None, None, False)`.
    """
    groups = data["message_id"].to_numpy()
    last_res = None
    last_method = None
    for method in ("newton", "lbfgs"):
        try:
            np.random.seed(seed)
            model = smf.logit(formula, data=data)
            res = model.fit(
                method=method,
                disp=0,
                maxiter=200,
                cov_type="cluster",
                cov_kwds={"groups": groups},
            )
        except Exception:  # noqa: BLE001 - degenerate fit; try the next method
            continue

        converged = bool(res.mle_retvals.get("converged", True)) if hasattr(res, "mle_retvals") else True
        last_res, last_method = res, method
        if converged:
            return res, method, True

    if last_res is not None:
        return last_res, last_method, False
    return None, None, False


def _condition_odds_ratios(res) -> dict:
    """Extract the `C(condition, Treatment("forced"))[T.<cond>]` MAIN-EFFECT
    terms (never the `:cer_target` interaction terms -- the regex anchors on
    end-of-string) into a per-condition odds-ratio/CI/p dict."""
    out = {}
    for term in res.params.index:
        m = _CONDITION_TERM_RE.match(term)
        if not m:
            continue
        coef = float(res.params[term])
        se = float(res.bse[term])
        out[m.group("cond")] = {
            "odds_ratio": float(np.exp(coef)),
            "ci_lower": float(np.exp(coef - _Z * se)),
            "ci_upper": float(np.exp(coef + _Z * se)),
            "p": float(res.pvalues[term]),
        }
    return out


def drift_condition_model(labeled_all: pd.DataFrame) -> dict:
    """Message-clustered logistic regression of drift on interface condition
    (reference = `forced`) and `cer_target`.

    Declined rows are DROPPED before fitting (an abstention is not a drift
    outcome one way or the other -- see the module docstring). Tries
    `C(condition, Treatment("forced")) * cer_target` (the condition x CER
    interaction) first; if that does not converge (or fails to fit at all),
    falls back to the main-effects-only model `C(condition,
    Treatment("forced")) + cer_target` and notes the fallback. Either way,
    the reported odds ratios are the condition MAIN-EFFECT terms (in the
    interaction model, that is the OR at cer_target == 0); cluster-robust SE
    on `message_id`.

    Returns:
        dict: `{"formula_used": "interaction" | "main_effects_only" | None,
        "interaction_converged": bool, "converged": bool,
        "method_used": "newton" | "lbfgs" | None, "reference_condition":
        "forced", "condition_odds_ratios": {condition: {"odds_ratio",
        "ci_lower", "ci_upper", "p"}}, "n_obs": int, "n_clusters": int,
        "n_declined_excluded": int, "notes": str,
        "main_effects_reference": {"converged": bool, "method_used": str |
        None, "condition_odds_ratios": {...}}}`. `condition_odds_ratios`
        is empty and `condition_odds_ratios`/`converged` reflect a failed
        fit if BOTH formulas fail (e.g. every non-reference condition
        subgroup is degenerate).

        `main_effects_reference` always carries the main-effects-only fit,
        even when the interaction model converged and supplied the headline
        `condition_odds_ratios`. The two answer different questions and can
        disagree in SIGN: an interaction-model condition term is the odds
        ratio at `cer_target == 0`, where drift is rare, while the
        main-effects term is the CER-averaged effect. Report the averaged
        one whenever the claim is about a condition overall.
    """
    n_declined_excluded = int((labeled_all["outcome"] == _DECLINED).sum())
    non_declined = labeled_all[labeled_all["outcome"] != _DECLINED].copy()
    non_declined["drift"] = (non_declined["outcome"] == "drift").astype(int)

    interaction_formula = "drift ~ C(condition, Treatment('forced')) * cer_target"
    main_formula = "drift ~ C(condition, Treatment('forced')) + cer_target"

    int_res, int_method, int_converged = _fit_clustered_logit(interaction_formula, non_declined)
    main_res, main_method, main_converged = _fit_clustered_logit(main_formula, non_declined)

    main_effects_reference = {
        "converged": bool(main_converged),
        "method_used": main_method,
        "condition_odds_ratios": _condition_odds_ratios(main_res) if main_res is not None else {},
    }

    if int_converged:
        res, method_used, converged, formula_used = int_res, int_method, True, "interaction"
        notes = ""
    else:
        res, method_used, converged = main_res, main_method, main_converged
        formula_used = "main_effects_only"
        notes = (
            "condition x cer_target interaction did not converge (or failed to fit); "
            "fell back to the main-effects-only model."
        )

    if res is None:
        return {
            "formula_used": None,
            "interaction_converged": bool(int_converged),
            "converged": False,
            "method_used": None,
            "reference_condition": "forced",
            "condition_odds_ratios": {},
            "n_obs": int(len(non_declined)),
            "n_clusters": int(non_declined["message_id"].nunique()),
            "n_declined_excluded": n_declined_excluded,
            "notes": (notes + " Both the interaction and main-effects logit failed to fit.").strip(),
            "main_effects_reference": main_effects_reference,
        }

    return {
        "formula_used": formula_used,
        "interaction_converged": bool(int_converged),
        "converged": bool(converged),
        "method_used": method_used,
        "reference_condition": "forced",
        "condition_odds_ratios": _condition_odds_ratios(res),
        "n_obs": int(len(non_declined)),
        "n_clusters": int(non_declined["message_id"].nunique()),
        "n_declined_excluded": n_declined_excluded,
        "notes": notes,
        "main_effects_reference": main_effects_reference,
    }


# ---------------------------------------------------------------------------
# 4. candidate_recovery -- candidate_list primary vs any-candidate faithful
# ---------------------------------------------------------------------------


def _candidate_rates(sub: pd.DataFrame) -> dict:
    n = int(len(sub))
    if n == 0:
        return {"n": 0, "primary_faithful_rate": float("nan"), "any_candidate_faithful_rate": float("nan"), "recovery_lift": float("nan")}
    primary = float((sub["outcome"] == "faithful").mean())
    any_faithful = float(sub["any_candidate_faithful"].fillna(False).astype(bool).mean())
    return {
        "n": n,
        "primary_faithful_rate": primary,
        "any_candidate_faithful_rate": any_faithful,
        "recovery_lift": any_faithful - primary,
    }


def candidate_recovery(labeled_all: pd.DataFrame) -> dict:
    """For `candidate_list` (P4) rows only: how much offering alternates
    recovers over the primary (best) candidate alone.

    `primary_faithful_rate` = P(outcome == "faithful") (the row's own best
    candidate); `any_candidate_faithful_rate` = P(any_candidate_faithful)
    (primary OR any alternate faithful); `recovery_lift` is their
    difference -- always >= 0 by construction, since `any_candidate_faithful`
    is an OR that includes the primary outcome (see
    `idrift.substudy_label.assemble_labeled`).

    Returns:
        dict: `{"n_total", "primary_faithful_rate", "any_candidate_faithful_rate",
        "recovery_lift", "by_cer_target": {cer_target: <same four keys plus n>}}`.
    """
    df = labeled_all[labeled_all["condition"] == "candidate_list"]

    overall = _candidate_rates(df)
    by_cer = {
        float(cer): _candidate_rates(sub) for cer, sub in df.groupby("cer_target", sort=True)
    }

    return {
        "n_total": overall["n"],
        "primary_faithful_rate": overall["primary_faithful_rate"],
        "any_candidate_faithful_rate": overall["any_candidate_faithful_rate"],
        "recovery_lift": overall["recovery_lift"],
        "by_cer_target": by_cer,
    }


# ---------------------------------------------------------------------------
# 5. benefit_harm_by_condition -- transition_matrix reused, grouped by condition
# ---------------------------------------------------------------------------


def _rank_by_net_metric(pooled_rows: list[dict]) -> list[dict]:
    """Sort per-condition pooled rows by `faithful_gained_per_drift_introduced`
    descending (higher = more rescue per unit of newly-introduced drift =
    better on the harm axis); `None`/NaN (undefined, 0/0) sorts last, not
    first, since it is neither good nor bad -- just uninformative."""

    def _key(rec):
        v = rec.get("faithful_gained_per_drift_introduced")
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return float("-inf")
        return v

    ranked = sorted(pooled_rows, key=_key, reverse=True)
    return [
        {"condition": r["condition"], "faithful_gained_per_drift_introduced": r["faithful_gained_per_drift_introduced"]}
        for r in ranked
    ]


def benefit_harm_by_condition(labeled_all: pd.DataFrame, raw_decode_labels) -> dict:
    """Per-condition benefit-harm transition rates, reusing
    `idrift.analysis.benefit_harm.transition_matrix` (never reimplemented).

    Declined rows are excluded before the cross-tab: an abstention has no
    raw-decode transition meaning (there is no LLM output to compare
    against the raw decode). `raw_decode_labels` may be `None` (e.g. the
    heavy raw-decode labeling job has not run yet) -- in that case this
    returns a skipped-section marker instead of crashing.

    Args:
        labeled_all: the assembled 6-condition labeled frame.
        raw_decode_labels: path, DataFrame, or `None`. See
            `idrift.analysis.benefit_harm.label_raw_decode`'s output shape.

    Returns:
        dict: `{"skipped": False, "n_declined_excluded": int,
        "by_condition_corpus_cer": {"by_group": [...], "notes": [...]},
        "by_condition_pooled": {"by_group": [...], "overall": {...},
        "notes": [...]}, "ranked_by_net_metric": [{"condition",
        "faithful_gained_per_drift_introduced"}, ...]}` (best condition on
        the harm axis first), or `{"skipped": True, "reason": str}` when
        `raw_decode_labels` is `None`.
    """
    if raw_decode_labels is None:
        return {"skipped": True, "reason": "no raw-decode labels provided; benefit-harm section skipped"}

    raw = _as_df(raw_decode_labels)

    n_declined_excluded = int((labeled_all["outcome"] == _DECLINED).sum())
    non_declined = labeled_all[labeled_all["outcome"] != _DECLINED].copy()

    per_group = transition_matrix(
        non_declined, raw, output_label_col="outcome", group_cols=("condition", "corpus", "cer_target")
    )
    pooled = transition_matrix(non_declined, raw, output_label_col="outcome", group_cols=("condition",))

    pooled_rows = pooled["by_group"].to_dict(orient="records")

    return {
        "skipped": False,
        "n_declined_excluded": n_declined_excluded,
        "by_condition_corpus_cer": {
            "by_group": per_group["by_group"].to_dict(orient="records"),
            "notes": per_group["notes"],
        },
        "by_condition_pooled": {
            "by_group": pooled_rows,
            "overall": pooled["overall"],
            "notes": pooled["notes"],
        },
        "ranked_by_net_metric": _rank_by_net_metric(pooled_rows),
    }


# ---------------------------------------------------------------------------
# 6. run -- orchestrate all 6 conditions + all 4 analyses, write the digest
# ---------------------------------------------------------------------------


def run(
    substudy_labeled_path,
    v3_labeled_path,
    v2_labeled_path,
    subset_parquet,
    raw_decode_path=None,
    out_path: str = "output/interface_substudy_digest.json",
) -> dict:
    """Load the substudy's P1-P5 labels, assemble the P0/`forced` reference
    condition, run the four analyses, write the digest JSON, and return the
    same dict.

    Args:
        substudy_labeled_path: `substudy_attempts_labeled.parquet` (Task D4
            output; P1-P5 rows). Path or DataFrame.
        v3_labeled_path, v2_labeled_path: `attempts_v3_labeled.parquet` /
            `attempts_v2_labeled.parquet`, forwarded to `assemble_forced`.
        subset_parquet: `substudy_exposures.parquet` (or any frame with
            `message_id` + `corpus`) -- forwarded whole to
            `assemble_forced`, which matches on the `(message_id, corpus)`
            pair.
        raw_decode_path: path/DataFrame/`None`, forwarded to
            `benefit_harm_by_condition`.
        out_path: where to write the digest JSON.

    Returns:
        dict: `{"per_condition_rates", "drift_condition_model",
        "candidate_recovery", "benefit_harm_by_condition", "n_rows_total",
        "n_rows_substudy", "n_rows_forced_assembled"}` -- the same dict
        written to `out_path`.
    """
    substudy = _as_df(substudy_labeled_path)
    subset = _as_df(subset_parquet)

    # pass the (message_id, corpus) pairs, never bare ids -- see assemble_forced
    forced = assemble_forced(subset, v3_labeled_path, v2_labeled_path)

    labeled_all = pd.concat([substudy, forced], ignore_index=True, sort=False)

    raw_labels = None if raw_decode_path is None else _as_df(raw_decode_path)

    digest = {
        "per_condition_rates": per_condition_rates(labeled_all),
        "drift_condition_model": drift_condition_model(labeled_all),
        "candidate_recovery": candidate_recovery(labeled_all),
        "benefit_harm_by_condition": benefit_harm_by_condition(labeled_all, raw_labels),
        "n_rows_total": int(len(labeled_all)),
        "n_rows_substudy": int(len(substudy)),
        "n_rows_forced_assembled": int(len(forced)),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2, default=_default, sort_keys=True))

    return digest
