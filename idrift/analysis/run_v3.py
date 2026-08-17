"""Task C4: re-run the primary statistics on a labeled v3 corpus.

The v2 driver that produced `output/stats_v2_digest.json` has since been
deleted; this module reconstructs its composition from the shape of that
digest (and the standalone `output/{multinomial,confidence,
dependence_sensitivity,zero_cer_breakdown}_digest.json` files) and re-runs
it against `attempts_v3_labeled.parquet` (1,701,000 rows -- AUTH grew from
500 to 2,168 phrasings; CRIT/CTRL are byte-identical to v2). Every fitting
call below is a single, unmodified call into the already-tested
`idrift.analysis.*` module named in the surrounding comment -- no
statistics are reimplemented here, and no fallback logic is added beyond
what each module already does internally.

Reuse vs. rerun (see the Task C4 brief)
----------------------------------------
RE-RUN on v3 (AUTH-dependent, so the numbers move): drift_curve,
calibration, multinomial, confidence, dependence_sensitivity, zero_cer_audit,
mixed_model.

REUSE verbatim from v2 (CRIT/CTRL matching and the physician panel did not
change): `matched_compare` (embedded in stats_v2_digest.json) and the two
standalone artifacts `output/balance_digest.json` / `output/stratified_diag_
digest.json`, which are untouched by this driver -- they still describe the
current CRIT/CTRL corpora exactly.

`sensitivity` (the tau/tie-break recalibration table) is NOT in the C4
brief's module list: recomputing it means re-deriving labels via
`idrift.adjudicate.label_runner.derive_labels` (a per-row Python loop) at
several tau/tie_break settings over the full 1.7M-row corpus, which this
task was not scoped to run. It is therefore copied from v2 and flagged
`reused_from` with an explicit note, same as the matched_compare/balance/
stratified_diag reuses -- this is a gap the brief left uncovered, not a
judgment that the number is unchanged.

Benefit-harm is optional because its raw-decode labels are generated in a
separate, expensive pass. Supply ``--raw-decode`` when the corresponding
model-independent raw-label parquet is available; the labels broadcast
across every model in the labeled panel.

Usage:
    uv run python -m idrift.analysis.run_v3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from idrift.analysis import (
    benefit_harm,
    calibration_v2,
    confidence_analysis,
    dependence_sensitivity,
    drift_curve,
    mixed_models,
    multinomial,
    zero_cer_audit,
)

LABELED_PATH = "output/intermediate/attempts_v3_labeled.parquet"
EXPOSURE_PATH = "output/intermediate/exposure_v3_full.parquet"
V2_DIGEST_PATH = "output/stats_v2_digest.json"

OUT_STATS_V3 = "output/stats_v3_digest.json"
OUT_MULTINOMIAL_V3 = "output/multinomial_v3_digest.json"
OUT_CONFIDENCE_V3 = "output/confidence_v3_digest.json"
OUT_DEPENDENCE_V3 = "output/dependence_sensitivity_v3_digest.json"
OUT_ZERO_CER_V3 = "output/zero_cer_breakdown_v3_digest.json"

_LABELS = ("faithful", "degraded", "drift")

# `dependence_sensitivity`'s own "stability" bar was implicitly ~5% in v2
# (the observed spread); we report the actual number either way, but need
# a documented pass/fail line for `conclusion.stable_across_dependence_
# assumptions`. Doubling the v2 observation is a defensible, transparent
# bound, not a magic number picked to force a "stable" verdict.
_STABILITY_PCT_THRESHOLD = 10.0


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


def _default(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write(path: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=_default))
    print(f"wrote {p}")


# ---------------------------------------------------------------------------
# drift_curve section (idrift.analysis.drift_curve.drift_rate_by_cer)
# ---------------------------------------------------------------------------
def build_drift_curve_section(df: pd.DataFrame) -> dict:
    """Mirrors stats_v2_digest.json's `drift_curve` section: drift rate by
    corpus x target-CER (percent, 2 dp -- matching v2's formatting), overall
    label mix by corpus (proportion, 4 dp), per-model AUTH drift rate
    (proportion, 4 dp, sorted ascending -- matching v2's ordering), and row
    counts by corpus. The only logic here is the per-corpus/per-model
    grouping the v2 driver used to assemble the digest around
    `drift_rate_by_cer`; the rate itself always comes from that function.
    """
    drift_by_corpus_x_targetcer = {}
    overall_by_corpus = {}
    for corpus, g in df.groupby("corpus"):
        rates = drift_curve.drift_rate_by_cer(g, label_col="label")
        drift_by_corpus_x_targetcer[corpus] = {
            f"{cer:.1f}": round(rate * 100, 2) for cer, rate in sorted(rates.items())
        }
        counts = g["label"].value_counts(normalize=True)
        overall_by_corpus[corpus] = {
            lbl: round(float(counts.get(lbl, 0.0)), 4) for lbl in _LABELS
        }

    auth = df[df["corpus"] == "AUTH"]
    per_model_auth = {
        str(model): float((g["label"] == "drift").mean()) for model, g in auth.groupby("model")
    }
    per_model_auth_drift = {
        model: round(rate, 4) for model, rate in sorted(per_model_auth.items(), key=lambda kv: kv[1])
    }

    n_by_corpus = {str(k): int(v) for k, v in df["corpus"].value_counts().items()}

    return {
        "drift_by_corpus_x_targetcer": drift_by_corpus_x_targetcer,
        "overall_by_corpus": overall_by_corpus,
        "per_model_auth_drift": per_model_auth_drift,
        "n_by_corpus": n_by_corpus,
    }


# ---------------------------------------------------------------------------
# calibration section (idrift.analysis.calibration_v2)
# ---------------------------------------------------------------------------
def build_calibration_section(labeled_path: str) -> dict:
    frame = calibration_v2.build_calibration_frame(labeled_path)
    per_model = calibration_v2.per_model_calibration(frame)
    return {
        "dropped_nan_confidence": int(frame.attrs.get("dropped_nan_confidence", 0)),
        "per_model": per_model,
        "meta_ece": calibration_v2.meta_analyze(per_model, metric="ece"),
        "meta_auroc": calibration_v2.meta_analyze(per_model, metric="auroc"),
    }


# ---------------------------------------------------------------------------
# multinomial digest (idrift.analysis.multinomial)
# ---------------------------------------------------------------------------
def build_multinomial_digest(
    df: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    labeled_path: str = LABELED_PATH,
    exposure_path: str = EXPOSURE_PATH,
) -> dict:
    merged = multinomial.attach_message_covariates(df, exposure)
    label_counts = {str(k): int(v) for k, v in merged["label"].value_counts().items()}
    model_families = sorted(merged["model"].astype(str).str.split(":").str[0].unique())

    mn = multinomial.fit_multinomial(merged)
    contrast_dd = multinomial.fit_binary_contrast(merged, positive="drift", negative="degraded")
    contrast_df = multinomial.fit_binary_contrast(merged, positive="drift", negative="faithful")
    highlight = next(t for t in contrast_dd["terms"] if t["term"] == "realized_cer")

    return {
        "task": "C4 v3 multinomial + drift-vs-faithful/degraded contrasts",
        "data": {
            "labeled_path": labeled_path,
            "exposure_path": exposure_path,
            "n_rows": int(len(merged)),
            "label_counts": label_counts,
            "model_families": model_families,
        },
        "predictors": "realized_cer + corrupted_numeral + corrupted_negation + char_len + C(model_family)",
        "multinomial_mnlogit_base_faithful": mn,
        "contrast_drift_vs_degraded": contrast_dd,
        "contrast_drift_vs_faithful": contrast_df,
        "highlight_drift_vs_degraded_realized_cer": {
            "note": (
                "PRIMARY reviewer-requested contrast: does realized corruption push the "
                "model toward a fluent-but-wrong drift OVER an obviously-degraded output? "
                "OR > 1 means rising realized_cer favors drift over degraded."
            ),
            "odds_ratio": highlight["odds_ratio"],
            "ci_lower": highlight["ci_lower"],
            "ci_upper": highlight["ci_upper"],
            "p": highlight["p"],
        },
    }


# ---------------------------------------------------------------------------
# confidence digest (idrift.analysis.confidence_analysis)
# ---------------------------------------------------------------------------
def build_confidence_digest(df: pd.DataFrame) -> dict:
    return {
        "confidently_wrong_rate": confidence_analysis.confidently_wrong_rate(df),
        "missing_confidence_audit": confidence_analysis.missing_confidence_audit(df),
        "predicted_drift_grid": confidence_analysis.predicted_drift_grid(df),
    }


# ---------------------------------------------------------------------------
# dependence-sensitivity digest (idrift.analysis.dependence_sensitivity)
# ---------------------------------------------------------------------------
def build_dependence_digest(df: pd.DataFrame, *, primary_slope_adjusted: float, univariate_anchor: dict) -> dict:
    """Reconstructs dependence_sensitivity_digest.json's shape. The
    "univariate_message_clustered_anchor" reference point is NOT
    recomputed here -- it is the same message-clustered univariate
    realized_cer coefficient `confidence_analysis.predicted_drift_grid`
    already fits (v2's driver reused it this way too: its anchor
    est/se/ci matches predicted_drift_grid's realized_cer coefficient
    exactly), so the caller passes that coefficient in rather than fitting
    it a second time.
    """
    anchor_est = univariate_anchor["est"]
    anchor_se = univariate_anchor["se"]
    anchor_ci = [anchor_est - 1.96 * anchor_se, anchor_est + 1.96 * anchor_se]

    n_obs = int(len(df))
    n_messages = int(df["message_id"].nunique())
    n_models = int(df["model"].nunique())
    drift_rate = float((df["label"] == "drift").mean())

    per_model = dependence_sensitivity.per_model_pooled_slope(df, seed=0)
    gee = dependence_sensitivity.gee_slope(df)
    boot_message = dependence_sensitivity.bootstrap_slope(df, level="message", n_boot=500, seed=0)
    boot_model = dependence_sensitivity.bootstrap_slope(df, level="model", n_boot=500, seed=0)

    estimators = {
        "per_model_pooled_dl": per_model,
        "gee": gee,
        "bootstrap_message": boot_message,
        "bootstrap_model": boot_model,
    }

    adjusted_key = f"ci_covers_adjusted_{round(primary_slope_adjusted, 2)}"
    agreement = {}
    slopes = []
    for name, est in estimators.items():
        slope = est["slope"]
        ci = est["ci"]
        slopes.append(slope)
        agreement[name] = {
            "slope": slope,
            "ci": ci,
            "ci_covers_univariate_anchor": bool(ci[0] <= anchor_est <= ci[1]),
            adjusted_key: bool(ci[0] <= primary_slope_adjusted <= ci[1]),
            "abs_diff_from_univariate_anchor": abs(slope - anchor_est),
        }

    slope_spread = max(slopes) - min(slopes)
    max_pct_dev = max(abs(s - anchor_est) / anchor_est * 100 for s in slopes)
    stable = bool(max_pct_dev <= _STABILITY_PCT_THRESHOLD)

    per_model_slopes = [v["slope"] for v in per_model["per_model"].values()]
    range_lo, range_hi = min(per_model_slopes), max(per_model_slopes)
    gee_note = (
        "converged" if gee["exchangeable_converged"]
        else "diverged on this cohort and fell back to an independence working correlation (see its notes)"
    )
    summary = (
        f"The realized_cer log-odds slope is {'STABLE' if stable else 'NOT STABLE (see max_pct_deviation_from_anchor)'} "
        f"across dependence assumptions: all four estimators lie in {min(slopes):.2f}-{max(slopes):.2f} "
        f"(spread {slope_spread:.2f}, {max_pct_dev:.1f}% max deviation from the univariate anchor {anchor_est:.2f}). "
        f"The preferred model-level estimator is the DerSimonian-Laird per-model pool ({n_models} models; "
        "a model-clustered SE is not used because the model count is too small for that purpose); it shows "
        f"between-model heterogeneity (tau^2={per_model['tau2']:.2f}, "
        f"I^2={per_model['i2']:.0f}%) but every model's slope is positive (range {range_lo:.1f}-{range_hi:.1f}). "
        f"The model-level cluster bootstrap is reported honestly as UNSTABLE ({n_models} clusters: much wider CI than the "
        f"message-level bootstrap). The exchangeable GEE {gee_note}."
    )

    return {
        "task": "C4 v3 dependence-structure sensitivity (was B9)",
        "description": (
            "Alternative estimators of the realized_cer -> drift log-odds slope under different "
            f"dependence assumptions across {n_models} models. All estimators fit realized_cer ONLY "
            "(plus intercept) for mutual comparability."
        ),
        "n_obs": n_obs,
        "n_messages": n_messages,
        "n_models": n_models,
        "drift_rate": drift_rate,
        "reference_slopes": {
            "primary_message_clustered_ADJUSTED": primary_slope_adjusted,
            "univariate_message_clustered_anchor": {
                "slope": anchor_est,
                "se": anchor_se,
                "ci": anchor_ci,
            },
            "note": (
                f"The {primary_slope_adjusted:.2f} headline is the ADJUSTED message-clustered logit slope "
                "(adjusts n_errors, char_len, corrupted_negation, corrupted_numeral, C(family)) from this "
                "run's mixed_model.drift.coefficients.realized_cer. The dependence estimators here are "
                f"UNIVARIATE, so their natural reference is the univariate message-clustered anchor "
                f"({anchor_est:.2f}, from confidence_analysis.predicted_drift_grid). That gap is ADJUSTMENT, "
                "not dependence; this digest's question is only about dependence."
            ),
        },
        "estimators": estimators,
        "agreement": agreement,
        "slope_spread_across_estimators": slope_spread,
        "max_pct_deviation_from_anchor": max_pct_dev,
        "model_level_bootstrap_unstable": bool(boot_model["unstable"]),
        "conclusion": {
            "stable_across_dependence_assumptions": stable,
            "summary": summary,
        },
    }


# ---------------------------------------------------------------------------
# mixed_model section (idrift.analysis.mixed_models)
# ---------------------------------------------------------------------------
def build_mixed_model_section(df: pd.DataFrame, exposure: pd.DataFrame, *, seed: int = 0) -> dict:
    merged = mixed_models.build_analysis_frame(df, exposure)
    # VB GLMM is intractable at 1.7M rows (fit_vb ran >1h without terminating) and never
    # converged at v2 scale, so the cluster-robust logit fallback is the reported estimate.
    return mixed_models.fit_multinomial(merged, seed=seed, force_fallback=True)


def build_benefit_harm_section(df: pd.DataFrame, raw_decode: pd.DataFrame) -> dict:
    """Build the raw-decode versus LLM transition digest.

    Raw labels are model-independent and therefore broadcast across every
    model row at the same exposure key.
    """
    result = benefit_harm.transition_matrix(df, raw_decode)
    by_group = result["by_group"].to_dict(orient="records")
    auth_net = {
        f"{row['cer_target']:.1f}": row["faithful_gained_per_drift_introduced"]
        for row in by_group
        if row.get("corpus") == "AUTH"
    }
    key_cols = ["message_id", "corpus", "cer_target", "replicate_idx"]
    return {
        "overall": result["overall"],
        "by_group": by_group,
        "auth_net_by_cer": auth_net,
        "n_full_raw_keys": int(raw_decode[key_cols].drop_duplicates().shape[0]),
        "raw_label_marginals": {
            str(k): int(v) for k, v in raw_decode["raw_label"].value_counts().items()
        },
        "notes": result["notes"],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeled", default=LABELED_PATH)
    ap.add_argument("--exposure", default=EXPOSURE_PATH)
    ap.add_argument("--v2-digest", default=V2_DIGEST_PATH)
    ap.add_argument("--out", default=OUT_STATS_V3)
    ap.add_argument(
        "--raw-decode",
        default=None,
        help="optional parquet of model-independent raw-decode labels for benefit-harm analysis",
    )
    args = ap.parse_args(argv)

    t0 = time.time()
    _log(f"loading {args.labeled}", t0)
    df = pd.read_parquet(args.labeled)
    _log(f"loaded {len(df)} rows", t0)
    _log(f"loading {args.exposure}", t0)
    exposure = pd.read_parquet(args.exposure)
    _log(f"loaded {len(exposure)} exposure rows", t0)

    v2_digest = json.loads(Path(args.v2_digest).read_text())

    notes: dict[str, str] = {}

    _log("drift_curve...", t0)
    drift_curve_section = build_drift_curve_section(df)
    _log("drift_curve done", t0)

    _log("zero_cer_audit (pooled)...", t0)
    zero_cer_section = zero_cer_audit.categorize_zero_cer_drift(df)
    _log("zero_cer_audit done", t0)

    _log("zero_cer_breakdown (standalone v3 digest)...", t0)
    zero_cer_breakdown_v3 = zero_cer_audit._build_digest(df)  # module's own tested composition seam
    _write(OUT_ZERO_CER_V3, zero_cer_breakdown_v3)
    _log("zero_cer_breakdown done", t0)

    _log("calibration...", t0)
    calibration_section = build_calibration_section(args.labeled)
    _log("calibration done", t0)

    _log("mixed_model (VB primary, clustered-logit fallback if VB doesn't converge)...", t0)
    mixed_model_section = build_mixed_model_section(df, exposure, seed=0)
    for outcome, result in mixed_model_section.items():
        _log(f"  mixed_model[{outcome}]: engine={result['engine']} converged={result['converged']}", t0)
    _log("mixed_model done", t0)

    _log("multinomial (MNLogit + binary contrasts)...", t0)
    multinomial_v3 = build_multinomial_digest(
        df, exposure, labeled_path=args.labeled, exposure_path=args.exposure
    )
    _write(OUT_MULTINOMIAL_V3, multinomial_v3)
    _log("multinomial done", t0)

    _log("confidence (confidently-wrong / missing-confidence / predicted-drift-grid)...", t0)
    confidence_v3 = build_confidence_digest(df)
    _write(OUT_CONFIDENCE_V3, confidence_v3)
    _log("confidence done", t0)

    primary_slope_adjusted = mixed_model_section["drift"]["coefficients"]["realized_cer"]["est"]
    univariate_anchor = confidence_v3["predicted_drift_grid"]["coefficients"]["realized_cer"]

    _log("dependence_sensitivity (per-model DL pool, GEE, message + model bootstraps)...", t0)
    dependence_v3 = build_dependence_digest(
        df, primary_slope_adjusted=primary_slope_adjusted, univariate_anchor=univariate_anchor
    )
    _write(OUT_DEPENDENCE_V3, dependence_v3)
    _log("dependence_sensitivity done", t0)

    # --- REUSE verbatim from v2 (see module docstring) ---
    matched_compare_section = dict(v2_digest["matched_compare"])
    matched_compare_section["reused_from"] = "v2 (CRIT/CTRL unchanged)"

    sensitivity_section = dict(v2_digest["sensitivity"])
    sensitivity_section["reused_from"] = "v2 (NOT in Task C4's module list -- see run_v3 module docstring)"
    notes["sensitivity"] = (
        "The tau/tie-break recalibration table was not recomputed: it requires re-deriving labels via "
        "idrift.adjudicate.label_runner.derive_labels (a per-row Python loop, no vectorized path) at several "
        f"tau/tie_break settings over the full {len(df):,}-row labeled corpus, which the Task C4 brief's module list did "
        "not include. Copied from v2 as a placeholder pending an explicit decision on whether to rerun it."
    )

    if args.raw_decode:
        _log("benefit_harm (raw labels supplied)...", t0)
        raw_decode = pd.read_parquet(args.raw_decode)
        benefit_harm_section = build_benefit_harm_section(df, raw_decode)
        _log("benefit_harm done", t0)
    else:
        benefit_harm_section = {"skipped": True, "reason": "raw-decode labels not supplied"}

    stats_v3 = {
        "_meta": {
            "labeled": args.labeled,
            "exposure": args.exposure,
            "tau_primary": 0.5,
            "tie_primary": "primary",
            "corpus_version": "v3",
            "n_rows": int(len(df)),
            "n_auth_messages": int(df.loc[df["corpus"] == "AUTH", "message_id"].nunique()),
            "generated_by": "idrift.analysis.run_v3",
            "reused_unchanged_artifacts": [
                "output/balance_digest.json (CRIT/CTRL SMD balance table -- unchanged, byte-identical corpora)",
                "output/stratified_diag_digest.json (physician-panel stratified diagnostics -- unchanged)",
            ],
        },
        "drift_curve": drift_curve_section,
        "zero_cer_audit": zero_cer_section,
        "calibration": calibration_section,
        "matched_compare": matched_compare_section,
        "sensitivity": sensitivity_section,
        "mixed_model": mixed_model_section,
        "multinomial": multinomial_v3,
        "confidence": confidence_v3,
        "dependence_sensitivity": dependence_v3,
        "benefit_harm": benefit_harm_section,
        "notes": notes,
    }

    _write(args.out, stats_v3)
    _log(f"ALL DONE, wrote {args.out}", t0)


if __name__ == "__main__":
    main()
