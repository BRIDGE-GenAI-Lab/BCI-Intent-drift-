"""End-to-end analysis: model outputs -> adjudication -> drift/calibration -> Figure 1.

Composes the tested modules (adjudicate.nli_metric, adjudicate.reconcile,
analysis.{drift_curve,calibration,contrasts,digests}, figures.fig1_drift) over
the real model outputs retrieved from O2. NLI adjudication runs locally (CPU is
fine for a few thousand short pairs); it lazily imports sentence-transformers +
transformers, so install those before running the --adjudicate step:
    uv pip install sentence-transformers transformers torch

Usage:
    uv run python -m idrift.analysis.run_pipeline --outputs output/intermediate/outputs_full_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from idrift.adjudicate.nli_metric import label_row, make_model_fns
from idrift.adjudicate.reconcile import assign_final
from idrift.analysis.drift_curve import drift_rate_by_cer, critical_threshold, boot_ci
from idrift.analysis.calibration import ece, auroc_conf_faithful, reliability
from idrift.analysis.contrasts import paired_bootstrap
from idrift.analysis.digests import write_digests
from idrift.figures.fig1_drift import make_fig1
from idrift.lib.io_utils import _dir


def load_outputs(patterns: list[str]) -> pd.DataFrame:
    rows = []
    for pat in patterns:
        for path in glob.glob(pat):
            model_tag = Path(path).stem.replace("outputs_full_", "").replace("outputs_", "")
            bad = 0
            for line in open(path):
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1          # skip lines corrupted by concurrent writes
                    continue
                d.setdefault("model_id", model_tag)
                rows.append(d)
            if bad:
                print(f"{Path(path).name}: skipped {bad} malformed line(s)")
    df = pd.DataFrame(rows)
    # drop error rows (model call failures) from the analysis, count them
    n_err = df["output_text"].astype(str).str.startswith("__ERROR__").sum()
    df = df[~df["output_text"].astype(str).str.startswith("__ERROR__")].reset_index(drop=True)
    if n_err:
        print(f"dropped {n_err} __ERROR__ rows")
    return df


def adjudicate(df: pd.DataFrame) -> pd.DataFrame:
    """NLI+cosine automated labeling -> nli_label; reconcile -> final_label, critical."""
    cos_fn, nli_fn = make_model_fns()
    df = df.copy()
    labels = []
    n = len(df)
    for i, r in enumerate(df.itertuples()):
        labels.append(label_row(r.intended_text, r.output_text, cos_fn, nli_fn))
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"  adjudicated {i + 1}/{n}", flush=True)
    df["nli_label"] = labels
    df["judge_label"] = None
    df["human_label"] = None
    return assign_final(df)


def analyze(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["is_drift"] = (df["final_label"] == "drift").astype(int)
    df["is_faithful"] = (df["final_label"] == "faithful").astype(int)

    curve_by_class = {m: drift_rate_by_cer(g) for m, g in df.groupby("model_id")}
    overall_curve = drift_rate_by_cer(df)
    crit = df[df["category"] == "message_critical"]
    critical_curve = drift_rate_by_cer(crit) if len(crit) else {}

    # calibration (needs verbalized_conf present)
    conf_df = df.dropna(subset=["verbalized_conf"])
    cal = {}
    if len(conf_df):
        cal = {
            "ece": ece(conf_df["verbalized_conf"].values, conf_df["is_faithful"].values),
            "auroc_conf_faithful": auroc_conf_faithful(conf_df["verbalized_conf"].values,
                                                       conf_df["is_faithful"].values),
        }
        rel = reliability(conf_df["verbalized_conf"].values, conf_df["is_faithful"].values)
        cal["reliability"] = rel.to_dict("records")

    # message-critical threshold with a bootstrap CI on the overall drift rate
    def overall_drift(d):
        return float((d["final_label"] == "drift").mean())
    drift_ci = boot_ci(overall_drift, df, n=1000, seed=0)

    return {
        "n": int(len(df)),
        "overall_drift_rate": overall_drift(df),
        "overall_drift_ci": drift_ci,
        "drift_curve_overall": {str(k): v for k, v in overall_curve.items()},
        "curve_by_class": {m: {str(k): v for k, v in c.items()} for m, c in curve_by_class.items()},
        "critical_curve": {str(k): v for k, v in critical_curve.items()},
        "critical_threshold_tol0.01": critical_threshold(df, tol=0.01),
        "calibration": cal,
        "reliability": cal.get("reliability", []),
        "label_counts": df["final_label"].value_counts().to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", nargs="+", required=True, help="JSONL glob(s) of model outputs")
    ap.add_argument("--skip-adjudicate", action="store_true",
                    help="inputs already carry final_label (skip NLI)")
    args = ap.parse_args()

    df = load_outputs(args.outputs)
    print(f"loaded {len(df)} outputs across models: {sorted(df['model_id'].unique())}")

    if not args.skip_adjudicate:
        df = adjudicate(df)
    df.to_parquet(_dir() / "attempts_labeled.parquet", index=False)

    results = analyze(df)
    print(json.dumps({k: v for k, v in results.items()
                      if k in ("n", "overall_drift_rate", "overall_drift_ci",
                               "drift_curve_overall", "critical_threshold_tol0.01",
                               "calibration", "label_counts")}, indent=2, default=str))

    write_digests({"results_digest": results, "stats_digest": results.get("calibration", {})})
    Path("output/figures").mkdir(parents=True, exist_ok=True)
    make_fig1(results, "output/figures/Figure1.pdf")
    print("wrote output/results_digest.json, output/stats_digest.json, output/figures/Figure1.pdf")


if __name__ == "__main__":
    main()
