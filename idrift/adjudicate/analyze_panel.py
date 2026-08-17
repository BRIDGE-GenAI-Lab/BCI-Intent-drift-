"""Turn the completed physician panel into the manuscript's pending numbers.

Run AFTER both physicians (and, for any disagreed items, the third adjudicator)
return their filled sheets. This composes the already-tested adjudication pieces
-- it adds no new statistics of its own:

  * ``reconcile.merge_ratings`` / ``cohen_kappa``  -> inter-rater agreement
  * ``reconcile.validation_frame``                 -> automated-vs-human join
    (the structural guard against dropping disagreement-resolved items)
  * ``validation_metrics.confusion`` / ``class_metrics`` -> class-specific
    agreement (precision, recall, F1, per-class kappa) and the human x auto
    confusion matrix
  * ``misclassification.corrected_prevalence``     -> the Rogan-Gladen
    misclassification-corrected drift estimate, per corpus

The four physician labels map onto the study's 3-class taxonomy plus a
criticality flag:

    Faithful               -> faithful
    Degraded               -> degraded
    Drift                  -> drift   (critical=False)
    Message-critical drift -> drift   (critical=True)

Consensus human label per item: the shared label where the two physicians agree;
the third adjudicator's label where they disagree. Items still lacking a
resolved label (a disagreement with no adjudicator entry) are reported and
excluded from the automated-vs-human comparison rather than guessed.

Outputs a JSON digest (``panel_results.json``) whose fields map one-to-one onto
the manuscript's pending statements.

Usage (agreement only, before adjudication):
    uv run python -m idrift.adjudicate.analyze_panel \\
        --rater-a .../physician_1/rating_sheet.csv \\
        --rater-b .../physician_2/rating_sheet.csv \\
        --key .../_KEY_DO_NOT_SHARE/key.csv \\
        --out .../panel_results.json

Full run (adds the corrected drift estimate):
    ... --adjudicator .../adjudicator/rating_sheet.csv \\
        --cohort output/intermediate/attempts_v2_for_rating.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from idrift.adjudicate.misclassification import corrected_prevalence
from idrift.adjudicate.reconcile import cohen_kappa, load_rater_csv, merge_ratings, validation_frame
from idrift.adjudicate.validation_metrics import CLASS_ORDER, class_metrics, confusion

_HUMAN_MAP = {
    "faithful": ("faithful", False),
    "degraded": ("degraded", False),
    "drift": ("drift", False),
    "message-critical drift": ("drift", True),
}


def map_human_label(raw: object) -> tuple:
    """Map one physician label string to (3-class label, critical flag).

    Case- and whitespace-insensitive. An unrecognized or missing value maps to
    (None, None) so callers can exclude it rather than silently miscount it.
    """
    if not isinstance(raw, str):
        return (None, None)
    return _HUMAN_MAP.get(raw.strip().lower(), (None, None))


def _mapped_series(rater: pd.DataFrame) -> pd.DataFrame:
    """Return item_id -> (class3, critical) for a completed rater sheet."""
    m = rater[["item_id", "rating"]].copy()
    mapped = m["rating"].map(map_human_label)
    m["class3"] = mapped.map(lambda t: t[0])
    m["critical"] = mapped.map(lambda t: t[1])
    return m


def interrater(rater_a: pd.DataFrame, rater_b: pd.DataFrame) -> dict:
    """Inter-rater agreement between the two physicians on the 3-class label.

    Restricted to items both rated with a recognized label. Reports the raw
    percent agreement, overall Cohen's kappa, and a per-class one-vs-rest kappa.
    """
    a = _mapped_series(rater_a).rename(columns={"class3": "a"})
    b = _mapped_series(rater_b).rename(columns={"class3": "b"})
    merged = a[["item_id", "a"]].merge(b[["item_id", "b"]], on="item_id", how="inner")
    both = merged.dropna(subset=["a", "b"])
    n = len(both)
    result = {
        "n_items_both_rated": n,
        "raw_percent_agreement": float((both["a"] == both["b"]).mean()) if n else float("nan"),
        "cohen_kappa_overall": cohen_kappa(both["a"], both["b"]) if n else float("nan"),
        "cohen_kappa_by_class": {},
    }
    for c in CLASS_ORDER:
        if n:
            result["cohen_kappa_by_class"][c] = cohen_kappa(both["a"] == c, both["b"] == c)
        else:
            result["cohen_kappa_by_class"][c] = float("nan")
    return result


def consensus_labels(rater_a: pd.DataFrame, rater_b: pd.DataFrame, adjudicator: pd.DataFrame | None) -> pd.DataFrame:
    """One consensus 3-class human label per item.

    Agreement -> the shared label. Disagreement (or single-rated) -> the
    adjudicator's label if present. Returns columns item_id, human, critical,
    resolution ('agreed' | 'adjudicated' | 'unresolved').
    """
    a = _mapped_series(rater_a).set_index("item_id")
    b = _mapped_series(rater_b).set_index("item_id")
    adj = _mapped_series(adjudicator).set_index("item_id") if adjudicator is not None else None

    all_ids = sorted(set(a.index) | set(b.index))
    rows = []
    for iid in all_ids:
        av = a["class3"].get(iid)
        bv = b["class3"].get(iid)
        acrit = a["critical"].get(iid)
        bcrit = b["critical"].get(iid)
        if av is not None and av == bv:
            rows.append((iid, av, bool(acrit) and bool(bcrit), "agreed"))
        elif adj is not None and adj["class3"].get(iid) is not None:
            rows.append((iid, adj["class3"].get(iid), bool(adj["critical"].get(iid)), "adjudicated"))
        else:
            rows.append((iid, None, None, "unresolved"))
    return pd.DataFrame(rows, columns=["item_id", "human", "critical", "resolution"])


def auto_vs_human(consensus: pd.DataFrame, key: pd.DataFrame) -> dict:
    """Class-specific automated-vs-human agreement + the confusion matrix.

    Joins the resolved consensus labels onto the automated labels from the key
    (via ``validation_frame`` so no resolved item is silently dropped), then
    reports overall/per-class agreement and the human(rows) x auto(cols)
    confusion counts.
    """
    resolved = consensus.dropna(subset=["human"]).copy()
    auto = key.rename(columns={"automated_final_label": "auto_label"})[["item_id", "auto_label"]]
    vf = validation_frame(resolved, auto, on_missing="raise", auto_col="auto_label")
    human, autolab = vf["human"], vf["auto_label"]

    cm = confusion(human, autolab, labels=CLASS_ORDER)
    metrics = class_metrics(human, autolab, labels=CLASS_ORDER)
    return {
        "n_resolved": int(len(resolved)),
        "n_unresolved": int(consensus["resolution"].eq("unresolved").sum()),
        "overall_percent_agreement": float((human.values == autolab.values).mean()),
        "cohen_kappa_overall": cohen_kappa(human, autolab),
        "cohen_kappa_by_class": {c: cohen_kappa(human == c, autolab == c) for c in CLASS_ORDER},
        "class_metrics": metrics.to_dict(orient="index"),
        "confusion_human_rows_auto_cols": cm.to_dict(orient="index"),
        "_confusion_frame": cm,          # kept for the corrected-prevalence step
        "_val_frame": vf[["item_id", "human", "auto_label"]],
    }


def corrected_drift_by_corpus(confusion_frame, key, cohort, val_frame, seed: int = 0) -> dict:
    """Rogan-Gladen misclassification-corrected drift prevalence per corpus.

    ONE labeler error model is used throughout: the confusion matrix is a
    property of the pipeline, estimated once on the FULL validation subset (all
    1,295 resolved items), and both the point estimate and the message-clustered
    bootstrap CI use that same pooled matrix. Only the naive label counts and the
    cohort resample vary by corpus. Estimating a separate confusion matrix per
    corpus was tried and rejected: the per-corpus validation subsets are small
    (CTRL n=70), so their 3x3 inversions were unstable and, because the point
    estimate used the pooled matrix while the CI resampled a per-corpus one, the
    two were inconsistent (the AUTH point estimate fell outside its own CI). A
    shared error model applied to corpus-specific observed drift rates is the
    standard, stable form of the correction; the assumption that the pipeline's
    confusion pattern is corpus-independent is a reported limitation.
    """
    key = key.copy()
    key["corpus"] = key["category"].where(key["category"] != "message_critical", "CRIT")
    val = val_frame.merge(key[["item_id", "corpus"]], on="item_id", how="left")
    val = val.rename(columns={"auto_label": "auto"})
    val_pooled = val[["message_id", "human", "auto"]]

    out = {}
    for corp in ["AUTH", "CRIT", "CTRL", "ALL"]:
        cohort_c = cohort if corp == "ALL" else cohort[cohort["corpus"] == corp]
        auto_counts = cohort_c["auto_label"].value_counts().to_dict()
        if not auto_counts:
            out[corp] = {"note": "insufficient data"}
            continue
        res = corrected_prevalence(
            auto_counts=auto_counts,
            confusion_from_human=confusion_frame,   # pooled point estimate
            n_boot=1000,
            seed=seed,
            target="drift",
            cohort_frame=cohort_c[["message_id", "auto_label"]],
            val_frame=val_pooled,                   # pooled CI: same error model
        )
        res["validation_n_pooled"] = int(len(val))
        out[corp] = res
    return out


def run(rater_a_path, rater_b_path, key_path, out_path,
        adjudicator_path=None, cohort_path=None, seed: int = 0) -> dict:
    """End-to-end panel analysis; writes ``out_path`` and returns the digest."""
    # completed sheets still carry the blinded text columns; slim to the label
    # columns so merge_ratings (rating/notes only) does not collide on them.
    _slim = lambda d: d[[c for c in ["item_id", "rating", "notes"] if c in d.columns]]
    a = _slim(load_rater_csv(rater_a_path))
    b = _slim(load_rater_csv(rater_b_path))
    adj = _slim(load_rater_csv(adjudicator_path)) if adjudicator_path else None
    key = pd.read_csv(key_path)

    merged = merge_ratings(a, b)
    digest = {
        "n_items": int(len(set(a["item_id"]) | set(b["item_id"]))),
        "raw_pair_agreement": float(merged["agree"].mean()),
        "interrater": interrater(a, b),
    }

    cons = consensus_labels(a, b, adj)
    digest["resolution_counts"] = cons["resolution"].value_counts().to_dict()

    avh = auto_vs_human(cons, key)
    confusion_frame = avh.pop("_confusion_frame")
    val_frame = avh.pop("_val_frame")
    digest["automated_vs_human"] = avh

    if cohort_path and digest["resolution_counts"].get("unresolved", 0) == 0:
        # base-message id for clustering: strip the "#rN" replicate suffix
        cohort = pd.read_parquet(cohort_path)
        cohort = cohort.rename(columns={"label": "auto_label"})
        cohort["corpus"] = cohort["category"].where(cohort["category"] != "message_critical", "CRIT")
        cohort["message_id"] = cohort["orig_message_id"] if "orig_message_id" in cohort else cohort["message_id"]
        # val_frame message_id must also be the base message for clustered CI
        vf = val_frame.merge(key[["item_id", "orig_attempt_id"]], on="item_id", how="left")
        vf["message_id"] = vf["orig_attempt_id"].str.split("::").str[0].str.split("#r").str[0]
        val_for_corr = vf[["item_id", "message_id", "human", "auto_label"]]
        digest["corrected_drift_by_corpus"] = corrected_drift_by_corpus(
            confusion_frame, key, cohort, val_for_corr, seed=seed
        )
    else:
        digest["corrected_drift_by_corpus"] = {
            "note": "skipped: provide --cohort and resolve all disagreements (need adjudicator) first"
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(digest, indent=2, default=str))
    return digest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rater-a", required=True)
    ap.add_argument("--rater-b", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--adjudicator", default=None, help="third-rater sheet for disagreed items")
    ap.add_argument("--cohort", default=None,
                    help="full-cohort parquet (attempts_v2_for_rating.parquet) for the corrected estimate")
    ap.add_argument("--out", required=True, help="destination panel_results.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    digest = run(
        args.rater_a, args.rater_b, args.key, args.out,
        adjudicator_path=args.adjudicator, cohort_path=args.cohort, seed=args.seed,
    )
    ir = digest["interrater"]
    print(f"items {digest['n_items']} | inter-rater kappa {ir['cohen_kappa_overall']:.3f} "
          f"(raw {ir['raw_percent_agreement']:.3f})")
    avh = digest["automated_vs_human"]
    print(f"auto-vs-human kappa {avh['cohen_kappa_overall']:.3f} "
          f"(resolved {avh['n_resolved']}, unresolved {avh['n_unresolved']})")
    cd = digest["corrected_drift_by_corpus"]
    if "note" not in cd:
        for corp in ["AUTH", "CRIT", "CTRL", "ALL"]:
            r = cd.get(corp, {})
            if "corrected" in r:
                print(f"  corrected drift [{corp}]: naive {r['naive']:.3f} -> "
                      f"corrected {r['corrected']:.3f} CI {r['ci'][0]:.3f}-{r['ci'][1]:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
