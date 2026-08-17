"""Build the third-adjudicator sheet from two completed physician sheets.

Run AFTER both physicians return their filled ``rating_sheet.csv``. It aligns
the two raters on ``item_id`` (via ``reconcile.merge_ratings``), takes every
item where they disagree (or where only one rater scored the item), and writes a
blinded sheet of just those items for the third physician. The adjudicator rates
those items fresh, blinded, exactly like the first two; their label becomes the
consensus label for the disagreed items (see ``analyze_panel``).

The adjudicator sheet carries only the blinded columns (intended message +
system output). The two prior ratings are deliberately withheld so the third
read is independent and unanchored.

Usage:
    uv run python -m idrift.adjudicate.make_adjudication_subset \\
        --rater-a output/human_rating_v2/physician_1/rating_sheet.csv \\
        --rater-b output/human_rating_v2/physician_2/rating_sheet.csv \\
        --blinded output/human_rating_v2/_source/rating_sheet_raterA.csv \\
        --out output/human_rating_v2/adjudicator/rating_sheet.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from idrift.adjudicate.build_rating_sheet import BLINDED_COLUMNS
from idrift.adjudicate.reconcile import load_rater_csv, merge_ratings


def adjudication_items(rater_a: pd.DataFrame, rater_b: pd.DataFrame) -> list:
    """Return the sorted item_ids the two raters did not agree on.

    Includes items only one rater scored (an incomplete pair cannot establish
    agreement), matching ``merge_ratings``' ``agree=False`` convention.
    """
    merged = merge_ratings(rater_a, rater_b)
    return sorted(merged.loc[~merged["agree"], "item_id"].tolist())


def build_subset(rater_a_path, rater_b_path, blinded_path, out_path) -> dict:
    """Write the blinded adjudicator sheet for the disagreed items.

    Returns a small summary dict (counts) for the console.
    """
    a = load_rater_csv(rater_a_path)
    b = load_rater_csv(rater_b_path)
    # a completed sheet still carries the blinded text columns
    # (intended_message/system_output); merge_ratings only handles rating/notes
    # and would collide on the rest, so slim to what it needs.
    keep = lambda d: d[[c for c in ["item_id", "rating", "notes"] if c in d.columns]]
    ids = adjudication_items(keep(a), keep(b))

    blinded = pd.read_csv(blinded_path)
    subset = blinded[blinded["item_id"].isin(ids)].copy()
    # blank the rating/notes so the adjudicator starts clean, keep column order
    subset["rating"] = ""
    subset["notes"] = ""
    subset = subset[BLINDED_COLUMNS]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out, index=False)

    n_pairs = len(set(a["item_id"]) | set(b["item_id"]))
    return {
        "items_total": n_pairs,
        "disagreements": len(ids),
        "agreement_rate": (n_pairs - len(ids)) / n_pairs if n_pairs else float("nan"),
        "adjudicator_sheet": str(out),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rater-a", required=True)
    ap.add_argument("--rater-b", required=True)
    ap.add_argument(
        "--blinded", required=True,
        help="an original blinded sheet (item_id/intended_message/system_output) to pull text from",
    )
    ap.add_argument("--out", required=True, help="destination adjudicator rating_sheet.csv")
    args = ap.parse_args(argv)

    summary = build_subset(args.rater_a, args.rater_b, args.blinded, args.out)
    print(
        f"items {summary['items_total']} | disagreements {summary['disagreements']} "
        f"| raw agreement {summary['agreement_rate']:.3f}"
    )
    print(f"wrote {summary['adjudicator_sheet']}")


if __name__ == "__main__":
    main()
