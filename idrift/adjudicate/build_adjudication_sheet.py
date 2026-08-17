"""Build the blinded third-rater (adjudicator) sheet for the 16-model panel.

The two physician raters (Alon Gorenshtein = rater A, Mahmud Omar =
rater B) each rated all 2,281 blinded items (2,133 stratified + 148
zero-CER). ``analyze_panel16.consensus_labels`` resolves an item only when
both raters gave the SAME 3-class label; every other item is "unresolved"
and needs a third rater to break the tie before the weighted Rogan-Gladen
correction can run.

This script reproduces that exact unresolved set (so the adjudicator rates
precisely the items the analyzer will later look for) and writes it out in
the SAME blinded column layout the two raters saw
(``item_id, intended_text, noisy_text, output_message, sampling_weight,
rating, notes``) with ``rating``/``notes`` blank. It adds no model, corpus,
CER, label, or prior-rater columns: the adjudicator rates independently and
blind, exactly as the first two raters did. The returned sheet drops
straight into ``analyze_panel16 --adjudicator`` with no reshaping.

An analyst-only audit of what each rater actually chose (with the unblinded
model/corpus/CER) is written under ``_KEY_DO_NOT_SHARE/`` -- NOT into the
adjudicator folder -- for later reconciliation only.

Usage:
    ./.venv/bin/python -m idrift.adjudicate.build_adjudication_sheet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from idrift.adjudicate.analyze_panel16 import _load_sheets, _mapped_series, consensus_labels

BASE = Path("output/human_rating_v3plus")

# Rater A = Alon Gorenshtein; Rater B = Mahmud Omar (corrected by the
# corresponding author after an earlier misattribution to Yosef Adiniaev,
# who was not one of the two raters). Same file lists passed to analyze_panel16.
RATER_A = [BASE / "panel_stratified_rated.csv", BASE / "panel_zerocer_rated.csv"]
RATER_B = [BASE / "sheet_rated_modelrater_2.csv", BASE / "zerocer_rated_modelrater_2.csv"]
KEY = BASE / "_KEY_DO_NOT_SHARE" / "key.csv"

OUT_DIR = BASE / "panel_adjudication"
OUT_SHEET = OUT_DIR / "adjudication_sheet.csv"
ANALYST_AUDIT = BASE / "_KEY_DO_NOT_SHARE" / "_adjudication_disagreements_ANALYST_ONLY.csv"

# The blinded display columns the two raters saw, in order; `rating`/`notes`
# are emitted blank for the adjudicator to fill.
DISPLAY_COLS = ["item_id", "intended_text", "noisy_text", "output_message", "sampling_weight"]


def build() -> dict:
    a = _load_sheets(RATER_A)
    b = _load_sheets(RATER_B)

    # Exactly the pipeline's own unresolved set (no adjudicator yet).
    cons = consensus_labels(a, b, None)
    unresolved_ids = cons.loc[cons["resolution"] == "unresolved", "item_id"].tolist()

    # Blinded display content is identical across raters (same instrument);
    # pull it from rater A. Guard that the two raters really did see the
    # same rendered item before we rely on A's copy.
    a_disp = a.set_index("item_id")[DISPLAY_COLS[1:]]
    b_disp = b.set_index("item_id")[DISPLAY_COLS[1:]]
    common = a_disp.index.intersection(b_disp.index)
    mism = (a_disp.loc[common, "output_message"].fillna("") != b_disp.loc[common, "output_message"].fillna(""))
    if mism.any():
        raise ValueError(f"raters saw different output_message for {int(mism.sum())} items; blinding/source mismatch")

    sheet = a.set_index("item_id").loc[unresolved_ids, DISPLAY_COLS[1:]].reset_index()
    sheet = sheet.sort_values("item_id").reset_index(drop=True)
    sheet["rating"] = ""
    sheet["notes"] = ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(OUT_SHEET, index=False)

    # Analyst-only audit (kept OUT of the adjudicator folder): what each
    # rater chose + the unblinded stratum, for later reconciliation.
    am = _mapped_series(a).set_index("item_id")["class3"].rename("rater_a_alon")
    bm = _mapped_series(b).set_index("item_id")["class3"].rename("rater_b_yosef")
    key = pd.read_csv(KEY)[["item_id", "panel", "model", "corpus", "cer_target", "realized_cer", "label"]]
    audit = (
        key[key["item_id"].isin(unresolved_ids)]
        .merge(am, on="item_id")
        .merge(bm, on="item_id")
        .rename(columns={"label": "automated_label"})
        .sort_values("item_id")
    )
    audit.to_csv(ANALYST_AUDIT, index=False)

    disagreement_matrix = (
        audit.groupby(["rater_a_alon", "rater_b_yosef"]).size().rename("n").reset_index()
    )

    return {
        "n_unresolved": len(unresolved_ids),
        "n_stratified": sum(i.startswith("S") for i in unresolved_ids),
        "n_zerocer": sum(i.startswith("Z") for i in unresolved_ids),
        "sheet_path": str(OUT_SHEET),
        "audit_path": str(ANALYST_AUDIT),
        "disagreement_matrix": disagreement_matrix,
    }


if __name__ == "__main__":
    r = build()
    print(f"unresolved (disagreement) items: {r['n_unresolved']} "
          f"(stratified {r['n_stratified']}, zero-CER {r['n_zerocer']})")
    print(f"wrote adjudicator sheet: {r['sheet_path']}")
    print(f"wrote analyst-only audit: {r['audit_path']}")
    print("\nrater A (Alon) x rater B (Mahmud) disagreement breakdown:")
    print(r["disagreement_matrix"].to_string(index=False))
