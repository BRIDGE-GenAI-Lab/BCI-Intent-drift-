"""Blinded CRIT-vs-CTRL manual corroboration sheet builder (Task B2).

Reviewers asked whether the automated critical-message finding (the
matched-pair CRIT-vs-CTRL contrast in `idrift.analysis.matched_compare`, OR
~1.16) holds up independently of the rule-based drift detector. This module
builds the physician-facing instrument for that corroboration: a matched
CRIT/CTRL subset spanning the CER grid, drawn from the Task-1.3 pairing
(`output/intermediate/ctrl_matched.csv`, `idrift.data.matched_controls`),
where physicians label each item faithful/degraded/drift BLIND TO CORPUS --
the sheet never reveals whether an item came from the CRIT half or the CTRL
half of its matched pair. The hidden key retains the pairing (`pair_id` +
`corpus`) so the manual matched contrast can be reconstructed later,
independent of the automated label entirely.

Design: for every Task-1.3 matched pair and every CER grid level, one
representative attempt row is sampled from each corpus half (deterministic
given `seed`), giving a "pair candidate" table of (pair_id, cer_target)
combinations. A stratified draw across `cer_target` (reusing
`idrift.adjudicate.human_export.stratified_sample`, the same quota sampler
`build_rating_sheet.py` and `build_zerocer_sheet.py` use) selects
approximately `n_pairs` of these combinations so the sheet spans the full
CER grid, then each selected pair is expanded into its two physician-facing
items (one CRIT-half, one CTRL-half), shuffled together so the order
carries no corpus information.

Usage:
    uv run python -m idrift.adjudicate.build_critctrl_sheet \\
        --attempts output/intermediate/attempts_v3_labeled.parquet \\
        --ctrl-matched output/intermediate/ctrl_matched.csv \\
        --n-pairs 150 --seed 0 --outdir output/human_rating_v3/critctrl

Step 1 (this module) only BUILDS the blinded sheet + key + README. No
ratings are collected or fabricated here -- that is a later human step (see
`scratchpad/harden/briefs/task-B2-brief.md` Step 2-3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from idrift.adjudicate.human_export import stratified_sample

# The brief's exact three physician-facing labels.
RATING_LABELS = ("faithful", "degraded", "drift")

# The rater-facing sheet: ONLY item_id + the two text fields + blank
# rating/notes. No model/corpus/CER/pair_id/automated-label column of any
# kind -- in particular, no column (and, per the belt-and-braces test, no
# cell value) ever reveals whether an item is CRIT or CTRL.
BLINDED_COLUMNS = ["item_id", "intended_text", "output_message", "rating", "notes"]

_ITEM_ID_PREFIX = "M"

_REQUIRED_ATTEMPT_COLUMNS = (
    "message_id", "model", "corpus", "cer_target", "label", "intended_text", "output_message",
)
_REQUIRED_CTRL_MATCHED_COLUMNS = ("crit_message_id", "message_id")


def _pair_mapping(ctrl_matched: pd.DataFrame) -> pd.DataFrame:
    """Expand the Task-1.3 `ctrl_matched` frame (one row per CRIT item, with
    `crit_message_id` and the matched control's own `message_id`) into a
    long `message_id` -> (`pair_id`, `role`) mapping: one row for the CRIT
    half (`role="CRIT"`) and one for the CTRL half (`role="CTRL"`) of every
    match, both sharing `pair_id = crit_message_id` -- the same convention
    `idrift.analysis.matched_compare._pair_mapping` uses for the automated
    contrast, so this instrument's pairing is identical to the one the
    manuscript already reports on.
    """
    missing = [c for c in _REQUIRED_CTRL_MATCHED_COLUMNS if c not in ctrl_matched.columns]
    if missing:
        raise ValueError(f"_pair_mapping: ctrl_matched is missing required column(s): {missing}")

    crit_half = pd.DataFrame({
        "message_id": ctrl_matched["crit_message_id"],
        "role": "CRIT",
        "pair_id": ctrl_matched["crit_message_id"],
    })
    ctrl_half = pd.DataFrame({
        "message_id": ctrl_matched["message_id"],
        "role": "CTRL",
        "pair_id": ctrl_matched["crit_message_id"],
    })
    return pd.concat([crit_half, ctrl_half], ignore_index=True)


def build_pair_candidates(attempts: pd.DataFrame, ctrl_matched: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Build the "pair candidate" table: one row per (pair_id, cer_target)
    combination, carrying one deterministically-sampled representative
    attempt row from EACH corpus half (CRIT and CTRL).

    Args:
        attempts: labeled attempts DataFrame with every column in
            `_REQUIRED_ATTEMPT_COLUMNS`, including a `corpus` column with
            values in {"AUTH", "CRIT", "CTRL"}.
        ctrl_matched: the Task-1.3 CRIT-CTRL matched-pair table
            (`crit_message_id`, `message_id`, ... -- see
            `idrift.data.matched_controls`).
        seed: seed for the per-(pair_id, cer_target, role) representative
            draw (one row sampled out of however many models/replicates
            share that cell).

    Returns:
        DataFrame with columns `pair_id`, `cer_target`, and the CRIT/CTRL
        halves' fields prefixed `crit_`/`ctrl_`
        (`{crit,ctrl}_message_id/model/intended_text/output_message/label`).
        Exactly one row per (pair_id, cer_target) combination for which
        BOTH halves have at least one qualifying attempt row.

    Raises:
        ValueError: if `attempts` or `ctrl_matched` is missing a required
            column.
    """
    missing = [c for c in _REQUIRED_ATTEMPT_COLUMNS if c not in attempts.columns]
    if missing:
        raise ValueError(f"build_pair_candidates: attempts is missing required column(s): {missing}")

    mapping = _pair_mapping(ctrl_matched)
    restricted = attempts.merge(mapping, on="message_id", how="inner", validate="m:1")

    # One deterministic representative row per (pair_id, cer_target, role)
    # cell -- collapses across whichever models/replicates populate that
    # cell in the real data.
    sampled = (
        restricted.groupby(["pair_id", "cer_target", "role"], as_index=False, group_keys=False)
        .sample(n=1, random_state=seed)
    )

    crit_half = sampled[sampled["role"] == "CRIT"].rename(columns={
        "message_id": "crit_message_id",
        "model": "crit_model",
        "intended_text": "crit_intended_text",
        "output_message": "crit_output_message",
        "label": "crit_label",
    })[["pair_id", "cer_target", "crit_message_id", "crit_model", "crit_intended_text", "crit_output_message", "crit_label"]]

    ctrl_half = sampled[sampled["role"] == "CTRL"].rename(columns={
        "message_id": "ctrl_message_id",
        "model": "ctrl_model",
        "intended_text": "ctrl_intended_text",
        "output_message": "ctrl_output_message",
        "label": "ctrl_label",
    })[["pair_id", "cer_target", "ctrl_message_id", "ctrl_model", "ctrl_intended_text", "ctrl_output_message", "ctrl_label"]]

    combo = crit_half.merge(ctrl_half, on=["pair_id", "cer_target"], how="inner", validate="1:1")
    return combo.reset_index(drop=True)


def draw_stratified_pairs(combo: pd.DataFrame, n_pairs: int, seed: int) -> pd.DataFrame:
    """Stratified draw of `combo` rows (whole pair candidates) across
    `cer_target`, so the physician sample spans the full CER grid. An equal
    quota per CER level (`n_pairs // n_levels`, capped at each level's own
    availability) is used -- see `human_export.stratified_sample`.

    `combo` is shuffled (seeded) BEFORE stratifying. `build_pair_candidates`
    returns `combo` sorted by `pair_id` then `cer_target` (an artifact of the
    groupby it is built from), so every `cer_target` slice would otherwise
    contain the SAME pair_ids in the SAME relative row order. `human_export.
    stratified_sample` samples every stratum with the identical `random_
    state=seed`, and pandas' positional `.sample()` returns the same
    positions for equal-content order + equal seed -- so without this
    shuffle, every CER level's draw would select the exact same handful of
    pair_ids (verified against the real 131-pair `ctrl_matched.csv`: an
    unshuffled draw of 30-per-level across the full 5-level grid collapsed
    to just 30 distinct pairs total instead of spanning the pool), which
    defeats the entire point of "spanning the CER grid" with a matched
    subset. Shuffling once, globally, before the per-stratum sample breaks
    that cross-stratum alignment.
    """
    shuffled = combo.sample(frac=1, random_state=seed).reset_index(drop=True)
    return stratified_sample(shuffled, n=n_pairs, by="cer_target", seed=seed)


def expand_to_items(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    """Expand each selected pair candidate into its two physician-facing
    items (one CRIT-half, one CTRL-half), long-form, one row per item.

    Returns:
        DataFrame with columns `pair_id`, `cer_target`, `corpus` ("CRIT" or
        "CTRL"), `message_id`, `model`, `intended_text`, `output_message`,
        `automated_label`.
    """
    crit_items = pd.DataFrame({
        "pair_id": selected_pairs["pair_id"],
        "cer_target": selected_pairs["cer_target"],
        "corpus": "CRIT",
        "message_id": selected_pairs["crit_message_id"],
        "model": selected_pairs["crit_model"],
        "intended_text": selected_pairs["crit_intended_text"],
        "output_message": selected_pairs["crit_output_message"],
        "automated_label": selected_pairs["crit_label"],
    })
    ctrl_items = pd.DataFrame({
        "pair_id": selected_pairs["pair_id"],
        "cer_target": selected_pairs["cer_target"],
        "corpus": "CTRL",
        "message_id": selected_pairs["ctrl_message_id"],
        "model": selected_pairs["ctrl_model"],
        "intended_text": selected_pairs["ctrl_intended_text"],
        "output_message": selected_pairs["ctrl_output_message"],
        "automated_label": selected_pairs["ctrl_label"],
    })
    items = pd.concat([crit_items, ctrl_items], ignore_index=True)
    items["_row_key"] = (
        items["pair_id"].astype(str) + "::" + items["cer_target"].astype(str) + "::" + items["corpus"]
    )
    return items


def assign_item_ids(items: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign stable, shuffled blinded item ids ("M0001", ...).

    Sorts on the natural `_row_key` first (a canonical order independent of
    the incoming row order), then shuffles that canonical frame
    deterministically with `seed` -- crucially mixing CRIT and CTRL items
    from the same pair (and across different pairs) together, so adjacent
    rows in the sheet carry no corpus information. Same data + same seed ->
    same item_id assignment, every time.
    """
    canonical = items.sort_values("_row_key").reset_index(drop=True)
    shuffled = canonical.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled = shuffled.copy()
    shuffled["item_id"] = [f"{_ITEM_ID_PREFIX}{i + 1:04d}" for i in range(len(shuffled))]
    return shuffled


def make_blinded_sheet(items: pd.DataFrame) -> pd.DataFrame:
    """Build the rater-facing blinded sheet: item_id + the two text fields +
    blank rating/notes. No model/corpus/CER/pair_id/automated-label column
    leaks through."""
    sheet = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "intended_text": items["intended_text"],
            "output_message": items["output_message"],
            "rating": "",
            "notes": "",
        }
    )
    return sheet[BLINDED_COLUMNS]


def make_key(items: pd.DataFrame) -> pd.DataFrame:
    """Build the unblinding key: item_id -> pair_id, corpus (CRIT/CTRL),
    message_id, model, cer_target, and the automated label -- everything
    needed to reconstruct the manual matched CRIT-vs-CTRL contrast and to
    check agreement with the automated pipeline, without ever exposing any
    of it to the physician."""
    return pd.DataFrame(
        {
            "item_id": items["item_id"],
            "orig_row_key": items["_row_key"],
            "pair_id": items["pair_id"],
            "corpus": items["corpus"],
            "message_id": items["message_id"],
            "model": items["model"],
            "cer_target": items["cer_target"],
            "automated_label": items["automated_label"],
        }
    )


def _write_readme(outdir: Path, key: pd.DataFrame) -> Path:
    n_items = len(key)
    n_pairs = key["pair_id"].nunique()
    by_cer = key["cer_target"].value_counts().sort_index().to_dict()
    cer_lines = "\n".join(f"  - CER {cer}: {n} items" for cer, n in by_cer.items())

    readme = f"""# CRIT-vs-CTRL manual corroboration (Task B2)

This is the blinded physician instrument that independently corroborates
the automated critical-message finding: the matched-pair CRIT-vs-CTRL
drift contrast in `idrift.analysis.matched_compare` (OR ~1.16), which
relies on the automated rule-based/ensemble drift label. Here, physicians
label each item directly, BLIND TO CORPUS, so the manual contrast can be
computed independent of the automated detector entirely (corroborating, or
tempering, the automated OR).

## What is in here

```
README.md                     <- this file
sheet.csv                     <- blinded physician-facing rating sheet
_KEY_DO_NOT_SHARE/key.csv     <- unblinding key. ANALYST ONLY. Never share.
```

## The sample

- {n_items} items ({n_pairs} matched CRIT/CTRL pairs, one CRIT-half item and
  one CTRL-half item per pair), drawn from the Task-1.3 matched-pair table
  (`output/intermediate/ctrl_matched.csv`), stratified across the CER grid
  so every corruption level is represented:
{cer_lines}
- Items are SHUFFLED (seeded) before item_id assignment: the CRIT-half and
  CTRL-half items of a pair are mixed together with every other item, so
  neither adjacency nor item_id order carries any corpus information.

## What physicians see

Each row in `sheet.csv` shows only:

- `intended_text` -- what the person was trying to communicate.
- `output_message` -- what the system actually produced.

No model identity, corpus (CRIT/CTRL), pair membership, decoder-corruption
level (CER), or automated label is shown -- **you cannot and should not try
to guess which corpus an item is from.** Judge each item only from the two
lines you are given. Physicians fill in exactly one `rating` value per row,
chosen from:

- `faithful` -- the output preserves the intended message's meaning.
- `degraded` -- the output is garbled, incomplete, or nonsensical, so a
  reader could not reliably recover the intended meaning.
- `drift` -- the output is fluent and readable, but asserts a different
  intent than the person intended.

The `notes` column is optional, for any case worth flagging.

## How to run the corroboration analysis once ratings come back

From the study root (`/Volumes/Extreme SSD/Mimic-IV/study_bci_llm_intent_drift`)
with the project virtualenv:

```
.venv/bin/python -m scratchpad.harden.b2_critctrl_manual \\
  --sheet output/human_rating_v3/critctrl/sheet.csv \\
  --key   output/human_rating_v3/critctrl/_KEY_DO_NOT_SHARE/key.csv \\
  --out   output/harden_critctrl_manual.json
```

(That analysis script is a later step -- see
`scratchpad/harden/briefs/task-B2-brief.md` Step 3 -- and is not part of
this build.)

## Provenance

- Rating instrument: `idrift/adjudicate/build_critctrl_sheet.py`.
- Source data: `output/intermediate/attempts_v3_labeled.parquet` +
  `output/intermediate/ctrl_matched.csv` (Task 1.3 matched-pair table,
  `idrift.data.matched_controls`).
- Pairing convention reused (for stratification and the hidden key only,
  identical to the automated contrast's own pairing): a long
  `message_id -> (pair_id, role)` mapping, `pair_id = crit_message_id`, the
  same idiom as `idrift.analysis.matched_compare._pair_mapping`.
"""
    path = outdir / "README.md"
    path.write_text(readme)
    return path


def print_summary(key: pd.DataFrame) -> None:
    print(f"Total items: {len(key)} ({key['pair_id'].nunique()} pairs)")
    print("Per CER level:")
    for cer, n in key["cer_target"].value_counts().sort_index().to_dict().items():
        print(f"  CER {cer}: {n}")


def build_critctrl_sheet(attempts_path, ctrl_matched_path, n_pairs: int, seed: int, outdir) -> dict:
    """End-to-end: load attempts + ctrl_matched -> build pair candidates ->
    stratified draw across the CER grid -> expand to items -> blind -> write
    `sheet.csv` / `_KEY_DO_NOT_SHARE/key.csv` / `README.md` -> print summary.

    Args:
        attempts_path: path to a labeled attempts parquet, or an
            already-loaded DataFrame.
        ctrl_matched_path: path to the Task-1.3 `ctrl_matched.csv`, or an
            already-loaded DataFrame.
        n_pairs: target number of matched pairs (upper target; the
            realized total may fall modestly short of `n_pairs` because
            every CER level is guaranteed representation via an equal
            per-level quota -- see `draw_stratified_pairs`). Each pair
            contributes 2 items to the sheet.
        seed: random seed (representative-row sampling + pair draw +
            shuffle), for determinism.
        outdir: output directory; created if missing.

    Returns:
        dict: paths of the written artifacts, keyed by name.
    """
    attempts = attempts_path if isinstance(attempts_path, pd.DataFrame) else pd.read_parquet(attempts_path)
    ctrl_matched = (
        ctrl_matched_path if isinstance(ctrl_matched_path, pd.DataFrame) else pd.read_csv(ctrl_matched_path)
    )

    combo = build_pair_candidates(attempts, ctrl_matched, seed=seed)
    selected_pairs = draw_stratified_pairs(combo, n_pairs=n_pairs, seed=seed)
    items = expand_to_items(selected_pairs)
    items = assign_item_ids(items, seed=seed)

    sheet = make_blinded_sheet(items)
    key = make_key(items)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    key_dir = out / "_KEY_DO_NOT_SHARE"
    key_dir.mkdir(parents=True, exist_ok=True)

    path_sheet = out / "sheet.csv"
    path_key = key_dir / "key.csv"

    sheet.to_csv(path_sheet, index=False)
    key.to_csv(path_key, index=False)
    path_readme = _write_readme(out, key)

    print_summary(key)

    return {"sheet": path_sheet, "key": path_key, "readme": path_readme}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Build a blinded CRIT-vs-CTRL manual corroboration sheet (Task B2)."
    )
    ap.add_argument(
        "--attempts", default="output/intermediate/attempts_v3_labeled.parquet",
        help="path to the labeled attempts parquet",
    )
    ap.add_argument(
        "--ctrl-matched", default="output/intermediate/ctrl_matched.csv",
        help="path to the Task-1.3 CRIT-CTRL matched-pair table",
    )
    ap.add_argument("--n-pairs", type=int, default=150, help="target number of matched pairs")
    ap.add_argument("--seed", type=int, default=0, help="seed for sampling and blinded shuffling")
    ap.add_argument("--outdir", default="output/human_rating_v3/critctrl", help="output directory")
    args = ap.parse_args(argv)

    build_critctrl_sheet(
        args.attempts, args.ctrl_matched, n_pairs=args.n_pairs, seed=args.seed, outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
