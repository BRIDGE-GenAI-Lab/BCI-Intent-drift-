"""Blinded zero-CER drift cause-adjudication sheet builder (Task B1).

Answers the reviewer's Figure 2b / zero-corruption critique: the automated
`idrift.analysis.zero_cer_audit` module bins every `realized_cer==0 &
label=="drift"` row into a rule-based cause taxonomy (critical_substitution
> formatting > paraphrase > overgenerative > other), but that taxonomy is
fully automated string/threshold logic, never a human judgment call. This
module builds the physician-facing instrument that lets clinicians assign
their own cause independently, BLIND to the automated call, so the
manuscript can report agreement between the rule-based categorization and
clinician adjudication (or demote Figure 2b to a supplement eFigure with
example cases if agreement is poor, per the reviewer).

Reuses `idrift.analysis.zero_cer_audit._classify_cause` -- the exact
automated rule-based taxonomy -- purely to STRATIFY the draw across all five
causes and to populate the hidden key. The physician-facing sheet never
carries the automated cause, the corpus, the model identity, the CER, or
the automated drift label.

Usage:
    uv run python -m idrift.adjudicate.build_zerocer_sheet \\
        --attempts output/intermediate/attempts_v3_labeled.parquet \\
        --n 200 --seed 0 --outdir output/human_rating_v3/zerocer

Step 1 (this module) only BUILDS the blinded sheet + key + README. No
ratings are collected or fabricated here -- that is a later human step (see
`scratchpad/harden/briefs/task-B1-brief.md` Step 2-3).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from idrift.adjudicate.human_export import stratified_sample
from idrift.analysis.zero_cer_audit import CAUSES, REQUIRED_COLUMNS, _classify_cause

# Physician-facing cause options (brief's exact wording), one-to-one with
# the automated taxonomy's five internal cause names (CAUSES, imported
# above): "other" is relabeled "unresolved" for the clinician-facing sheet,
# matching the "unresolved_or_mixed" honesty relabel zero_cer_audit itself
# applies for reporting (see that module's `_rename_residual`).
CAUSE_OPTIONS = ("critical-substitution", "formatting", "overgenerative", "paraphrase", "unresolved")

_CAUSE_TO_PHYSICIAN_OPTION = {
    "critical_substitution": "critical-substitution",
    "formatting": "formatting",
    "paraphrase": "paraphrase",
    "overgenerative": "overgenerative",
    "other": "unresolved",
}

# The rater-facing sheet: ONLY the three text fields plus a blank cause/notes
# pair. No model/corpus/CER/automated-label column of any kind.
BLINDED_COLUMNS = ["item_id", "intended_text", "noisy_text", "output_message", "cause", "notes"]

_ITEM_ID_PREFIX = "Z"


def select_zero_cer_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to `realized_cer==0 & label=="drift"` and attach a stable
    `_row_key` (unique per row, independent of any content duplication) plus
    the automated `_cause` (used only for stratification and the hidden
    key -- never exposed to physicians).

    Args:
        df: Labeled attempts DataFrame with every column in
            `idrift.analysis.zero_cer_audit.REQUIRED_COLUMNS` plus
            `message_id`, `model`, `corpus`, `cer_target`.

    Returns:
        DataFrame: the qualifying subset, with `_row_key` and `_cause` added.

    Raises:
        ValueError: if a required column is missing.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"select_zero_cer_drift: df is missing required column(s): {missing}")

    # A synthetic, guaranteed-unique row key: the real attempts_v3_labeled
    # parquet has genuine duplicate rows on every business-key combination
    # tried (message_id/model/cer_target/replicate_idx/prompt_id/temperature/
    # noisy_text/output_message all identical for a handful of rows -- see
    # the Task B1 investigation), so a positional key on the reset index is
    # the only robust choice here.
    work = df.reset_index(drop=True).copy()
    work["_row_key"] = "row" + work.index.astype(str)

    subset = work[(work["realized_cer"] == 0) & (work["label"] == "drift")].copy()
    subset["_cause"] = subset.apply(lambda r: _classify_cause(r.to_dict()), axis=1)
    return subset


def draw_stratified_sample(subset: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified draw across the five automated cause categories (`_cause`)
    so all are represented in the physician sample, up to `n` rows total.

    An equal quota per cause (`n // 5`, capped at each cause's own
    population) is used -- see `human_export.stratified_sample`. When a
    cause's population is smaller than the quota (e.g. `paraphrase` is
    always the rarest zero-CER-drift cause), the realized total falls
    modestly below `n`; this is a deliberate consequence of guaranteeing
    every cause is represented, not a bug.
    """
    return stratified_sample(subset, n=n, by="_cause", seed=seed)


def assign_item_ids(selected: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign stable, shuffled blinded item ids ("Z0001", ...).

    Sorts on the natural `_row_key` first (a canonical order independent of
    the incoming row/selection order), then shuffles that canonical frame
    deterministically with `seed`. Same data + same seed -> same item_id
    assignment, every time.
    """
    canonical = selected.sort_values("_row_key").reset_index(drop=True)
    shuffled = canonical.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled = shuffled.copy()
    shuffled["item_id"] = [f"{_ITEM_ID_PREFIX}{i + 1:04d}" for i in range(len(shuffled))]
    return shuffled


def make_blinded_sheet(items: pd.DataFrame) -> pd.DataFrame:
    """Build the rater-facing blinded sheet: item_id + the three text
    fields + blank cause/notes. No model/corpus/CER/automated-cause/
    automated-label column leaks through."""
    sheet = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "intended_text": items["intended_text"],
            "noisy_text": items["noisy_text"],
            "output_message": items["output_message"],
            "cause": "",
            "notes": "",
        }
    )
    return sheet[BLINDED_COLUMNS]


def make_key(items: pd.DataFrame) -> pd.DataFrame:
    """Build the unblinding key: item_id -> message_id, model, corpus,
    cer_target, and the automated rule-based cause (raw internal name, e.g.
    "other", not the physician-facing "unresolved" relabel -- the analyst
    doing the agreement comparison needs the exact taxonomy name)."""
    key = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "orig_row_key": items["_row_key"],
            "message_id": items["message_id"],
            "model": items["model"],
            "corpus": items["corpus"],
            "cer_target": items["cer_target"],
            "automated_cause": items["_cause"],
        }
    )
    return key


def _write_readme(outdir: Path, key: pd.DataFrame) -> Path:
    n_total = len(key)
    by_cause = key["automated_cause"].value_counts().to_dict()
    cause_lines = "\n".join(
        f"  - `{cause}` (shown to physicians as `{_CAUSE_TO_PHYSICIAN_OPTION[cause]}`): {by_cause.get(cause, 0)}"
        for cause in CAUSES
    )
    readme = f"""# Zero-CER drift cause adjudication (Task B1)

This is the blinded physician instrument answering the reviewer's Figure 2b
critique: why does the automated ensemble call "drift" on a message whose
realized, decoder-level corruption was zero (`realized_cer == 0`)? The
existing `idrift.analysis.zero_cer_audit` module answers this with a fully
automated, rule-based taxonomy. This instrument lets a physician assign a
cause independently, blind to that automated call, so the manuscript can
report agreement (or, if agreement is poor, demote the automated Figure 2b
to a supplement eFigure with example cases, per the reviewer).

## What is in here

```
README.md                     <- this file
sheet.csv                     <- blinded physician-facing rating sheet
_KEY_DO_NOT_SHARE/key.csv     <- unblinding key. ANALYST ONLY. Never share.
```

## The sample

- {n_total} items drawn from the {len(key)} rows this run sampled from the
  full `realized_cer==0 & label=="drift"` pool (4,889 in the real
  `attempts_v3_labeled.parquet` run), stratified across the five automated
  cause categories so every category is represented in the physician sample:
{cause_lines}
- Items are SHUFFLED (seeded) before item_id assignment, so item order
  carries no information about the automated cause.

## What physicians see

Each row in `sheet.csv` shows only:

- `intended_text` -- what the person was trying to communicate.
- `noisy_text` -- the (possibly corrupted) decoder input the model received.
- `output_message` -- what the system actually produced.

No model identity, corpus, decoder-corruption level (CER), or automated
label/cause is shown. Physicians fill in exactly one `cause` value per row,
chosen from:

- `critical-substitution` -- the output changes a safety-relevant detail
  (a reversed yes/no, a wrong number/dose/name/recipient, a changed
  urgency, or a dropped actionable instruction) even though the visible
  input was clean.
- `formatting` -- the output differs from the intended message only
  cosmetically (punctuation, capitalization, whitespace), not in meaning.
- `overgenerative` -- the output adds substantial unrequested content
  beyond the intended message, without necessarily changing its core
  meaning.
- `paraphrase` -- the output reworks the wording but preserves the
  intended meaning; a fluent, semantically equivalent restatement.
- `unresolved` -- none of the above cleanly applies, or the cause is
  ambiguous/mixed.

The `notes` column is optional, for any case worth flagging.

## How to run the agreement analysis once ratings come back

From the study root (`/Volumes/Extreme SSD/Mimic-IV/study_bci_llm_intent_drift`)
with the project virtualenv:

```
.venv/bin/python -m scratchpad.harden.b1_zerocer_adjudicated \\
  --sheet output/human_rating_v3/zerocer/sheet.csv \\
  --key   output/human_rating_v3/zerocer/_KEY_DO_NOT_SHARE/key.csv \\
  --out   output/harden_zerocer_adjudicated.json
```

(That analysis script is a later step -- see
`scratchpad/harden/briefs/task-B1-brief.md` Step 3 -- and is not part of
this build.)

## Provenance

- Rating instrument: `idrift/adjudicate/build_zerocer_sheet.py`.
- Source data: `output/intermediate/attempts_v3_labeled.parquet`.
- Automated cause taxonomy reused (for stratification and the hidden key
  only): `idrift.analysis.zero_cer_audit._classify_cause` (precedence:
  critical_substitution > formatting > paraphrase > overgenerative > other).
"""
    path = outdir / "README.md"
    path.write_text(readme)
    return path


def print_summary(key: pd.DataFrame) -> None:
    print(f"Total items: {len(key)}")
    print("Per automated cause (hidden from physicians):")
    for cause, n in key["automated_cause"].value_counts().to_dict().items():
        print(f"  {cause}: {n}")


def build_zerocer_sheet(attempts_path, n: int, seed: int, outdir) -> dict:
    """End-to-end: load parquet -> filter to realized_cer==0 & label=="drift"
    -> stratified draw across the five automated causes -> blind -> write
    `sheet.csv` / `_KEY_DO_NOT_SHARE/key.csv` / `README.md` -> print summary.

    Args:
        attempts_path: path to a labeled attempts parquet, or an
            already-loaded DataFrame (accepted directly so tests can build
            one in memory or via a temp parquet round trip).
        n: target total sample size (upper target; the realized total may
            fall modestly short because every cause is guaranteed
            representation via an equal per-cause quota -- see
            `draw_stratified_sample`).
        seed: random seed (sampling + shuffle), for determinism.
        outdir: output directory; created if missing.

    Returns:
        dict: paths of the written artifacts, keyed by name.
    """
    df = attempts_path if isinstance(attempts_path, pd.DataFrame) else pd.read_parquet(attempts_path)

    subset = select_zero_cer_drift(df)
    sampled = draw_stratified_sample(subset, n=n, seed=seed)
    items = assign_item_ids(sampled, seed=seed)

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
        description="Build a blinded zero-CER drift cause-adjudication sheet (Task B1)."
    )
    ap.add_argument(
        "--attempts", default="output/intermediate/attempts_v3_labeled.parquet",
        help="path to the labeled attempts parquet",
    )
    ap.add_argument("--n", type=int, default=200, help="target total sample size across all five causes")
    ap.add_argument("--seed", type=int, default=0, help="seed for sampling and blinded shuffling")
    ap.add_argument("--outdir", default="output/human_rating_v3/zerocer", help="output directory")
    args = ap.parse_args(argv)

    build_zerocer_sheet(args.attempts, n=args.n, seed=args.seed, outdir=args.outdir)


if __name__ == "__main__":
    main()
