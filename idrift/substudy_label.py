"""Abstention-aware labeling of the interface substudy, P1-P5 (revision Task D4).

The interface-condition substudy (Task D2/prompts.py P1-P5) generates its own
JSONL output rows via `idrift.models.o2_runner_v2` -- one file per
(model, prompt, shard), named `outputs_v3sub_<PROMPT>_<SAFE>_part<SHARD>.jsonl`.
This module turns those raw rows into a labeled parquet, reusing the two
pieces already built rather than reimplementing either:

- `idrift.interfaces.conditions` (Task D1): `condition_for_prompt` (the
  runner's own `condition` field is just the prompt id string, e.g. "P1" --
  see `scratchpad/o2_v3_stage/o2_v3_substudy.sbatch`'s `--condition "$PROMPT"`
  -- so it is not the interface-condition name and must be overwritten, not
  trusted), `normalize_record` (candidate-list parsing), and the abstention
  safety guard `resolve_outcome`/`is_declined`.
- `idrift.adjudicate.label_runner`: `compute_signals` (the only layer that
  touches a model) and `derive_labels` (pure, reconstructs the
  faithful/degraded/drift label from precomputed signals).

`forced`/P0 is NOT handled here -- it is reused from the already-labeled
main/v2 run at analysis time (see the D5 brief), so this module only ever
sees prompt ids P1-P5.

The safety property this module exists to protect: an abstained row must
never reach the labeler, in either the main (intended, output) pair or the
candidate-list alternates. `assemble_labeled` routes every row's outcome
through `resolve_outcome`, which returns "declined" WITHOUT calling the
injected label lookup at all when the row is abstained -- so a `main_labels`
mapping that simply omits an abstained row's key is enough to prove the row
was never consulted (a lookup that shouldn't happen raises KeyError loudly
rather than silently producing a plausible-looking label).

Pure pandas/numpy/pyarrow + stdlib, except for the two reused model-layer
calls inside `run` (the one function this module allows to touch a model).

American spelling; no em dashes (double hyphens for asides, per house
style).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from idrift.adjudicate.label_runner import compute_signals, derive_labels
from idrift.interfaces.conditions import condition_for_prompt, normalize_record, resolve_outcome


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_substudy_outputs(paths: list[str]) -> pd.DataFrame:
    """Read the substudy JSONL output files into one DataFrame.

    Concatenates all `paths` in order (stable within and across files) into a
    fresh 0..n-1 index -- that index is the `row_key` every other function in
    this module keys on. Overwrites `condition` with
    `condition_for_prompt(prompt_id)` (the runner's own `condition` field is
    just the prompt id, see the module docstring) and adds a `candidates`
    column via `normalize_record` (non-None only for candidate_list/P4 rows).
    All other columns (model, prompt_sha256, temperature, message_id, corpus,
    cer_target, replicate_idx, realized_cer, intended_text, noisy_text,
    raw_output, output_message, confidence, abstained) pass through
    unchanged.
    """
    rows = []
    for path in paths:
        rows.extend(_read_jsonl(path))
    df = pd.DataFrame(rows).reset_index(drop=True)

    df["condition"] = df["prompt_id"].map(condition_for_prompt)
    df["candidates"] = [normalize_record(rec)["candidates"] for rec in df.to_dict("records")]
    return df


def pairs_to_label(df: pd.DataFrame) -> pd.DataFrame:
    """The subframe of NON-abstained rows that must go through the labeler.

    Just `intended_text`/`output_message`, indexed by the same row_key as
    `df` (a caller joins back via that index). Abstained rows are excluded
    here -- they become "declined" in `assemble_labeled` without ever being
    handed to a labeler.
    """
    abstained = df["abstained"].fillna(False).astype(bool)
    return df.loc[~abstained, ["intended_text", "output_message"]]


def candidate_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Explode candidate_list (P4) rows' `candidates` into one row per
    (row_key, intended_text, candidate_text) pair, so each alternate can be
    labeled the same way a main pair is. Empty frame (with the right
    columns) if `df` has no candidate_list rows.
    """
    records = []
    mask = df["condition"] == "candidate_list"
    for row_key, row in df.loc[mask, ["intended_text", "candidates"]].iterrows():
        cands = row["candidates"]
        # A candidate_list row may still have no parseable candidates: parse_candidates
        # returns None, which becomes NaN (a float) once it is a pandas column. Guard on
        # `isinstance(list)` -- `cands or []` would NOT work because float('nan') is truthy.
        if not isinstance(cands, list):
            continue
        for candidate_text in cands:
            records.append({"row_key": row_key, "intended_text": row["intended_text"], "candidate_text": candidate_text})
    return pd.DataFrame(records, columns=["row_key", "intended_text", "candidate_text"])


def _label_lookup(main_labels) -> dict:
    """Normalize `main_labels` (dict, Series, or a `label`-column DataFrame,
    all indexed/keyed by row_key) into a plain `{row_key: label}` dict."""
    if isinstance(main_labels, pd.DataFrame):
        return main_labels["label"].to_dict()
    if isinstance(main_labels, pd.Series):
        return main_labels.to_dict()
    return dict(main_labels)


def assemble_labeled(df: pd.DataFrame, main_labels, candidate_labels: pd.DataFrame) -> pd.DataFrame:
    """Combine the loaded substudy frame with its labels into the final
    labeled frame.

    - `outcome`: routed through `resolve_outcome` per row, so an abstained
      row becomes "declined" WITHOUT its (intended, output) pair ever being
      looked up in `main_labels`; every other row's outcome is
      `main_labels[row_key]`.
    - `any_candidate_faithful`: for candidate_list rows only, True iff the
      row's own outcome (the best candidate, `output_message`) OR any
      alternate in `candidate_labels` is labeled "faithful"; pandas <NA> for
      every other condition (a scalar for a set of conditions that never
      offered alternatives in the first place, not a meaningful False).

    `candidate_labels` is a frame shaped like `candidate_pairs`'s output plus
    a `label` column (one row per candidate; possibly empty, possibly
    multiple rows per row_key).

    Returns all of `df`'s original columns plus `outcome` and
    `any_candidate_faithful`.
    """
    label_map = _label_lookup(main_labels)

    outcomes = []
    for row_key, row in df.iterrows():
        rec = row.to_dict()

        def label_fn(_rec, _row_key=row_key):
            return label_map[_row_key]

        outcomes.append(resolve_outcome(rec, label_fn))

    out = df.copy()
    out["outcome"] = outcomes

    alt_faithful: dict = {}
    if len(candidate_labels):
        alt_faithful = (
            candidate_labels.groupby("row_key")["label"].apply(lambda labels: bool((labels == "faithful").any())).to_dict()
        )

    is_candidate_list = out["condition"] == "candidate_list"
    any_candidate_faithful = []
    for row_key in out.index:
        if not is_candidate_list.loc[row_key]:
            any_candidate_faithful.append(pd.NA)
            continue
        primary_faithful = out.at[row_key, "outcome"] == "faithful"
        any_candidate_faithful.append(bool(primary_faithful or alt_faithful.get(row_key, False)))
    out["any_candidate_faithful"] = pd.array(any_candidate_faithful, dtype="boolean")

    return out


def run(
    paths: list[str], out_parquet: str, *, device: str = "mps", batch_size: int = 64, checkpoint_dir: str | None = None
) -> pd.DataFrame:
    """Orchestrate the full pipeline end to end. The only function in this
    module that touches a model (via `compute_signals`), invoked on O2/MPS
    once the substudy generation lands.

    load_substudy_outputs -> pairs_to_label + candidate_pairs ->
    compute_signals + derive_labels on BOTH pair sets -> assemble_labeled ->
    write `out_parquet`.
    """
    df = load_substudy_outputs(paths)
    to_label = pairs_to_label(df)
    cands = candidate_pairs(df)

    checkpoint_main = checkpoint_cands = None
    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_main = str(Path(checkpoint_dir) / "substudy_main_signals_checkpoint.parquet")
        checkpoint_cands = str(Path(checkpoint_dir) / "substudy_candidate_signals_checkpoint.parquet")

    main_signals = compute_signals(to_label, device=device, batch_size=batch_size, checkpoint_path=checkpoint_main)
    main_merged = pd.concat([to_label.reset_index(drop=True), main_signals.reset_index(drop=True)], axis=1)
    main_merged.index = to_label.index
    main_labels = pd.Series(derive_labels(main_merged)["label"].values, index=to_label.index)

    if len(cands):
        cand_input = cands.rename(columns={"candidate_text": "output_message"})[["intended_text", "output_message"]]
        cand_signals = compute_signals(cand_input, device=device, batch_size=batch_size, checkpoint_path=checkpoint_cands)
        cand_merged = pd.concat([cand_input.reset_index(drop=True), cand_signals.reset_index(drop=True)], axis=1)
        candidate_labels = cands.assign(label=derive_labels(cand_merged)["label"].values)
    else:
        candidate_labels = pd.DataFrame(columns=["row_key", "intended_text", "candidate_text", "label"])

    result = assemble_labeled(df, main_labels, candidate_labels)
    result.to_parquet(out_parquet)
    return result


if __name__ == "__main__":
    import argparse
    import glob

    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="glob for outputs_v3sub_*.jsonl shards")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--checkpoint-dir", default=None)
    args = ap.parse_args()

    input_paths = sorted(glob.glob(args.glob))
    labeled = run(input_paths, args.out, device=args.device, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir)
    print(f"substudy_label.run: {len(labeled)} rows from {len(input_paths)} files -> {args.out}")
