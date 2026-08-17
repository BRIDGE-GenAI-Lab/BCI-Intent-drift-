"""Interface-policy substudy exposure subset (Task D2).

The interface-policy substudy re-runs the baseline `forced`/P0 condition
under alternative interface policies, so it needs a smaller exposure set --
but it must NOT re-corrupt anything: reusing the main run's `forced`/P0
generations requires that the substudy's `noisy_text` be BYTE-IDENTICAL to
the main run's, for the same (message_id, cer_target, replicate_idx). This
module therefore only SUBSETS the already-materialized, already-corrupted
exposure parquets (`exposure_v3_auth_full`, `exposure_v2_full`) -- it never
calls the noise model. The only genuine sampling here is which 300 AUTH
message_ids (of the 2168-message Costello corpus) are drawn into the
substudy, stratified proportionally across the corpus's 40 categories.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from idrift.data.corpus_sample import _hamilton_apportion
from idrift.lib.io_utils import _dir, load_checkpoint, save_checkpoint

SUBSTUDY_SEED = 20260720
N_SUBSTUDY_AUTH = 300
MAX_REPLICATE = 10  # keep replicate_idx < 10 of the available 0..19
N_SHARDS = 4

# The 7 keys `idrift.models.o2_runner_v2` reads from each exposure row.
_RUNNER_KEYS = ("message_id", "corpus", "text", "cer_target", "replicate_idx", "realized_cer", "noisy_text")


def _apportion_with_floor(sizes: dict, n: int) -> dict:
    """Largest-remainder (Hamilton) apportionment of `n` across `sizes`
    (category -> available count), with an explicit floor of 1 per
    category that never exceeds that category's own size.

    Reserves 1 unit for every category up front, then runs the existing
    `idrift.data.corpus_sample._hamilton_apportion` (the same
    largest-remainder method already used for the Task 1.1 category x
    length-bin sample) on the REMAINING `n - len(sizes)` units,
    proportional to each category's remaining capacity (size - 1). Since
    that reused apportionment already guarantees its allocation never
    exceeds a group's size, adding the reserved 1 back keeps every
    category's final allocation <= its own size.
    """
    keys = list(sizes.keys())
    if n < len(keys):
        raise ValueError(f"n={n} is smaller than the {len(keys)} categories; cannot floor every one at 1")
    if n > sum(sizes.values()):
        raise ValueError(f"n={n} exceeds total available size {sum(sizes.values())}")

    remaining_capacity = {k: sizes[k] - 1 for k in keys}
    extra = _hamilton_apportion(remaining_capacity, n - len(keys))
    return {k: 1 + extra[k] for k in keys}


def draw_auth_subset(corpus_df: pd.DataFrame, n: int = N_SUBSTUDY_AUTH, seed: int = SUBSTUDY_SEED) -> list[str]:
    """Stratified, seeded draw of `n` AUTH message_ids from `corpus_df`.

    Allocates `n` across `corpus_df`'s `category` values proportional to
    category size via `_apportion_with_floor` (floor of 1 per category,
    never exceeding a category's size, largest remainder for the rest).
    Within each category, the allocated count is drawn WITHOUT replacement
    from a single `numpy.random.default_rng(seed)`, consuming categories in
    a fixed sorted order -- so the same (corpus_df, n, seed) always returns
    the same 300 message_ids.
    """
    sizes = corpus_df.groupby("category")["message_id"].size().to_dict()
    alloc = _apportion_with_floor(sizes, n)

    rng = np.random.default_rng(seed)
    drawn: list[str] = []
    for cat in sorted(sizes):
        pool = corpus_df.loc[corpus_df["category"] == cat, "message_id"].to_numpy()
        chosen = rng.choice(pool, size=alloc[cat], replace=False)
        drawn.extend(chosen.tolist())
    return drawn


def _as_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the intended-text column to `text` if the source used
    `intended_text` instead (neither of this task's two source parquets
    actually needs this -- both already carry `text` -- but the assembly
    step guards it anyway since the spec calls for it)."""
    if "text" not in df.columns and "intended_text" in df.columns:
        return df.rename(columns={"intended_text": "text"})
    return df


def assemble_substudy_exposure(
    corpus_df: pd.DataFrame,
    auth_full: pd.DataFrame,
    v2_full: pd.DataFrame,
    n_auth: int = N_SUBSTUDY_AUTH,
    seed: int = SUBSTUDY_SEED,
    max_replicate: int = MAX_REPLICATE,
) -> pd.DataFrame:
    """Assemble the interface-policy substudy exposure subset by SUBSETTING
    (never re-corrupting) the existing exposure parquets.

    - AUTH: `n_auth` message_ids drawn by `draw_auth_subset` (stratified over
      `corpus_df`'s categories, seeded), rows of `auth_full` with
      `replicate_idx < max_replicate`.
    - CRIT / CTRL: EVERY message_id already in `v2_full` for that corpus
      (they are the small, fixed critical/control probe sets -- not
      subsampled), same `replicate_idx < max_replicate` cap.

    CTRL is a matched-control subset drawn FROM the same Costello corpus as
    AUTH (its message_ids are `costello_XXXX`, not a separate namespace like
    CRIT's `probe_XXXX`), so the AUTH draw pool excludes any message_id
    already reserved by CRIT/CTRL in `v2_full` -- otherwise the same
    message_id could land in the subset twice under two different corpus
    labels, breaking the "each message_id contributes exactly 50 rows"
    invariant. The three corpora are therefore disjoint by construction.

    Returns the concatenation, sorted stably by
    (message_id, cer_target, replicate_idx).
    """
    auth_full = _as_text(auth_full)
    v2_full = _as_text(v2_full)

    reserved_ids = set(v2_full.loc[v2_full["corpus"].isin(["CRIT", "CTRL"]), "message_id"])
    eligible_corpus_df = corpus_df[~corpus_df["message_id"].isin(reserved_ids)]
    auth_ids = set(draw_auth_subset(eligible_corpus_df, n=n_auth, seed=seed))
    auth_rows = auth_full[auth_full["message_id"].isin(auth_ids) & (auth_full["replicate_idx"] < max_replicate)]
    crit_rows = v2_full[(v2_full["corpus"] == "CRIT") & (v2_full["replicate_idx"] < max_replicate)]
    ctrl_rows = v2_full[(v2_full["corpus"] == "CTRL") & (v2_full["replicate_idx"] < max_replicate)]

    subset = pd.concat([auth_rows, crit_rows, ctrl_rows], ignore_index=True)
    return subset.sort_values(["message_id", "cer_target", "replicate_idx"], kind="mergesort").reset_index(drop=True)


def _to_native(value):
    """Cast a numpy scalar/array to a plain Python type -- `json.dumps`
    doesn't accept numpy int64/float64/bool_ directly."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _runner_record(row: pd.Series) -> dict:
    return {k: _to_native(row[k]) for k in _RUNNER_KEYS}


def write_shards(df: pd.DataFrame, out_dir: Path, name: str, n_shards: int = N_SHARDS) -> list[Path]:
    """Split `df` (already stably sorted by the caller) into `n_shards`
    CONTIGUOUS slices, one exposure-per-line jsonl each, keyed by the 7
    fields `idrift.models.o2_runner_v2` requires."""
    paths = []
    for i, idx in enumerate(np.array_split(np.arange(len(df)), n_shards)):
        p = Path(out_dir) / f"{name}.part{i}.jsonl"
        with open(p, "w") as f:
            for _, row in df.iloc[idx].iterrows():
                f.write(json.dumps(_runner_record(row)) + "\n")
        paths.append(p)
    return paths


def main() -> pd.DataFrame:
    """Materialize `output/intermediate/substudy_exposures.parquet` and its
    4 `substudy_exposure.part{0..3}.jsonl` shards."""
    d = _dir()
    corpus_df = load_checkpoint("corpus_costello")
    auth_full = load_checkpoint("exposure_v3_auth_full")
    v2_full = load_checkpoint("exposure_v2_full")

    subset = assemble_substudy_exposure(corpus_df, auth_full, v2_full)
    save_checkpoint(subset, "substudy_exposures")
    shard_paths = write_shards(subset, d, "substudy_exposure")

    print(f"rows={len(subset)} msgids={subset['message_id'].nunique()}")
    for corpus, cnt in subset["corpus"].value_counts().items():
        print(f"  {corpus}: {cnt}")
    for p in shard_paths:
        with open(p) as f:
            n_lines = sum(1 for _ in f)
        print(f"  {p.name}: {n_lines} lines")

    return subset


if __name__ == "__main__":
    main()
