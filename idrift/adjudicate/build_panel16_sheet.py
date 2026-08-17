"""Prespecified stratified 16-model human-rating validation instrument +
dedicated zero-CER sheet (Task 6, reviewer major #1).

Reviewer major #1 asked for validation of all 16 models with a
PRESPECIFIED, stratified, sampling-weighted human sample covering each
corpus, the full CER range, and each automated outcome class -- plus
direct adjudication of the zero-CER condition (does a human, blind to the
fact realized_cer == 0, still call these items "drift"?). This module
builds that instrument. It only builds rating sheets; no ratings are
collected here and no inference is run.

Two panels are produced from a single cached, labeled attempts parquet
(`output/intermediate/attempts_v3plus_labeled.parquet`, 3,888,000 rows, 16
models):

    panel_stratified -- one blinded item sheet drawn from EVERY nonempty
        cell of the model(16) x corpus(3) x cer_target(5) x automated
        label(3) stratification (up to 720 cells; 711 are nonempty in the
        real data), up to `n_per_cell` items per cell. Each item carries a
        `sampling_weight` = stratum population / stratum sample size, so
        downstream analyses can reweight the rated sample back to the
        population it was drawn from.

    panel_zerocer -- a dedicated draw from the `realized_cer == 0 &
        label == "drift"` subset (17,971 rows in the real data), stratified
        by model x automated cause bucket (reusing
        `idrift.analysis.zero_cer_audit._classify_cause`, exactly as
        `build_zerocer_sheet.py` already does, purely to STRATIFY the draw
        -- never exposed to raters). Every item in this panel has
        `realized_cer == 0`; the point is to check whether a rater, blind
        to that fact, still calls the item "drift" using the SAME 4-label
        protocol as the main panel (this is a validation check on the
        automated zero-CER-drift calls themselves, distinct from the
        separate cause-taxonomy adjudication `build_zerocer_sheet.py`
        already builds).

Both panels reuse the replicate-aware attempt key convention documented in
`idrift.adjudicate.adapt_v2_for_rating`: synthetic corruption is redrawn
per `replicate_idx` even at temp0, so the 20 replicates of a given
(message, model, CER) cell are DISTINCT generations. Folding `replicate_idx`
into `message_id` before any keying/joining keeps each generation its own
rate-able item instead of silently collapsing 19 of every 20 onto one.

Blinding (critical, per the binding constraints): the rater-facing sheets
(`panel_stratified/sheet.csv`, `panel_zerocer/sheet.csv`) NEVER carry
`model`, `cer_target`, `label`, or `realized_cer` -- those columns live
only in the combined `_KEY_DO_NOT_SHARE/key.csv`. Item order in both sheets
is a stable shuffle keyed on SHA-256(seed, item_id) -- not
`.sample(frac=1, random_state=...)` (whose row order depends on the
installed numpy PRNG implementation and is not guaranteed stable across
numpy versions) -- so the same data + seed always reproduce the exact same
row order, on any machine, forever.

Usage:
    uv run python -m idrift.adjudicate.build_panel16_sheet \\
        --attempts output/intermediate/attempts_v3plus_labeled.parquet \\
        --n-per-cell 3 --seed 0 --outdir output/human_rating_v3plus
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from idrift.analysis.zero_cer_audit import CAUSES
from idrift.analysis.zero_cer_audit import REQUIRED_COLUMNS as _ZEROCER_REQUIRED_COLUMNS
from idrift.analysis.zero_cer_audit import _classify_cause

# The brief's exact four physician-facing labels.
RATING_LABELS = ("Faithful", "Degraded", "Drift", "Message-critical")

STRATUM_COLUMNS = ("model", "corpus", "cer_target", "label")
ZEROCER_STRATUM_COLUMNS = ("model", "_cause")

# The rater-facing sheet: item_id + the three text fields + the sampling
# weight (needed for reweighted analysis, not identifying) + blank
# rating/notes. No model/corpus/CER/automated-label column of any kind.
BLINDED_COLUMNS = ["item_id", "intended_text", "noisy_text", "output_message", "sampling_weight", "rating", "notes"]

_STRAT_ITEM_PREFIX = "S"
_ZEROCER_ITEM_PREFIX = "Z"

_REQUIRED_ATTEMPT_COLUMNS = (
    "model", "corpus", "cer_target", "realized_cer", "label", "message_id",
    "replicate_idx", "intended_text", "noisy_text", "output_message",
)


def _stable_hash(*parts) -> str:
    """SHA-256 hex digest of `parts`, joined -- stable across interpreter
    runs, machines, and Python/numpy versions (unlike Python's salted
    `hash()` or a numpy RandomState/Generator, whose stream can change
    across versions)."""
    key = "|".join(str(p) for p in parts).encode()
    return hashlib.sha256(key).hexdigest()


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the replicate-aware attempt id and a guaranteed-unique
    positional row key.

    The replicate-aware id folds `replicate_idx` into `message_id`
    (`idrift.adjudicate.adapt_v2_for_rating`'s convention): synthetic
    corruption is redrawn per replicate even at temp0, so collapsing on the
    bare `message_id` would silently treat 20 distinct generations as one.
    The original id is preserved as `orig_message_id`.

    `_row_key` is a synthetic, guaranteed-unique positional key ("row0",
    "row1", ...) on the reset index -- the real attempts_v3plus_labeled
    parquet carries genuine duplicate rows on every business-key
    combination (see `build_zerocer_sheet.select_zero_cer_drift`'s
    docstring), so a natural-key sort/dedup is not robust here.
    """
    missing = [c for c in _REQUIRED_ATTEMPT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"attempts parquet is missing required column(s): {missing}")

    work = df.reset_index(drop=True).copy()
    work["orig_message_id"] = work["message_id"].astype(str)
    work["message_id"] = work["orig_message_id"] + "#r" + work["replicate_idx"].astype(str)
    work["_row_key"] = "row" + work.index.astype(str)
    return work


def _sample_stratum(group: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    """Sample up to `n_per_cell` rows from one stratum (capped at the
    stratum's own population), attach `sampling_weight` = population /
    realized sample size to every sampled row."""
    population = len(group)
    n = min(n_per_cell, population)
    sampled = group.sample(n=n, random_state=seed).copy()
    sampled["sampling_weight"] = population / len(sampled)
    return sampled


def draw_stratified_panel(df: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    """Draw the main panel: every nonempty (model, corpus, cer_target,
    label) cell contributes up to `n_per_cell` items.

    Iterates groups and concatenates (rather than `groupby(...).apply(...)`)
    -- under pandas 3.0.3, `DataFrameGroupBy.apply` excludes the grouping
    columns from the sub-frame passed to the callable by default and
    `include_groups=True` is no longer accepted, so the naive apply
    approach would silently drop the stratum columns (see
    `idrift.adjudicate.human_export.stratified_sample`'s docstring for the
    same gotcha).

    Args:
        df: prepared attempts DataFrame (see `_prepare`), with every
            column in `STRATUM_COLUMNS`.
        n_per_cell: target sample size per nonempty stratum cell (capped at
            that cell's own population -- never a shortfall silently
            padded from elsewhere).
        seed: random seed for the per-stratum draw.

    Returns:
        DataFrame: union of all per-stratum draws, with `sampling_weight`
        attached.
    """
    parts = [
        _sample_stratum(group, n_per_cell, seed)
        for _, group in df.groupby(list(STRATUM_COLUMNS), sort=True, observed=True)
    ]
    return pd.concat(parts, ignore_index=False)


def select_zero_cer_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to `realized_cer == 0 & label == "drift"` and attach the
    automated cause bucket (`_cause`, via `zero_cer_audit._classify_cause`)
    -- used only to stratify the zero-CER draw and to populate the hidden
    key, never exposed to raters.

    Args:
        df: prepared attempts DataFrame with every column in
            `idrift.analysis.zero_cer_audit.REQUIRED_COLUMNS`.

    Returns:
        DataFrame: the qualifying subset, with `_cause` added.

    Raises:
        KeyError: if a required signal column is missing.
    """
    missing = [c for c in _ZEROCER_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"select_zero_cer_drift: df is missing required column(s): {missing}")

    subset = df[(df["realized_cer"] == 0) & (df["label"] == "drift")].copy()
    subset["_cause"] = subset.apply(lambda r: _classify_cause(r.to_dict()), axis=1)
    return subset


def draw_zerocer_panel(subset: pd.DataFrame, n_per_cell: int, seed: int) -> pd.DataFrame:
    """Draw the zero-CER panel: every nonempty (model, cause) cell
    contributes up to `n_per_cell` items. Same per-stratum quota + weight
    convention as `draw_stratified_panel`."""
    parts = [
        _sample_stratum(group, n_per_cell, seed)
        for _, group in subset.groupby(list(ZEROCER_STRATUM_COLUMNS), sort=True, observed=True)
    ]
    return pd.concat(parts, ignore_index=False)


def assign_item_ids(items: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Assign stable, content-derived item ids ("S0001", "Z0001", ...).

    Sorted on the natural `_row_key` first (a canonical, seed-independent
    base order), then numbered sequentially. This assignment step is
    intentionally NOT the blinding shuffle -- see
    `stable_shuffle_by_item_id`, which reorders the sheet's rows AFTER ids
    are assigned.
    """
    canonical = items.sort_values("_row_key").reset_index(drop=True).copy()
    canonical["item_id"] = [f"{prefix}{i + 1:04d}" for i in range(len(canonical))]
    return canonical


def stable_shuffle_by_item_id(items: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Reorder rows deterministically by SHA-256(seed, item_id).

    Not `.sample(frac=1, random_state=seed)`: that ordering depends on the
    installed numpy PRNG implementation, which is not a stability
    guarantee across numpy versions or machines. SHA-256 is stable
    forever, so the same items + seed always produce the same row order,
    everywhere -- and the resulting order carries no information about the
    stratum a row came from.
    """
    sort_key = items["item_id"].map(lambda item_id: _stable_hash(seed, item_id))
    return (
        items.assign(_shuffle_key=sort_key)
        .sort_values("_shuffle_key")
        .drop(columns="_shuffle_key")
        .reset_index(drop=True)
    )


def make_blinded_sheet(items: pd.DataFrame) -> pd.DataFrame:
    """Build the rater-facing blinded sheet: item_id + the three text
    fields + sampling_weight + blank rating/notes. No model/corpus/CER/
    automated-label/automated-cause column leaks through."""
    sheet = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "intended_text": items["intended_text"],
            "noisy_text": items["noisy_text"],
            "output_message": items["output_message"],
            "sampling_weight": items["sampling_weight"],
            "rating": "",
            "notes": "",
        }
    )
    return sheet[BLINDED_COLUMNS]


def make_stratified_key(items: pd.DataFrame) -> pd.DataFrame:
    """Unblinding key rows for the stratified panel: item_id -> panel,
    model/corpus/cer_target/label/realized_cer, the replicate-aware and
    original message ids, and the sampling weight."""
    return pd.DataFrame(
        {
            "item_id": items["item_id"],
            "panel": "stratified",
            "message_id": items["message_id"],
            "orig_message_id": items["orig_message_id"],
            "replicate_idx": items["replicate_idx"],
            "model": items["model"],
            "corpus": items["corpus"],
            "cer_target": items["cer_target"],
            "realized_cer": items["realized_cer"],
            "label": items["label"],
            "automated_cause": pd.NA,
            "sampling_weight": items["sampling_weight"],
        }
    )


def make_zerocer_key(items: pd.DataFrame) -> pd.DataFrame:
    """Unblinding key rows for the zero-CER panel: same shape as
    `make_stratified_key` plus the automated cause bucket used to
    stratify the draw."""
    return pd.DataFrame(
        {
            "item_id": items["item_id"],
            "panel": "zerocer",
            "message_id": items["message_id"],
            "orig_message_id": items["orig_message_id"],
            "replicate_idx": items["replicate_idx"],
            "model": items["model"],
            "corpus": items["corpus"],
            "cer_target": items["cer_target"],
            "realized_cer": items["realized_cer"],
            "label": items["label"],
            "automated_cause": items["_cause"],
            "sampling_weight": items["sampling_weight"],
        }
    )


def _panel_coverage(key: pd.DataFrame, all_models: set, all_labels: set | None = None) -> dict:
    n_models = key["model"].nunique()
    coverage = {
        "n_items": int(len(key)),
        "n_models_covered": int(n_models),
        "n_models_expected": int(len(all_models)),
        "all_models_covered": bool(n_models == len(all_models)),
    }
    if all_labels is not None:
        n_labels = key["label"].nunique()
        coverage["n_classes_covered"] = int(n_labels)
        coverage["n_classes_expected"] = int(len(all_labels))
        coverage["all_classes_covered"] = bool(n_labels == len(all_labels))
    return coverage


def _build_manifest(
    df: pd.DataFrame, strat_key: pd.DataFrame, zerocer_key: pd.DataFrame, n_per_cell: int, seed: int,
) -> dict:
    all_models = set(df["model"].unique())
    all_labels = set(df["label"].unique())

    n_strata_possible = df["model"].nunique() * df["corpus"].nunique() * len(df["cer_target"].unique())
    n_strata_possible *= len(all_labels)
    n_strata_nonempty = len(strat_key.groupby(["model", "corpus", "cer_target", "label"]))

    n_cells_nonempty = len(zerocer_key.groupby(["model", "automated_cause"])) if len(zerocer_key) else 0

    panel_stratified = _panel_coverage(strat_key, all_models, all_labels)
    panel_stratified["n_strata_nonempty"] = int(n_strata_nonempty)
    panel_stratified["n_strata_possible"] = int(n_strata_possible)

    panel_zerocer = _panel_coverage(zerocer_key, all_models)
    panel_zerocer["n_cells_nonempty"] = int(n_cells_nonempty)
    causes_covered = sorted(zerocer_key["automated_cause"].dropna().unique().tolist())
    panel_zerocer["causes_covered"] = causes_covered
    panel_zerocer["causes_expected"] = list(CAUSES)
    panel_zerocer["all_causes_covered"] = bool(set(causes_covered) == set(CAUSES))

    return {
        "n_per_cell": n_per_cell,
        "seed": seed,
        "panel_stratified": panel_stratified,
        "panel_zerocer": panel_zerocer,
        "total_items": int(len(strat_key) + len(zerocer_key)),
    }


def _write_readme(path: Path, manifest: dict) -> None:
    ps = manifest["panel_stratified"]
    pz = manifest["panel_zerocer"]
    readme = f"""# Prespecified stratified 16-model human-rating validation instrument (Task 6)

Answers reviewer major #1: validate all 16 models with a PRESPECIFIED,
stratified, sampling-weighted human sample covering each corpus, the full
CER range, and each automated outcome class, plus direct adjudication of
the zero-CER condition.

## What is in here

```
README.md                     <- this file
panel_stratified/sheet.csv     <- blinded main panel (model x corpus x CER x label strata)
panel_zerocer/sheet.csv        <- blinded zero-CER-drift adjudication panel
_KEY_DO_NOT_SHARE/key.csv     <- unblinding key for BOTH panels. ANALYST ONLY. Never share.
```

## Rating protocol

Every item, in both panels, is rated on the SAME four-label ordinal scale,
BLIND to model identity and decoder-corruption level (CER):

- `Faithful` -- the output preserves the intended message's meaning.
- `Degraded` -- the output is garbled, incomplete, or nonsensical, so a
  reader could not reliably recover the intended meaning.
- `Drift` -- the output is fluent and readable, but asserts a different
  intent than the person intended.
- `Message-critical` -- the output changes a safety- or care-relevant
  detail (a reversed yes/no, a wrong number/dose/name/recipient, a changed
  urgency, or a dropped actionable instruction).

Two independent raters score every item; an adjudicator resolves any
disagreement. Raters see only `intended_text` / `noisy_text` /
`output_message` (plus `sampling_weight`, which carries no model/CER/label
identity) and fill in `rating` + optional `notes`. No model identity,
corpus, CER, or automated label/cause is ever shown to a rater.

## panel_stratified

Stratified across model (16) x corpus (AUTH/CRIT/CTRL) x cer_target
(0.0-0.4) x automated label (faithful/degraded/drift): every nonempty cell
of that {ps['n_strata_possible']}-cell grid contributes up to
{manifest['n_per_cell']} items each (`n_per_cell`), capped at that cell's
own population. This run: {ps['n_strata_nonempty']} nonempty cells
contributed, {ps['n_items']} items total. Each item carries a
`sampling_weight` = (stratum population) / (stratum sample size), so a
rated sample can be reweighted back to the full labeled population it was
drawn from.

- Models covered: {ps['n_models_covered']}/{ps['n_models_expected']} ({'ALL' if ps['all_models_covered'] else 'INCOMPLETE -- see manifest'}).
- Automated classes covered: {ps['n_classes_covered']}/{ps['n_classes_expected']} ({'ALL' if ps['all_classes_covered'] else 'INCOMPLETE -- see manifest'}).

## panel_zerocer

A dedicated draw from the `realized_cer == 0 & label == "drift"` subset
(the audited set the reviewer specifically asked to see adjudicated
directly), stratified by model x automated cause bucket (reusing
`idrift.analysis.zero_cer_audit`'s rule-based taxonomy purely to spread the
draw across causes -- never shown to raters). Every item in this panel has
`realized_cer == 0`; raters are blind to that fact and rate it on the
SAME four-label scale above. This checks whether the automated "drift"
call at zero decoder-level corruption survives independent human review,
complementing (not replacing) the separate cause-taxonomy adjudication in
`output/human_rating_v3/zerocer/`.

- Items this run: {pz['n_items']} across {pz['n_cells_nonempty']} nonempty (model, cause) cells.
- Models covered: {pz['n_models_covered']}/{pz['n_models_expected']} ({'ALL' if pz['all_models_covered'] else 'INCOMPLETE -- see manifest'}).
- Causes covered: {', '.join(pz['causes_covered'])} ({'ALL 5' if pz['all_causes_covered'] else 'INCOMPLETE -- see manifest'}).

## Provenance

- Rating instrument: `idrift/adjudicate/build_panel16_sheet.py`.
- Source data: `output/intermediate/attempts_v3plus_labeled.parquet`
  (3,888,000 rows, 16 models).
- Replicate-aware attempt id: `idrift.adjudicate.adapt_v2_for_rating`'s
  convention (fold `replicate_idx` into `message_id`).
- Zero-CER cause taxonomy reused (stratification + hidden key only):
  `idrift.analysis.zero_cer_audit._classify_cause`.
- Item order in both sheets is a stable shuffle by SHA-256(seed, item_id),
  not a numpy-PRNG-dependent `.sample(frac=1, ...)`.
- Sampling: seed={manifest['seed']}, n_per_cell={manifest['n_per_cell']}.
"""
    path.write_text(readme)


def log_summary(manifest: dict) -> None:
    """Print the total item count and explicit all-16-models / all-3-classes
    coverage confirmation, so an incomplete draw is never silently capped."""
    ps = manifest["panel_stratified"]
    pz = manifest["panel_zerocer"]

    print(f"panel_stratified: {ps['n_items']} items across {ps['n_strata_nonempty']} "
          f"nonempty strata (of {ps['n_strata_possible']} possible), n_per_cell={manifest['n_per_cell']}")
    print(f"  models covered: {ps['n_models_covered']}/{ps['n_models_expected']} "
          f"({'ALL COVERED' if ps['all_models_covered'] else 'INCOMPLETE'})")
    print(f"  automated classes covered: {ps['n_classes_covered']}/{ps['n_classes_expected']} "
          f"({'ALL COVERED' if ps['all_classes_covered'] else 'INCOMPLETE'})")

    print(f"panel_zerocer: {pz['n_items']} items across {pz['n_cells_nonempty']} nonempty (model, cause) cells")
    print(f"  models covered: {pz['n_models_covered']}/{pz['n_models_expected']} "
          f"({'ALL COVERED' if pz['all_models_covered'] else 'INCOMPLETE'})")
    print(f"  causes covered: {pz['causes_covered']}")

    print(f"TOTAL items (both panels): {manifest['total_items']}")


def build(attempts_path, out_dir, n_per_cell: int, seed: int) -> dict:
    """End-to-end: load parquet -> prepare replicate-aware key -> draw both
    panels -> blind -> write `panel_stratified/sheet.csv`,
    `panel_zerocer/sheet.csv`, a combined `_KEY_DO_NOT_SHARE/key.csv`, and
    `README.md` -> log the coverage summary.

    Args:
        attempts_path: path to the labeled attempts parquet, or an
            already-loaded DataFrame (accepted directly so tests can build
            one in memory or via a temp parquet round trip).
        out_dir: output directory; created if missing.
        n_per_cell: target items per nonempty stratum cell, for BOTH
            panels (model x corpus x cer_target x label for the main
            panel; model x cause for the zero-CER panel). Capped at each
            cell's own population.
        seed: random seed (per-stratum sampling + the SHA-256 shuffle key),
            for determinism.

    Returns:
        dict: the manifest (per-panel item counts, coverage, and output
        paths).
    """
    df = attempts_path if isinstance(attempts_path, pd.DataFrame) else pd.read_parquet(attempts_path)
    work = _prepare(df)

    strat_sampled = draw_stratified_panel(work, n_per_cell=n_per_cell, seed=seed)
    strat_items = assign_item_ids(strat_sampled, prefix=_STRAT_ITEM_PREFIX)
    strat_items = stable_shuffle_by_item_id(strat_items, seed=seed)
    strat_sheet = make_blinded_sheet(strat_items)
    strat_key = make_stratified_key(strat_items)

    zerocer_subset = select_zero_cer_drift(work)
    zerocer_sampled = draw_zerocer_panel(zerocer_subset, n_per_cell=n_per_cell, seed=seed)
    zerocer_items = assign_item_ids(zerocer_sampled, prefix=_ZEROCER_ITEM_PREFIX)
    zerocer_items = stable_shuffle_by_item_id(zerocer_items, seed=seed)
    zerocer_sheet = make_blinded_sheet(zerocer_items)
    zerocer_key = make_zerocer_key(zerocer_items)

    combined_key = pd.concat([strat_key, zerocer_key], ignore_index=True)

    out = Path(out_dir)
    strat_dir = out / "panel_stratified"
    zerocer_dir = out / "panel_zerocer"
    key_dir = out / "_KEY_DO_NOT_SHARE"
    for d in (strat_dir, zerocer_dir, key_dir):
        d.mkdir(parents=True, exist_ok=True)

    path_strat_sheet = strat_dir / "sheet.csv"
    path_zerocer_sheet = zerocer_dir / "sheet.csv"
    path_key = key_dir / "key.csv"
    path_readme = out / "README.md"

    strat_sheet.to_csv(path_strat_sheet, index=False)
    zerocer_sheet.to_csv(path_zerocer_sheet, index=False)
    combined_key.to_csv(path_key, index=False)

    manifest = _build_manifest(work, strat_key, zerocer_key, n_per_cell, seed)
    _write_readme(path_readme, manifest)
    log_summary(manifest)

    manifest["paths"] = {
        "panel_stratified_sheet": path_strat_sheet,
        "panel_zerocer_sheet": path_zerocer_sheet,
        "key": path_key,
        "readme": path_readme,
    }
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Build the prespecified stratified 16-model human-rating validation instrument "
        "+ zero-CER sheet (Task 6, reviewer major #1)."
    )
    ap.add_argument(
        "--attempts", default="output/intermediate/attempts_v3plus_labeled.parquet",
        help="path to the labeled attempts parquet",
    )
    ap.add_argument(
        "--n-per-cell", type=int, default=3,
        help="target items per nonempty stratum cell, for both panels (capped at each cell's own population)",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for sampling and the SHA-256 blinded shuffle")
    ap.add_argument("--outdir", default="output/human_rating_v3plus", help="output directory")
    args = ap.parse_args(argv)

    build(args.attempts, args.outdir, n_per_cell=args.n_per_cell, seed=args.seed)


if __name__ == "__main__":
    main()
