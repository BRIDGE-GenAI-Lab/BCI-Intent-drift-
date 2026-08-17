"""Raw-decode benefit-harm transition matrix (revision Task B1).

The top reviewer ask: does LLM post-editing of a noisy BCI decode HELP or
HARM relative to just sending the raw corrupted stream? To answer it we
label the same taxonomy on two things and compare them exposure-by-exposure:

- the LLM's reconstruction (`output_message`, already labeled upstream in
  `attempts_v2_labeled.parquet` -> `label`), and
- the RAW corrupted decode itself (`noisy_text`), labeled here with the
  IDENTICAL pipeline (`label_raw_decode`, treating `noisy_text` as the
  "output" and `intended_text` as the reference).

The raw decode is MODEL-INDEPENDENT: for a given (message_id, corpus,
cer_target, replicate_idx) every one of the 7 models saw the same
`noisy_text`. So a single raw label broadcasts across all models at that
key, and `transition_matrix` cross-tabulates raw-label -> output-label into
five clinically meaningful transitions, per corpus and per CER band:

    no_benefit       raw faithful  -> output faithful   (nothing to fix)
    rescue           raw degraded  -> output faithful   (LLM repaired it)
    silent_failure   raw degraded  -> output drift       (visible break -> fluent lie)
    llm_induced_harm raw faithful  -> output drift       (LLM invented an error)
    no_correction    raw degraded  -> output degraded    (still visibly broken)

A raw corrupted stream is usually DISFLUENT, so most non-faithful raw
decodes label `degraded` (correct: the raw stream is visibly broken, not a
fluent substitution). Any other raw label -- including a raw stream that is
itself `drift` -- is handled generally and counted under an `other__*`
category, never dropped.

The headline safety number is
    faithful_gained_per_drift_introduced = rescues / (llm_induced_harm + silent_failure)
i.e. how many visibly-broken decodes the LLM turns faithful for each new
silent (drift) failure it creates. Divide-by-zero is guarded (see
`_net_metric`).

Two functions:
    label_raw_decode(exposure_path, out_path) -- run the taxonomy pipeline on
        the raw decodes (heavy; run separately, NOT in tests).
    transition_matrix(labeled_attempts_df, labeled_raw_df) -- pure pandas
        cross-tab; no models.

American spelling; no em dashes (double hyphens for asides, per house style).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from idrift.adjudicate import label_runner

# Keys that identify one exposure. The raw decode is unique per key (it is
# model-independent), so it broadcasts across the 7 models' output rows.
_KEY_COLS = ["message_id", "corpus", "cer_target", "replicate_idx"]

# The five named clinical transitions, keyed (raw_label, output_label).
_NAMED = {
    ("faithful", "faithful"): "no_benefit",
    ("degraded", "faithful"): "rescue",
    ("degraded", "drift"): "silent_failure",
    ("faithful", "drift"): "llm_induced_harm",
    ("degraded", "degraded"): "no_correction",
}
# Stable order so the frame's columns are deterministic.
_CANONICAL = ["no_benefit", "rescue", "silent_failure", "llm_induced_harm", "no_correction"]


def _as_df(x) -> pd.DataFrame:
    """Accept an already-loaded DataFrame or a parquet path, matching the
    DataFrame-or-path convention used elsewhere in this package (e.g.
    `matched_compare.build_matched_frame`)."""
    if isinstance(x, pd.DataFrame):
        return x
    return pd.read_parquet(x)


# ---------------------------------------------------------------------------
# 1. label_raw_decode -- run the taxonomy pipeline on the raw corrupted decode.
#    HEAVY (loads NLI/embedding/fluency models via label_runner.compute_signals);
#    run separately, never in the test suite.
# ---------------------------------------------------------------------------


def label_raw_decode(
    exposure_path,
    out_path,
    *,
    device: str = "mps",
    batch_size: int = 64,
    checkpoint_path: str | None = None,
    tau: float = 0.5,
    tie_break: str = "primary",
) -> pd.DataFrame:
    """Label the RAW corrupted decodes (`noisy_text`) with the same taxonomy
    pipeline the LLM outputs went through, treating `noisy_text` as the
    "output_message" and `intended_text` as the reference.

    Reuses `idrift.adjudicate.label_runner.compute_signals` (the only layer
    that touches a model) and `label_runner.derive_labels` (pure decision
    logic) UNCHANGED -- it never reimplements the taxonomy. To avoid
    redundant inference, the unique `(intended_text, noisy_text)` pairs are
    labeled ONCE, then broadcast back to every exposure row sharing that
    pair. (Dedup is on the pair, not `noisy_text` alone, because at
    `cer_target == 0.0` the raw decode equals `intended_text`, so the same
    `noisy_text` string can belong to different intended references and must
    not collapse across them.)

    Args:
        exposure_path: parquet path (or an already-loaded DataFrame) with at
            least `message_id`, `corpus`, `cer_target`, `replicate_idx`,
            `intended_text`, `noisy_text` -- e.g.
            `output/intermediate/attempts_v2_labeled.parquet`.
        out_path: parquet path to write. The written frame is keyed by
            `(message_id, corpus, cer_target, replicate_idx)` with a single
            `raw_label` column (in {faithful, degraded, drift}).
        device, batch_size, checkpoint_path: forwarded to
            `label_runner.compute_signals`.
        tau, tie_break: forwarded to `label_runner.derive_labels`.

    Returns:
        The written frame (keys + `raw_label`), one row per exposure row.

    WARNING: This loads the heavy NLI/embedding/fluency stack and can take
    hours over the full cohort. It is invoked by a dedicated background job,
    NOT by the test suite -- the unit test monkeypatches
    `label_runner.compute_signals`/`derive_labels`.
    """
    df = _as_df(exposure_path)

    missing = [c for c in _KEY_COLS + ["intended_text", "noisy_text"] if c not in df.columns]
    if missing:
        raise ValueError(f"label_raw_decode: exposure frame missing columns: {missing}")

    # Unique (reference, raw-decode) pairs -- the pipeline scores each once.
    unique = df[["intended_text", "noisy_text"]].drop_duplicates().reset_index(drop=True)
    to_label = pd.DataFrame(
        {"intended_text": unique["intended_text"].values, "output_message": unique["noisy_text"].values}
    )

    signals = label_runner.compute_signals(
        to_label, device=device, batch_size=batch_size, checkpoint_path=checkpoint_path
    )
    merged = pd.concat([to_label.reset_index(drop=True), signals.reset_index(drop=True)], axis=1)
    labels = label_runner.derive_labels(merged, tau=tau, tie_break=tie_break)
    unique = unique.assign(raw_label=labels["label"].values)

    # Broadcast the per-pair label back to every exposure row (m:1: many
    # exposure rows can share one unique pair; each maps to exactly one).
    keyed = df[_KEY_COLS + ["intended_text", "noisy_text"]].merge(
        unique, on=["intended_text", "noisy_text"], how="left", validate="m:1"
    )
    out = keyed[_KEY_COLS + ["raw_label"]].reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


# ---------------------------------------------------------------------------
# 2. transition_matrix -- pure pandas cross-tab of raw-label -> output-label.
# ---------------------------------------------------------------------------


def _classify(raw_label: str, output_label: str) -> str:
    """Map one (raw_label, output_label) transition to a category. The five
    clinically named categories are returned by name; every other
    combination (e.g. a raw stream that is itself drift, or a faithful raw
    that the LLM merely degraded) gets a generic `other__<raw>_to_<out>`
    key so it is counted, never silently dropped."""
    named = _NAMED.get((raw_label, output_label))
    if named is not None:
        return named
    return f"other__{raw_label}_to_{output_label}"


def _net_metric(rescues: int, harm: int, silent: int):
    """faithful_gained_per_drift_introduced = rescues / (llm_induced_harm +
    silent_failure), with divide-by-zero guarded:

    - denominator 0 and some rescues -> float('inf') (net benefit, no new
      drift), with an explanatory note;
    - denominator 0 and no rescues -> None (0/0 undefined), with a note;
    - otherwise the finite ratio and an empty note.
    """
    denom = harm + silent
    if denom == 0:
        if rescues > 0:
            return float("inf"), "no drift introduced (denominator 0) with rescues>0; net benefit unbounded"
        return None, "no rescues and no drift introduced (0/0); net metric undefined"
    return rescues / denom, ""


def transition_matrix(
    labeled_attempts_df,
    labeled_raw_df,
    *,
    output_label_col: str = "label",
    raw_label_col: str = "raw_label",
    group_cols=("corpus", "cer_target"),
):
    """Cross-tabulate raw-decode labels against LLM-output labels into the
    five clinical benefit-harm transitions, per corpus and per CER band.

    Args:
        labeled_attempts_df: the LLM outputs, one row per (model, exposure),
            with `output_label_col` and the four `_KEY_COLS`. Path or
            DataFrame (e.g. `attempts_v2_labeled.parquet`).
        labeled_raw_df: the raw-decode labels from `label_raw_decode`, keyed
            by the four `_KEY_COLS` with `raw_label_col`. Path or DataFrame.
            Model-independent, so it broadcasts across every model at a key.
        output_label_col: LLM output label column (default `label`).
        raw_label_col: raw-decode label column (default `raw_label`).
        group_cols: grouping keys for the per-cell report (default
            `(corpus, cer_target)`; cer_target is discrete so each value IS
            a CER band).

    Returns:
        dict with:
          - "by_group": DataFrame, one row per group, with `n_total`, an
            `n_<category>` count and `rate_<category>` rate for each of the
            five named categories (always present) plus any observed
            `other__*` categories, the `faithful_gained_per_drift_introduced`
            net metric, and a `net_metric_note`.
          - "overall": dict of the same fields pooled over all rows (net
            metric is None when 0/0 undefined).
          - "categories": the category column names present.
          - "notes": diagnostics (e.g. count of attempts rows with no
            matching raw label).
    """
    attempts = _as_df(labeled_attempts_df).copy()
    raw = _as_df(labeled_raw_df).copy()
    group_cols = list(group_cols)
    notes: list[str] = []

    for name, frame, col in (("attempts", attempts, output_label_col), ("raw", raw, raw_label_col)):
        miss = [c for c in _KEY_COLS if c not in frame.columns] + ([col] if col not in frame.columns else [])
        if miss:
            raise ValueError(f"transition_matrix: {name} frame missing columns: {miss}")

    # Raw is unique per key; broadcast it across the models' attempt rows
    # (validate m:1 makes a fanned-out/duplicated raw frame fail loudly).
    raw_keyed = raw[_KEY_COLS + [raw_label_col]].drop_duplicates(subset=_KEY_COLS)
    if len(raw_keyed) != len(raw[_KEY_COLS].drop_duplicates()):
        notes.append("raw frame had duplicate keys; kept first per key")

    joined = attempts.merge(raw_keyed, on=_KEY_COLS, how="left", validate="m:1")

    n_unmatched = int(joined[raw_label_col].isna().sum())
    if n_unmatched:
        notes.append(f"{n_unmatched} attempt row(s) had no matching raw label and were dropped from the cross-tab")
        joined = joined[joined[raw_label_col].notna()]

    joined["transition"] = [
        _classify(r, o) for r, o in zip(joined[raw_label_col], joined[output_label_col])
    ]

    # Column universe: five named categories (always) + any observed others.
    observed_other = sorted(set(joined["transition"]) - set(_CANONICAL))
    all_cats = _CANONICAL + observed_other

    def _summarize(sub: pd.DataFrame) -> dict:
        counts = sub["transition"].value_counts().to_dict()
        n = int(len(sub))
        rescues = int(counts.get("rescue", 0))
        harm = int(counts.get("llm_induced_harm", 0))
        silent = int(counts.get("silent_failure", 0))
        val, note = _net_metric(rescues, harm, silent)
        rec: dict = {"n_total": n}
        for cat in all_cats:
            c = int(counts.get(cat, 0))
            rec[f"n_{cat}"] = c
            rec[f"rate_{cat}"] = (c / n) if n else float("nan")
        rec["faithful_gained_per_drift_introduced"] = val
        rec["net_metric_note"] = note
        return rec

    rows = []
    for gkey, sub in joined.groupby(group_cols, sort=True):
        gvals = gkey if isinstance(gkey, tuple) else (gkey,)
        rec = dict(zip(group_cols, gvals))
        rec.update(_summarize(sub))
        rows.append(rec)

    count_cols = [f"n_{c}" for c in all_cats]
    rate_cols = [f"rate_{c}" for c in all_cats]
    ordered_cols = group_cols + ["n_total"] + count_cols + rate_cols + [
        "faithful_gained_per_drift_introduced",
        "net_metric_note",
    ]
    by_group = pd.DataFrame(rows, columns=ordered_cols)
    # None (undefined net metric) -> NaN so the column stays numeric.
    by_group["faithful_gained_per_drift_introduced"] = by_group[
        "faithful_gained_per_drift_introduced"
    ].astype(float)

    overall = _summarize(joined)  # net metric here keeps None for 0/0
    if isinstance(overall["faithful_gained_per_drift_introduced"], float) and math.isnan(
        overall["faithful_gained_per_drift_introduced"]
    ):
        overall["faithful_gained_per_drift_introduced"] = None

    return {
        "by_group": by_group,
        "overall": overall,
        "categories": all_cats,
        "notes": notes,
    }
