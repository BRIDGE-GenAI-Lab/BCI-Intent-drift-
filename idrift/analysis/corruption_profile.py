"""Corruption-model transparency + a held-out-matrix sensitivity
(reviewer major #10).

Why this module exists
-----------------------
A reviewer asked for three things about the character-level corruption
process that turns a clean reference message into a noisy BCI-decode
message (`idrift.data.noise_model.inject` / `inject_with_metadata`):

1. State the EXACT insertion/deletion/substitution allocation actually
   used, not a vague "some insertions and deletions happen".
2. Justify the assumption that each character position is corrupted
   INDEPENDENTLY, relative to what is known about real P300-speller
   row/column error structure.
3. Show how sensitive the results are to the confusion matrix used --
   specifically, whether pooling all 29 participants' selection errors
   into one matrix (`output/intermediate/confusion_overall.npy`) hides
   participant-to-participant variability that a study-/participant-
   specific matrix would reveal.

This module answers all three from CACHED ARTIFACTS and the corruption
source code alone -- no new LLM inference, and no re-materialization of
the confusion matrices themselves (that would require re-walking the
source EDFs, which is out of scope for this task; see
`idrift.data.materialize_confusion`).

1. Exact allocation (`configured_allocation`)
----------------------------------------------
`inject`/`inject_with_metadata` split a message's per-character corruption
budget (`cer_target`) into an indel share and a substitution share via one
parameter, `ins_del_rate` (default 0.1, read here via `inspect.signature`
rather than re-typed as a constant, so this digest cannot silently drift
out of sync with the implementation if that default ever changes):

    indel_budget = cer_target * ins_del_rate        # deletion OR insertion
    sub_budget   = cer_target * (1 - ins_del_rate)  # substitution

Within the indel branch, the split between deletion and insertion is a
structural 50/50 coin flip in the algorithm itself (`noise_model.py`:
`if rng.random() < 0.5: ... delete ... else: ... insert ...`), not a
tunable parameter -- so it is reported here as a fixed fact read from the
source, not introspected via a signature default. Put together, the
allocation of the total corruption budget is:

    substitution_frac = 1 - ins_del_rate            # 0.90 at the default
    insertion_frac    = ins_del_rate / 2             # 0.05
    deletion_frac     = ins_del_rate / 2             # 0.05

which sums to exactly 1.0 by construction (verified, not assumed, by the
test suite).

2. Grid pass-through (`grid_passthrough_summary`)
---------------------------------------------------
The 36-cell speller grid (`idrift.data.grid.GRID_ALPHABET`) covers
A-Z/1-9/space. A character outside that alphabet (in practice: ordinary
English punctuation -- periods, commas, apostrophes) can never be
SUBSTITUTED (the substitution branch requires `ch in idx`), but it is NOT
otherwise protected: the indel branch fires before any grid-membership
check, so an out-of-grid character can still be deleted, or have a random
grid character inserted immediately before it. This function reports both
a structural count (the exact fraction of each cached AUTH message's
characters that are out-of-grid, independent of any corruption draw) and
an empirical cross-check (the fraction of characters that the ALREADY-
GENERATED `exposure_v2.parquet` records as having passed through fully
untouched, `n_passthrough_chars`), which is expected to run slightly
*below* the structural figure -- some out-of-grid characters are unlucky
enough to be touched by the indel branch even though they can never be
substituted.

3. Row/column confusion structure (`row_column_confusion_structure`)
----------------------------------------------------------------------
The reviewer's independence concern has two distinct parts, and only one
of them is actually addressed by the empirical confusion matrix:

  - WITHIN one character's substitution draw, does the model capture
    P300 row/column confusability (real speller errors cluster on grid
    cells sharing a stimulation row or column)? This function checks the
    materialized pooled matrix directly: it decomposes the off-diagonal
    probability mass into same-row / same-column / neither, and compares
    it against what a structure-free (uniform-over-other-cells) confusion
    matrix would give by chance. On the real matrix, off-diagonal mass is
    modestly enriched for grid neighbors (see computed numbers in the
    digest) relative to the 1/7-1/7-5/7 chance split -- so the per-
    character substitution step is NOT independent of grid geometry.
  - ACROSS character positions within a message, is one position's
    corruption draw independent of its neighbors'? Here the answer is
    yes by construction -- `inject`/`inject_with_metadata` draw a fresh
    `rng.random()` for every position in sequence, with no term coupling
    a position's outcome to its neighbors' outcomes. This is the
    assumption actually under scrutiny, and it is a genuine
    simplification: it does not model, e.g., attentional fatigue or
    drift-related error runs that would correlate consecutive positions'
    error probability. The digest states this explicitly rather than
    conflating it with the (better-supported) row/column structure.

4. Held-out-participant matrix sensitivity (`heldout_participant_sensitivity`)
---------------------------------------------------------------------------
Changing the confusion matrix used throughout the study (pooled across
all 29 participants) to a per-study or per-participant matrix would
require regenerating every downstream exposure/output/label artifact --
out of scope here. Instead, this function computes a DESCRIPTIVE bound:
it re-runs the SAME deterministic corruption function
(`idrift.data.noise_model.inject`) on the SAME cached AUTH message sample
(`output/intermediate/auth_sample.csv`, labels/messages unchanged), once
with the pooled matrix and once with each of the 29 cached per-
participant matrices (`output/intermediate/confusion_by_subject.npz`),
and reports how much the realized-CER distribution would have differed
had a single held-out participant's error profile been used instead of
the pooled cohort matrix. This is pure re-application of an existing,
cached, deterministic function to cached message text -- not new
inference and not a re-fit of anything -- so it is in scope even though a
literal per-study/per-participant analysis re-run is not.

Determinism: every corruption draw here uses a sha256-derived stable seed
(`_stable_seed`), the same idiom already used by
`idrift.data.build_exposure`/`idrift.data.exposure_v2` (Python's built-in
`hash()` is salted per interpreter process and would silently break
reproducibility). American spelling, no em dashes (double hyphens for
asides), per house style.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from idrift.data.grid import GRID_ALPHABET
from idrift.data.noise_model import inject, inject_with_metadata
from idrift.lib.cer import cer as measured_cer

DEFAULT_INTERMEDIATE = "output/intermediate"
DEFAULT_OUT = "output/corruption_profile_digest.json"

# The study's non-trivial cer_target grid (matches idrift.data.corpus_sample.
# CER_GRID minus the 0.0 level, which is a matrix-independent no-op: `inject`
# returns the text unchanged whenever cer_target <= 0, so it carries no
# information about matrix sensitivity).
CER_GRID = (0.1, 0.2, 0.3, 0.4)
DEFAULT_N_REP = 5

_ALLOCATION_NOTE = (
    "inject()/inject_with_metadata() split cer_target into indel_budget = "
    "cer_target * ins_del_rate and sub_budget = cer_target * (1 - ins_del_rate); "
    "within the indel budget, a structural (non-parametric) 50/50 coin flip in "
    "the algorithm itself chooses deletion vs. insertion. insertion_frac and "
    "deletion_frac are therefore each ins_del_rate / 2, and substitution_frac "
    "is 1 - ins_del_rate; the three sum to 1.0 by construction."
)

_INDEPENDENCE_NOTE = (
    "The per-character SUBSTITUTION step is not independent of grid geometry: "
    "the substitution target is drawn from that character's own row of the "
    "empirical confusion matrix, and the row_column_confusion_structure check "
    "below shows the matrix's off-diagonal mass is enriched for same-row/"
    "same-column grid neighbors relative to a structure-free (uniform) null, "
    "consistent with known P300 row/column confusability. What IS an "
    "independence assumption is ACROSS character positions within a message: "
    "inject()/inject_with_metadata() draw a fresh, uncorrelated random number "
    "for every position in sequence, with no mechanism coupling one position's "
    "corruption outcome to its neighbors'. This does not model within-message "
    "phenomena such as attentional fatigue or drift-related error runs that "
    "would correlate consecutive positions' error probability; it is reported "
    "here as a genuine simplifying limitation, not resolved by this module."
)


def _stable_seed(*parts) -> int:
    """Reproducible non-negative integer seed derived from `parts` via
    sha256 (Python's built-in `hash()` is salted per interpreter process
    and would silently break reproducibility across runs; same idiom as
    `idrift.data.build_exposure._stable_seed` / `idrift.data.exposure_v2.
    _stable_seed`)."""
    key = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def _default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def configured_allocation() -> dict:
    """Read the corruption process's EXACT insertion/deletion/substitution
    allocation from `idrift.data.noise_model` itself (via
    `inspect.signature`), not a re-typed constant.

    Returns:
        dict: `{ins_del_rate, insertion_frac, deletion_frac,
        substitution_frac, source, note}`. `insertion_frac + deletion_frac
        + substitution_frac == 1.0` by construction.

    Raises:
        ValueError: if `inject` and `inject_with_metadata`'s `ins_del_rate`
            defaults ever disagree (they must reproduce the same
            corruption process byte-for-byte; a mismatch would mean this
            digest could no longer speak for both).
    """
    rate_inject = inspect.signature(inject).parameters["ins_del_rate"].default
    rate_meta = inspect.signature(inject_with_metadata).parameters["ins_del_rate"].default
    if rate_inject != rate_meta:
        raise ValueError(
            "inject() and inject_with_metadata() ins_del_rate defaults disagree "
            f"({rate_inject!r} vs {rate_meta!r}); they must match to describe the "
            "same corruption process."
        )
    ins_del_rate = float(rate_inject)

    return {
        "ins_del_rate": ins_del_rate,
        "insertion_frac": ins_del_rate / 2.0,
        "deletion_frac": ins_del_rate / 2.0,
        "substitution_frac": 1.0 - ins_del_rate,
        "source": "idrift.data.noise_model.inject / inject_with_metadata (ins_del_rate default parameter)",
        "note": _ALLOCATION_NOTE,
    }


def grid_passthrough_summary(
    messages: pd.DataFrame, exposure_path: Path | str | None = None, alphabet=GRID_ALPHABET
) -> dict:
    """Structural + empirical pass-through summary for out-of-grid characters.

    Args:
        messages: DataFrame with a `text` column (e.g. the cached AUTH
            sample, `output/intermediate/auth_sample.csv`).
        exposure_path: optional path to a cached exposure parquet carrying
            `text`/`cer_target`/`n_passthrough_chars` columns (e.g.
            `output/intermediate/exposure_v2.parquet`); if given and
            present, an empirical cross-check is added. If absent, the
            empirical cross-check is omitted (never fabricated) and
            `empirical_check_available` is False.
        alphabet: the 36-cell grid alphabet.

    Returns:
        dict: `{description, alphabet_size, n_messages, mean_structural_oog_char_frac,
        median_structural_oog_char_frac, empirical_check_available,
        empirical_mean_oog_passthrough_frac (or None), empirical_source, note}`.
    """
    alpha_set = set(alphabet)

    def _oog_frac(text: str) -> float:
        text = str(text).upper()
        if len(text) == 0:
            return 0.0
        n_oog = sum(1 for c in text if c not in alpha_set)
        return n_oog / len(text)

    structural = messages["text"].apply(_oog_frac)

    out = {
        "description": (
            "Out-of-grid characters (not in the 36-cell alphabet, in practice "
            "ordinary punctuation) can never be SUBSTITUTED (the substitution "
            "branch requires grid membership), but they are NOT otherwise "
            "protected: the indel branch fires before any grid-membership "
            "check, so an out-of-grid character can still be deleted, or "
            "preceded by an inserted grid character."
        ),
        "alphabet_size": len(alphabet),
        "n_messages": int(len(messages)),
        "mean_structural_oog_char_frac": float(structural.mean()),
        "median_structural_oog_char_frac": float(structural.median()),
        "empirical_check_available": False,
        "empirical_mean_oog_passthrough_frac": None,
        "empirical_source": None,
        "note": (
            "mean_structural_oog_char_frac is the exact, corruption-draw-"
            "independent fraction of characters in the cached message sample "
            "that are outside the 36-cell grid. empirical_mean_oog_passthrough_"
            "frac (when available) is the fraction of characters the cached "
            "exposure artifact recorded as passing through COMPLETELY "
            "untouched (n_passthrough_chars); it is expected to sit slightly "
            "BELOW the structural figure, since some out-of-grid characters "
            "are unlucky enough to be touched by the indel branch even though "
            "they can never be substituted."
        ),
    }

    if exposure_path is not None and Path(exposure_path).exists():
        exp = pd.read_parquet(exposure_path)
        nonzero = exp[exp["cer_target"] > 0].copy()
        if len(nonzero):
            text_len = nonzero["text"].str.len().replace(0, np.nan)
            pass_frac = (nonzero["n_passthrough_chars"] / text_len).dropna()
            out["empirical_check_available"] = True
            out["empirical_mean_oog_passthrough_frac"] = float(pass_frac.mean())
            out["empirical_source"] = str(exposure_path)
            out["empirical_n_rows_checked"] = int(len(pass_frac))

    return out


def row_column_confusion_structure(
    confusion: np.ndarray, alphabet=GRID_ALPHABET, n_rows: int = 6, n_cols: int = 6
) -> dict:
    """Decompose the confusion matrix's off-diagonal mass into same-row,
    same-column, and neither, vs. what a structure-free (uniform-over-
    other-cells) confusion would give by chance.

    Args:
        confusion: row-normalized A x A confusion matrix (rows/cols in
            `alphabet`'s order, the fixed row-major 6x6 grid layout --
            see `idrift.data.grid`).
        alphabet: the grid alphabet (defines A and row-major position).
        n_rows, n_cols: grid dimensions (6x6 for the canonical speller).

    Returns:
        dict: `{off_diag_mass_same_row_frac, off_diag_mass_same_col_frac,
        off_diag_mass_neither_frac, uniform_expected_same_row_frac,
        uniform_expected_same_col_frac, uniform_expected_neither_frac,
        grid_shape, note}`. The three `off_diag_mass_*_frac` values sum to
        1.0 (they partition all off-diagonal mass); likewise the three
        `uniform_expected_*_frac` values.
    """
    A = len(alphabet)
    if confusion.shape != (A, A):
        raise ValueError(f"confusion shape {confusion.shape} != ({A}, {A})")
    if A != n_rows * n_cols:
        raise ValueError(f"alphabet size {A} != n_rows*n_cols ({n_rows*n_cols})")

    row_idx = np.arange(A) // n_cols
    col_idx = np.arange(A) % n_cols
    same_row = row_idx[:, None] == row_idx[None, :]
    same_col = col_idx[:, None] == col_idx[None, :]
    off_diag = ~np.eye(A, dtype=bool)

    off_diag_mass = float(confusion[off_diag].sum())
    same_row_mass = float(confusion[off_diag & same_row].sum())
    same_col_mass = float(confusion[off_diag & same_col].sum())
    neither_mass = off_diag_mass - same_row_mass - same_col_mass

    n_off_diag_cells = int(off_diag.sum())
    n_same_row_cells = int((off_diag & same_row).sum())
    n_same_col_cells = int((off_diag & same_col).sum())
    n_neither_cells = n_off_diag_cells - n_same_row_cells - n_same_col_cells

    def _safe_div(a, b):
        return float(a / b) if b else None

    return {
        "off_diag_mass_same_row_frac": _safe_div(same_row_mass, off_diag_mass),
        "off_diag_mass_same_col_frac": _safe_div(same_col_mass, off_diag_mass),
        "off_diag_mass_neither_frac": _safe_div(neither_mass, off_diag_mass),
        "uniform_expected_same_row_frac": _safe_div(n_same_row_cells, n_off_diag_cells),
        "uniform_expected_same_col_frac": _safe_div(n_same_col_cells, n_off_diag_cells),
        "uniform_expected_neither_frac": _safe_div(n_neither_cells, n_off_diag_cells),
        "grid_shape": [n_rows, n_cols],
        "note": (
            "off_diag_mass_* partitions the empirical pooled confusion matrix's "
            "off-diagonal probability mass by grid adjacency; uniform_expected_* "
            "is the same partition's null (structure-free/uniform-over-other-"
            "cells) expectation. same_row/same_col enrichment above the uniform "
            "expectation is evidence the per-character substitution step "
            "already encodes empirical P300 row/column confusability; it does "
            "not address independence ACROSS character positions within a "
            "message (see independence_assumption_note)."
        ),
    }


def heldout_participant_sensitivity(
    messages: pd.DataFrame,
    pooled_confusion: np.ndarray,
    subject_confusions: dict,
    *,
    cer_grid=CER_GRID,
    n_rep: int = DEFAULT_N_REP,
    alphabet=GRID_ALPHABET,
    seed: int = 0,
) -> dict:
    """Descriptive held-out-participant-matrix sensitivity for realized CER.

    Re-runs `idrift.data.noise_model.inject` (unmodified, cached source
    code) on the SAME cached message texts, once with `pooled_confusion`
    and once with EACH per-participant matrix in `subject_confusions`, and
    reports how much the realized-CER distribution would have differed had
    a single held-out participant's matrix been used instead of the pooled
    cohort matrix. Labels/messages are unchanged; only the confusion matrix
    varies. No LLM inference and no re-materialization of any matrix is
    performed here.

    Args:
        messages: DataFrame with `message_id`, `text` columns (the cached
            AUTH sample).
        pooled_confusion: the pooled A x A confusion matrix (`confusion_
            overall.npy`).
        subject_confusions: dict mapping participant key -> A x A confusion
            matrix (`confusion_by_subject.npz`, one entry per participant).
        cer_grid: target CER levels to sweep (0.0 excluded: matrix-
            independent no-op).
        n_rep: independently seeded replicates per (message, cer_target,
            matrix).
        alphabet: the grid alphabet shared by every matrix.
        seed: master seed folded into every stable per-draw sub-seed.

    Returns:
        dict: `{compute_gated: False, method, n_participants, n_messages,
        cer_grid, n_rep, by_cer_target: {level_str: {
        pooled_mean_realized_cer, heldout_participant_mean_realized_cer,
        heldout_participant_min, heldout_participant_max,
        heldout_participant_sd, max_abs_diff_vs_pooled,
        max_rel_diff_vs_pooled_pct}}, overall_max_abs_diff_vs_pooled, note}`.
    """
    texts = [
        (str(mid), str(text).upper())
        for mid, text in zip(messages["message_id"], messages["text"])
    ]

    def _mean_realized_cer_by_target(confusion: np.ndarray, tag: str) -> dict:
        sums = {ct: 0.0 for ct in cer_grid}
        n = {ct: 0 for ct in cer_grid}
        for mid, text in texts:
            for ct in cer_grid:
                for rep in range(n_rep):
                    s = _stable_seed("corruption_profile", tag, mid, ct, rep, seed)
                    noisy = inject(text, ct, confusion, alphabet, seed=s)
                    sums[ct] += measured_cer(text, noisy)
                    n[ct] += 1
        return {ct: sums[ct] / n[ct] for ct in cer_grid}

    pooled_means = _mean_realized_cer_by_target(pooled_confusion, "pooled")

    per_participant_means = {
        subj: _mean_realized_cer_by_target(conf, subj)
        for subj, conf in subject_confusions.items()
    }

    by_cer_target = {}
    overall_max_abs_diff = 0.0
    for ct in cer_grid:
        vals = np.array([per_participant_means[s][ct] for s in subject_confusions])
        pooled_val = pooled_means[ct]
        abs_diffs = np.abs(vals - pooled_val)
        max_abs_diff = float(abs_diffs.max())
        overall_max_abs_diff = max(overall_max_abs_diff, max_abs_diff)
        by_cer_target[f"{ct:.1f}"] = {
            "pooled_mean_realized_cer": float(pooled_val),
            "heldout_participant_mean_realized_cer": float(vals.mean()),
            "heldout_participant_min": float(vals.min()),
            "heldout_participant_max": float(vals.max()),
            "heldout_participant_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "max_abs_diff_vs_pooled": max_abs_diff,
            "max_rel_diff_vs_pooled_pct": (
                float(100.0 * max_abs_diff / pooled_val) if pooled_val > 0 else None
            ),
        }

    return {
        "compute_gated": False,
        "method": (
            "Re-applied idrift.data.noise_model.inject (unmodified) to the cached "
            "AUTH message sample, once per participant's cached confusion matrix "
            "(output/intermediate/confusion_by_subject.npz) and once with the "
            "pooled matrix (confusion_overall.npy), at matched cer_target levels "
            "with the same n_rep replicate count; message texts and cer_target "
            "levels are identical across matrices, only the confusion matrix "
            "varies. Reports the spread of per-participant mean realized CER "
            "around the pooled-matrix mean as a descriptive bound on matrix "
            "sensitivity."
        ),
        "n_participants": len(subject_confusions),
        "n_messages": len(texts),
        "cer_grid": list(cer_grid),
        "n_rep": n_rep,
        "by_cer_target": by_cer_target,
        "overall_max_abs_diff_vs_pooled": overall_max_abs_diff,
        "note": (
            "Descriptive bound only (reviewer major #10): this quantifies how "
            "differently the SAME messages would have been corrupted had a "
            "single held-out participant's own selection-error profile been "
            "used instead of the cohort-pooled matrix; it does not re-run the "
            "labeling/exposure pipeline itself, which would require full "
            "regeneration and is out of scope for this task."
        ),
    }


def run(
    intermediate_dir: str | Path = DEFAULT_INTERMEDIATE,
    out_path: str | Path = DEFAULT_OUT,
    *,
    n_rep: int = DEFAULT_N_REP,
    seed: int = 0,
) -> dict:
    """Build the corruption-profile digest and write it to `out_path`.

    Args:
        intermediate_dir: directory holding the cached confusion-matrix
            and AUTH-sample artifacts (`output/intermediate` by default).
        out_path: path to write the JSON digest to
            (`output/corruption_profile_digest.json` by default).
        n_rep: replicate count for `heldout_participant_sensitivity`.
        seed: master seed for the sensitivity sweep.

    Returns:
        dict: the full digest (see module docstring for the four
        component analyses). Top-level keys include `insertion_frac`,
        `deletion_frac`, `substitution_frac` (sum to 1.0), `n_participants`,
        `n_studies`, and `heldout_matrix_sensitivity`.
    """
    d = Path(intermediate_dir)

    pooled_confusion = np.load(d / "confusion_overall.npy")
    subj_npz = np.load(d / "confusion_by_subject.npz")
    subject_confusions = {k: subj_npz[k] for k in subj_npz.files}

    grid_alphabet = json.loads((d / "grid_alphabet.json").read_text())
    if grid_alphabet != GRID_ALPHABET:
        raise ValueError(
            "materialized grid_alphabet.json does not match idrift.data.grid.GRID_ALPHABET"
        )

    provenance_path = d / "provenance.json"
    cm_provenance = {}
    if provenance_path.exists():
        cm_provenance = json.loads(provenance_path.read_text()).get("confusion_matrix", {})

    studies = sorted({k.split("_")[0] for k in subject_confusions})
    per_study_n_participants = {
        s: sum(1 for k in subject_confusions if k.startswith(s + "_")) for s in studies
    }
    n_participants = len(subject_confusions)
    n_studies = len(studies)

    # Cross-check against the materialization's own provenance record (if
    # present); a mismatch would mean the cached artifacts are stale
    # relative to each other, which must be surfaced, not silently papered
    # over by trusting one source blindly.
    if cm_provenance.get("n_subjects") not in (None, n_participants):
        raise ValueError(
            f"n_participants mismatch: confusion_by_subject.npz has {n_participants} "
            f"entries but provenance.json records n_subjects={cm_provenance.get('n_subjects')}"
        )
    if cm_provenance.get("studies") not in (None, studies):
        raise ValueError(
            f"studies mismatch: confusion_by_subject.npz keys imply {studies} but "
            f"provenance.json records studies={cm_provenance.get('studies')}"
        )

    alloc = configured_allocation()

    messages = pd.read_csv(d / "auth_sample.csv")

    exposure_path = d / "exposure_v2.parquet"
    passthrough = grid_passthrough_summary(
        messages, exposure_path=exposure_path if exposure_path.exists() else None
    )

    rc_structure = row_column_confusion_structure(pooled_confusion, GRID_ALPHABET)

    sensitivity = heldout_participant_sensitivity(
        messages, pooled_confusion, subject_confusions, n_rep=n_rep, seed=seed
    )

    digest = {
        "insertion_frac": alloc["insertion_frac"],
        "deletion_frac": alloc["deletion_frac"],
        "substitution_frac": alloc["substitution_frac"],
        "ins_del_rate": alloc["ins_del_rate"],
        "allocation_source": alloc["source"],
        "allocation_note": alloc["note"],
        "n_participants": n_participants,
        "n_studies": n_studies,
        "studies": studies,
        "per_study_n_participants": per_study_n_participants,
        "confusion_matrix_provenance": cm_provenance,
        "grid_passthrough": passthrough,
        "row_column_confusion_structure": rc_structure,
        "independence_assumption_note": _INDEPENDENCE_NOTE,
        "heldout_matrix_sensitivity": sensitivity,
        "notes": (
            "Reviewer major #10 (corruption-model transparency + sensitivity). "
            "Reads cached confusion-matrix artifacts, the cached AUTH message "
            "sample, and idrift.data.noise_model's source directly; no new LLM "
            "inference and no re-materialization of any confusion matrix from "
            "the source EDFs is performed here."
        ),
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2, default=_default))

    print(
        f"corruption_profile: ins_del_rate={alloc['ins_del_rate']}, "
        f"sub/ins/del fracs={alloc['substitution_frac']}/{alloc['insertion_frac']}/{alloc['deletion_frac']}, "
        f"n_participants={n_participants}, n_studies={n_studies}, "
        f"heldout sensitivity overall_max_abs_diff={sensitivity['overall_max_abs_diff_vs_pooled']:.4f}"
    )

    return digest


if __name__ == "__main__":
    run()
