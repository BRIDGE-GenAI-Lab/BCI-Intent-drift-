"""Materialize the REAL ALS-P300 confusion matrix from bigP3BCI Studies F/L/N.

Walks every EDF in the companion source cache, keeps eligible Test-phase
selections from the three studies that reference the canonical 6x6 speller
(StudyF, StudyL, StudyN -- StudyB is a different device/protocol and is
intentionally excluded), reconstructs each trial's (target, selected) grid
stimulus code pair via the companion's own EDF/event reconstruction, maps
codes to characters via the fixed 6x6 layout (`idrift.data.grid`), and
materializes:

  - output/intermediate/confusion_overall.npy     36x36 row-normalized
                                                    P(selected_char | target_char)
  - output/intermediate/confusion_by_subject.npz  one 36x36 array per subject
                                                    (same fixed alphabet/order)
  - output/intermediate/grid_alphabet.json        the 36-char alphabet, in order

Files that fail to parse (unexpected path shape) or fail to reconstruct
(missing trial-stream channels, corrupt/unreadable EDF, etc.) are SKIPPED and
counted -- a single bad file never aborts the run, and skips are never
silently dropped from the provenance record.

This module only *imports* the companion study's code (never modifies it);
see `idrift.data.bigp3_adapter` for the sys.path wiring this reuses.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Importing bigp3_adapter (not just for DEFAULT_ARCHIVE_ROOT) also runs its
# module-level sys.path insertion for the companion's `src`, so the direct
# `bigp3_als.*` imports below resolve without duplicating that wiring here.
from idrift.data.bigp3_adapter import DEFAULT_ARCHIVE_ROOT
from idrift.data.confusion_matrix import build_confusion, overall_cer
from idrift.data.grid import GRID_ALPHABET, code_to_char
from idrift.lib.io_utils import _dir, log_provenance

from bigp3_als.edf import parse_source_path, select_edf_paths  # noqa: E402
from bigp3_als.trials import reconstruct_edf_trials  # noqa: E402

# StudyB uses a different device/grid and is out of scope here (see grid.py).
WANTED_STUDIES = ("StudyF", "StudyL", "StudyN")


def _sanitize_key(subject: str) -> str:
    """Make a subject id safe as an .npz array name (e.g. 'StudyF:F_03' -> 'StudyF_F_03')."""
    return re.sub(r"[^A-Za-z0-9_]", "_", subject)


def collect_eligible_rows(root: Path):
    """Walk every EDF under `root`; keep eligible Test-phase F/L/N selections.

    Returns (rows, n_files_used, n_files_skipped) where each row is
    (target_code, selected_code, study_participant_id, study, condition) --
    all ints/strs, straight off the companion's reconstructed trial table.
    """
    rows: list[tuple[int, int, str, str, str]] = []
    n_files_used = 0
    n_files_skipped = 0
    for edf_path in select_edf_paths(root):
        relative_path = edf_path.relative_to(root).as_posix()
        try:
            source = parse_source_path(relative_path)
        except ValueError:
            n_files_skipped += 1
            continue
        if source.phase != "Test" or source.study not in WANTED_STUDIES:
            continue
        try:
            trial_df = reconstruct_edf_trials(edf_path, relative_path)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any bad
            # file (missing channels, corrupt EDF, mne read failure, ...) is
            # skipped and counted, never a hard crash mid-run.
            print(f"SKIP (reconstruct failed): {relative_path}: {exc}")
            n_files_skipped += 1
            continue
        n_files_used += 1
        eligible = trial_df[trial_df["eligible"]]
        for row in eligible.itertuples(index=False):
            rows.append(
                (
                    int(row.target),
                    int(row.selected),
                    row.study_participant_id,
                    row.study,
                    row.condition,
                )
            )
    return rows, n_files_used, n_files_skipped


def _to_char_pairs(rows):
    """Convert (target_code, selected_code, ...) rows to (target_char, selected_char) pairs.

    Codes outside 1..36 (should not occur for a 6x6 grid; see grid.py) are
    dropped defensively and counted rather than raising, so a single stray
    code cannot crash a multi-minute run.
    """
    pairs = []
    n_out_of_grid = 0
    for target_code, selected_code, *_ in rows:
        try:
            pairs.append((code_to_char(target_code), code_to_char(selected_code)))
        except ValueError:
            n_out_of_grid += 1
    return pairs, n_out_of_grid


def _top_confusions(char_pairs, n=10):
    """Top-N off-diagonal (intended -> selected) confusions, by raw count."""
    counts = Counter(char_pairs)
    intended_totals = Counter(t for t, _ in char_pairs)
    off_diag = [(t, s, c) for (t, s), c in counts.items() if t != s]
    off_diag.sort(key=lambda row: row[2], reverse=True)
    out = []
    for t, s, c in off_diag[:n]:
        total = intended_totals[t]
        rate = c / total if total else 0.0
        out.append((t, s, c, rate))
    return out


def main():
    root = DEFAULT_ARCHIVE_ROOT
    print(f"Scanning companion source cache: {root}")
    rows, n_files_used, n_files_skipped = collect_eligible_rows(root)

    char_pairs, n_out_of_grid = _to_char_pairs(rows)
    if n_out_of_grid:
        print(f"WARNING: dropped {n_out_of_grid} pairs with out-of-grid codes (expected 1..36)")

    n_total_eligible = len(rows)
    per_study_eligible = {s: sum(1 for r in rows if r[3] == s) for s in WANTED_STUDIES}

    overall_cer_value = overall_cer(rows)
    per_study_cer = {
        s: overall_cer([r for r in rows if r[3] == s]) for s in WANTED_STUDIES
    }

    # Overall confusion matrix (all subjects pooled).
    overall_matrix = build_confusion(char_pairs, GRID_ALPHABET)

    # Per-subject confusion matrices, same fixed alphabet -> all 36x36, aligned.
    rows_by_subject = defaultdict(list)
    for target_code, selected_code, subject, study, condition in rows:
        rows_by_subject[subject].append((target_code, selected_code))
    by_subject_matrix = {}
    for subject, subject_rows in rows_by_subject.items():
        subject_char_pairs, _ = _to_char_pairs(subject_rows)
        by_subject_matrix[subject] = build_confusion(subject_char_pairs, GRID_ALPHABET)

    n_subjects = len(by_subject_matrix)

    # --- Save artifacts ---
    out_dir = _dir()
    np.save(out_dir / "confusion_overall.npy", overall_matrix)
    np.savez(
        out_dir / "confusion_by_subject.npz",
        **{_sanitize_key(subject): matrix for subject, matrix in by_subject_matrix.items()},
    )
    (out_dir / "grid_alphabet.json").write_text(json.dumps(GRID_ALPHABET, indent=2))

    log_provenance(
        {
            "confusion_matrix": {
                "n_total_eligible_selections": n_total_eligible,
                "per_study_eligible_selections": per_study_eligible,
                "n_subjects": n_subjects,
                "overall_cer": overall_cer_value,
                "per_study_cer": per_study_cer,
                "n_files_used": n_files_used,
                "n_files_skipped": n_files_skipped,
                "n_pairs_out_of_grid": n_out_of_grid,
                "grid": "6x6",
                "charmap": "".join(GRID_ALPHABET),
                "studies": list(WANTED_STUDIES),
            }
        }
    )

    top = _top_confusions(char_pairs, n=10)

    print("\n=== bigP3BCI F/L/N confusion-matrix materialization: summary ===")
    print(f"EDF files used:            {n_files_used}")
    print(f"EDF files skipped:         {n_files_skipped}")
    print(f"Total eligible selections: {n_total_eligible}")
    for s in WANTED_STUDIES:
        print(
            f"  {s}: eligible={per_study_eligible[s]:>6}  CER={per_study_cer[s]:.4f}"
        )
    print(f"Subjects:                  {n_subjects}")
    print(f"Overall CER:               {overall_cer_value:.4f}")
    print(f"Overall matrix shape:      {overall_matrix.shape}")
    print("Top off-diagonal confusions (intended -> selected, count, rate):")
    for t, s, c, rate in top:
        print(f"  {t!r:>4} -> {s!r:<4}  n={c:>4}  rate={rate:.3f}")
    print(f"\nSaved to: {out_dir.resolve()}")
    print("  confusion_overall.npy, confusion_by_subject.npz, grid_alphabet.json")


if __name__ == "__main__":
    main()
