"""Adapter: turn the companion bigP3BCI-ALS-calibration study's reconstructed
online-spelling trials into (intended, selected, subject) triples for
`idrift.data.confusion_matrix.build_confusion` / `overall_cer`.

This module only *imports* the companion study's code (never modifies it).
Companion repo: ../study_bigp3_als_calibration (src/bigp3_als/{edf,trials}.py).

Real companion API used here (confirmed by reading the companion source,
not guessed):
    - `bigp3_als.trials.build_online_trial_table(cache_path) -> pd.DataFrame`
      walks every Test-phase EDF under `cache_path` and reconstructs one row
      per feedback-phase (phase 3) trial, recovering the BCI's *target*
      (intended) and *selected* grid stimulus codes from the recorded event
      streams. Columns include: target, selected, correct, eligible,
      exclusion_reason, study, participant_id, study_participant_id,
      session_id, condition, relative_path.
    - `bigp3_als.edf.select_edf_paths(cache_path) -> list[Path]`

IMPORTANT -- integer code space, not characters:
`target`/`selected` are INTEGER 6x6-grid stimulus codes (P300 speller row/
column codes), NOT decoded characters. Mapping those integer codes to the
spelled character is a separate, downstream concern (the grid layout is a
protocol detail of the companion study) and is intentionally NOT done in
this module -- `build_confusion` is symbol-type-agnostic and operates
directly in integer-code space here.

IMPORTANT -- cache_path convention:
`build_online_trial_table`/`select_edf_paths` must be called with the
*parent* of the `bigP3BCI-data` directory (e.g.
".../study_bigp3_als_calibration/data/source_cache"), not the
`bigP3BCI-data` directory itself. The companion's `parse_source_path`
requires `bigP3BCI-data` to be the first component of each path *relative
to cache_path*; passing the `bigP3BCI-data` directory itself as cache_path
strips that component and every path fails to parse. Verified directly
against the on-disk cache (see task-3-report.md).

IMPORTANT -- study code convention:
Companion source paths encode the study segment as "StudyB"/"StudyF"/
"StudyL"/"StudyN" (the `study` column holds these full strings), not the
bare letters "B"/"F"/"L"/"N". This adapter's public `studies` parameter
accepts either form (bare letters, matching the rest of this pipeline, or
the companion's native "StudyX" strings) and normalizes before filtering.
"""
import os
import sys
from pathlib import Path

COMPANION_SRC = Path(os.environ.get("IDRIFT_COMPANION_SRC", "/Volumes/Extreme SSD/Mimic-IV/study_bigp3_als_calibration/src"))
if not COMPANION_SRC.exists():
    raise FileNotFoundError(f"Companion loader not found at {COMPANION_SRC}; set IDRIFT_COMPANION_SRC")
if str(COMPANION_SRC) not in sys.path:
    sys.path.insert(0, str(COMPANION_SRC))

from bigp3_als.trials import build_online_trial_table  # noqa: E402

# Companion cache root as it must be passed to build_online_trial_table /
# select_edf_paths (the *parent* of bigP3BCI-data -- see module docstring).
DEFAULT_ARCHIVE_ROOT = Path(os.environ.get("IDRIFT_BIGP3_ROOT", "/Volumes/Extreme SSD/Mimic-IV/study_bigp3_als_calibration/data/source_cache"))


def _normalize_study_code(value: str) -> str:
    """Strip an optional 'Study' prefix, e.g. 'StudyF' -> 'F'."""
    return value[len("Study"):] if value.startswith("Study") else value


def extract_pairs(archive_root, studies=("F", "L", "N")):
    """Return (intended_code, selected_code, subject) triples for eligible
    Test-phase selections in the requested studies.

    archive_root: cache_path as expected by the companion's
        `build_online_trial_table` -- the parent of `bigP3BCI-data`
        (see `DEFAULT_ARCHIVE_ROOT`).
    studies: bare study letter codes, e.g. ("F","L","N") (default); the
        companion's native "StudyF" etc. strings are also accepted.

    `subject` is the companion's `participant_id` (already of the form
    "F_03", i.e. study letter + participant number) -- this matches the
    `source_subject` convention used by the rest of the intent-drift
    pipeline (see docs/superpowers/plans/2026-07-19-intent-drift.md).
    """
    wanted = {_normalize_study_code(s) for s in studies}
    trials = build_online_trial_table(Path(archive_root))
    eligible = trials[trials["eligible"]]
    eligible = eligible[eligible["study"].map(_normalize_study_code).isin(wanted)]
    return [
        (int(row.target), int(row.selected), row.participant_id)
        for row in eligible.itertuples(index=False)
    ]
