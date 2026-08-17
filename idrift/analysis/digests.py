"""Serialize the manuscript's two JSON "digest" artifacts.

`write_digests` performs no analysis of its own -- it is a thin
serializer. Everything it writes (drift curves, critical thresholds with
CIs, per-model calibration/reliability, model-class contrasts with
BH-corrected p-values, and the adjudication validity table) is computed
upstream by `idrift.analysis.drift_curve`, `idrift.analysis.calibration`,
`idrift.analysis.contrasts`, and `idrift.adjudicate.reconcile`, then handed
in here as a single `results` dict with two top-level buckets:

    results = {
        "results_digest": {...},  # descriptive: curve_by_class, critical
                                   # threshold(s), calibration/reliability
                                   # tables, taxonomy shares, adjudication
                                   # validity -- everything Figure 1 and the
                                   # manuscript text quote directly.
        "stats_digest": {...},    # inferential: bootstrap CIs, contrast
                                   # diffs, raw + BH-corrected p-values.
    }

Either bucket may be omitted (defaults to an empty object in the written
file), so a partial digest -- e.g. calibration written before contrasts
are ready -- is allowed. This dict is the manuscript's single source of
truth: prose, tables, and figures should read numbers from the written
JSON rather than recomputing them.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _default(obj):
    """`json.dumps(..., default=...)` fallback for numpy/pandas types.

    Plain Python floats/ints/tuples/dicts already round-trip through the
    stdlib `json` encoder (and `np.float64` happens to subclass `float`,
    so it does too) -- this only has to cover the types that don't:
    `pandas.DataFrame`/`Series` (e.g. a `calibration.reliability()` table
    embedded in the digest) and `numpy.integer`/`bool_`/`ndarray`.
    """
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _out_dir(out_dir=None) -> Path:
    d = Path(out_dir) if out_dir is not None else Path(os.environ.get("IDRIFT_OUTPUT", "output"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_digests(results: dict, out_dir=None):
    """Write `results` to `results_digest.json` + `stats_digest.json`.

    Args:
        results: dict with optional "results_digest" and "stats_digest"
            keys (see module docstring for the expected contents of each).
            A missing key is written as an empty JSON object.
        out_dir: directory to write into. Defaults to the `IDRIFT_OUTPUT`
            env var, or "output" if unset -- mirrors the
            `IDRIFT_INTERMEDIATE` convention in `idrift.lib.io_utils`.

    Returns:
        tuple[Path, Path]: (path to results_digest.json, path to
            stats_digest.json).
    """
    d = _out_dir(out_dir)
    results_path = d / "results_digest.json"
    stats_path = d / "stats_digest.json"
    results_path.write_text(
        json.dumps(results.get("results_digest", {}), indent=2, default=_default, sort_keys=True)
    )
    stats_path.write_text(
        json.dumps(results.get("stats_digest", {}), indent=2, default=_default, sort_keys=True)
    )
    return results_path, stats_path
