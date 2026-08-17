"""Confidence-calibration metrics: ECE, confidence-vs-faithfulness AUROC,
and reliability curves.

Model confidence is captured two different ways upstream (Task 8): a
verbalized 0-100 confidence the model states in its own output, or a
0-1 value derived from token logprobs. Every metric here normalizes via
`_norm` so both scales are handled transparently.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _norm(conf):
    """Rescale confidence to [0, 1], detecting a 0-100 input scale.

    Args:
        conf: array-like confidence values, either already in [0, 1]
            (logprob-derived) or on a 0-100 scale (verbalized).

    Returns:
        np.ndarray: confidence values in [0, 1].
    """
    conf = np.asarray(conf, dtype=float)
    return conf / 100 if conf.max() > 1 else conf


def ece(conf, correct, bins=10):
    """Expected Calibration Error: bin-weighted |accuracy - confidence| gap.

    Args:
        conf: array-like confidence values (0-100 or 0-1; auto-normalized).
        correct: array-like binary correctness indicator, same length as
            `conf`.
        bins: number of equal-width bins spanning [0, 1] confidence.

    Returns:
        float: the expected calibration error, in [0, 1]. Bins with no
            observations contribute zero.
    """
    c = _norm(conf)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Right edge of the last bin is inclusive so a confidence of
        # exactly 1.0 lands in a bin instead of being silently dropped.
        m = (c >= lo) & (c <= hi if hi == 1 else c < hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - c[m].mean())
    return float(e)


def auroc_conf_faithful(conf, is_faithful):
    """AUROC of confidence as a predictor of a faithful (non-drift) output.

    Answers the calibration question this study cares about: can the
    model's own stated confidence tell you when its output is trustworthy?

    Args:
        conf: array-like confidence values (0-100 or 0-1; auto-normalized).
        is_faithful: array-like binary label, 1 = faithful output.

    Returns:
        float: AUROC, or NaN if `is_faithful` has fewer than 2 distinct
            classes (AUROC is undefined for a single-class outcome).
    """
    y = np.asarray(is_faithful, dtype=int)
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, _norm(conf)))


def reliability(conf, correct, bins=10):
    """Reliability-curve table: mean accuracy vs. mean confidence per bin.

    Args:
        conf: array-like confidence values (0-100 or 0-1; auto-normalized).
        correct: array-like binary correctness indicator, same length as
            `conf`.
        bins: number of equal-width bins spanning [0, 1] confidence.

    Returns:
        DataFrame: one row per non-empty bin, with columns `bin_mid` (bin
            midpoint), `acc` (mean correctness in the bin), `conf` (mean
            confidence in the bin), `n` (row count in the bin).
    """
    c = _norm(conf)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Same inclusive-last-bin fix as `ece`, so confidence == 1.0 is
        # counted rather than silently vanishing from every bin's `n`.
        m = (c >= lo) & (c <= hi if hi == 1 else c < hi)
        if m.sum():
            rows.append(
                {
                    "bin_mid": (lo + hi) / 2,
                    "acc": correct[m].mean(),
                    "conf": c[m].mean(),
                    "n": int(m.sum()),
                }
            )
    return pd.DataFrame(rows)
