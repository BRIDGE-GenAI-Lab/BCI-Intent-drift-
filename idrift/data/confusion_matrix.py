"""Row-normalized confusion-matrix builder and overall character-error-rate.

Symbol-type agnostic: `pairs` may carry any hashable symbol type (str
characters for synthetic/unit tests, or int grid-stimulus codes as emitted
by `idrift.data.bigp3_adapter.extract_pairs`). The alphabet passed in must
use the same symbol type as the pairs.
"""
import numpy as np


def build_confusion(pairs, alphabet):
    """Build an A x A row-normalized confusion matrix P(selected | intended).

    pairs: iterable of (intended, selected[, ...]) tuples (extra trailing
        elements, e.g. a subject id, are ignored).
    alphabet: ordered list of symbols defining matrix rows/columns.
    """
    idx = {c: i for i, c in enumerate(alphabet)}
    A = len(alphabet)
    M = np.zeros((A, A))
    for intended, selected in ((p[0], p[1]) for p in pairs):
        if intended in idx and selected in idx:
            M[idx[intended], idx[selected]] += 1
    rs = M.sum(1, keepdims=True)
    rs[rs == 0] = 1.0
    return M / rs


def overall_cer(pairs) -> float:
    """Fraction of pairs where selected != intended."""
    pairs = list(pairs)
    if not pairs:
        return 0.0
    return sum(1 for p in pairs if p[0] != p[1]) / len(pairs)
