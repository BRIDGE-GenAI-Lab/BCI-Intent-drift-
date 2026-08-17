"""Exposure-matrix builder: expand a message corpus across a CER grid, a
set of source subjects (confusion matrices), and repeated noise seeds.

Determinism note: the per-row noise seed MUST be derived from a stable
hash. Python's built-in `hash()` on str/tuple is salted per interpreter
process (PYTHONHASHSEED randomization), so `hash(...) % (2**31)` would
silently change on every run and violate the pipeline's determinism
constraint. `_stable_seed` uses sha256 instead, which is fixed across
processes, machines, and Python versions.
"""
import hashlib

import pandas as pd

from idrift.data.noise_model import inject
from idrift.lib.cer import cer


def _stable_seed(*parts) -> int:
    """Derive a reproducible non-negative 32-bit seed from `parts`.

    Unlike `hash()`, this is stable across interpreter runs (no
    PYTHONHASHSEED salt), across machines, and across Python versions.
    """
    key = "|".join(str(p) for p in parts).encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def build_exposure(corpus_df, confusion_by_subject, cer_grid, n_seeds, alphabet):
    """Expand `corpus_df` into one row per (message, source_subject, cer_target, seed).

    corpus_df: DataFrame with at least `message_id`, `intended_text` (extra
        columns, e.g. `corpus`/`category`, pass through onto every row).
    confusion_by_subject: dict mapping a subject id -> row-normalized
        confusion matrix (as consumed by `idrift.data.noise_model.inject`).
    cer_grid: iterable of target CER values.
    n_seeds: number of independent noise draws per (message, subject, cer_target).
    alphabet: ordered symbol alphabet matching the confusion matrices.

    Returns a DataFrame carrying every corpus_df column plus `cer_target`,
    `seed`, `source_subject`, `noisy_text`, `actual_cer`.
    """
    rows = []
    for _, m in corpus_df.iterrows():
        for subj, conf in confusion_by_subject.items():
            for ct in cer_grid:
                for s in range(n_seeds):
                    seed = _stable_seed(m.message_id, subj, ct, s)
                    noisy = inject(m.intended_text, ct, conf, alphabet, seed=seed)
                    rows.append({
                        **m.to_dict(),
                        "cer_target": ct,
                        "seed": s,
                        "source_subject": subj,
                        "noisy_text": noisy,
                        "actual_cer": cer(m.intended_text, noisy),
                    })
    return pd.DataFrame(rows)
