"""Replicate-aware exposure generator with realized-CER and error-location
metadata (revision Task 1.2).

The original exposure pipeline (`idrift.data.build_exposure`) drew exactly
ONE corruption per (message, subject, cer_target, seed-index) and reported
only the target CER, not what was actually realized. Reviewers flagged both
as fatal: a single draw cannot estimate within-cell variability, and the
target CER is a knob, not a measurement. This module fixes both problems --
it draws `n_rep` INDEPENDENTLY seeded corruption replicates per
(message, cer_target), and records, per replicate, the ACTUALLY realized
CER (measured by edit distance against the reference text, not assumed
from the target) plus where and what kind of corruption happened.

Determinism: every replicate's corruption seed is a stable sha256-derived
hash of (message_id, cer_target, replicate_idx, master_seed) -- see
`_stable_seed`. Python's built-in `hash()` is salted per process
(PYTHONHASHSEED) and would silently break reproducibility across runs.

Realized CER measurement: `idrift.lib.cer.cer`, which is backed by the
`python-Levenshtein` package (already an explicit pyproject.toml
dependency, and already used this same way by the rest of the exposure
pipeline) -- not a separate dependency-free implementation, since one
already exists in this codebase for exactly this purpose.

Grid casing: the 36-symbol speller alphabet (`idrift.data.grid.GRID_ALPHABET`)
is uppercase letters/digits/space. Messages are uppercased before injection
(matching the convention already established in `build_exposure_run.py`,
which uppercases `intended_text` for the same reason): if the reference
text were left in mixed case, the lowercase letters that make up most of
ordinary English text would all read as "out-of-grid" and pass through
uncorrupted regardless of `cer_target`, silently deflating realized CER.
The `text` column in this module's output IS that uppercased reference
(the corruption/adjudication reference M), not the original display
casing.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from idrift.data.grid import GRID_ALPHABET
from idrift.data.noise_model import inject_with_metadata
from idrift.lib.cer import cer as measured_cer

# Alnum runs, with an optional following "'letters" contraction suffix kept
# attached (so "don't"/"I'm" tokenize as one token, not split on the
# apostrophe). Punctuation and whitespace are never tokens: a corrupted
# position that lands only on punctuation/whitespace never sets a
# content-word/negation/numeral flag.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

# Negation tokens (case-insensitive exact match, OR any token containing the
# "n't" contraction suffix so novel contractions not explicitly listed --
# e.g. "mustn't" -- are still caught).
_NEGATION_WORDS = {
    "not", "no", "never",
    "cannot", "can't", "don't", "didn't", "doesn't", "won't", "wouldn't",
    "couldn't", "shouldn't", "isn't", "wasn't", "aren't", "weren't",
    "haven't", "hasn't", "hadn't",
}

# Common English function words, EXCLUDING them from the "content word"
# count. Negation words ("not", "no") are deliberately included here too --
# they are surfaced via the dedicated `corrupted_negation` flag instead, so
# they are not double-counted as content words as well.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "am",
    "be", "been", "being", "to", "of", "in", "on", "at", "for", "with",
    "as", "by", "that", "this", "these", "those", "it", "i", "you", "he",
    "she", "we", "they", "my", "your", "his", "her", "its", "our", "their",
    "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "should", "not", "no",
}


def _stable_seed(message_id, cer_target, replicate_idx, master_seed) -> int:
    """Derive a reproducible non-negative integer sub-seed from
    (message_id, cer_target, replicate_idx, master_seed) via sha256.

    Same rationale as `idrift.data.build_exposure._stable_seed`: Python's
    built-in `hash()` is salted per interpreter process (PYTHONHASHSEED
    randomization) and would silently change across runs, violating the
    pipeline's determinism constraint. sha256 is fixed across processes,
    machines, and Python versions.
    """
    key = f"{message_id}|{cer_target}|{replicate_idx}|{master_seed}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def _tokenize_with_spans(text: str):
    """Return [(token_text, start, end)] for every token in `text`
    (`_TOKEN_RE`), spans in original-text character offsets."""
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _is_content_word(token: str) -> bool:
    """A token counts as a "content word" if it is not a stopword and is
    not purely numeric (numeral tokens are surfaced via the dedicated
    `corrupted_numeral` flag instead, not double-counted here)."""
    low = token.lower()
    return low not in _STOPWORDS and not token.isdigit()


def _corruption_flags(text: str, error_positions: list[int]):
    """Given the ORIGINAL `text` and the 0-based indices touched by a
    sub/ins/del operation, return
    (corrupted_content_words, corrupted_negation, corrupted_numeral).

    A token is "touched" if any corrupted position falls within its
    [start, end) span. `corrupted_content_words` counts each touched
    content-word OCCURRENCE once, even if multiple characters within that
    same occurrence were corrupted -- it is not a count of distinct word
    types across the whole message.
    """
    if not error_positions:
        return 0, False, False

    positions = set(error_positions)
    n_content = 0
    negation = False
    numeral = False
    for token, start, end in _tokenize_with_spans(text):
        if not any(p in positions for p in range(start, end)):
            continue
        low = token.lower()
        if low in _NEGATION_WORDS or "n't" in low:
            negation = True
        if any(c.isdigit() for c in token):
            numeral = True
        if _is_content_word(token):
            n_content += 1
    return n_content, negation, numeral


def build_exposure_v2(messages, cer_grid, n_rep, confusion, seed) -> pd.DataFrame:
    """Expand `messages` into one row per (message, cer_target, replicate).

    Args:
        messages: iterable of dict-like objects, each with at least
            `message_id` and `text` (a `corpus` key -- e.g. "AUTH"/"CRIT"/
            "CTRL" -- defaults to "AUTH" if absent).
        cer_grid: iterable of target CER values.
        n_rep: number of independently seeded corruption replicates drawn
            per (message, cer_target) cell. This is the reviewer-mandated
            fix for "one corruption draw per condition": every replicate
            gets its own stable sub-seed (`_stable_seed`), so repeated
            draws are independent, not copies of a single arbitrary draw.
        confusion: row-normalized confusion matrix aligned to
            `idrift.data.grid.GRID_ALPHABET` (the fixed 36-symbol grid
            alphabet; not a parameter here, since it must match `confusion`
            exactly and this module always corrupts against that grid).
        seed: master seed. Together with (message_id, cer_target,
            replicate_idx) it fully determines every replicate's noisy_text
            (see `_stable_seed`); the same four values always reproduce it.

    Returns a DataFrame with one row per (message, cer_target, replicate),
    columns: message_id, corpus, text, cer_target, replicate_idx, seed,
    noisy_text, realized_cer, n_errors, error_positions, sub_count,
    ins_count, del_count, corrupted_content_words, corrupted_negation,
    corrupted_numeral, n_passthrough_chars.

    `text` is the UPPERCASED reference (see module docstring); `seed` holds
    the per-row DERIVED sub-seed actually used to produce `noisy_text`
    (not the `seed` argument here, the master seed) -- so any single row is
    independently reproducible via
    `inject_with_metadata(row.text, row.cer_target, confusion,
    GRID_ALPHABET, seed=row.seed)` without recomputing the hash.
    `realized_cer` is a measurement (edit distance between `text` and
    `noisy_text`, divided by len(text)), not the `cer_target` knob --
    the reviewer-mandated "realized, message-level corruption is the
    primary predictor, not target CER".
    """
    rows = []
    for m in messages:
        message_id = m["message_id"]
        corpus = m["corpus"] if "corpus" in m else "AUTH"
        text = str(m["text"]).upper()

        for ct in cer_grid:
            for rep in range(n_rep):
                sub_seed = _stable_seed(message_id, ct, rep, seed)
                meta = inject_with_metadata(text, ct, confusion, GRID_ALPHABET, seed=sub_seed)
                noisy_text = meta["noisy_text"]
                realized_cer = measured_cer(text, noisy_text)
                n_content, negation, numeral = _corruption_flags(text, meta["error_positions"])

                rows.append({
                    "message_id": message_id,
                    "corpus": corpus,
                    "text": text,
                    "cer_target": ct,
                    "replicate_idx": rep,
                    "seed": sub_seed,
                    "noisy_text": noisy_text,
                    "realized_cer": realized_cer,
                    "n_errors": meta["sub_count"] + meta["ins_count"] + meta["del_count"],
                    "error_positions": meta["error_positions"],
                    "sub_count": meta["sub_count"],
                    "ins_count": meta["ins_count"],
                    "del_count": meta["del_count"],
                    "corrupted_content_words": n_content,
                    "corrupted_negation": negation,
                    "corrupted_numeral": numeral,
                    "n_passthrough_chars": meta["n_passthrough_chars"],
                })

    return pd.DataFrame(rows)


# --- Step 5 runner -----------------------------------------------------------

def _load_auth_messages() -> list[dict]:
    """Load the Task 1.1 prespecified stratified AUTH sample
    (`output/intermediate/auth_sample.csv`: message_id, text, category,
    length_bin) as AUTH-tagged messages."""
    from idrift.lib.io_utils import _dir

    df = pd.read_csv(_dir() / "auth_sample.csv")
    return [
        {"message_id": r.message_id, "corpus": "AUTH", "text": r.text}
        for r in df.itertuples(index=False)
    ]


def _load_crit_messages() -> list[dict]:
    """Load the frozen message-critical probe set
    (`output/intermediate/probe_set.json`) as CRIT-tagged messages, using
    each item's `intended_text` (the grounded reference message; consistent
    with the original exposure pipeline's `probe` corpus, which likewise
    corrupted `intended_text` and did not separately corrupt
    `dangerous_variant`)."""
    import json

    from idrift.lib.io_utils import _dir

    items = json.loads((_dir() / "probe_set.json").read_text())
    return [
        {"message_id": item["message_id"], "corpus": "CRIT", "text": item["intended_text"]}
        for item in items
    ]


def main(n_rep: int = 20, seed: int = 0) -> pd.DataFrame:
    """Build `output/intermediate/exposure_v2.parquet` for AUTH + CRIT x the
    prespecified 5-level CER grid x `n_rep` replicates.

    CTRL (the matched non-critical controls) is built by Task 1.3, not yet
    available at this task's implementation time; this run covers AUTH and
    CRIT only, and the printed summary says so explicitly so the gap is not
    silently left out of the record. A later run (Task 1.3+) appends CTRL
    rows to this same checkpoint.
    """
    import json as _json

    import numpy as np

    from idrift.data.corpus_sample import CER_GRID
    from idrift.lib.io_utils import _dir, log_provenance, save_checkpoint

    d = _dir()
    confusion = np.load(d / "confusion_overall.npy")
    grid_alphabet = _json.loads((d / "grid_alphabet.json").read_text())
    assert grid_alphabet == GRID_ALPHABET, "materialized grid_alphabet.json must match idrift.data.grid.GRID_ALPHABET"

    messages = _load_auth_messages() + _load_crit_messages()
    n_auth = sum(1 for m in messages if m["corpus"] == "AUTH")
    n_crit = sum(1 for m in messages if m["corpus"] == "CRIT")

    exposure = build_exposure_v2(messages, list(CER_GRID), n_rep=n_rep, confusion=confusion, seed=seed)
    save_checkpoint(exposure, "exposure_v2")

    realized = exposure.groupby("cer_target")["realized_cer"].agg(["mean", "std", "count"]).round(4)
    realized_summary = {
        str(k): {"mean": float(v["mean"]), "std": float(v["std"]), "n": int(v["count"])}
        for k, v in realized.to_dict("index").items()
    }

    log_provenance({
        "exposure_v2": {
            "rows": int(len(exposure)),
            "n_messages": int(len(messages)),
            "n_auth": n_auth,
            "n_crit": n_crit,
            "n_ctrl": 0,
            "ctrl_note": "CTRL (Task 1.3 matched non-critical controls) not yet built; appended in a later run.",
            "cer_grid": list(CER_GRID),
            "n_rep": n_rep,
            "master_seed": seed,
            "realized_cer_by_target": realized_summary,
        }
    })

    print(f"rows={len(exposure)} messages={len(messages)} (AUTH={n_auth} CRIT={n_crit} CTRL=0 [Task 1.3])")
    print("CTRL will be appended once Task 1.3 (matched non-critical controls) is built.")
    print("realized CER distribution by target (mean / std / n):")
    for ct in sorted(realized_summary):
        s = realized_summary[ct]
        print(f"  target={ct:<5} mean={s['mean']:.4f}  std={s['std']:.4f}  n={s['n']}")

    return exposure


if __name__ == "__main__":
    main()
