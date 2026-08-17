import numpy as np


def inject_with_metadata(text, cer_target, confusion, alphabet, seed, ins_del_rate=0.1):
    """Like `inject`, but also returns per-character corruption metadata.

    This reproduces `inject`'s exact corruption algorithm and RNG draw order
    step for step (same `text`/`cer_target`/`confusion`/`alphabet`/`seed`/
    `ins_del_rate` -> byte-identical `noisy_text` as `inject`), and
    additionally records, for every 0-based index `i` in the ORIGINAL
    `text`, whether that position was left alone or touched by a
    substitution/insertion/deletion. `inject` itself is intentionally left
    unmodified (a plain wrapper around this function would work too, but
    duplicating the ~25-line loop here keeps `inject`'s existing callers and
    tests provably untouched rather than depending on a refactor being
    exactly behavior-preserving).

    Returns a dict:
      noisy_text: str -- identical to `inject(...)` for the same arguments.
      sub_count, ins_count, del_count: int -- counts of each operation type
          actually applied by the corruption process.
      error_positions: sorted list[int] of original-text indices touched by
          a sub/ins/del (NOT plain passthrough). For an insertion, the
          position recorded is the original index the random character was
          inserted immediately before (that original character is itself
          still emitted, but the position counts as touched by the
          insertion event).
      n_passthrough_chars: int -- count of OUT-OF-GRID original characters
          (not in `alphabet`) that passed straight through completely
          uncorrupted. Out-of-grid characters are never eligible for
          substitution (the grid confusion matrix has no row for them) but
          CAN still be deleted or preceded by an insertion, since the
          indel branch does not check grid membership; this count only
          quantifies the untouched case, so the reviewer-flagged silent
          pass-through limitation is measured rather than hidden.
    """
    idx = {c: i for i, c in enumerate(alphabet)}

    if cer_target <= 0:
        n_oog = sum(1 for ch in text if ch not in idx)
        return {
            "noisy_text": text,
            "sub_count": 0,
            "ins_count": 0,
            "del_count": 0,
            "error_positions": [],
            "n_passthrough_chars": n_oog,
        }

    rng = np.random.default_rng(seed)
    sub_budget = cer_target * (1 - ins_del_rate)
    indel_budget = cer_target * ins_del_rate
    out = []
    sub_count = ins_count = del_count = 0
    error_positions = []
    n_passthrough_chars = 0

    for i, ch in enumerate(text):
        r = rng.random()
        if r < indel_budget:                                   # deletion or insertion
            if rng.random() < 0.5:
                del_count += 1
                error_positions.append(i)
                continue                                        # delete
            out.append(alphabet[rng.integers(len(alphabet))])
            out.append(ch)
            ins_count += 1
            error_positions.append(i)
        elif ch in idx and r < indel_budget + sub_budget:      # substitution from confusion row
            row = confusion[idx[ch]].copy()
            row[idx[ch]] = 0
            if row.sum() == 0:
                # Empty confusion row (char unseen / always-correct in the anchor data):
                # fall back to a uniform substitution to a DIFFERENT grid character so the
                # manipulated CER is still realized. Real confusion structure is used
                # wherever it exists; only sparse/unseen rows use the fallback.
                choices = [j for j in range(len(alphabet)) if j != idx[ch]]
                out.append(alphabet[choices[rng.integers(len(choices))]])
            else:
                out.append(alphabet[rng.choice(len(alphabet), p=row / row.sum())])
            sub_count += 1
            error_positions.append(i)
        else:
            out.append(ch)
            if ch not in idx:
                n_passthrough_chars += 1

    return {
        "noisy_text": "".join(out),
        "sub_count": sub_count,
        "ins_count": ins_count,
        "del_count": del_count,
        "error_positions": sorted(error_positions),
        "n_passthrough_chars": n_passthrough_chars,
    }


def inject(text, cer_target, confusion, alphabet, seed, ins_del_rate=0.1):
    if cer_target <= 0:
        return text
    rng = np.random.default_rng(seed)
    idx = {c: i for i, c in enumerate(alphabet)}
    sub_budget = cer_target * (1 - ins_del_rate)
    indel_budget = cer_target * ins_del_rate
    out = []
    for ch in text:
        r = rng.random()
        if r < indel_budget:                                   # deletion or insertion
            if rng.random() < 0.5:
                continue                                        # delete
            out.append(alphabet[rng.integers(len(alphabet))])
            out.append(ch)
        elif ch in idx and r < indel_budget + sub_budget:      # substitution from confusion row
            row = confusion[idx[ch]].copy()
            row[idx[ch]] = 0
            if row.sum() == 0:
                # Empty confusion row (char unseen / always-correct in the anchor data):
                # fall back to a uniform substitution to a DIFFERENT grid character so the
                # manipulated CER is still realized. Real confusion structure is used
                # wherever it exists; only sparse/unseen rows use the fallback.
                choices = [i for i in range(len(alphabet)) if i != idx[ch]]
                out.append(alphabet[choices[rng.integers(len(choices))]])
            else:
                out.append(alphabet[rng.choice(len(alphabet), p=row / row.sum())])
        else:
            out.append(ch)
    return "".join(out)
