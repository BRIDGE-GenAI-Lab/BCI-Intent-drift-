"""Coherent faithful/degraded/drift outcome taxonomy + fluency gate
(revision Task 3.1).

Reviewers rejected the earlier NLI+cosine-only labeler
(`idrift.adjudicate.nli_metric.label_row`, kept as-is and still used
elsewhere) on two grounds this module fixes:

1. It had no notion of FLUENCY. This study defines "drift" specifically as
   a FLUENT-but-WRONG reconstruction: an LLM confidently asserting the
   wrong intent. A garbled, visibly broken output that also happens to
   assert a different meaning is a different failure mode (`degraded`), not
   drift, because nothing about it is "confidently wrong" -- it is simply
   broken. `label()` below adds that fluency gate.
2. It could silently label an actionable-content OMISSION as `faithful`
   (high lexical/cosine overlap with the intended message even though a
   clinically important clause was dropped, e.g. "I am in pain and need
   help" -> "I am in pain"). `idrift.adjudicate.critical_rules.detect()`
   (Task 3.2) now feeds into the meaning-difference check here specifically
   so an omission (and four other clinically critical substitution types)
   counts as a meaning difference even when NLI/cosine alone would miss it.

Outcome taxonomy is EXACTLY {faithful, degraded, drift} -- three mutually
exclusive labels, nothing else. "Message-critical" (whether the INPUT
message was one of the study's prespecified high-stakes probe messages) is
a separate, pre-existing INPUT attribute computed elsewhere in the
pipeline, not a fourth output class; this module never emits it and never
special-cases on it.

How the four signals compose
------------------------------
`label(intended, output, cos_fn, nli_fn, fluency_fn)` composes:

- `nli_fn` (bidirectional): a contradiction in EITHER direction is a
  meaning difference, exactly as `nli_metric.label_row` already treats it.
- `critical_rules.detect()` (called directly here, not injected -- it is
  fast, deterministic spaCy/regex, already a project dependency since Task
  3.2, so this module has a real, non-optional dependency on it): ANY of
  its five boolean flags (negation_flip, numeral_change, recipient_change,
  urgency_change, actionable_omission) is ALSO a meaning difference, even
  when NLI and cosine alone would not flag one. This check happens
  regardless of the NLI outcome so it can override a false "entailment" --
  catching exactly the cases NLI/cosine quietly miss is the entire reason
  `critical_rules.py` exists (see its module docstring).
- `nli_fn` bidirectional entailment, or `cos_fn` >= `nli_metric.COS_HI`
  (imported and reused, not redefined): if neither an NLI contradiction nor
  any critical rule fired, this reproduces the pre-existing cosine-threshold
  rule verbatim to decide whether meaning is preserved.
- `fluency_fn`: gates what a meaning difference MEANS, and also gates
  `faithful`. A meaning difference in a fluent output
  (`fluency_fn(output) >= TAU`) is `drift` (confidently wrong); the same
  meaning difference in a disfluent output is `degraded` (visibly broken,
  not confidently anything). When meaning is NOT different, a fluent
  output is `faithful`; a disfluent one is still `degraded` -- a garbled
  reconstruction is not "faithful" merely because its words happen to still
  parse as meaning-equivalent by NLI/cosine, it is simply a broken
  reconstruction that did not corrupt the intended MEANING. This keeps
  `degraded` meaning "not fluent, whatever the meaning verdict" and
  reserves `faithful` for a clean, fluent, meaning-preserving
  reconstruction.

Only `critical_rules.detect()` is called directly; `cos_fn`/`nli_fn`/
`fluency_fn` are injected exactly as the brief specifies, so this module
has no direct dependency on `sentence_transformers`/`transformers` and
imports cleanly without either installed (their heavy imports live inside
each signal's own `make_*_fn()` factory elsewhere, not here).

Invariants a reviewer can check directly against `_meaning_differs`/`label`
below:
- No path can return `{"label": "drift", "fluent": False}` -- both the
  `drift` and `faithful` branches are reached only under the `fluent`
  guard in `label()`; the `not fluent` branch always returns `degraded`.
- `actionable_omission` (and the other four critical-rule flags) route to
  `drift` whenever the output is fluent -- they are never silently
  `faithful`, closing exactly the gap reviewers flagged.
- `"label"` is always one of {"faithful", "degraded", "drift"}; there is no
  fourth branch and no "message-critical" value anywhere in this module.
- `label()` is a pure function of its five arguments (no randomness, no
  hidden state) -- calling it twice with the same inputs always returns an
  identical dict.

TAU is pinned as a module constant here (mirrors `nli_metric.COS_HI`
sitting next to the label logic that consumes it, rather than inside the
similarity function itself): TAU = 0.5, the fluency threshold calibrated
and justified in `idrift/adjudicate/fluency.py`'s module docstring against
a real gpt2 perplexity calibration set (0 of 8 word-salad strings score
>= 0.5; 11 of 13 grammatical calibration strings do -- see that module and
the accompanying report for the full table). Tests in
`tests/adjudicate/test_taxonomy.py` inject stub `fluency_fn` values rather
than depending on that calibration directly, so this module's decision
LOGIC is tested independent of the real model's numeric behavior.

Deterministic; American spelling; no em dashes (double hyphens for
asides, per house style).
"""
from __future__ import annotations

from idrift.adjudicate.critical_rules import detect
from idrift.adjudicate.nli_metric import COS_HI

TAU = 0.5

_CRITICAL_KEYS = (
    "negation_flip",
    "numeral_change",
    "recipient_change",
    "urgency_change",
    "actionable_omission",
)


def _meaning_differs(intended: str, output: str, cos_fn, nli_fn, crit: dict | None = None) -> tuple[bool, str]:
    """Return (differs, reason). Mirrors `nli_metric.label_row`'s
    contradiction/entailment/cosine logic, with the critical-rules check
    (Task 3.2) inserted so it can flag a meaning difference NLI/cosine alone
    would miss; that check runs regardless of the NLI outcome, so it can
    override a false "entailment" the way a plain NLI+cosine rule cannot.

    `crit`, if given, is used in place of calling `detect(intended, output)`
    internally (see `label`'s docstring -- a precomputed pass-through of the
    same result, not a different logic path)."""
    fwd = nli_fn(intended, output)
    bwd = nli_fn(output, intended)

    if "contradict" in (fwd, bwd):
        return True, "nli_contradiction"

    flags = detect(intended, output) if crit is None else crit
    fired = tuple(key for key in _CRITICAL_KEYS if flags.get(key))
    if fired:
        return True, "critical_rule:" + "+".join(fired)

    if fwd == "entail" and bwd == "entail":
        return False, "nli_bidirectional_entailment"

    if cos_fn(intended, output) < COS_HI:
        return True, "cosine_below_threshold"

    return False, "cosine_at_or_above_threshold"


def label(intended: str, output: str, cos_fn, nli_fn, fluency_fn, *, crit: dict | None = None) -> dict:
    """Label an (intended, output) pair with the coherent
    faithful/degraded/drift outcome taxonomy (revision Task 3.1).

    Args:
        intended: the ground-truth / reference message.
        output: the model's reconstructed message.
        cos_fn: injected (str, str) -> float cosine-similarity function.
        nli_fn: injected (str, str) -> str NLI function, one of "entail" |
            "contradict" | "neutral".
        fluency_fn: injected (str) -> float fluency function in [0, 1] (see
            `idrift.adjudicate.fluency.fluency_score`).
        crit: optional precomputed pass-through of
            `idrift.adjudicate.critical_rules.detect(intended, output)`'s
            five-boolean-flag dict (revision Task 5.0). When `None` (the
            default), behavior is EXACTLY as before this parameter existed:
            `detect(intended, output)` is called internally. When given, its
            flags are used in place of that internal call -- this is a pure
            pass-through of the SAME result a caller could have obtained by
            calling `detect` itself, not an alternate decision path, so a
            batched runner that already computed `detect()` once per unique
            (intended, output) pair can skip re-parsing spaCy per row.

    Returns:
        {"label": "faithful" | "degraded" | "drift", "fluent": bool,
         "reason": str}
    """
    differs, reason = _meaning_differs(intended, output, cos_fn, nli_fn, crit=crit)
    fluent = fluency_fn(output) >= TAU

    if differs:
        return {"label": "drift" if fluent else "degraded", "fluent": fluent, "reason": reason}
    return {"label": "faithful" if fluent else "degraded", "fluent": fluent, "reason": reason}
