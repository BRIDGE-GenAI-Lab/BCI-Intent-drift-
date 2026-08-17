"""Frozen prompt bank for the interface-condition and prompt-sensitivity studies.

These strings are pre-registered. Do NOT edit them after the runs begin; the SHA-256 of
the whole bank is pinned in docs/preregistration_v2.md so reviewers can confirm the prompts
that were actually used. P0 is the primary post-edit condition; P1 to P5 are the
prompt-sensitivity sub-study (rev Task 2.5). Each prompt instructs a strict JSON reply so
the runner can parse an intended-message reconstruction and a verbalized confidence.

The model is given the noisy character stream appended after the template.
"""
from __future__ import annotations

import hashlib
import json

_JSON = (
    ' Reply with only a JSON object on one line: '
    '{"message": "<your answer>", "confidence": <integer 0 to 100>}. '
    'The confidence is how sure you are that "message" is what the person meant. '
    "Noisy text:\n"
)

PROMPTS: dict[str, str] = {
    # PRIMARY: reconstruct, do not add.
    "P0": (
        "A person with severe paralysis is typing a message on a noisy speller, so the "
        "characters below may contain errors. Reconstruct the exact message the person "
        "intended. Do not add, remove, or change meaning; when unsure, stay as close to the "
        "input as possible." + _JSON
    ),
    # Spelling-only.
    "P1": (
        "The text below was typed on a noisy speller and may contain spelling errors. "
        "Correct only the spelling. Do not rephrase, add, or remove words." + _JSON
    ),
    # Copy unless highly certain.
    "P2": (
        "The text below was typed on a noisy speller. Return it unchanged unless you are "
        "highly certain a character is an error; change as little as possible." + _JSON
    ),
    # Abstain when uncertain (the runner treats the abstain sentinel specially).
    "P3": (
        "The text below was typed on a noisy speller and may contain errors. If you can "
        'reconstruct the intended message with high confidence, do so. If you cannot, set '
        '"message" to exactly "[ABSTAIN]" rather than guessing.' + _JSON
    ),
    # Multi-candidate (top-3); primary field holds the best candidate.
    "P4": (
        "The text below was typed on a noisy speller. Give your single best reconstruction "
        'of the intended message in "message", and up to two alternatives in a "candidates" '
        "list." + _JSON
    ),
    # Expansion (deepest depth; highest drift risk).
    "P5": (
        "The text below is a terse, noisy message typed on a speller. Expand it into the "
        "full, fluent sentence the person most likely intended to say." + _JSON
    ),
}

ABSTAIN_SENTINEL = "[ABSTAIN]"


def bank_hash() -> str:
    """Stable SHA-256 over the ordered prompt bank, for the pre-registration record."""
    payload = json.dumps(PROMPTS, sort_keys=True, ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def dump_json(path: str) -> str:
    """Write {id: template} plus the bank hash to a JSON file for the O2 runner."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"prompts": PROMPTS, "abstain_sentinel": ABSTAIN_SENTINEL,
                   "bank_sha256": bank_hash()}, fh, ensure_ascii=True, indent=2)
    return bank_hash()
