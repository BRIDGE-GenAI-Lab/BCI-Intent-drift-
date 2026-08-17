"""Interface-condition registry, candidate-list parser, and the abstention
safety guard (revision Task D1).

The 6 interface conditions in this substudy each map to one prompt id in
the frozen bank (`idrift.interfaces.prompts.PROMPTS`); `CONDITIONS` /
`CONDITION_ORDER` below are that registry. `forced` (P0, reconstruct-no-add)
is the primary baseline reused from the main run and is listed first as the
reference condition; `minimal_edit` (P1), `copy_when_uncertain` (P2),
`abstain_enabled` (P3), `candidate_list` (P4), and `expansion` (P5) are the
prompt-sensitivity sub-study conditions.

`idrift.models.o2_runner_v2.parse_reply` already extracts
`output_message`/`confidence`/`abstained` from a raw model reply; it does
NOT extract a `candidates` list (only `candidate_list`/P4 replies carry
one). `parse_candidates`/`normalize_record` here add that, and
`is_declined`/`resolve_outcome` add the safety guard: an abstained row must
never reach the faithful/degraded/drift labeler.

Abstention ("declined") sits OUTSIDE the `{faithful, degraded, drift}`
outcome taxonomy (`idrift.adjudicate.taxonomy`): a declined row is excluded
from that denominator and counted separately. `resolve_outcome` is the one
place that boundary is enforced -- it returns "declined" without ever
calling the injected `label_fn` for an abstained row, so a labeler that is
expensive (a real NLI/fluency model) or has side effects is never invoked
on a row that was never scored in the first place.

Pure functions only: stdlib (json, re) plus the frozen prompt bank. No I/O,
no models.

American spelling; no em dashes (double hyphens for asides, per house
style).
"""
from __future__ import annotations

import json
import re

from idrift.interfaces.prompts import ABSTAIN_SENTINEL, PROMPTS  # noqa: F401 (re-exported for callers)

CONDITIONS: dict[str, dict] = {
    "forced": {"prompt_id": "P0", "permits_abstention": False, "permits_candidates": False},
    "minimal_edit": {"prompt_id": "P1", "permits_abstention": False, "permits_candidates": False},
    "copy_when_uncertain": {"prompt_id": "P2", "permits_abstention": False, "permits_candidates": False},
    "abstain_enabled": {"prompt_id": "P3", "permits_abstention": True, "permits_candidates": False},
    "candidate_list": {"prompt_id": "P4", "permits_abstention": False, "permits_candidates": True},
    "expansion": {"prompt_id": "P5", "permits_abstention": False, "permits_candidates": False},
}

# "forced" first: it is the reference condition (reused baseline, P0). The
# rest follow the P1-P5 numbering of the prompt bank.
CONDITION_ORDER: list[str] = [
    "forced", "minimal_edit", "copy_when_uncertain", "abstain_enabled", "candidate_list", "expansion",
]

_PROMPT_TO_CONDITION: dict[str, str] = {spec["prompt_id"]: name for name, spec in CONDITIONS.items()}

_JSON_OBJ = re.compile(r"\{.*\}", re.S)
_CANDIDATES_ARRAY = re.compile(r'"candidates"\s*:\s*\[(.*?)\]', re.S)
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def prompt_for_condition(name: str) -> str:
    """Condition name -> prompt id. Raises KeyError for an unknown name."""
    try:
        return CONDITIONS[name]["prompt_id"]
    except KeyError:
        raise KeyError(f"unknown interface condition: {name!r}") from None


def condition_for_prompt(prompt_id: str) -> str:
    """Prompt id -> condition name (inverse of `prompt_for_condition`).
    Raises KeyError for a prompt id not mapped to any of the 6 conditions."""
    try:
        return _PROMPT_TO_CONDITION[prompt_id]
    except KeyError:
        raise KeyError(f"unknown prompt id: {prompt_id!r}") from None


def _extract_candidates(obj) -> list[str] | None:
    if not isinstance(obj, dict):
        return None
    candidates = obj.get("candidates")
    if not isinstance(candidates, list):
        return None
    out = [c for c in candidates if isinstance(c, str) and c.strip()]
    return out or None


def parse_candidates(raw_output: str | None) -> list[str] | None:
    """Extract the "candidates" array from a raw model reply.

    JSON-first: try the whole reply, then the greedy brace-matched substring
    (mirrors `o2_runner_v2.parse_reply`'s own JSON recovery). If both fail to
    parse (malformed JSON -- e.g. a trailing comma), fall back to a targeted
    regex that recovers a well-formed "candidates": [...] array even when the
    JSON around it is broken. Documented behavior: this fallback only
    recovers actual quoted strings inside the captured array; if no such
    array is found at all, the reply is treated as truly unrecoverable and
    this returns None (same as "no candidates present").

    Returns a list of non-empty strings, or None if absent/unparseable.
    """
    if not raw_output:
        return None

    obj = None
    try:
        obj = json.loads(raw_output)
    except (ValueError, TypeError):
        m = _JSON_OBJ.search(raw_output)
        if m:
            try:
                obj = json.loads(m.group(0))
            except (ValueError, TypeError):
                obj = None

    if obj is not None:
        found = _extract_candidates(obj)
        if found is not None:
            return found
        if isinstance(obj, dict) and "candidates" not in obj:
            return None  # clean JSON, genuinely no candidates key

    m = _CANDIDATES_ARRAY.search(raw_output)
    if not m:
        return None
    items = [s for s in _QUOTED.findall(m.group(1)) if s.strip()]
    return items or None


def normalize_record(rec: dict) -> dict:
    """Normalize one runner output row.

    candidates is populated ONLY when the row's condition (derived from
    `rec["prompt_id"]`, not the runner's own unrelated `condition` field --
    see module docstring) permits_candidates (candidate_list/P4); for every
    other condition it is None even if raw_output happens to contain a
    stray "candidates" key.
    """
    permits_candidates = False
    prompt_id = rec.get("prompt_id")
    if prompt_id is not None:
        try:
            permits_candidates = CONDITIONS[condition_for_prompt(prompt_id)]["permits_candidates"]
        except KeyError:
            permits_candidates = False

    candidates = parse_candidates(rec.get("raw_output")) if permits_candidates else None

    return {
        "output_message": rec.get("output_message"),
        "abstained": bool(rec.get("abstained")),
        "candidates": candidates,
    }


def is_declined(rec_or_normalized: dict) -> bool:
    """True iff the row's `abstained` field is truthy."""
    return bool(rec_or_normalized.get("abstained"))


def resolve_outcome(rec: dict, label_fn) -> str:
    """THE SAFETY GUARD: an abstained row is never scored as drift.

    If `rec` is declined (abstained), returns "declined" WITHOUT calling
    `label_fn` at all. Otherwise returns `label_fn(rec)`, unwrapping a
    dict-shaped result's "label" key (mirrors
    `idrift.adjudicate.taxonomy.label`'s {"label": ..., ...} return shape) so
    callers can pass either that labeler or a plain str-returning one.
    """
    if is_declined(rec):
        return "declined"
    result = label_fn(rec)
    if isinstance(result, dict):
        return result["label"]
    return result
