"""Rule-based critical-error detectors (revision Task 3.2).

The automated ordinal labeler (`idrift.adjudicate.nli_metric`, NLI +
cosine) is the pipeline's primary faithful/degraded/drift outcome measure,
but reviewers flagged it as unable to reliably surface the specific
substitution TYPES that matter clinically: a negated instruction read back
as its opposite, a dose or quantity changed, the addressee of a request
swapped, an urgency marker dropped, or an actionable request silently
omitted. `detect()` below is a second, independent, rule-based layer that
flags those five substitution types explicitly. It feeds BOTH the coherent
error taxonomy (Task 3.1) and stands alone as a labeled error-subtype field
in the adjudication output -- it is not a replacement for the NLI+cosine
outcome label.

spaCy vs. a dependency-light fallback
--------------------------------------
The brief asks to try spaCy first (`uv add spacy` +
`python -m spacy download en_core_web_sm`) and only fall back to a
dependency-light regex/lexicon implementation if the model cannot be made
to load (this workspace sits on an exFAT/network volume that generates
macOS `._*` AppleDouble sidecar files, which previously broke `wordfreq`'s
data-directory globbing in Task 1.3). Both `uv add spacy` and the
`en_core_web_sm` download completed cleanly here, and `spacy.load` /
parsing worked correctly on first try (verified manually against every
sentence used in the test suite before writing this module), so spaCy IS
used -- for the two sub-problems where a real dependency parse and lemma
buys genuine robustness over a hand-rolled regex:

- `negation_flip`: uses spaCy's `neg` dependency label (catches "not"
  attached to its governing verb/aux) UNIONed with the same negation
  lexicon Task 1.2 already uses for exposure metadata (see below), rather
  than a from-scratch negation list.
- `recipient_change` / `actionable_omission`: uses spaCy's lemmatizer so
  inflected forms (e.g. "nurses", "calling", "needed") normalize to the
  same lexicon entry ("nurse", "call", "need") as their base forms,
  which a bare substring/regex lexicon match would not do.

`numeral_change` and `urgency_change` are deliberately NOT spaCy-dependent:
digit extraction and a short fixed urgency-phrase list are solved
adequately (and more transparently) by regex, and the brief's own
interface line lists "number extraction" as a thing separate from the
spaCy dependency/POS parsing, not a spaCy sub-task. This is a genuine
hybrid, not spaCy-for-show: two of the five detectors go through the
parser, three do not, because that is where a real parse actually adds
value over a lexicon.

Reuse of Task 1.2's negation lexicon
-------------------------------------
`idrift.data.exposure_v2` defines `_NEGATION_WORDS` (the negation lexicon
used to flag `corrupted_negation` in the replicate-aware exposure
metadata). The name is module-private by convention (leading underscore)
but is a Python-importable value, not truly inaccessible, so this module
imports it directly rather than re-defining a second negation lexicon that
could silently drift out of sync with the one exposure metadata is keyed
on. `exposure_v2` also defines `_TOKEN_RE`, its alnum-run tokenizer used to
map corruption character-positions back onto whole tokens; that tokenizer
is NOT reused here, because it solves a different problem (which token
does character index N fall inside) than this module needs (does a
negation/numeral/role/action token appear anywhere in this whole string).
spaCy's own tokenizer already supplies the tokens `_has_negation` and the
lemma lookups below walk over, so a second, independent tokenizer would be
redundant rather than reused for a real purpose. Numeral extraction below
also does not use `_TOKEN_RE`: it would fold a token like "5mg" into one
alnum run and defeat isolating the digit run for a numeric comparison, so
a dedicated digit-only regex is used instead (see `_NUMERAL_RE` below).

Definitions implemented (mirrors the brief's plain-language spec exactly)
--------------------------------------------------------------------------
- `negation_flip`: a negation marker (spaCy `neg` dependency OR the
  Task-1.2 negation lexicon) is present in exactly one of
  {intended, output} -- added or removed.
- `numeral_change`: the multiset of numbers (compared numerically, e.g.
  "5" and "5.0" are the same number; regex-extracted digit runs) differs
  between intended and output.
- `recipient_change`: the set of recipient/role lemmas (a curated
  addressee lexicon: nurse, doctor, family roles, etc.) present differs
  between intended and output.
- `urgency_change`: an urgency marker (now, immediately, right away,
  urgent(ly), asap, hurry, quickly) is present in one but not the other,
  matched case-insensitively on whole words/phrases (word-boundary regex,
  so "now" does not match inside "know").
- `actionable_omission`: for each occurrence of an actionable-request verb
  (need, want, help, call, bring, get, give, stop, wait, check) in
  intended, its CORE CONTENT ARGUMENT -- the object noun of the request
  (spaCy `dobj`/`attr`/`oprd`/`dative`, or the noun inside a `prep`+`pobj`
  child), or, when the verb has no object noun (e.g. "want to move"), its
  clausal complement (`xcomp`/`ccomp`) -- must be absent from output for
  that occurrence to count as omitted. Recipient/role nouns (the
  `_ROLE_LEMMAS` lexicon) are excluded from the content argument, since a
  role word changing is `recipient_change`'s job, not an omission; when a
  clause's only object is a role noun (e.g. "call the nurse", no other
  content argument), the check falls back to whether the verb ITSELF
  (lemma, not argument) survives anywhere in output. This keys the flag on
  surviving MEANING rather than the specific verb lemma, so a faithful
  synonym paraphrase that keeps the request's content intact (bring water
  -> get water; want to move -> would like to move) does NOT flag, while a
  clause whose content is genuinely dropped (need help -> [nothing]; call
  the nurse -> unrelated content, verb gone too) still does. Directional by
  design -- a NEW actionable request appearing only in the output is a
  different failure mode (fabrication), not an omission, and is out of
  scope for this flag. (Revised post-review; see "Known limitations" below
  for the residual gap this narrower rule still has.)

Deterministic; no randomness anywhere in this module. American spelling
throughout. No em dashes (the module uses double hyphens for asides, per
house style).

Known limitations (documented per the brief's request, not silently
assumed away): the curated lexicons (roles, urgency phrases, actionable
verbs) are illustrative, not exhaustive, so both false negatives (an
addressee word or urgency phrase this module does not know about) and
occasional false positives (e.g. a role word used generically rather than
as the addressee of a request, such as "the nurse said...") are possible.
`recipient_change` compares LEMMA SETS, not argument structure, so it
cannot tell that a role word changed which verb it attaches to versus
simply appearing/disappearing elsewhere in the sentence. See the
accompanying report for concrete failure-mode examples.

`actionable_omission` (post-review revision, see fix note in the report
dated after commit 0e7e35e): the rule now keys on the actionable clause's
surviving CONTENT ARGUMENT rather than the verb lemma, which fixes the
false-positive on verb-synonym paraphrases (bring/get, want/would like)
that the original lemma-set version produced. This narrower rule has its
own residual limitations, honestly documented rather than assumed away:
- **Noun-level synonym substitutions may still (incorrectly) flag.** The
  content-argument match is exact lemma equality, not semantic similarity,
  so a faithful-in-context noun paraphrase -- e.g. "bring me water" ->
  "bring me a drink" -- still flags as an omission, because "water" is
  simply absent from the output and the rule has no synonym/embedding
  layer to recognize "drink" as covering the same request. This is the
  same class of gap as the pre-existing role/urgency lexicon limitations
  above, just relocated to the content-noun side of this one detector.
- **A clause whose actionable verb has no object AND no clausal
  complement** (rare in this study's short single-clause register, but
  possible) falls back to checking whether the verb lemma itself survives
  in output; that fallback is exactly the original lemma-only behavior for
  that narrow case, with the same verb-synonym blind spot it always had
  (e.g. a bare "help!" paraphrased as a bare "assist!" would fall back to
  lemma comparison and could still misfire). This case is expected to be
  uncommon given the lexicon's verbs almost always take an object or
  infinitival complement in practice.
- **Spelled-out numerals are still a separate, unrelated miss** (this is
  `numeral_change`'s limitation, not `actionable_omission`'s, called out
  here only because an earlier version of the accompanying report
  mis-stated it): a change from a digit to a digit, OR a digit to a
  spelled-out number, is still CAUGHT -- e.g. "5 mg" -> "ten mg" IS
  flagged, because "5" simply vanishes from the digit-run comparison (the
  multiset goes from {5} to {}, which is already unequal) even though
  "ten" itself is never parsed as a number. The actual miss is narrower:
  BOTH sides fully spelled out, e.g. "five mg" -> "ten mg", where neither
  string contains a digit run at all, so both multisets are empty and
  compare equal.
"""
from __future__ import annotations

import re

import spacy

from idrift.data.exposure_v2 import _NEGATION_WORDS

_NLP = None


def _nlp():
    """Lazily load and cache the spaCy English pipeline (one load per
    process, not per `detect()` call)."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


# Numeral extraction: digit runs with an optional decimal part, independent
# of `_TOKEN_RE` (which would fold an adjacent unit like "5mg" into one
# alnum token and defeat isolating the digits).
_NUMERAL_RE = re.compile(r"\d+(?:\.\d+)?")

# Urgency markers, matched as whole words/phrases (word-boundary regex) so
# "now" does not match inside "know" and "urgent" also catches "urgently".
_URGENCY_PATTERNS = [
    re.compile(pat, re.IGNORECASE)
    for pat in (
        r"\bnow\b",
        r"\bimmediately\b",
        r"\bright away\b",
        r"\burgent(?:ly)?\b",
        r"\basap\b",
        r"\bhurry\b",
        r"\bquickly\b",
    )
]

# Recipient/addressee role lexicon (lemma-matched). Illustrative and
# curated, not exhaustive -- see module docstring "Known limitations".
_ROLE_LEMMAS = {
    "nurse", "doctor", "physician", "daughter", "son", "wife", "husband",
    "mother", "mom", "father", "dad", "sister", "brother", "spouse",
    "aide", "caregiver", "therapist", "attendant", "family", "friend",
}

# Actionable-request lemma lexicon (lemma-matched, verb or noun form).
# Illustrative and curated, not exhaustive -- see module docstring "Known
# limitations".
_ACTION_LEMMAS = {
    "need", "want", "help", "call", "bring", "get", "give", "stop",
    "wait", "check",
}


def _has_negation(doc) -> bool:
    """True if `doc` contains a spaCy `neg` dependency arc OR a token
    matching the Task-1.2 negation lexicon (exact match, or containing the
    "n't" contraction suffix, mirroring `exposure_v2._corruption_flags`)."""
    for tok in doc:
        if tok.dep_ == "neg":
            return True
        low = tok.text.lower()
        if low in _NEGATION_WORDS or "n't" in low:
            return True
    return False


def _numerals(text: str) -> list[float]:
    """Extract the multiset of numbers in `text` as a sorted list of
    floats (so "5" and "5.0" compare equal, and multiset order does not
    matter for the equality check in `detect()`)."""
    return sorted(float(m) for m in _NUMERAL_RE.findall(text))


def _has_urgency(text: str) -> bool:
    return any(pat.search(text) for pat in _URGENCY_PATTERNS)


def _role_lemmas(doc) -> set[str]:
    return {tok.lemma_.lower() for tok in doc if tok.lemma_.lower() in _ROLE_LEMMAS}


# Dependency labels that mark a direct content-argument child of an
# actionable verb (its object). `dative` covers the indirect-object slot in
# a ditransitive like "bring me water" ("me" is dative, "water" is dobj);
# it is listed here too because on rare non-pronoun datives (e.g. "bring
# Mary water") the same NOUN/PROPN + non-role filter below should apply.
_CONTENT_OBJECT_DEPS = ("dobj", "attr", "oprd", "dative")


def _action_content_clauses(doc) -> list[tuple[str, set[str]]]:
    """For every occurrence of an actionable-request verb in `doc`, return
    an (verb_lemma, content_lemmas) pair, one per occurrence (not merged
    across repeats), where `content_lemmas` is that specific occurrence's
    core content argument(s):

    - the lemma of a direct object/attribute/dative child
      (`_CONTENT_OBJECT_DEPS`), or the object of a prepositional child
      (`prep` -> `pobj`), restricted to NOUN/PROPN children (so a bare
      pronoun like "me" is never mistaken for meaningful content) and
      excluding recipient/role nouns (`_ROLE_LEMMAS` -- a role word
      changing is `recipient_change`'s concern, not an omission); and/or
    - the lemma of a clausal complement (`xcomp`/`ccomp`), for verbs like
      "want"/"need" that take an infinitival complement instead of a noun
      object ("want to move" has no object noun, but "move" is the
      request's actual content).

    An empty content set for an occurrence means it has no qualifying
    content argument at all (e.g. "call the nurse", whose only object is a
    role noun) -- `detect()` falls back to checking the verb lemma itself
    for that occurrence, since there is nothing else to key on.
    """
    clauses: list[tuple[str, set[str]]] = []
    for tok in doc:
        if tok.pos_ not in ("VERB", "AUX"):
            continue
        verb_lemma = tok.lemma_.lower()
        if verb_lemma not in _ACTION_LEMMAS:
            continue
        contents: set[str] = set()
        for child in tok.children:
            if child.dep_ in _CONTENT_OBJECT_DEPS:
                lemma = child.lemma_.lower()
                if child.pos_ in ("NOUN", "PROPN") and lemma not in _ROLE_LEMMAS:
                    contents.add(lemma)
            elif child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ != "pobj":
                        continue
                    lemma = grandchild.lemma_.lower()
                    if grandchild.pos_ in ("NOUN", "PROPN") and lemma not in _ROLE_LEMMAS:
                        contents.add(lemma)
            elif child.dep_ in ("xcomp", "ccomp"):
                lemma = child.lemma_.lower()
                if lemma not in _ROLE_LEMMAS:
                    contents.add(lemma)
        clauses.append((verb_lemma, contents))
    return clauses


def _actionable_omission(doc_i, doc_o) -> bool:
    """True if any actionable-request occurrence in `doc_i` is omitted from
    `doc_o`: its content argument (if it has one) is absent from output's
    lemma set, or -- for a clause with no qualifying content argument --
    the verb lemma itself is absent from output. See
    `_action_content_clauses` for how each occurrence's content is
    determined, and the module docstring for the surviving-argument
    rationale (fixes the verb-synonym false positive from the original
    lemma-set version)."""
    output_lemmas = {tok.lemma_.lower() for tok in doc_o}
    for verb_lemma, contents in _action_content_clauses(doc_i):
        if contents:
            if not (contents & output_lemmas):
                return True
        elif verb_lemma not in output_lemmas:
            return True
    return False


def detect(intended: str, output: str) -> dict:
    """Flag five clinically critical substitution types between an
    intended message and a reconstructed output.

    Args:
        intended: the ground-truth / reference message.
        output: the model's reconstructed message.

    Returns:
        dict with boolean keys: negation_flip, numeral_change,
        recipient_change, urgency_change, actionable_omission. See the
        module docstring for the exact definition of each.
    """
    nlp = _nlp()
    doc_i = nlp(intended)
    doc_o = nlp(output)

    negation_flip = _has_negation(doc_i) != _has_negation(doc_o)
    numeral_change = _numerals(intended) != _numerals(output)
    recipient_change = _role_lemmas(doc_i) != _role_lemmas(doc_o)
    urgency_change = _has_urgency(intended) != _has_urgency(output)
    actionable_omission = _actionable_omission(doc_i, doc_o)

    return {
        "negation_flip": negation_flip,
        "numeral_change": numeral_change,
        "recipient_change": recipient_change,
        "urgency_change": urgency_change,
        "actionable_omission": actionable_omission,
    }
