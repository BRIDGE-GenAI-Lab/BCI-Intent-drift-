"""Fluency scoring via causal-LM perplexity (revision Task 3.1).

`fluency_score(text) -> float` maps a piece of text to a grammaticality
proxy in [0, 1]: higher means more fluent (a plausible, well-formed English
utterance a causal language model would not find surprising); lower means
less fluent (word salad, garbled token order, an obviously broken
reconstruction). This is a FLUENCY check only -- it never looks at meaning
or correctness, that is `idrift.adjudicate.taxonomy`'s job, which combines
this signal with NLI/cosine/critical-rules meaning comparison.

Method
------
Perplexity of `text` under a small pretrained causal LM (gpt2 by default)
is the base signal: perplexity = exp(mean token-level cross-entropy),
computed with the model's own next-token prediction over `text` (Hugging
Face's `labels=input_ids` convention, which shifts internally). Lower
perplexity means the model found the text less surprising, i.e. more
fluent.

Two preprocessing steps are applied before scoring (these affect ONLY the
fluency measurement in this module; `taxonomy.py`'s meaning comparison
always sees the raw, unmodified text):

1. Capitalize the first letter and append terminal punctuation (".") if
   missing. This study's probe messages are often short, bare, unpunctuated
   fragments (e.g. "call the nurse now"). gpt2 was trained on full,
   punctuated documents and penalizes an unpunctuated fragment's perplexity
   for reasons unrelated to grammaticality; punctuating normalizes that
   away.
2. Prepend the tokenizer's BOS/EOS token before computing next-token loss.
   This gives the model a neutral, consistent left context instead of
   scoring the very first content token with zero context, and it avoids a
   NaN loss on a single-token input (a length-1 sequence has no valid
   shifted (input, label) pair for HF's causal-LM loss convention).

Perplexity -> [0, 1] mapping
-----------------------------
Raw perplexity is unbounded and its useful range is model- and
domain-dependent, so it is rescaled IN LOG SPACE between two anchors:

    score(ppl) = clip((ln(PPL_HI) - ln(ppl)) / (ln(PPL_HI) - ln(PPL_LO)), 0, 1)

    PPL_LO = 40.0    # perplexity at/below which fluency saturates at 1.0
    PPL_HI = 2500.0  # perplexity at/above which fluency saturates at 0.0

PPL_LO/PPL_HI were chosen by calibrating gpt2 (with the preprocessing
above) against a hand-built set of 13 clearly fluent short clinical-register
sentences matching this study's probe-message style (e.g. "I am in pain and
need help.", "call the nurse now", "bring me water") and 8 clearly
non-fluent word-salad strings (e.g. "asdkj qpwoe zznrx blaghfut", "zzt qux
blorp fneg"). On that set, gpt2 perplexities were 25-2560 (fluent; median
~135) and 433-11916 (word salad; median ~1346) -- see
`.superpowers/sdd/rev-task-3.1-report.md` for the full calibration table.
gpt2 was chosen over distilgpt2 (the brief's other suggested option)
because it gave a markedly cleaner separation on the same calibration set.

TAU (the "is this fluent" decision threshold) is defined in
`idrift.adjudicate.taxonomy`, not here, mirroring how
`idrift.adjudicate.nli_metric` owns its own `COS_HI` threshold next to the
label logic that consumes it; this module only ever returns the raw [0, 1]
score. For reference, at TAU = 0.5: 0 of the 8 word-salad calibration
strings score >= 0.5 (no false positives), and 11 of the 13 fluent
calibration strings do (2 false negatives -- see "Known limitations").

Known limitations
------------------
- A single bare content word with no surrounding context (e.g. the literal
  string "move") scores near the low end of this proxy's range even though
  it is grammatical, because a generic prose-trained causal LM finds a
  one-word "document" surprising regardless of grammaticality. In
  `taxonomy.py` this pushes such an output toward `degraded` rather than
  `drift`/`faithful`, which is the conservative direction for this study (it
  never inflates the `drift` count), but it is a real, documented proxy
  limitation, not a solved problem.
- Perplexity is a fluency (grammaticality/plausibility) proxy, not a
  semantic-meaning check: a fluent sentence with the wrong meaning scores
  HIGH here on purpose. Meaning is checked elsewhere
  (`idrift.adjudicate.taxonomy`); this module only ever answers "is this
  well-formed English", never "is this correct".
- gpt2 is an English-only, general-prose model, not fine-tuned on the
  ALS/BCI short-message clinical register used in this study, so this
  calibration is illustrative, not definitive -- the same posture this
  codebase already takes with its other rule-based/heuristic proxies (see
  `idrift/adjudicate/critical_rules.py`'s "Known limitations" section).

Heavy imports (`torch`, `transformers`) are kept function-local inside
`make_fluency_fn()` so this module imports cleanly without those packages
installed, mirroring `idrift.adjudicate.nli_metric.make_model_fns`.

Deterministic (model kept in eval mode, no sampling anywhere). American
spelling throughout. No em dashes (double hyphens for asides, per house
style).
"""
from __future__ import annotations

import math

PPL_LO = 40.0
PPL_HI = 2500.0


def _normalize_text(text: str) -> str:
    """Capitalize the first letter and ensure terminal punctuation, so a
    bare unpunctuated fragment (this study's typical probe-message style)
    is not penalized by the LM for reasons unrelated to grammaticality."""
    stripped = text.strip()
    if not stripped:
        return stripped
    normalized = stripped[0].upper() + stripped[1:]
    if normalized[-1] not in ".!?":
        normalized = normalized + "."
    return normalized


def _ppl_to_score(ppl: float) -> float:
    """Map a raw perplexity value to a fluency score in [0, 1] via the
    log-linear rescaling between the calibrated PPL_LO/PPL_HI anchors (see
    module docstring). Non-finite or non-positive input maps to 0.0
    (treated as non-fluent); this should not occur in practice once
    `make_fluency_fn`'s BOS-prefix handling is applied, but is guarded here
    defensively rather than allowed to propagate a NaN into the taxonomy."""
    if not math.isfinite(ppl) or ppl <= 0:
        return 0.0
    score = (math.log(PPL_HI) - math.log(ppl)) / (math.log(PPL_HI) - math.log(PPL_LO))
    return max(0.0, min(1.0, score))


def make_fluency_fn(model_name: str = "gpt2"):
    """Load a causal LM once and return a ready `fluency_score(text)`
    callable (mirrors `idrift.adjudicate.nli_metric.make_model_fns`).

    Args:
        model_name: HuggingFace model ID for the causal LM used as the
            perplexity/grammaticality proxy. Defaults to "gpt2", chosen
            over "distilgpt2" (the brief's other suggested option) after
            calibrating both against the fluent/word-salad set described in
            the module docstring: gpt2 gave a markedly cleaner separation
            (see the accompanying report for the full comparison).

    Returns:
        `fluency_score(text) -> float`, a callable computing the [0, 1]
        fluency proxy described in the module docstring.

    Note:
        Heavy imports (torch, transformers) are kept inside this function
        so importing `fluency.py` for testing does not require those
        packages to be installed.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    def fluency_score(text: str) -> float:
        """Grammaticality/perplexity proxy for `text`, in [0, 1]; higher is
        more fluent. See module docstring for the normalization, BOS-prefix,
        and perplexity-to-score mapping."""
        normalized = _normalize_text(text)
        if not normalized:
            return 0.0
        token_ids = [bos_id] + tokenizer(normalized)["input_ids"]
        input_ids = torch.tensor([token_ids], device=device)
        with torch.no_grad():
            output = model(input_ids, labels=input_ids)
        ppl = float(torch.exp(output.loss))
        return _ppl_to_score(ppl)

    return fluency_score
