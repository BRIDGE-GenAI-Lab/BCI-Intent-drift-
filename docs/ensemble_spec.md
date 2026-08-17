# Evaluator-ensemble specification (reproducibility, revision Task B7)

Canonical, full specification of the automated evaluator ensemble that produced
every outcome label in `attempts_v2_labeled.parquet`. A compact version is folded
into the supplement (eMethods). Source of truth = the code in
`idrift/adjudicate/` (`label_runner.py`, `ensemble.py`, `nli_metric.py`,
`fluency.py`, `taxonomy.py`, `critical_rules.py`).

## Signals computed per (intended_text, output_message) pair

`label_runner.compute_signals` computes six model-derived signal blocks
(checkpointed independently); each names the exact Hugging Face checkpoint:

| Signal | Model / checkpoint | Notes |
|---|---|---|
| `nli_deberta_fwd`, `nli_deberta_bwd` | `microsoft/deberta-large-mnli` | 3-class MNLI head; forward = premise intended, hypothesis output, and the reverse |
| `nli_roberta_fwd`, `nli_roberta_bwd` | `roberta-large-mnli` | second, architecturally distinct MNLI checkpoint |
| `cos_mpnet` | `sentence-transformers/all-mpnet-base-v2` | cosine similarity between intended and output embeddings |
| `cos_minilm` | `sentence-transformers/all-MiniLM-L6-v2` | second, smaller embedding checkpoint |
| `fluency_raw` | `gpt2` causal LM (perplexity) | one shared fluency score in [0,1]; see below |
| `crit_*` | rule-based (spaCy `en_core_web_sm`) | five safety-relevant-subtype detectors |

NLI label canonicalisation (`nli_metric`): the 3-class head's arg-max label is
mapped `entailment -> entail`, `contradiction -> contradict`, `neutral ->
neutral`. Threshold constant `COS_HI = 0.75`.

## Fluency score (shared across all four evaluators)

`fluency.fluency_score` = gpt2 perplexity rescaled in log space:
`score = clip((ln(PPL_HI) - ln(ppl)) / (ln(PPL_HI) - ln(PPL_LO)), 0, 1)` with
`PPL_LO = 40.0`, `PPL_HI = 2500.0`. Preprocessing before scoring only: capitalise
first letter + append a period if missing; prepend BOS/EOS for a neutral left
context. Calibrated on 13 fluent + 8 word-salad short clinical-register strings
(gpt2 chosen over distilgpt2 for cleaner separation). Decision threshold `TAU =
0.5` lives in `taxonomy`.

## Per-evaluator label (`taxonomy.label`)

Each evaluator combines one NLI checkpoint + one embedding checkpoint + the shared
fluency function + the five critical rules into one of {faithful, degraded,
drift} plus a fluency flag:

- Meaning differs if: NLI contradiction in either direction, OR cosine <
  `COS_HI`, OR any critical rule fires (negation flip, numeral/dose change,
  recipient change, urgency change, actionable omission). NLI is retained
  because a negation flip can keep cosine high while being a logical
  contradiction.
- Fluency gate applied first: `fluent = fluency_raw >= TAU (0.5)`.
- Label: fluent AND meaning same -> **faithful**; fluent AND meaning differs ->
  **drift**; not fluent -> **degraded** (a garbled output that asserts no clear
  alternative). The comparator for "meaning" is always the INTENDED message.

## Ensemble vote and tie-break

`ensemble.default_evaluators` builds the 2 x 2 = **4 evaluators**
({deberta, roberta} x {mpnet, minilm}), each wired through `taxonomy.label` with
the one shared fluency function. The final `label` is the majority vote across
the four; on a 2-2 tie the `tie_break="primary"` rule takes the primary
evaluator (deberta-large-mnli x all-mpnet-base-v2). The parquet records the four
individual votes (`vote__<nli>+<emb>`), `agreement` (mean pairwise agreement),
`tie` (bool), and `tied_labels`.

## Handling of malformed / missing outputs

`o2_runner_v2.parse_reply` extracts `{message, confidence, abstained}` JSON-first
with a regex fallback; a reply equal to the abstain sentinel sets `abstained`.
Empty/None output falls through to `degraded` (not fluent). Rows with missing
verbalised confidence (119 in the v2 cohort) are dropped from calibration only,
and reported.

## Important caveat (state in the manuscript)

Mean pairwise inter-evaluator agreement was 0.979. This is **internal
consistency among four related checkpoints**, not independent validation: the two
NLI checkpoints and two embedding checkpoints can share systematic errors. The
external evaluation is the blinded physician panel (eTable 6), against which the
ensemble is characterised, not the inter-evaluator agreement.

## Provenance to record at submission (human)

Checkpoint revision hashes and tokenizer versions should be pinned in the
released repository (`requirements`/lockfile + a `models.lock` listing each HF
repo id and commit sha). The frozen prompt bank SHA is `97d7a1f9...`.
