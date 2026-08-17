# Reproducibility release manifest (revision Task F2)

Both reviewers require that the preregistration, code, and data be available to
editors and reviewers at submission (not "on request" or "on acceptance"). This
file lists exactly what is released and how. Items marked `[HUMAN]` need an
author action before submission.

## Released artifacts (public repository + OSF)

- **Analysis code**: the `idrift/` package (data engineering, exposure
  generation, taxonomy/ensemble labeling, all analysis modules) and the test
  suite.
- **Frozen exposure**: the corruption confusion matrix
  (`confusion_overall.npy`, derived from bigP3BCI) with its grid alphabet, the
  exposure generator, and deterministic seeds; the confirmatory exposure is
  reproducible byte-for-byte from these.
- **Frozen prompt bank**: `prompts.json` (SHA `97d7a1f9...`), P0-P5.
- **Ensemble specification**: `docs/ensemble_spec.md` (exact checkpoints,
  thresholds, voting and tie-break rules) plus a `models.lock` `[HUMAN: pin each
  Hugging Face repo id + commit sha]`.
- **Physician panel**: the blinded item set, both physicians' completed labels,
  the third adjudicator's labels, the resolved consensus, the automated
  predictions, and the final confusion matrix (`output/human_rating_v2/`,
  key withheld from the blinded packet).
- **Result digests**: every `output/*_digest.json` (drift curve, calibration,
  matched, benefit-harm, multinomial, confidence, stratified diagnostics,
  dependence sensitivity, zero-CER, panel).
- **Deviation table**: `docs/deviation_table.md` (eTable 1) and the
  preregistration chronology (`docs/preregistration_chronology.md`).

Not redistributed (cited only): the raw Costello/BCH ALS message-banking phrase
list (see ethics statement). Derived outputs are released.

## Model and inference provenance (for the methods/supplement)

- **Models under test (7, Ollama tags with quantization)**: `gemma4:e4b`,
  `gemma4:12b`, `gemma4:31b`, `phi4:mini`, `phi4:14b`, `qwen3.5:27b-q4`,
  `mistral-small:24b`. Run-level provenance (model digest, quantization,
  parameter size, Ollama version) is captured per run in each output's
  `.provenance.json` by `o2_runner_v2.py`.
- **Decoding**: temperature 0, `num_predict` capped at 160, `think=false` for
  reasoning models (else long chain-of-thought traces); JSON reconstruction +
  verbalized confidence parsed with a regex fallback.
- **Hardware**: generation on HMS O2 NVIDIA L40S (48 GB) GPUs via SLURM;
  labeling on the same. Fluency scorer is `gpt2` perplexity; NLI checkpoints
  `microsoft/deberta-large-mnli` and `roberta-large-mnli`; embeddings
  `all-mpnet-base-v2` and `all-MiniLM-L6-v2`.
- **Missing confidence**: 119 of 533,400 outputs (v2 cohort) had no parseable
  verbalized confidence, tabulated by model x CER in `output/confidence_digest.json`
  (concentrated in `phi4:mini`); dropped from calibration only.

## Access at submission `[HUMAN]`

- Register the OSF preregistration and obtain the DOI; insert it in the title
  page, methods, and supplement (currently placeholders).
- Publish the analysis repository (or create an anonymous double-blind review
  link) and insert the URL; the cover letter and supplement already state the
  materials are available at submission.
- Confirm the license for the released code and derived data.
