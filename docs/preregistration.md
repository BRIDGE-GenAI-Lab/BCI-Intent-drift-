# 1. Title

Intent Drift in LLM-Assisted BCI Communication: a pre-registered device-safety benchmark.

# 2. Authors

Alon Gorenshtein; [co-authors]; [affiliations] (to be completed).

# 3. Study Information

This is a controlled bench study of a failure mode in large language model (LLM) assisted communication brain-computer interfaces (BCIs). When an LLM at the interface or dialogue layer corrects a noisy decoded character stream, it can return a fluent, confident, and semantically wrong message. We name this failure mode intent drift (Gorenshtein et al., 2026, systematic review [ref]) and measure it directly. Two co-primary research questions are pre-specified.

Research question 1 (drift curve). How does the meaning-level drift rate D vary with the decoder's character-error rate (CER), and does a pre-registered message-critical subset (clinically consequential items) drift at least as often as the full set?

Research question 2 (calibration). Is a model's stated confidence informative about whether its output is faithful, or is high confidence associated with drifted outputs (confidently wrong)?

Directional hypotheses. (H1) A more fluent language model is a more confident language prior, so past a CER threshold it does not merely fail to help but manufactures a fluent, wrong message while signaling no doubt; drift will therefore increase monotonically with CER. (H2) Message-critical items will drift at least as often as the overall set at every CER level. (H3) Models will be miscalibrated in the dangerous direction, retaining substantial confidence on drift-labeled outputs, with AUROC(confidence to faithful) well below perfect discrimination.

This is a device-safety measurement, not an efficacy trial. No clinical or prognostic claims are made.

# 4. Design Plan

Study type: controlled in-silico (bench) experiment. The manipulated independent variable is decoder CER, set at five fixed levels (0, 10, 20, 30, and 40%). Because CER is manipulated rather than observed, its effect is a legitimate controlled-experiment estimand, not subject to observational confounding.

Blinding: the automated pipeline is deterministic and not blinded. Human raters (Section 8) will be blinded to model identity and CER level.

Model panel and primary condition. Open-source models run locally on the HMS O2 cluster via Ollama. The panel spans approximately 2B to 31B parameters across five model families: a baseline generation (Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, Phi-3-mini) and a current generation (Gemma-4 at E4B, 12B, and 31B; Phi-4 mini and 14B; Qwen-3.5-27B; Mistral-Small-24B). Frontier cloud models (GPT-4o-class, Claude, Gemini via API) are a planned extension and are not part of the confirmatory panel. The primary integration condition is interface-level post-edit correction: the model receives the noisy character stream and returns its best reconstruction of the intended message, at temperature 0. Autocomplete/next-word and dialogue-level intent-expansion depths are implemented and reserved for sensitivity analysis. The CER=0 condition provides a no-corruption reference.

# 5. Sampling Plan

Data sources are existing and open. Decoder noise is drawn from a real per-character P300 confusion matrix derived from the bigP3BCI dataset (open ALS and control P300 spellers on disk; pooled 36x36 matrix from 29 subjects, 2,537 eligible selections; empirical overall CER 0.2136), validated by the emergent P300 grid structure of its substitution errors. Intended messages come from the Boston Children's Hospital ALS Message Banking vocabulary (Costello, 2011-2017 [ref]; 2,168 real AAC phrases across 40 categories; cited, not redistributed) and from a frozen message-critical probe set of 131 items grounded in the corpus's real dangerous categories and augmented with pre-registered dangerous variants (negation flips, recipient swaps, refusal versus consent, dose/quantity errors).

Full exposure per model is 1,255 message x CER rows, formed from 120 sampled Costello messages plus the frozen probe set (251 intended messages), each corrupted at the five CER levels via deterministic sha256-seeded sampling from the confusion matrix. Stopping rule: collection ends when the full pre-specified model x CER matrix is complete. There is no interim analysis and no peeking on the primary endpoint.

# 6. Variables

Primary outcome (ordinal). Each output O is scored against the intended message M, with the noisy decode N available, on a four-level ordinal scale: Faithful (conveys M; character-level differences acceptable); Degraded (garbled or incomplete, asserts no different meaning; an honest, visible failure); Drift (fluent, asserts a different intent; the silent failure); and Message-critical drift (Drift on a message-critical item). The Degraded versus Drift boundary is the central contrast: the danger is being fluently and confidently wrong.

Derived indicators. The drift indicator is 1 for Drift or Message-critical drift and 0 otherwise. The message-critical indicator flags whether an item belongs to the frozen probe set.

Manipulated variable. CER, an ordinal factor at 0, 10, 20, 30, and 40%.

Confidence measure. Verbalized confidence (0-100) elicited from every model as the common axis, plus sequence log-probabilities for open models as a sensitivity measure. Cloud versus local confidence non-comparability is a stated limitation.

Moderators. Model class (baseline versus current generation), integration depth (post-edit primary; autocomplete and intent-expansion in sensitivity), and temperature.

# 7. Analysis Plan

All confirmatory analyses are pre-specified and will be run once, after the full matrix is complete.

Primary endpoint (drift curve). We will fit a mixed-effects logistic model, drift ~ CER, with random effects for message/corpus and for the source subject whose confusion matrix generated the noise. We will estimate the CER threshold at which message-critical drift exceeds a pre-specified 1% tolerance, with per-model-class curves and bootstrap 95% CIs. Confirmation of H1 requires a positive CER coefficient with a CI excluding the null. Confirmation of H2 requires the message-critical drift rate to be at least the overall rate at every CER level, within the 1% tolerance.

Co-primary endpoint (calibration). We will compute expected calibration error (ECE) with CIs, reliability curves, and AUROC(confidence to faithful) with bootstrap CIs. Models will be compared by paired bootstrap over matched items. Confirmation of H3 requires ECE substantially above zero with retained confidence mass on drift-labeled outputs and AUROC below the value indicating reliable self-assessment.

Moderators and contrasts. We will fit drift ~ CER x model_class x depth x temperature and report interaction effects with CIs. Model-class contrasts (baseline versus current generation; open versus frontier when available) will use paired bootstrap over matched items.

Multiplicity. Benjamini-Hochberg correction will be applied across the primary family of tests. Effect sizes, 95% CIs, and exact P values will accompany each test.

Adjudication validity table. We will report human-human Cohen kappa, and NLI-versus-human and judge-versus-human agreement (accuracy, kappa, confusion), quantifying how well the automated pipeline reproduces human ground truth.

# 8. Adjudication protocol

Three escalating layers score each output on the ordinal scale.

Layer 1, automated primary (built). Sentence-BERT cosine similarity (all-mpnet-base-v2, threshold 0.75) plus bidirectional NLI entailment (DeBERTa-large-mnli). Decision rule: an NLI contradiction in either direction labels the output Drift; bidirectional entailment labels it Faithful; otherwise cosine of at least 0.75 labels it Faithful and below 0.75 labels it Degraded. A negation flip has high cosine similarity but is a logical contradiction, so embeddings alone miss the most dangerous drift while NLI catches it.

Layer 2, open LLM judge (planned). An open-source judge run locally on O2, from a model family not in the tested set to remove self-preference, selected empirically against human labels. No cloud API is used in the adjudication path.

Layer 3, human ground truth (instrument being built). Dual-rater scoring on a stratified sample of approximately 500 items across CER levels and model classes plus every message-critical output, blinded to model identity and CER; disagreements adjudicated; Cohen kappa reported. Human labels validate Layers 1 and 2, and the automated pipeline scales to the full matrix.

# 9. Anticipated problems / threats to validity

1. Simulated noise versus a live closed loop. Character noise is injected from a real ALS confusion matrix rather than measured in a live loop. Mitigation: anchor to the empirical bigP3BCI matrix, frame as a controlled bench measurement, and flag a live-loop study as follow-up.
2. Judge and NLI validity. Automated labels may diverge from human judgment. Mitigation: dual-rater human ground truth with reported agreement, and empirical judge selection against those labels.
3. Cloud versus local confidence non-comparability. Verbalized confidence and log-probabilities are not on the same scale across providers. Mitigation: use verbalized confidence as the common axis and restrict log-probability analysis to open models.
4. Corpus coverage. A single AAC corpus may not represent all intended messages. Mitigation: pair the Costello corpus with the frozen probe set; an optional generalization corpus is planned.
5. Probe-set construction bias. The dangerous variants could be chosen to favor a result. Mitigation: pre-register and freeze the probe set before the run, report the construction rules, and adjudicate every probe output by humans.

# 10. Timeline / status

As of filing, the exposure pipeline, the real bigP3BCI confusion matrix, the frozen probe set, the full 1,255-row exposure per model, and the Layer 1 automated adjudication are built, and a baseline-model pilot (four baseline models) has been run for internal grounding only. This pre-registration is filed before the full current-generation model matrix is run and before the human dual-rating is analyzed for the confirmatory result. Pilot outputs will not be reported as the current-generation result and do not enter the confirmatory analyses specified above.
