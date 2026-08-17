# Preregistration deviation table

This is a living document. It maps every analysis named in the original
preregistration (`docs/preregistration.md`) to its current status during the
reviewer-driven major revision, with a one-line reason for any change.
Entries are added as revision tasks complete; final counts and effect
estimates populate once the full labeled cohort (currently being produced by
a remote labeling job) is analyzed. No preregistered analysis is left
planned-but-absent: each row below is Run, Modified, or explicitly carries a
reason it could not be run as originally specified. Analyses not in the
original preregistration but added during revision are marked Added.

| Preregistered analysis | Status | What changed / why |
|---|---|---|
| drift ~ CER mixed model | Run | Fit as a statsmodels one-vs-rest GLMM (cross-classified binomial, `idrift.analysis.mixed_models`); the Bayesian `bambi` engine specified in the original plan is unavailable in the compute environment, so the statsmodels fallback is used and documented in the module itself. |
| Primary predictor = target CER | Modified | Replaced by realized message-level corruption (`realized_cer`), the actual character-error rate observed after the synthetic corruption process runs, not the CER value it targeted. |
| One corruption draw per condition | Modified | Replaced by 20 replicates per message x CER cell, so within-cell sampling variability is estimable rather than assumed away. |
| Calibration (ECE, AUROC) | Run, extended | Computed per model (`idrift.analysis.calibration_v2`), with message-clustered bootstrap confidence intervals and DerSimonian-Laird random-effects pooling across models; a comparability caveat is documented because models differ in scale and training data. |
| Message-critical vs overall drift | Modified | Replaced by a matched CRIT-vs-CTRL conditional logistic comparison (`idrift.analysis.matched_compare`), which removes the length/negation/numeral confound between the message-critical probe set and the corpus at large that the original raw contrast could not separate from criticality itself. |
| Automated labels treated as ground truth | Modified | The automated pipeline is evaluated against human labels (class-specific sensitivity, specificity, PPV, NPV, F1, confusion; `idrift.adjudicate.validation_metrics`), never asserted as ground truth, and a misclassification-corrected prevalence is added alongside the raw automated rate. |
| Misclassification-corrected prevalence | Added | Not in the original preregistration. Uses a multiclass Rogan-Gladen correction with clustered bootstrap intervals (`idrift.adjudicate.misclassification`) so the full-cohort estimate accounts for the automated pipeline's own measured error rates. |
| Matched controls | Added | Not in the original preregistration. One non-critical control message is matched to each message-critical item on character length, word count, numeral presence, negation presence, and mean word frequency (`idrift.data.matched_controls`), enabling the matched CRIT-vs-CTRL contrast above. |
| Zero-CER forensic audit | Added | Not in the original preregistration. Every drift label on a message with zero realized decoder error is categorized by cause (critical substitution, formatting, paraphrase, overgenerative, other; `idrift.analysis.zero_cer_audit`), answering why drift occurs on an input the decoder corrupted perfectly. |
| Fluency gate in outcome taxonomy | Added | Not in the original preregistration. The faithful/degraded/drift taxonomy now requires the output to be fluent before it can be called faithful or degraded, so a fluent-but-different output and a garbled, non-fluent output are never conflated under the same label. |
| Interface conditions (prompts P1-P5) | Modified | The original plan reserved the five non-P0 prompts for sensitivity analysis. They were instead run as a full substudy on a 562-message subset against all seven models, so the interface policy is measured as a design variable rather than assumed fixed; forced reconstruction (P0) is reused from the main run at identical exposures as the reference arm. |

## Notes

- Status definitions: **Run** = executed exactly as originally planned, subject only to a documented engine substitution. **Modified** = the analysis runs, but its predictor, unit of analysis, or comparison changed from the original plan for a stated reason. **Added** = not part of the original preregistration; introduced during the reviewer-driven revision to address a specific reviewer concern. No row is "Not run" as of this revision; if a future task cannot complete a preregistered analysis, it is recorded here rather than silently dropped from the Methods.
- This table is authored ahead of the full-cohort analysis run (the labeling job was still executing remotely at the time of writing). The Status column reflects what code exists and is tested today; specific numeric results (effect sizes, confidence intervals, counts) are reported in the manuscript once that run completes, not in this table.
