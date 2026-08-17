# Preregistration chronology (revision Task F1)

Reviewers require an explicit timeline so "preregistered" cannot be read as a
quality label applied to analyses chosen after seeing results. This file states
what was frozen when, and labels every analysis confirmatory vs post-hoc. Dates
marked `[DATE: ...]` are HUMAN-ONLY placeholders the authors must fill from the
OSF record before submission.

## Timeline

1. **Original registration** `[DATE: original OSF registration]` — endpoints
   (co-primary: drift-vs-CER dose-response and confidence calibration), the
   ordinal-to-3-class outcome taxonomy, the ensemble decision rules, the model
   list, and the CER grid were registered. Drafted at `docs/preregistration.md`.
   OSF DOI: `[OSF DOI to be inserted]`.
2. **Data state at registration** `[DATE]` — at registration, only the pilot
   (baseline 4-model, 5,020-generation) outputs existed; the confirmatory
   large-scale exposure had not been generated.
3. **Prompt bank frozen** — the six-prompt bank was frozen with SHA `97d7a1f9...`
   before the confirmatory run; P0 (forced reconstruction) is the primary
   condition. The other five prompts drive the interface substudy (post-hoc,
   reviewer-motivated; see below).
4. **Confirmatory exposure frozen** `[DATE]` — the corruption confusion matrix
   (`confusion_overall.npy`, derived from bigP3BCI), the exposure grid, seeds,
   and outcome rules were frozen; the confirmatory 533,400-generation run (7
   current-generation models x 76,200 exposures x P0) was then executed. The
   full-corpus extension to 2,168 AUTH phrases reuses the identical seed scheme
   (verified byte-identical for the original sample).
5. **Addendum** `[DATE]` — a preregistration addendum
   (`docs/preregistration_v2.md`) recorded the outcome-taxonomy overhaul (fluency
   gate; drift only when fluent AND meaning differs), the replicate design (20
   corruptions per cell), the matched-control design, and the additions listed
   below. Every change from the original plan is enumerated with its reason in
   the deviation table (eTable 1).

## "Reviewer-driven" clarification

Where the manuscript refers to a "reviewer-driven" or adversarial revision, this
was an **internal adversarial critique by the study team** used to harden the
analysis, not an external journal peer review. Analyses added in response to it
are labeled post-hoc, not preregistered.

## Confirmatory vs post-hoc labeling

**Confirmatory** (frozen before the confirmatory run): the drift-vs-realized-CER
dose-response, the per-corpus reporting (AUTH/CRIT/CTRL), the confidence
calibration (ECE, AUROC), the matched CRIT-vs-CTRL criticality contrast, and the
prespecified tie-break and fluency-threshold sensitivity analyses.

**Post-hoc / reviewer-motivated** (added during revision, labeled as such in
text): the raw-decode benefit-harm transition analysis, the multinomial and
drift-vs-degraded contrasts, the confidently-wrong rate, the misclassification
(Rogan-Gladen) correction, the physician validation panel, the dependence-
structure sensitivity analyses, the stratified classifier diagnostics, and the
interface/prompt-policy substudy.

## Use of "preregistered"

The title and abstract use "preregistered" only for the confirmatory endpoints
and design frozen before the confirmatory run. Post-hoc analyses are never
described as preregistered.
