# Semantic intent substitution during language-model post-editing under P300-informed simulated corruption

Analysis code, dated analysis plan, and result digests for an in-silico benchmark of **intent drift**:
a fluent language-model reconstruction of a noisy brain-computer-interface decode that asserts a
*different* intent from the one the user attempted.

Gorenshtein A, Adiniaev Y, Omar M, Liba T, Klang E, Daniel O. *Semantic Intent Substitution During
Language-Model Post-Editing Under P300-Informed Simulated Corruption: An In-Silico Benchmark.*
Manuscript under review. Open Science Framework: [DOI to be inserted on deposit].

This is not a prospective preregistration: the analysis plan was drafted and timestamped in version
control before the reported runs (`docs/preregistration*.md` and the deviation table are that dated
record), and the manuscript makes no prospective-registration claim.

## What the study measures

Twenty open-weight models (4-120 billion parameters, ten-plus architecture families) post-edit character
streams that have been corrupted by a synthetic process anchored to an empirical P300-speller confusion
matrix from ALS and control participants. Each output is scored **faithful**, **degraded**, or **drift**
by a multi-evaluator ensemble that was itself benchmarked against a blinded physician panel.

**No participant used a brain-computer interface in this study.** The decoder noise is real in origin;
the deployment is simulated. Nothing here is a clinical or device-safety claim.

| | |
|---|---|
| Open-weight models evaluated | 20 (17 on the full grid, 3 reasoning models on a reduced-replicate grid) |
| Labeled generations (main benchmark) | 4,252,326 |
| Exposures per model | 243,000 (2,168 AUTH + 131 CRIT + 131 matched CTRL, 5 corruption levels, 20 replicates); 40,500 for the 3 reduced-replicate reasoning models |
| Interface substudy | 562 messages re-run under 6 policies (seven-model panel) |
| Physician validation panel | 2,281 blinded items, 2 raters + adjudicator (16 of the 20 models) |

## Headline results

- **Drift scales with decoder corruption**, in all three corpora, from 2.2% with no corruption to 60.3%
  at a 40% target character error rate in the authentic corpus (odds ratio 2.30 per 10-percentage-point
  rise in target CER).
- **Confidence does not reliably flag drift.** Verbalized confidence discriminates faithful output
  reasonably well (meta-analytic AUROC 0.83, 95% CI 0.80-0.85) but is poorly calibrated (ECE 0.32,
  95% CI 0.27-0.37); 28.4% of outputs at confidence 90 or higher were not faithful, so a confidence gate
  would pass a share of the substitutions it was meant to catch.
- **Post-editing is a trade, not a free win.** The ratio of faithful rescues to fluent errors introduced
  exceeds 1 at low corruption but falls below 1 at 20-30% target CER.
- **Message-critical content carries a small residual excess** after matching that survives removal of
  the rule-based detector (rule-free OR 1.10, 95% CI 1.08-1.11; detector-inclusive OR 1.16, 1.13-1.19,
  from the earlier validated seven-model panel).
- **No interface policy removes drift.** Conservative editing and permitted abstention lower it
  (OR 0.75, 95% CI 0.74-0.77 and 0.85, 0.83-0.86 vs forced reconstruction); offering alternatives or
  expanding the message raise it (1.13 and 1.25). The best policy still drifts on 18.0 of every 100
  messages attempted.

## Layout

```
idrift/            analysis package
  data/            corpus, confusion matrix, corruption model, exposure builder
  models/          O2/SLURM inference runners (Ollama)
  adjudicate/      outcome taxonomy, fluency gate, critical-error rules, evaluator ensemble
  interfaces/      frozen prompt bank (P0-P5) and the interface-condition registry
  analysis/        every statistical module; run_v3.py is the top-level pipeline
  figures/         figure builders (fig_v3 = Figure 1; fig_validated_examples = Figure 2;
                   fig_supplement_calibration = Figure 3 + eFigures 3-4)
tests/             pytest suite
docs/              analysis plan, deviation table, ensemble spec, rater instructions
results/digests/   the JSON digest behind every number in the manuscript
results/figures/   Figure 1-3 and eFigure 1-4 (PDF, vector)
scripts/           reproduction and verification entry points
```

`stats_v3plus_digest.json` is the primary source of truth; `calibration_16_v3_digest.json` carries the
per-model calibration and its message-clustered 95% CIs (drawn on Figure 3). Figure 1b's per-model slope
CIs are cached in `results/digests/fig1b_slope_ci.json` (a 1,000-replicate message-clustered bootstrap,
fixed seed).

## Reproducing

```bash
uv sync                                    # or: pip install -e .
python -m pytest tests -q                  # tests that load the non-redistributed parquets skip
python scripts/verify_manuscript_numbers.py    # every manuscript number vs its digest
python scripts/verify_substudy_numbers.py      # the interface-substudy numbers vs the digest
```

The suite is green in the authoring environment (about 510 tests). The tests that load the materialized
main-run parquets skip on a fresh checkout, because those files are not redistributed (see **Data**).

Regenerating the digests from scratch requires the intermediate generation files
(`output/intermediate/*.parquet`, several hundred GB of model output), which are not distributed here.
`scripts/run_d5_substudy.py` and `scripts/rerun_dependence_v3.py` show the exact entry points; the full
set of model generations is available from the corresponding author on request.

## Data

- **bigP3BCI** (P300 confusion matrix anchor) is openly available from PhysioNet and is not
  redistributed here. DOI [10.13026/0byy-ry86](https://doi.org/10.13026/0byy-ry86).
- **Boston Children's Hospital / Costello ALS Message Banking vocabulary** is cited, not
  redistributed. The raw phrase list is not in this repository.
- Human-validation materials (blinded rating sheets, instructions, panel results) are deposited on the
  Open Science Framework rather than here.

## Two things worth knowing if you reuse this code

Both were real defects caught during the analysis and are pinned by regression tests:

1. **`analysis/interface_substudy.assemble_forced` matches on the `(message_id, corpus)` pair, not on
   `message_id` alone.** The message-critical probes are drawn *from* the message bank, so a probe item
   and an authentic item routinely share an identifier. Filtering `corpus == "AUTH"` and
   `message_id.isin(ids)` independently also admits the authentic twin of every probe, which silently
   unbalances the reference arm.
2. **A condition term in an interaction model is the effect at the interaction variable = 0.**
   `drift_condition_model` therefore always reports a CER-averaged main-effects fit alongside the
   interaction fit; the two can disagree in sign, and the manuscript quotes the averaged one.

A third, in `analysis/dependence_sensitivity.py`: statsmodels reported `converged=True` for a GEE fit
that returned a slope of -3.6e22 with a standard error of exactly zero. The fitter now applies a
plausibility guard on magnitude and standard error rather than trusting the convergence flag.

## License

MIT (see `LICENSE`). The license covers this code. It does not extend to the bigP3BCI dataset or to the
message-banking vocabulary, both of which carry their own terms.
