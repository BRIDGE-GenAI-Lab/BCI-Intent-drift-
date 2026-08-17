# Preregistration v2 (reviewer-driven major revision addendum)

This document records amendments to `docs/preregistration.md` made during the
reviewer-driven major revision. It is a living document extended one revision
task at a time; each section below is scoped to a specific revision task and
does not speak for tasks not yet completed.

## Revision Task 1.1: prespecified stratified authentic-corpus sample + power calculation

### What changed and why

The original exposure pipeline (`idrift/data/build_exposure_run.py`, full
mode) drew its authentic-corpus messages by uniform random sampling of 120
phrases from the 2,168-phrase Costello corpus (`rng.choice(len(costello),
n_corpus, replace=False)`). Reviewers rejected this as an ad-hoc convenience
sample: it is not stratified by category or message length, its size was not
justified by a power calculation, and it was not prespecified.

This task replaces that sampling step with:

1. A prespecified, proportional stratified sample over category x
   length_bin, implemented in `idrift/data/corpus_sample.py`
   (`sample_authentic`), deterministic by seed, with no duplicate texts.
2. A simulation-based power calculation for the primary drift ~ CER slope
   (`power_for_slope` in the same module), so the sample size is justified
   rather than assumed.

### Sampling design

`sample_authentic(corpus_df, n, seed)` draws from the full digitized
Costello corpus (`output/intermediate/corpus_costello.parquet`; 2,168
phrases, 40 categories). Length bins are terciles of character length
computed within the corpus (labeled short/medium/long). Allocation uses
Hamilton (largest-remainder) apportionment in two stages -- first across
categories, then within each category across its length bins -- so every
stratum's allocation is within +-1 of its exact proportional quota, and
category-level marginal proportionality is preserved by construction.
Duplicate texts (the same phrase filed under two categories, which occurs in
the real corpus) are collapsed before sampling, so the sample never contains
the same text twice. Sampling is driven by a single seeded
`numpy.random.Generator`, so the same corpus, n, and seed always return the
same sample.

The frozen production sample was drawn with `n=500`, `seed=0`, and written to
`output/intermediate/auth_sample.csv` (500 rows; 39 of 40 corpus categories
represented -- one category, "Health and Safety", had only 2 unique phrases in
the de-duplicated sampling frame of 1,977 unique phrases, giving a proportional
quota of 0.51, so it received 0 under the largest-remainder allocation. That is
the correct behavior for a below-one quota, not a sampling error, and it is
within the prespecified +-1 tolerance).

### Power calculation

`power_for_slope(effect, n_msg, n_rep, alpha=0.05)` estimates power for the
fixed-effect CER slope in the primary drift ~ CER mixed-effects logistic
model (`docs/preregistration.md`, Analysis Plan) via Monte Carlo simulation.
Per simulated dataset: `n_msg` messages are each observed at every level of
the prespecified CER exposure grid (0.0, 0.1, 0.2, 0.3, 0.4), replicated
`n_rep` times per message x CER cell; each message carries its own random
intercept (between-message heterogeneity in baseline drift-proneness,
Normal(0, 0.5) on the log-odds scale, a documented placeholder in the
absence of a fitted variance component from the pilot); the fixed intercept
is anchored to the pilot's CER = 0 drift rate (~0.6%); and `effect` is a
per-10-percentage-point-of-CER odds ratio converted to a logit-scale slope.
Each simulated dataset is analyzed with a cluster-robust (clustered on
message id) logistic regression, a fast, numerically stable proxy for the
fixed-effect slope test in the full mixed-effects model. Power is the
fraction of simulated datasets whose two-sided slope p-value is below alpha
with the correct (positive) sign.

The effect size used is anchored to the current pilot result, not assumed:
the pilot's overall drift rate rises from about 0.6% at CER = 0 to about
46.3% at CER = 0.4 (`docs/manuscript_facts.md`). Converting those two
endpoints to the logit scale and rescaling to a per-10-percentage-point CER
odds ratio gives `PILOT_OR_PER_10PT_CER` = 3.457 (module constant, with its
derivation documented inline).

### Achieved power

At the production sample size (`n_msg=500`, `n_rep=20`, `alpha=0.05`,
`PILOT_OR_PER_10PT_CER` = 3.457, 200 Monte Carlo replicates, seed 0):

**achieved power = 1.00**

This scale is comfortably overpowered given the pilot's large effect size.
For context, the power curve saturates quickly at this effect size even at
much smaller scale (`n_rep=1` throughout, same effect and alpha):

| n_msg | power |
|------:|------:|
|     5 | 0.66  |
|    10 | 0.865 |
|    20 | 0.99  |
|    30 | 1.00  |
|    50 | 1.00  |

Power is monotonically non-decreasing in `n_msg` (verified in
`tests/data/test_corpus_sample.py`, `test_power_monotone_increasing_with_n_msg`).
This result should be read as a check that the design is adequately powered
to detect a slope of the magnitude seen in the pilot, not as evidence about
the true effect size in the full confirmatory run, which has not yet been
analyzed.

### Code and tests

- `idrift/data/corpus_sample.py`: `sample_authentic`, `power_for_slope`,
  `CER_GRID`, `PILOT_OR_PER_10PT_CER`, and a `main()` runner that produces
  `output/intermediate/auth_sample.csv` from the checkpointed corpus.
- `tests/data/test_corpus_sample.py`: proportional/deterministic/no-duplicate
  stratified-sampling tests plus a power-monotonicity test. All tests pass
  (`uv run pytest tests/data/test_corpus_sample.py`).

### Frozen prompt bank

The interface prompts are frozen in `idrift/interfaces/prompts.py`. The primary condition uses P0 (reconstruct, do not add). The full six-prompt bank (P0 primary; P1 spelling-only; P2 copy-unless-certain; P3 abstain-when-uncertain; P4 multi-candidate; P5 expansion) has SHA-256 `97d7a1f9f8dc7fba731537028a1271ab8f53fecdeb21bcd927378524ed0f7696`, recorded here so reviewers can confirm the exact prompts used.
