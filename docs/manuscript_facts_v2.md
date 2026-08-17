# Manuscript facts v2 (source of truth) -- from output/stats_v2_digest.json, 533,400 labeled generations

Cohort: 7 open models x 76,200 exposure (AUTH 50,000 + CRIT 13,100 + CTRL 13,100) x P0/temp0. Ensemble label tau=0.5, tie=primary. Mean inter-evaluator agreement 0.979, tie rate 0.039.

## Drift x target CER (%), by corpus (co-primary dose-response)
- AUTH: CER0.0=1.01, CER0.1=13.67, CER0.2=31.46, CER0.3=49.86, CER0.4=63.36
- CRIT: CER0.0=2.36, CER0.1=15.57, CER0.2=34.09, CER0.3=51.44, CER0.4=63.36
- CTRL: CER0.0=0.58, CER0.1=14.1, CER0.2=30.87, CER0.3=48.41, CER0.4=62.0

## Outcome shares by corpus
- AUTH: faithful 0.5817, degraded 0.0996, drift 0.3187
- CRIT: faithful 0.5303, degraded 0.1361, drift 0.3337
- CTRL: faithful 0.5517, degraded 0.1363, drift 0.3119

## Per-model drift (AUTH, reproducible primary panel)
- gemma4:31b: 23.9%
- gemma4:e4b: 24.9%
- gemma4:12b: 29.2%
- qwen3.5:27b-q4: 29.7%
- mistral-small:24b: 33.4%
- phi4:14b: 37.6%
- phi4:mini: 44.4%

## Mixed model (drift-vs-rest, cluster-robust logit fallback after VB non-convergence; message-clustered SE; n=533,400)
- Intercept: est -3.1474 [-3.308, -2.987]
- corrupted_negation[T.True]: est 0.1147 [-0.01, 0.24]
- corrupted_numeral[T.True]: est 1.6782 [1.627, 1.73]
- C(family)[T.mistral-small]: est 0.5472 [0.515, 0.579]
- C(family)[T.phi4]: est 1.0421 [1.008, 1.076]
- C(family)[T.qwen3.5]: est 0.2816 [0.253, 0.31]
- realized_cer: est 4.7917 [4.16, 5.424]
- n_errors: est 0.22 [0.19, 0.25]
- char_len: est -0.0107 [-0.018, -0.003]
  (engine=clustered_logit_fallback, converged=True, variance_components unavailable in fallback)

## Calibration (per-model ECE / AUROC(confidence->faithful))
- gemma4:12b: ECE 0.312, AUROC 0.882
- gemma4:31b: ECE 0.176, AUROC 0.904
- gemma4:e4b: ECE 0.332, AUROC 0.874
- mistral-small:24b: ECE 0.335, AUROC 0.82
- phi4:14b: ECE 0.36, AUROC 0.875
- phi4:mini: ECE 0.515, AUROC 0.737
- qwen3.5:27b-q4: ECE 0.226, AUROC 0.889
- META (random-effects): ECE 0.322 [0.245, 0.4]; AUROC 0.854 [0.803, 0.906]
- dropped 119 rows with missing verbalized confidence

## Zero-CER drift forensic audit (drift at realized_cer==0)
- total 1462 cases; by cause: critical_substitution 640, formatting 90, paraphrase 20, overgenerative 52, other 660

## Matched CRIT-vs-CTRL conditional logistic (finer identical-corruption strata: pair x CER x replicate x model; discordant matched pairs)
- n_obs 55624, discordant pairs 27812, converged True
- critical: est 0.1443 [0.118, 0.171], p=1.0e-26
- char_len: est 0.001 [-0.017, 0.019], p=9.1e-01
- n_errors: est 0.1382 [0.117, 0.159], p=7.0e-39
- corrupted_negation: est 0.041 [-0.114, 0.196], p=6.0e-01
- corrupted_numeral: est 0.6792 [0.116, 1.242], p=1.8e-02
- realized_cer: est 4.7428 [4.317, 5.169], p=1.4e-105
  critical OR = exp(0.144) = 1.155

## Sensitivity (drift rate)
- tie-break: primary 0.3201, drift_averse 0.287, severity 0.3265
- tau: 0.3 0.3587, 0.4 0.3426, 0.5 0.3201, 0.6 0.2926