"""Light smoke tests for the Task C4 v3 stats driver (`idrift.analysis.run_v3`).

This module is a COMPOSITION layer over already-tested analysis modules
(`drift_curve`, `calibration_v2`, `multinomial`, `confidence_analysis`,
`dependence_sensitivity`, `zero_cer_audit`, `mixed_models`) -- the modules
themselves already have their own unit tests. What is genuinely new here,
and worth testing on small synthetic fixtures (never the real 1.7M-row
parquet -- that is exercised by actually running the driver, per the
Task C4 brief), is the ASSEMBLY logic: the per-corpus/per-model grouping
in `build_drift_curve_section`, and the agreement/spread arithmetic in
`build_dependence_digest`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from idrift.analysis.run_v3 import build_dependence_digest, build_drift_curve_section


def test_build_drift_curve_section_hand_computable():
    """4 corpora x cer_target cells, small enough to check by hand."""
    df = pd.DataFrame(
        {
            "corpus": ["AUTH"] * 4 + ["CRIT"] * 2,
            "model": ["modelA", "modelA", "modelB", "modelB", "modelA", "modelB"],
            "cer_target": [0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            "label": ["drift", "faithful", "drift", "drift", "faithful", "drift"],
        }
    )
    out = build_drift_curve_section(df)

    # AUTH @ cer_target 0.0: 3/4 drift = 75%
    assert out["drift_by_corpus_x_targetcer"]["AUTH"]["0.0"] == 75.0
    # CRIT @ cer_target 0.1: 1/2 drift = 50%
    assert out["drift_by_corpus_x_targetcer"]["CRIT"]["0.1"] == 50.0

    # overall_by_corpus sums to 1 per corpus and is keyed on all three labels
    for corpus_stats in out["overall_by_corpus"].values():
        assert set(corpus_stats) == {"faithful", "degraded", "drift"}
        assert abs(sum(corpus_stats.values()) - 1.0) < 1e-9

    # per_model_auth_drift is AUTH-only and sorted ascending by rate
    assert set(out["per_model_auth_drift"]) == {"modelA", "modelB"}
    rates = list(out["per_model_auth_drift"].values())
    assert rates == sorted(rates)

    assert out["n_by_corpus"] == {"AUTH": 4, "CRIT": 2}


def _make_dependence_fixture(seed=0, n_messages=40, n_models=4, n_rep=6):
    """Small hierarchical frame with a planted realized_cer effect -- same
    idiom as tests/analysis/test_dependence_sensitivity.py's own fixture,
    just sized for a fast unit test."""
    rng = np.random.default_rng(seed)
    models = [f"model{i}" for i in range(n_models)]
    rows = []
    for msg_i in range(n_messages):
        message_id = f"msg{msg_i}"
        cer = rng.uniform(0.0, 0.4)
        for model in models:
            for rep in range(n_rep):
                logit = -2.0 + 8.0 * cer + rng.normal(0, 0.3)
                p = 1 / (1 + np.exp(-logit))
                label = "drift" if rng.random() < p else "faithful"
                rows.append(
                    {
                        "message_id": message_id,
                        "model": model,
                        "realized_cer": cer,
                        "label": label,
                    }
                )
    return pd.DataFrame(rows)


def test_build_dependence_digest_shape_and_arithmetic():
    df = _make_dependence_fixture()
    out = build_dependence_digest(df, primary_slope_adjusted=4.0, univariate_anchor={"est": 8.0, "se": 0.5})

    assert set(out["estimators"]) == {
        "per_model_pooled_dl",
        "gee",
        "bootstrap_message",
        "bootstrap_model",
    }
    assert set(out["agreement"]) == set(out["estimators"])

    slopes = [e["slope"] for e in out["estimators"].values()]
    assert out["slope_spread_across_estimators"] == max(slopes) - min(slopes)

    anchor_est = 8.0
    expected_max_pct = max(abs(s - anchor_est) / anchor_est * 100 for s in slopes)
    assert abs(out["max_pct_deviation_from_anchor"] - expected_max_pct) < 1e-9

    # the dynamic "ci_covers_adjusted_<slope>" key names off the passed-in
    # primary_slope_adjusted, rounded to 2 dp
    for entry in out["agreement"].values():
        assert "ci_covers_adjusted_4.0" in entry

    assert out["model_level_bootstrap_unstable"] is True  # n_models=4 < 15
    assert isinstance(out["conclusion"]["stable_across_dependence_assumptions"], bool)
