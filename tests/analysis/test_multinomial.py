"""Tests for idrift.analysis.multinomial (revision Task B2).

Reviewers objected that collapsing {faithful, degraded} into one "not drift"
reference hides that degraded (a VISIBLE failure) and faithful (a success)
carry opposite safety meaning. These tests exercise the two models that keep
them separate:

  * `fit_multinomial` -- an MNLogit with base category = "faithful", giving
    two equations (degraded|faithful, drift|faithful).
  * `fit_binary_contrast` -- a two-class clustered logit that lets a caller
    ask the drift-vs-degraded and drift-vs-faithful contrasts directly.

The fixture plants a realized_cer -> drift relationship (drift becomes much
more likely as realized_cer rises) while making degraded independent of
realized_cer, so the recovered drift|faithful realized_cer odds ratio should
be well above 1 and the degraded|faithful one should not carry the same
signal.
"""

import numpy as np
import pandas as pd
import pytest

from idrift.analysis.multinomial import fit_binary_contrast, fit_multinomial


def _synthetic_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    realized_cer = rng.uniform(0.0, 1.0, size=n)
    model_family = rng.choice(["gemma4", "phi4", "qwen3.5"], size=n)
    corrupted_numeral = rng.integers(0, 2, size=n).astype(bool)
    corrupted_negation = rng.integers(0, 2, size=n).astype(bool)
    # A char length the module can also derive from intended_text; supplied
    # directly here so the fixture does not depend on that derivation.
    char_len = rng.integers(12, 42, size=n)
    message_id = [f"msg_{i}" for i in rng.integers(0, 60, size=n)]

    # Planted structure: drift probability climbs steeply with realized_cer;
    # degraded is assigned INDEPENDENTLY of realized_cer among the non-drift
    # rows, so drift (not degraded) carries the realized_cer signal.
    p_drift = 1.0 / (1.0 + np.exp(-(-1.2 + 4.5 * realized_cer)))
    labels = []
    for i in range(n):
        if rng.uniform() < p_drift[i]:
            labels.append("drift")
        elif rng.uniform() < 0.45:
            labels.append("degraded")
        else:
            labels.append("faithful")

    return pd.DataFrame(
        {
            "label": labels,
            "realized_cer": realized_cer,
            "corrupted_numeral": corrupted_numeral,
            "corrupted_negation": corrupted_negation,
            "char_len": char_len,
            "model_family": model_family,
            "message_id": message_id,
        }
    )


def _finite(x):
    return np.isfinite(x) and not np.isnan(x)


def _term_record(records, term):
    matches = [r for r in records if r["term"] == term]
    assert matches, f"term {term!r} not found among {[r['term'] for r in records]}"
    return matches[0]


def test_fit_multinomial_returns_two_equations():
    df = _synthetic_frame()
    res = fit_multinomial(df, seed=0)

    assert res["base_category"] == "faithful"
    assert set(res["equations"].keys()) == {"degraded", "drift"}
    assert res["n_obs"] == len(df)

    for cls in ("degraded", "drift"):
        records = res["equations"][cls]
        assert len(records) >= 2  # at least intercept + realized_cer
        for rec in records:
            assert _finite(rec["odds_ratio"])
            assert _finite(rec["ci_lower"])
            assert _finite(rec["ci_upper"])
            assert _finite(rec["p"])
            assert rec["ci_lower"] <= rec["odds_ratio"] <= rec["ci_upper"]
            assert rec["odds_ratio"] > 0.0


def test_fit_multinomial_recovers_planted_drift_signal():
    df = _synthetic_frame()
    res = fit_multinomial(df, seed=0)

    drift_cer = _term_record(res["equations"]["drift"], "realized_cer")
    degraded_cer = _term_record(res["equations"]["degraded"], "realized_cer")

    # Planted: realized_cer strongly raises drift odds vs faithful ...
    assert drift_cer["odds_ratio"] > 1.5
    # ... and the drift signal exceeds the degraded one (degraded is flat in cer).
    assert drift_cer["odds_ratio"] > degraded_cer["odds_ratio"]


def test_fit_binary_contrast_drift_vs_degraded_uses_only_those_rows():
    df = _synthetic_frame()
    expected_n = int(df["label"].isin(["drift", "degraded"]).sum())

    res = fit_binary_contrast(df, "drift", "degraded", seed=0)

    assert res["positive"] == "drift"
    assert res["negative"] == "degraded"
    assert res["n_obs"] == expected_n  # faithful rows excluded

    cer = _term_record(res["terms"], "realized_cer")
    assert _finite(cer["odds_ratio"])
    assert cer["odds_ratio"] > 0.0
    assert cer["ci_lower"] <= cer["odds_ratio"] <= cer["ci_upper"]


def test_fit_binary_contrast_drift_vs_faithful_runs():
    df = _synthetic_frame()
    expected_n = int(df["label"].isin(["drift", "faithful"]).sum())

    res = fit_binary_contrast(df, "drift", "faithful", seed=0)

    assert res["n_obs"] == expected_n
    cer = _term_record(res["terms"], "realized_cer")
    assert _finite(cer["odds_ratio"])
    # Planted drift-vs-faithful gradient in realized_cer is strongly positive.
    assert cer["odds_ratio"] > 1.5


def test_fit_binary_contrast_rejects_unknown_label():
    df = _synthetic_frame()
    with pytest.raises(ValueError):
        fit_binary_contrast(df, "drift", "nonsense", seed=0)
