"""Tests for dependence-structure sensitivity analyses (revision Task B9):
alternative estimators of the realized-CER -> drift log-odds slope under
different dependence assumptions (per-model DL pooling, GEE with an
exchangeable working correlation, and a nonparametric cluster bootstrap).

All tests run on a SMALL SYNTHETIC frame with a PLANTED realized_cer slope
and >= 3 models, never on the real labeled parquet. The point of the task
(and these tests) is that the slope is STABLE across dependence
assumptions, so each estimator is checked for recovering a slope near the
planted truth (or, for the bootstrap, a CI that brackets its own point).
"""
import numpy as np
import pandas as pd
import pytest

from idrift.analysis.dependence_sensitivity import (
    bootstrap_slope,
    gee_slope,
    per_model_pooled_slope,
)


def _make_synthetic(
    *, seed=0, beta=3.0, intercept=-1.5, n_messages=90, n_models=4, n_rep=8
):
    """Hierarchical synthetic frame with a known realized_cer log-odds slope.

    drift ~ Bernoulli(sigmoid(intercept + beta*realized_cer + msg_RE +
    model_RE)). Messages carry both a shared realized_cer level (so
    clustering by message is meaningful) and a random intercept; models
    carry their own random intercept. Returns a frame with the same
    schema the real estimators consume: `model`, `message_id`,
    `realized_cer`, `label` (in {"drift", "faithful"}).
    """
    rng = np.random.default_rng(seed)
    # Family = model.split(":")[0]; two families, two sizes each -> >= 3 models.
    models = [f"fam{m // 2}:size{m}" for m in range(n_models)]
    msg_ids = [f"msg_{i}" for i in range(n_messages)]
    msg_cer = {mid: float(rng.uniform(0.0, 1.0)) for mid in msg_ids}
    msg_re = {mid: float(rng.normal(0.0, 0.4)) for mid in msg_ids}
    model_re = {m: float(rng.normal(0.0, 0.5)) for m in models}

    rows = []
    for mid in msg_ids:
        base_cer = msg_cer[mid]
        for m in models:
            for _ in range(n_rep):
                rc = float(np.clip(base_cer + rng.normal(0.0, 0.02), 0.0, 1.0))
                eta = intercept + beta * rc + msg_re[mid] + model_re[m]
                p = 1.0 / (1.0 + np.exp(-eta))
                y = int(rng.random() < p)
                rows.append((m, mid, rc, y))

    df = pd.DataFrame(rows, columns=["model", "message_id", "realized_cer", "_y"])
    df["label"] = np.where(df["_y"] == 1, "drift", "faithful")
    return df.drop(columns="_y")


def test_per_model_pooled_slope_recovers_truth_with_finite_tau2():
    beta = 3.0
    df = _make_synthetic(seed=1, beta=beta, n_models=4)
    out = per_model_pooled_slope(df, seed=0)

    # Pooled slope on the log-odds scale, near the planted (marginal-
    # attenuated) truth. The clustered logit is a marginal model, so a mild
    # attenuation from the random intercepts is expected -- a generous
    # tolerance keeps the test about correctness, not exact recovery.
    assert np.isfinite(out["slope"])
    assert abs(out["slope"] - beta) < 1.5

    # A finite, non-negative between-model variance component.
    assert out["tau2"] is not None
    assert np.isfinite(out["tau2"])
    assert out["tau2"] >= 0.0

    # A finite CI that brackets the pooled point, one slope per model, label.
    lo, hi = out["ci"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= out["slope"] <= hi
    assert out["k_models"] == 4
    assert len(out["per_model"]) == 4
    assert isinstance(out["label"], str) and out["label"]


def test_gee_slope_returns_finite_slope_and_se():
    beta = 3.0
    df = _make_synthetic(seed=2, beta=beta, n_models=4)
    out = gee_slope(df)

    assert np.isfinite(out["slope"])
    assert np.isfinite(out["se"])
    assert out["se"] > 0.0
    lo, hi = out["ci"]
    assert lo <= out["slope"] <= hi
    # Broadly recovers the planted slope on the log-odds scale.
    assert abs(out["slope"] - beta) < 1.5
    assert isinstance(out["label"], str) and out["label"]


def test_bootstrap_slope_message_level_ci_brackets_point():
    df = _make_synthetic(seed=3, beta=3.0, n_models=4)
    out = bootstrap_slope(df, level="message", n_boot=60, seed=0)

    assert np.isfinite(out["slope"])
    lo, hi = out["ci"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= out["slope"] <= hi
    assert lo < hi  # a non-degenerate interval
    assert out["level"] == "message"
    assert isinstance(out["label"], str) and out["label"]


def test_bootstrap_slope_model_level_flagged_unstable():
    df = _make_synthetic(seed=4, beta=3.0, n_models=4)
    out = bootstrap_slope(df, level="model", n_boot=60, seed=0)

    # Only a handful of model clusters -> the estimator itself must flag the
    # instability; the CI is still returned (wide), and must be well-formed.
    assert out["level"] == "model"
    assert out["unstable"] is True
    lo, hi = out["ci"]
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo <= hi


def test_bootstrap_slope_rejects_bad_level():
    df = _make_synthetic(seed=5, n_models=3)
    with pytest.raises(ValueError, match="level"):
        bootstrap_slope(df, level="subject", n_boot=10, seed=0)


def test_per_model_pooled_slope_is_deterministic():
    df = _make_synthetic(seed=6, n_models=4)
    a = per_model_pooled_slope(df, seed=0)
    b = per_model_pooled_slope(df, seed=0)
    assert a["slope"] == b["slope"]
    assert a["ci"] == b["ci"]


# --- regression: a converged-but-diverged exchangeable fit must fall back ---
#
# On the full-corpus cohort the exchangeable working correlation returned a
# slope of -3.6e22 with a standard error of exactly 0 while statsmodels
# reported convergence=True. The finite-value guard passed it through, the
# reported spread across estimators became ~1e22, and the digest conclusion
# flipped to "NOT STABLE" on the strength of a fit that had blown up. The
# guard must reject an implausible magnitude or a degenerate standard error
# even when the optimizer claims to have converged.


def test_fit_gee_rejects_converged_but_diverged_fit(monkeypatch):
    from idrift.analysis import dependence_sensitivity as ds

    frame = _make_synthetic(seed=3)

    class _Res:
        params = {"realized_cer": -3.5871413189157216e22}
        bse = {"realized_cer": 0.0}
        converged = True

    class _Model:
        def fit(self, *a, **k):
            return _Res()

    monkeypatch.setattr(ds.smf, "gee", lambda *a, **k: _Model())
    slope, se, converged = ds._fit_gee(ds._with_outcome(frame), object())
    assert slope is None, "a slope of -3.6e22 must not be accepted as a valid fit"
    assert converged is False


def test_fit_gee_rejects_zero_standard_error(monkeypatch):
    from idrift.analysis import dependence_sensitivity as ds

    frame = _make_synthetic(seed=4)

    class _Res:
        params = {"realized_cer": 3.0}
        bse = {"realized_cer": 0.0}
        converged = True

    class _Model:
        def fit(self, *a, **k):
            return _Res()

    monkeypatch.setattr(ds.smf, "gee", lambda *a, **k: _Model())
    slope, _, converged = ds._fit_gee(ds._with_outcome(frame), object())
    assert slope is None, "a zero standard error signals a degenerate fit"
    assert converged is False


def test_fit_gee_accepts_a_plausible_fit(monkeypatch):
    from idrift.analysis import dependence_sensitivity as ds

    frame = _make_synthetic(seed=5)

    class _Res:
        params = {"realized_cer": 3.1}
        bse = {"realized_cer": 0.2}
        converged = True

    class _Model:
        def fit(self, *a, **k):
            return _Res()

    monkeypatch.setattr(ds.smf, "gee", lambda *a, **k: _Model())
    slope, se, converged = ds._fit_gee(ds._with_outcome(frame), object())
    assert slope == pytest.approx(3.1)
    assert se == pytest.approx(0.2)
    assert converged is True
