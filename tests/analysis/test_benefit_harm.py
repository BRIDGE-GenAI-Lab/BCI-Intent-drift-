"""Tests for the raw-decode benefit-harm transition matrix (revision Task B1).

TDD: these were written before `idrift.analysis.benefit_harm` existed. The
core assertions pin the five clinical transition categories and the net
`faithful_gained_per_drift_introduced` arithmetic; `label_raw_decode` is
covered only by a mocked unit test (its real run loads heavy NLI/embedding
models and is executed separately, never in the test suite).
"""
import math

import pandas as pd
import pytest

from idrift.analysis.benefit_harm import label_raw_decode, transition_matrix


# ---------------------------------------------------------------------------
# transition_matrix
# ---------------------------------------------------------------------------


def _one_group_raw():
    """Raw (model-independent) labels for one (corpus, cer_target) group:
    three degraded raw decodes (E, B, A) and two faithful ones (C, D)."""
    return pd.DataFrame(
        [
            {"message_id": "A", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "raw_label": "degraded"},
            {"message_id": "B", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "raw_label": "degraded"},
            {"message_id": "C", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "raw_label": "faithful"},
            {"message_id": "D", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "raw_label": "faithful"},
            {"message_id": "E", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "raw_label": "degraded"},
        ]
    )


def _one_group_attempts(model="m1"):
    """One LLM's output labels for the same five exposures, arranged to hit
    exactly one of each of the five clinical categories:
    A degraded->faithful (rescue), B degraded->drift (silent_failure),
    C faithful->drift (llm_induced_harm), D faithful->faithful (no_benefit),
    E degraded->degraded (no_correction)."""
    return pd.DataFrame(
        [
            {"model": model, "message_id": "A", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "label": "faithful"},
            {"model": model, "message_id": "B", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "label": "drift"},
            {"model": model, "message_id": "C", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "label": "drift"},
            {"model": model, "message_id": "D", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "label": "faithful"},
            {"model": model, "message_id": "E", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "label": "degraded"},
        ]
    )


def test_classifies_five_clinical_categories_and_net_metric():
    res = transition_matrix(_one_group_attempts(), _one_group_raw())
    bg = res["by_group"]
    row = bg[(bg["corpus"] == "AUTH") & (bg["cer_target"] == 0.2)].iloc[0]

    # degraded -> faithful == rescue
    assert row["n_rescue"] == 1
    # degraded -> drift == silent_failure
    assert row["n_silent_failure"] == 1
    # faithful -> drift == llm_induced_harm
    assert row["n_llm_induced_harm"] == 1
    # faithful -> faithful == no_benefit
    assert row["n_no_benefit"] == 1
    # degraded -> degraded == no_correction
    assert row["n_no_correction"] == 1
    assert row["n_total"] == 5

    # net = rescues / (llm_induced_harm + silent_failure) = 1 / (1 + 1)
    assert row["faithful_gained_per_drift_introduced"] == pytest.approx(0.5)

    # rates are counts / group total
    assert row["rate_rescue"] == pytest.approx(1 / 5)


def test_raw_labels_broadcast_across_models():
    # Same raw decode broadcasts to every model at the same key. Two models,
    # both rescuing A, both harming C -> counts double, net metric unchanged.
    attempts = pd.concat([_one_group_attempts("m1"), _one_group_attempts("m2")], ignore_index=True)
    res = transition_matrix(attempts, _one_group_raw())
    row = res["by_group"].iloc[0]
    assert row["n_rescue"] == 2
    assert row["n_llm_induced_harm"] == 2
    assert row["n_silent_failure"] == 2
    assert row["n_total"] == 10
    assert row["faithful_gained_per_drift_introduced"] == pytest.approx(0.5)
    # overall bucket agrees with the single group
    assert res["overall"]["n_rescue"] == 2


def test_net_metric_divide_by_zero_guarded():
    # A group with a rescue but no introduced drift -> infinite net benefit.
    raw = pd.DataFrame(
        [{"message_id": "A", "corpus": "CRIT", "cer_target": 0.1, "replicate_idx": 0, "raw_label": "degraded"}]
    )
    attempts = pd.DataFrame(
        [{"model": "m1", "message_id": "A", "corpus": "CRIT", "cer_target": 0.1, "replicate_idx": 0, "label": "faithful"}]
    )
    res = transition_matrix(attempts, raw)
    row = res["by_group"].iloc[0]
    assert math.isinf(row["faithful_gained_per_drift_introduced"])
    assert row["net_metric_note"] != ""

    # A group with neither rescues nor introduced drift -> undefined (NaN in
    # the frame; None in the overall dict), still with a note.
    raw2 = pd.DataFrame(
        [{"message_id": "A", "corpus": "CRIT", "cer_target": 0.1, "replicate_idx": 0, "raw_label": "faithful"}]
    )
    attempts2 = pd.DataFrame(
        [{"model": "m1", "message_id": "A", "corpus": "CRIT", "cer_target": 0.1, "replicate_idx": 0, "label": "faithful"}]
    )
    res2 = transition_matrix(attempts2, raw2)
    row2 = res2["by_group"].iloc[0]
    assert math.isnan(row2["faithful_gained_per_drift_introduced"])
    assert res2["overall"]["faithful_gained_per_drift_introduced"] is None


def test_handles_non_degraded_raw_labels_generally():
    # A raw stream that is itself drift (fluent-but-wrong) is not one of the
    # five named categories; it must still be counted, under a generic key,
    # never silently dropped.
    raw = pd.DataFrame(
        [{"message_id": "A", "corpus": "AUTH", "cer_target": 0.3, "replicate_idx": 0, "raw_label": "drift"}]
    )
    attempts = pd.DataFrame(
        [{"model": "m1", "message_id": "A", "corpus": "AUTH", "cer_target": 0.3, "replicate_idx": 0, "label": "faithful"}]
    )
    res = transition_matrix(attempts, raw)
    row = res["by_group"].iloc[0]
    assert row["n_total"] == 1
    # counted somewhere other than the five named categories
    named = row[["n_no_benefit", "n_rescue", "n_silent_failure", "n_llm_induced_harm", "n_no_correction"]].sum()
    assert named == 0
    assert res["overall"]["n_total"] == 1


def test_groups_by_corpus_and_cer_band():
    raw = pd.concat(
        [_one_group_raw(), _one_group_raw().assign(corpus="CRIT", cer_target=0.4)], ignore_index=True
    )
    attempts = pd.concat(
        [_one_group_attempts(), _one_group_attempts().assign(corpus="CRIT", cer_target=0.4)], ignore_index=True
    )
    res = transition_matrix(attempts, raw)
    bg = res["by_group"]
    assert len(bg) == 2
    assert set(zip(bg["corpus"], bg["cer_target"])) == {("AUTH", 0.2), ("CRIT", 0.4)}


# ---------------------------------------------------------------------------
# label_raw_decode -- mocked only (never runs real models in the test suite)
# ---------------------------------------------------------------------------


def test_label_raw_decode_maps_noisy_to_pipeline(tmp_path, monkeypatch):
    captured = {}

    def fake_compute_signals(df, **kwargs):
        # record what the labeling layer was asked to score
        captured["input"] = df.copy()
        return pd.DataFrame({"fluency_raw": [0.9] * len(df)}, index=df.index)

    def fake_derive_labels(signals_df, **kwargs):
        # trivial faithful/degraded rule keyed on the pair the pipeline sees
        labels = [
            "faithful" if i == o else "degraded"
            for i, o in zip(signals_df["intended_text"], signals_df["output_message"])
        ]
        return pd.DataFrame({"label": labels}, index=signals_df.index)

    monkeypatch.setattr("idrift.adjudicate.label_runner.compute_signals", fake_compute_signals)
    monkeypatch.setattr("idrift.adjudicate.label_runner.derive_labels", fake_derive_labels)

    exposure = pd.DataFrame(
        [
            {"message_id": "A", "corpus": "AUTH", "cer_target": 0.0, "replicate_idx": 0, "intended_text": "turn me", "noisy_text": "turn me"},
            {"message_id": "A", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 0, "intended_text": "turn me", "noisy_text": "trn me"},
            {"message_id": "A", "corpus": "AUTH", "cer_target": 0.2, "replicate_idx": 1, "intended_text": "turn me", "noisy_text": "trn me"},
            {"message_id": "B", "corpus": "CRIT", "cer_target": 0.2, "replicate_idx": 0, "intended_text": "no water", "noisy_text": "no wxer"},
        ]
    )
    out_path = tmp_path / "raw_labeled.parquet"
    out = label_raw_decode(exposure, str(out_path))

    # the pipeline was fed the noisy_text AS output_message, intended as ref,
    # de-duplicated over unique (intended, noisy) pairs -> 3 unique of 4 rows
    inp = captured["input"]
    assert set(inp.columns) >= {"intended_text", "output_message"}
    assert len(inp) == 3
    assert set(inp["output_message"]) == {"turn me", "trn me", "no wxer"}

    # output is keyed and broadcast back to all four exposure rows
    assert out_path.exists()
    written = pd.read_parquet(out_path)
    assert list(written.columns) == ["message_id", "corpus", "cer_target", "replicate_idx", "raw_label"]
    assert len(written) == 4
    lab = {
        (r.message_id, r.cer_target, r.replicate_idx): r.raw_label
        for r in written.itertuples(index=False)
    }
    assert lab[("A", 0.0, 0)] == "faithful"   # noisy == intended
    assert lab[("A", 0.2, 0)] == "degraded"
    assert lab[("A", 0.2, 1)] == "degraded"   # dup pair broadcast
    assert lab[("B", 0.2, 0)] == "degraded"
