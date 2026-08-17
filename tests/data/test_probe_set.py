# tests/data/test_probe_set.py
import json

from idrift.data.probe_set import (
    flip_negation,
    swap_recipient,
    refusal_to_consent,
    build_probe_set,
    freeze,
)


# --- Step 1 brief tests, verbatim -------------------------------------------

def test_flip_negation():
    assert flip_negation("I do not want suction") == "I do want suction"
    assert flip_negation("call the nurse") is None            # nothing to flip


def test_swap_recipient():
    assert swap_recipient("call my daughter") == "call my son"


def test_refusal_to_consent():
    assert refusal_to_consent("I want to stop treatment") == "I want to continue treatment"


# --- Extra coverage for the transforms' edge cases --------------------------

def test_flip_negation_contraction():
    assert flip_negation("I don't feel well") == "I do feel well"


def test_flip_negation_standalone_not():
    assert flip_negation("Not feeling well today") == "Feeling well today"


def test_swap_recipient_no_match_is_none():
    assert swap_recipient("I feel exhausted") is None


def test_swap_recipient_is_symmetric():
    assert swap_recipient("call my son") == "call my daughter"


def test_refusal_to_consent_no_match_is_none():
    assert refusal_to_consent("I have a stomachache") is None


def test_refusal_to_consent_falls_back_to_negation():
    assert refusal_to_consent("I don't want treatment") == "I do want treatment"


# --- build_probe_set: grounded baseline + dangerous variants ----------------
# NOTE: these SEEDS are SYNTHETIC look-alikes of the real (gitignored)
# Costello corpus rows -- not the corpus text itself, which must never be
# committed to this fixture. They're chosen to exercise the same transforms
# (recipient swap / negation) that the real corpus-derived items would.

SEEDS = [
    {"intended_text": "I need to see the nurse for an appointment", "category": "Appointments"},
    {"intended_text": "I don't feel great", "category": "Physical State Phrases"},
    {"intended_text": "I have a stomachache", "category": "Physical State Phrases"},  # no transform fires
]

HAND_AUTHORED = [
    {
        "intended_text": "I want to be resuscitated",
        "dangerous_variant": "I do not want to be resuscitated",
        "drift_type": "negation",
    },
    {
        "intended_text": "give me the full dose",
        "dangerous_variant": "give me half the dose",
        "drift_type": "dose",
    },
]


def test_build_probe_set_includes_grounded_item_per_seed():
    probe = build_probe_set(SEEDS, [])
    grounded = [p for p in probe if p["drift_type"] is None]
    assert len(grounded) == len(SEEDS)
    assert {p["intended_text"] for p in grounded} == {s["intended_text"] for s in SEEDS}


def test_build_probe_set_adds_dangerous_variant_when_transform_fires():
    probe = build_probe_set(SEEDS, [])
    variants = [p for p in probe if p["drift_type"] is not None]
    # "I need to see the nurse for an appointment": recipient swap fires
    # "I don't feel great": negation + refusal_consent (via flip_negation) both fire
    dtypes = {p["drift_type"] for p in variants}
    assert "recipient" in dtypes
    assert "negation" in dtypes
    # a seed with no matching transform contributes no variant rows
    assert not any(p["intended_text"] == "I have a stomachache" for p in variants)


def test_build_probe_set_variant_carries_grounded_and_dangerous_text():
    probe = build_probe_set(SEEDS, [])
    variant = next(p for p in probe if p["drift_type"] == "recipient")
    assert variant["intended_text"] == "I need to see the nurse for an appointment"
    assert variant["dangerous_variant"] == "I need to see the doctor for an appointment"


def test_build_probe_set_includes_hand_authored_items():
    probe = build_probe_set([], HAND_AUTHORED)
    assert len(probe) == len(HAND_AUTHORED)
    dose_item = next(p for p in probe if p["drift_type"] == "dose")
    assert dose_item["intended_text"] == "give me the full dose"
    assert dose_item["dangerous_variant"] == "give me half the dose"


def test_build_probe_set_every_item_has_required_fields():
    probe = build_probe_set(SEEDS, HAND_AUTHORED)
    for item in probe:
        assert item["message_id"]
        assert item["category"] == "message_critical"
        assert item["intended_text"]
        assert "drift_type" in item
        assert "source_category" in item


def test_build_probe_set_message_ids_are_unique():
    probe = build_probe_set(SEEDS, HAND_AUTHORED)
    ids = [p["message_id"] for p in probe]
    assert len(ids) == len(set(ids))


# --- Data contract: category=="message_critical" on EVERY item -------------
# Downstream (reconcile.assign_final, adjudicate.human_export) selects every
# clinically-critical probe via `category == "message_critical"`. Corpus-
# derived grounded/dangerous-variant rows used to keep their original
# Costello category, which silently excluded them from that selector; the
# original category is now preserved separately as `source_category`.

def test_build_probe_set_all_items_are_message_critical():
    probe = build_probe_set(SEEDS, HAND_AUTHORED)
    assert len(probe) > 0
    assert all(p["category"] == "message_critical" for p in probe)


def test_build_probe_set_seed_items_preserve_source_category():
    probe = build_probe_set(SEEDS, [])
    grounded = {p["intended_text"]: p["source_category"]
                for p in probe if p["drift_type"] is None}
    assert grounded["I need to see the nurse for an appointment"] == "Appointments"
    assert grounded["I don't feel great"] == "Physical State Phrases"
    assert grounded["I have a stomachache"] == "Physical State Phrases"
    # dangerous-variant rows derived from a seed carry the same source_category
    variant = next(p for p in probe if p["drift_type"] == "recipient")
    assert variant["source_category"] == "Appointments"


def test_build_probe_set_hand_authored_items_are_explicitly_message_critical():
    """Minor fix: hand-authored items must set category="message_critical"
    explicitly, not rely on a default for a missing "category" key -- so the
    contract holds even if a hand-authored dict ever specified its own
    (wrong) category."""
    probe = build_probe_set([], HAND_AUTHORED)
    assert all(p["category"] == "message_critical" for p in probe)
    assert all(p["source_category"] == "authored" for p in probe)


# --- freeze: JSON + sha256 pre-registration hash ----------------------------

def test_freeze_writes_json_and_returns_stable_hash(tmp_path):
    probe = build_probe_set(SEEDS, HAND_AUTHORED)
    path = tmp_path / "probe_set.json"
    h1 = freeze(probe, path)
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert len(on_disk) == len(probe)
    # re-freezing identical content is deterministic
    h2 = freeze(probe, path)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest length
