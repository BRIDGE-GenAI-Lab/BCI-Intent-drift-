"""Tests for the coherent faithful/degraded/drift outcome taxonomy
(revision Task 3.1).

`cos_fn`/`nli_fn`/`fluency_fn` are all STUBS (canned return values) here,
so these tests are fast, deterministic, and do not require the real
embedding/NLI/fluency models. `critical_rules.detect()` is NOT stubbed --
it runs for real (fast, deterministic spaCy/regex, no heavy ML) on the
literal `intended`/`output` strings each test passes in, since exercising
the real critical-rules composition is the whole point of this module.
Every (intended, output) pair used below was verified directly against
`idrift.adjudicate.critical_rules.detect()` before being used here (see the
revision Task 3.1 report for the confirmation transcript).
"""
from idrift.adjudicate.critical_rules import detect
from idrift.adjudicate.taxonomy import TAU, label


def _stub(nli_result="neutral", cosine=0.9, fluent=0.9):
    """Build canned cos_fn/nli_fn/fluency_fn. `nli_result` is returned for
    both NLI directions (this module's own bidirectionality logic is
    exercised by the real `nli_metric.label_row`/`test_nli_metric.py`; here
    we only need to select entail/contradict/neutral as a fixed input)."""
    def nli_fn(premise, hypothesis):
        return nli_result

    def cos_fn(a, b):
        return cosine

    def fluency_fn(text):
        return fluent

    return cos_fn, nli_fn, fluency_fn


def test_actionable_omission_is_not_faithful_drift_when_fluent():
    # "I am in pain and need help" -> "I am in pain": the old NLI+cosine-only
    # labeler would call this faithful (high lexical/cosine overlap, no
    # contradiction). NLI is stubbed "neutral" (neither entail nor
    # contradict) so this case is decided purely by critical_rules'
    # actionable_omission flag, proving that flag -- not NLI -- is what
    # catches the omission.
    cos_fn, nli_fn, fluency_fn = _stub(nli_result="neutral", cosine=0.95, fluent=0.9)
    result = label("I am in pain and need help", "I am in pain", cos_fn, nli_fn, fluency_fn)

    assert result["label"] == "drift"
    assert result["label"] != "faithful"
    assert result["fluent"] is True
    assert "actionable_omission" in result["reason"]


def test_high_cosine_paraphrase_is_faithful():
    cos_fn, nli_fn, fluency_fn = _stub(nli_result="entail", cosine=0.95, fluent=0.9)
    result = label("I am in pain", "I'm suffering", cos_fn, nli_fn, fluency_fn)

    assert result == {"label": "faithful", "fluent": True, "reason": "nli_bidirectional_entailment"}


def test_negation_flip_is_drift_and_fluent_true():
    # NLI stubbed "neutral" so the negation flip is caught by critical_rules,
    # not by NLI.
    cos_fn, nli_fn, fluency_fn = _stub(nli_result="neutral", cosine=0.9, fluent=0.9)
    result = label("do not move", "move", cos_fn, nli_fn, fluency_fn)

    assert result["label"] == "drift"
    assert result["fluent"] is True
    assert "negation_flip" in result["reason"]


def test_word_salad_contradiction_is_degraded():
    cos_fn, nli_fn, fluency_fn = _stub(nli_result="contradict", fluent=0.1)
    result = label("I am in pain", "blarg not the xyzzy", cos_fn, nli_fn, fluency_fn)

    assert result == {"label": "degraded", "fluent": False, "reason": "nli_contradiction"}


def test_meaning_differs_but_not_fluent_is_degraded_never_drift():
    # Coherence invariant: whatever the meaning-difference signal, a
    # disfluent output must never be labeled "drift". Three independent
    # meaning-difference paths (NLI contradiction, a critical rule, and the
    # cosine fallback), all with fluent=False.
    cases = [
        ("I am in pain", "blarg not the xyzzy", _stub(nli_result="contradict", fluent=0.1)),
        ("do not move", "move", _stub(nli_result="neutral", fluent=0.1)),
        ("the sky is blue", "purple elephants dance wildly",
         _stub(nli_result="neutral", cosine=0.1, fluent=0.1)),
    ]
    for intended, output, (cos_fn, nli_fn, fluency_fn) in cases:
        result = label(intended, output, cos_fn, nli_fn, fluency_fn)
        assert result["fluent"] is False
        assert result["label"] == "degraded"
        assert result["label"] != "drift"


def test_meaning_preserved_but_not_fluent_is_degraded_not_faithful():
    # A disfluent output must never be "faithful" either, even when NLI/
    # cosine see no meaning difference: "degraded" means "not fluent",
    # independent of the meaning verdict.
    cos_fn, nli_fn, fluency_fn = _stub(nli_result="entail", cosine=0.95, fluent=0.1)
    result = label("call my doctor", "Please call my doctor.", cos_fn, nli_fn, fluency_fn)

    assert result["label"] == "degraded"
    assert result["fluent"] is False


def test_label_is_always_one_of_the_three_taxonomy_values_never_message_critical():
    combos = [
        _stub(nli_result="entail", cosine=0.95, fluent=0.9),
        _stub(nli_result="contradict", cosine=0.1, fluent=0.9),
        _stub(nli_result="neutral", cosine=0.1, fluent=0.1),
        _stub(nli_result="neutral", cosine=0.95, fluent=0.1),
    ]
    for cos_fn, nli_fn, fluency_fn in combos:
        result = label("call my doctor", "Please call my doctor.", cos_fn, nli_fn, fluency_fn)
        assert result["label"] in {"faithful", "degraded", "drift"}
        assert "message-critical" not in result
        assert set(result.keys()) == {"label", "fluent", "reason"}


def test_tau_is_pinned_and_is_the_exact_fluency_decision_boundary():
    cos_fn, nli_fn, _ = _stub(nli_result="contradict", cosine=0.1)
    just_below = label("I am in pain", "I am not in pain", cos_fn, nli_fn, lambda text: TAU - 0.01)
    just_at = label("I am in pain", "I am not in pain", cos_fn, nli_fn, lambda text: TAU)

    assert just_below["fluent"] is False and just_below["label"] == "degraded"
    assert just_at["fluent"] is True and just_at["label"] == "drift"


def test_crit_injection_is_equivalent_to_internal_detect():
    # Step A (rev Task 5.0): label() grows an optional `crit=` keyword so a
    # batched runner can inject a precomputed critical_rules.detect() result
    # instead of re-parsing spaCy per row. This must be a pure pass-through,
    # not a different logic path: injecting the SAME dict detect() would
    # have produced internally must return an identical result to calling
    # label() without `crit` at all, for every case below (a negation flip,
    # a clean paraphrase, and an omission).
    cases = [
        ("do not move", "move", _stub(nli_result="neutral", cosine=0.9, fluent=0.9)),
        ("I am in pain", "I'm suffering", _stub(nli_result="entail", cosine=0.95, fluent=0.9)),
        ("I am in pain and need help", "I am in pain", _stub(nli_result="neutral", cosine=0.95, fluent=0.9)),
    ]
    for intended, output, (cos_fn, nli_fn, fluency_fn) in cases:
        without_crit = label(intended, output, cos_fn, nli_fn, fluency_fn)
        injected = label(intended, output, cos_fn, nli_fn, fluency_fn, crit=detect(intended, output))
        assert injected == without_crit


def test_label_is_deterministic():
    cos_fn, nli_fn, fluency_fn = _stub(nli_result="neutral", cosine=0.95, fluent=0.9)
    first = label("I am in pain and need help", "I am in pain", cos_fn, nli_fn, fluency_fn)
    second = label("I am in pain and need help", "I am in pain", cos_fn, nli_fn, fluency_fn)

    assert first == second
