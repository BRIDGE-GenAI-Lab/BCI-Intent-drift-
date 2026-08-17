"""Tests for the rule-based critical-error detectors (revision Task 3.2).

Every case below asserts the FULL returned dict (not just the one flag the
case is named for), so a detector that over-fires on an unrelated input is
caught here rather than silently passing a narrower assertion.
"""
from idrift.adjudicate.critical_rules import detect

ALL_FALSE = {
    "negation_flip": False,
    "numeral_change": False,
    "recipient_change": False,
    "urgency_change": False,
    "actionable_omission": False,
}


def _expect(intended, output, **true_flags):
    expected = dict(ALL_FALSE)
    expected.update(true_flags)
    assert detect(intended, output) == expected


def test_negation_flip_do_not_move_to_move():
    _expect("do not move", "move", negation_flip=True)


def test_numeral_change_dose():
    _expect("5 mg", "15 mg", numeral_change=True)


def test_recipient_change_nurse_to_daughter():
    _expect("call the nurse", "call my daughter", recipient_change=True)


def test_urgency_change_now_dropped():
    _expect("call the nurse now", "call the nurse", urgency_change=True)


def test_actionable_omission_need_help_dropped():
    _expect("I am thirsty and need help", "I am thirsty", actionable_omission=True)


def test_clean_paraphrase_triggers_none():
    _expect("call my doctor", "Please call my doctor.")


def test_negation_flip_pain_statement():
    _expect("I am in pain", "I am not in pain", negation_flip=True)


def test_omission_and_urgency_both_fire():
    _expect(
        "I am in pain and need help now",
        "I am in pain",
        actionable_omission=True,
        urgency_change=True,
    )


def test_negation_flip_is_symmetric_added_or_removed():
    # Negation added going the other direction should also flip.
    _expect("move", "do not move", negation_flip=True)


def test_recipient_change_not_triggered_by_shared_role():
    _expect("please call the nurse", "call the nurse please")


# --- Regression tests: rev Task 3.2 review, "faithful verb-synonym
# paraphrase wrongly flagged as actionable_omission" -----------------------
#
# The original implementation keyed `actionable_omission` on the action-verb
# LEMMA set alone, so any verb-synonym paraphrase (bring -> get, want -> like)
# flagged as an omission even though the actionable content (the object of
# the request) survived untouched. The fixed rule keys on the surviving
# content ARGUMENT of the actionable clause (its object noun, or its
# clausal complement when the verb has no object), not the verb lemma, so a
# synonym verb with the same surviving argument must NOT flag.


def test_actionable_omission_not_triggered_by_verb_synonym_bring_get():
    # "bring" -> "get": different verb lemmas, but "water" (the actionable
    # content) survives untouched -- must NOT be flagged as an omission.
    _expect("bring me water", "get me water")


def test_actionable_omission_not_triggered_by_verb_synonym_please_bring_get():
    _expect("please bring my water", "please get my water")


def test_actionable_omission_not_triggered_by_verb_synonym_want_would_like():
    # "want" -> "would like": different verb lemmas and no object noun at
    # all, but the clausal complement "move" (the actionable content)
    # survives untouched -- must NOT be flagged as an omission.
    _expect("I want to move", "I would like to move")


def test_actionable_omission_still_triggered_when_content_is_dropped_pain():
    # Genuine omission: "need help" (the actionable clause) is dropped
    # entirely, not just paraphrased with a synonym verb.
    _expect(
        "I am in pain and need help now",
        "I am in pain",
        actionable_omission=True,
        urgency_change=True,
    )


def test_actionable_omission_still_triggered_when_content_is_dropped_thirsty():
    _expect("I am thirsty and need help", "I am thirsty", actionable_omission=True)


def test_actionable_omission_still_triggered_when_whole_request_is_dropped():
    # Genuine omission: the entire actionable request ("call the nurse") is
    # replaced by unrelated content, with neither the verb "call" nor any
    # surviving content argument present in the output. "nurse" also
    # vanishing legitimately fires recipient_change too -- both flags are
    # correct here, not a conflict.
    _expect(
        "call the nurse",
        "I am fine",
        actionable_omission=True,
        recipient_change=True,
    )


def test_actionable_omission_synonym_mechanism_generalizes_to_new_noun():
    # Same verb-synonym pair as the "water" case above (both "bring" and
    # "get" are already in the actionable-verb lexicon), but with a
    # different object noun never used elsewhere in this test file --
    # confirms the fix is a general surviving-argument rule, not a lookup
    # keyed on the literal reviewed test strings.
    _expect("bring my glasses", "get my glasses")
