"""Tests for the fluency/grammaticality proxy (revision Task 3.1).

Loads the real gpt2 model once (module-scoped fixture) and keeps the
assertions to a minimal, deterministic check that the proxy actually
discriminates grammatical text from word salad, per the brief.
`tests/adjudicate/test_taxonomy.py` covers the label() decision logic with
STUB fluency values and does not depend on this module's real numeric
behavior, so this file stays small and does not need to be re-run to
validate the taxonomy's logic.
"""
import pytest

from idrift.adjudicate.fluency import make_fluency_fn
from idrift.adjudicate.taxonomy import TAU

FLUENT_TEXT = "I am in pain and need help."
WORD_SALAD_TEXT = "asdkj qpwoe zznrx blaghfut"


@pytest.fixture(scope="module")
def fluency_score():
    return make_fluency_fn()


def test_fluency_proxy_discriminates_grammatical_from_word_salad(fluency_score):
    fluent = fluency_score(FLUENT_TEXT)
    word_salad = fluency_score(WORD_SALAD_TEXT)

    assert fluent > word_salad
    assert fluent >= TAU
    assert word_salad < TAU
