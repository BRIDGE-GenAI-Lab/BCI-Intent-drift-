# tests/data/test_corpus_digitize.py
from idrift.data.corpus_digitize import parse_corpus, dedupe_rows

FIX = """Physical State Phrases
Careful, you are hurting me
My head hurts
Phone Conversation Phrases
Call me back when you can
Boston Children's Hospital
Message Banking examples from people with ALS  © 2017"""


def test_parses_categories_and_drops_boilerplate():
    rows = parse_corpus(FIX)
    cats = {r["category"] for r in rows}
    texts = {r["text"] for r in rows}
    assert "Physical State Phrases" in cats
    assert "Careful, you are hurting me" in texts
    assert not any("Boston Children" in t for t in texts)      # boilerplate removed
    assert not any("Message Banking examples" in t for t in texts)


# --- Real-PDF messiness: the naive Title-Case/ALL-CAPS heuristic from the
# brief mis-parses the actual alspharsnet.pdf. These tests lock in the fixes
# needed against realistic excerpts (see task-4-report.md for the full
# analysis of the real document).

REAL_EXCERPT = """EXPRESSIONS
Idioms
It's not my cup of tea
Just like a dream
Social Requests
Come talk with me
I want a hug.
Humor
Is that your real name?
Expressions of feelings
Angry
Better
Good
Sad
Time of Day Based Expressions
Good morning
What time is it?
Boston Children's Hospital
Message Banking examples from people with ALS © 2017"""


def test_lowercase_of_headers_are_recognized_as_categories():
    rows = parse_corpus(REAL_EXCERPT)
    cats = {r["category"] for r in rows}
    assert "Expressions of feelings" in cats
    assert "Time of Day Based Expressions" in cats


def test_single_word_feeling_entries_are_phrases_not_categories():
    rows = parse_corpus(REAL_EXCERPT)
    feelings = {r["text"] for r in rows if r["category"] == "Expressions of feelings"}
    assert {"Angry", "Better", "Good", "Sad"} <= feelings
    cats = {r["category"] for r in rows}
    # single-word entries must never be promoted to their own category
    assert "Angry" not in cats
    assert "Good" not in cats


def test_section_marker_is_dropped_not_treated_as_phrase_or_category():
    rows = parse_corpus(REAL_EXCERPT)
    cats = {r["category"] for r in rows}
    texts = {r["text"] for r in rows}
    assert "EXPRESSIONS" not in cats
    assert "EXPRESSIONS" not in texts


def test_dedupe_rows_drops_exact_duplicate_category_text_pairs():
    rows = [
        {"category": "A", "text": "Hi"},
        {"category": "A", "text": "Hi"},
        {"category": "B", "text": "Hi"},  # same text, different category: kept
        {"category": "A", "text": "Bye"},
    ]
    deduped = dedupe_rows(rows)
    assert deduped == [
        {"category": "A", "text": "Hi"},
        {"category": "B", "text": "Hi"},
        {"category": "A", "text": "Bye"},
    ]
