"""Digitize the Costello/Boston Children's Hospital ALS Message Banking
vocabulary PDF (`alspharsnet.pdf`) into a categorized intended-message
corpus.

`parse_corpus(text)` turns raw page text into `{"category","text"}` rows.
The real source PDF is messier than a synthetic fixture: category headers
like "Expressions of feelings" and "Time of Day Based Expressions" contain
lowercase "of"/"feelings" and so are NOT `.istitle()`, while many single-word
PHRASE entries under those categories (e.g. "Angry", "Good", "Alright") DO
satisfy a naive Title-Case/ALL-CAPS-no-terminal-punctuation heuristic and
would be wrongly promoted to their own categories. Verified directly against
the real document (see task-4-report.md): every one of the 41 real category
headers below appears exactly once, verbatim, as its own line -- so matching
against this known finite header set is the robust approach for this
specific source document, in preference to the generic heuristic.

The runner (`digitize` / `main`) reads the real PDF via pdfplumber, strips
front matter (definitions/terminology preceding the vocabulary) and the
trailing "Additional recordings completed..." freeform section, parses,
dedupes exact (category, text) duplicate rows, assigns
`message_id=f"costello_{i:04d}"`, and checkpoints to
`output/intermediate/corpus_costello.parquet`.
"""
import os
import re
from pathlib import Path

import pandas as pd

from idrift.lib.io_utils import save_checkpoint, sha256_file, log_provenance

# Running header/footer text repeated on every page of the source PDF, plus
# copyright-line fragments -- never real phrases.
BOILER = ("Boston Children", "Message Banking examples", "ALS Augmentative", "© ")

# The real Costello/BCH ALS Message Banking vocabulary (alspharsnet.pdf,
# John M. Costello, Boston Children's Hospital, © 2011-2017) uses this fixed,
# finite set of category headers. Confirmed against the actual PDF text:
# each string below appears exactly once, verbatim, as a standalone line.
COSTELLO_HEADERS = frozenset({
    "Idioms", "Social Requests", "Humor", "Expressions of feelings",
    "Time of Day Based Expressions", "Topic Continuations", "Appointments",
    "Equipment Related Phrases", "Physical State Phrases",
    "Ice Breaker(Conversation Opener) Phrases", "Phone Conversation Phrases",
    "Goodbye/Farewell Phrases", "Request for Assistance", "Exclamations",
    "Encourage/Discourage Comments", "Location Marker Phrases",
    "Conversation Modifiers/Repairs", "Interpersonal Comments",
    "Temporal Markers", "Opinion/Perspective Phrases",
    "Requests for Specific Information", "Generic Request Phrases",
    "Conversation Control Phrases", "Social Amenities",
    "Generic Responses Phrases", "Nourishment/Food", "Likes/Dislikes",
    "Appreciation", "Expressions of Love", "Conversing About ALS",
    "Health and Safety", "Family and Close Friends' Names", "Compassion",
    "Environmental/Elements", "Occasions/Holidays/Celebrations",
    "Personal Care/Needs", "Self Determination", "Suggestions/Initiations",
    "Family Routines", "Modifying Other's Behavior",
    "Agreement/Disagreement Phrases",
})

# Front-matter section markers that introduce the vocabulary but are not
# themselves categories -- dropped rather than treated as headers or phrases.
SECTION_MARKERS = ("DEFINITIONS", "TERMINOLOGY", "THE VOCABULARY", "EXPRESSIONS")

# Generic fallback heuristic (only used when known_headers is falsy/None --
# not exercised by the real Costello corpus, which always supplies the known
# header set above). Kept for callers digitizing a different, unknown-header
# corpus with this same module.
HEADER = re.compile(r"^[A-Z][A-Za-z '/()]+$")


def _is_header(line: str, known_headers=COSTELLO_HEADERS) -> bool:
    if known_headers:
        # Known-header mode: exact match only. Do NOT fall back to the fuzzy
        # heuristic here -- it would wrongly promote single-word phrase
        # entries (e.g. "Angry", "Good", "Alright") to categories.
        return line in known_headers
    return bool(HEADER.match(line)) and (line.istitle() or line.isupper()) and not line.endswith(('.', '?', '!'))


def parse_corpus(text: str, known_headers=COSTELLO_HEADERS):
    """Parse category-headed phrase text into [{"category","text"}, ...].

    known_headers: finite set of exact category-header strings to match
        against (default: the real Costello corpus headers). Pass None to
        fall back to the generic Title-Case/ALL-CAPS heuristic for a
        different/unknown corpus.
    """
    rows, cat = [], "Uncategorized"
    for raw in text.splitlines():
        line = raw.strip()
        if not line or any(b in line for b in BOILER) or line.isdigit():
            continue
        if line in SECTION_MARKERS:
            continue
        if _is_header(line, known_headers):
            cat = line
            continue
        rows.append({"category": cat, "text": line})
    return rows


def dedupe_rows(rows):
    """Drop exact duplicate (category, text) rows, preserving first-seen
    order. Cross-category duplicates (the same phrase legitimately filed
    under two categories in the source, e.g. "Good morning" under both
    "Time of Day Based Expressions" and "Interpersonal Comments") are kept:
    that reflects the source document's own categorization, not an error.
    """
    seen = set()
    out = []
    for r in rows:
        key = (r["category"], r["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _strip_front_and_trailing_matter(text: str) -> str:
    """Drop front-matter prose (definitions/terminology before the
    vocabulary proper) and the trailing freeform "Additional recordings..."
    section, neither of which is phrase-formatted category content."""
    lines = text.splitlines()
    markers = COSTELLO_HEADERS | set(SECTION_MARKERS)
    start = next((i for i, l in enumerate(lines) if l.strip() in markers), 0)
    end = next(
        (i for i, l in enumerate(lines) if "additional recordings completed" in l.lower()),
        len(lines),
    )
    return "\n".join(lines[start:end])


DEFAULT_PDF_PATH = Path(os.environ.get(
    "IDRIFT_COSTELLO_PDF", "/Volumes/Extreme SSD/Mimic-IV/alspharsnet.pdf"
))


def extract_pdf_text(pdf_path=DEFAULT_PDF_PATH) -> str:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join((page.extract_text() or "") for page in pdf.pages)


def digitize(pdf_path=DEFAULT_PDF_PATH) -> pd.DataFrame:
    """Full pipeline: extract -> strip front/trailing matter -> parse ->
    dedupe -> assign message_id -> return the checkpoint-ready DataFrame."""
    text = extract_pdf_text(pdf_path)
    body = _strip_front_and_trailing_matter(text)
    rows = dedupe_rows(parse_corpus(body))
    return pd.DataFrame([
        {
            "message_id": f"costello_{i:04d}",
            "corpus": "costello",
            "category": r["category"],
            "intended_text": r["text"],
        }
        for i, r in enumerate(rows)
    ])


REQUIRED_NONEMPTY_CATEGORIES = (
    "Physical State Phrases",
    "Phone Conversation Phrases",
    "Appointments",
    "Expressions of feelings",
)


def main(pdf_path=DEFAULT_PDF_PATH):
    df = digitize(pdf_path)

    assert len(df) >= 200, f"expected >=200 phrases, got {len(df)}"
    counts = df["category"].value_counts()
    for cat in REQUIRED_NONEMPTY_CATEGORIES:
        n = int(counts.get(cat, 0))
        assert n > 0, f"required category {cat!r} is empty"
    for boiler in BOILER:
        leaked = df["intended_text"].str.contains(re.escape(boiler), regex=True)
        assert not leaked.any(), f"boilerplate {boiler!r} leaked into phrases"

    print("Category -> phrase count:")
    for cat, n in counts.items():
        print(f"  {n:4d}  {cat}")
    print(f"Total phrases: {len(df)}")

    p = save_checkpoint(df, "corpus_costello")
    log_provenance({
        "corpus_costello": {
            "source_pdf": str(pdf_path),
            "source_sha256": sha256_file(pdf_path),
            "phrase_count": len(df),
            "checkpoint": str(p),
        }
    })
    return df


if __name__ == "__main__":
    main()
