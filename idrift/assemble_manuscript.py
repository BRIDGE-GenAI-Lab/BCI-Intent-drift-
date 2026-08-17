"""Assemble manuscript/manuscript.md from the section files + a header, refreshing word counts.

Preserves the existing title/author header block (everything before the first '---'),
updates the Word count line, then concatenates: Abstract -> Introduction -> Methods ->
Results -> Discussion -> References -> Figure legends. Sections are the single source of
truth; manuscript.md is generated. Run: uv run python idrift/assemble_manuscript.py
"""
from __future__ import annotations
import re
from pathlib import Path

M = Path(__file__).resolve().parent.parent / "manuscript"


def wc(t: str) -> int:
    # Journal-body word count, matching _pub_assets/wordcount.py: exclude
    # pipe-table rows and figure/table legend lines; strip heading markers and
    # emphasis but count the heading text.
    words = 0
    for s in (ln.strip() for ln in t.split("\n")):
        if not s or s.startswith("|"):
            continue
        if re.match(r"^\*{0,2}(figure|table|efigure|etable)\s", s, re.I):
            continue
        txt = re.sub(r"^#{1,6}\s*", "", s)
        txt = re.sub(r"[*_`]", "", txt)
        words += len(txt.split())
    return words


def main() -> None:
    rd = lambda f: (M / f).read_text().strip()
    abstract, intro = rd("50_abstract.md"), rd("10_introduction.md")
    methods, results, disc = rd("20_methods.md"), rd("30_results.md"), rd("40_discussion.md")
    abody = abstract.split("## Research in context")[0]
    main_wc = wc(intro) + wc(methods) + wc(results) + wc(disc)

    cur = (M / "manuscript.md").read_text()
    header = cur.split("\n---\n", 1)[0].rstrip("\n")
    header = re.sub(
        r"\*\*Word count:\*\*.*",
        f"**Word count:** Abstract {wc(abody)}; Main text {main_wc} "
        f"(Introduction + Methods + Results + Discussion)",
        header,
    )

    refs_path = M / "references.md"
    if refs_path.exists():
        raw = refs_path.read_text()
        # pull only the numbered list (drop the citation-mapping table and verification report)
        marker = "## References (numbered"
        if marker in raw:
            body = raw.split(marker, 1)[1]
            body = body.split("\n---", 1)[0]  # stop before the verification report
            refs = "## References\n" + body.rstrip()
        else:
            refs = raw.strip()
    else:
        refs = "## References\n\n[To be built and DOI-verified with the ama-citation-zotero skill.]"
    figs = (M / "figure_legends.md").read_text().strip() if (M / "figure_legends.md").exists() else (
        "## Figure legends\n\n"
        "**Figure 1.** Intent drift and confidence calibration versus decoder character-error "
        "rate across seven current-generation open language models (8,785 outputs). "
        "(a) Overall and message-critical drift rate versus CER, with the 1% tolerance marked. "
        "(b) Reliability curve of stated confidence versus observed faithful rate "
        "(ECE 0.254; AUROC 0.791). (c) Per-model drift curves.\n\n"
        "**Figure 2.** Generational comparison. (a) Overall drift versus CER, current versus "
        "prior generation. (b) Calibration (ECE, AUROC) and message-critical tolerance-crossing "
        "by generation."
    )

    out = "\n\n".join([header + "\n\n---", abstract, intro, methods, results, disc, refs, figs]) + "\n"
    (M / "manuscript.md").write_text(out)
    print(f"assembled manuscript.md | abstract {wc(abody)} | main {main_wc}")


if __name__ == "__main__":
    main()
