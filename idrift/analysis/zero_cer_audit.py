"""Zero-CER drift forensic categorizer (revision Task 5.4).

Answers Rev1 #4 / Rev2 #18: why does the automated ensemble call "drift" on
a message whose realized, decoder-level corruption was zero? A drift label
at `realized_cer == 0` means the model changed the intended meaning on its
own, with a perfectly clean input to work from -- this module bins every
such case into a documented cause taxonomy so the manuscript can report WHY,
not just THAT, zero-CER drift occurs.

Pure and deterministic: every cause is decided from already-computed signal
columns (the same `cos_*`/`nli_*`/`crit_*`/`fluency_raw` columns
`idrift.adjudicate.label_runner.compute_signals` produces) and plain string
normalization. No models are loaded and no re-inference happens here.

Cause taxonomy and precedence
------------------------------
Each qualifying row (`realized_cer == 0` and `label == "drift"`) is assigned
EXACTLY ONE cause, checked in this fixed precedence order -- a row that
would qualify for more than one cause (e.g. a genuine negation flip whose
embeddings also happen to sit above the paraphrase threshold) is resolved by
the FIRST cause it matches, not summed across causes:

    critical_substitution > formatting > paraphrase > overgenerative > other

- `critical_substitution`: any of the five `crit_*` flags
  (`crit_negation_flip`, `crit_numeral_change`, `crit_recipient_change`,
  `crit_urgency_change`, `crit_actionable_omission`) is True. A genuine
  intent change was detected even though the decoder introduced zero
  character error -- the model itself substituted the critical content.
- `formatting`: `output_message` equals `intended_text` after lowercasing,
  punctuation removal, and whitespace collapse. A benign, cosmetic-only
  difference the automated ensemble nonetheless called "drift" on -- a
  likely classifier false positive on a purely stylistic change.
- `paraphrase`: cosine similarity on BOTH embedders (`cos_mpnet` and
  `cos_minilm`) is at or above `nli_metric.COS_HI` AND at least one of the
  two NLI models entails in both directions (`nli_deberta_fwd`/
  `nli_deberta_bwd`, or `nli_roberta_fwd`/`nli_roberta_bwd`, both
  `"entail"`). Reached only when `critical_substitution` did not already
  match, so this is implicitly a no-crit-flag case. A semantically
  equivalent reword the ensemble still labeled "drift" -- a candidate
  classifier false positive or an overly strict boundary.
- `overgenerative`: `output_message` is at least `OVERGEN_LEN_RATIO` times
  longer (by character count) than `intended_text`. Reached only when none
  of the above matched. Prompt over-expansion: the model added unrequested
  content rather than changing the requested content.
- `other`: residual -- none of the above matched.

Input contract
---------------
`categorize_zero_cer_drift(df)` requires every column in
`REQUIRED_COLUMNS`: `realized_cer`, `label`, `intended_text`,
`output_message`, `cos_mpnet`, `cos_minilm`, `nli_deberta_fwd`,
`nli_deberta_bwd`, `nli_roberta_fwd`, `nli_roberta_bwd`, the five `crit_*`
bool columns, `fluency_raw`, `message_id`. `fluency_raw` is part of the
contract (it is one of `compute_signals`'s six signal blocks and downstream
Step-5 artifacts always carry it) but is not itself consulted by the cause
logic below: the fluency gate already shaped `label` upstream (a row cannot
be labeled "drift" without first passing the taxonomy's fluency check), so
re-checking it here would be redundant.

Deterministic; American spelling; no em dashes (double hyphens for asides,
per house style).
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

import pandas as pd

from idrift.adjudicate.nli_metric import COS_HI

# The five prespecified critical-content flags (see
# `idrift.adjudicate.label_runner._CRIT_KEY_TO_COL` for the canonical
# key -> column mapping this mirrors).
CRIT_COLUMNS = (
    "crit_negation_flip",
    "crit_numeral_change",
    "crit_recipient_change",
    "crit_urgency_change",
    "crit_actionable_omission",
)

REQUIRED_COLUMNS = (
    "realized_cer",
    "label",
    "intended_text",
    "output_message",
    "cos_mpnet",
    "cos_minilm",
    "nli_deberta_fwd",
    "nli_deberta_bwd",
    "nli_roberta_fwd",
    "nli_roberta_bwd",
) + CRIT_COLUMNS + ("fluency_raw", "message_id")

# Documented overgenerative threshold: output at least 1.5x the intended
# character length, with no crit flag, counts as prompt over-expansion.
OVERGEN_LEN_RATIO = 1.5

CAUSES = ("critical_substitution", "formatting", "paraphrase", "overgenerative", "other")

_MAX_EXAMPLES_PER_CAUSE = 10

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace -- used only to
    detect benign formatting-only differences, never for semantic
    comparison."""
    normalized = str(text).lower().translate(_PUNCT_TABLE)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _has_crit_flag(row) -> bool:
    return any(bool(row[col]) for col in CRIT_COLUMNS)


def _is_formatting_only(row) -> bool:
    return _normalize_text(row["output_message"]) == _normalize_text(row["intended_text"])


def _bidirectional_entail(fwd, bwd) -> bool:
    return fwd == "entail" and bwd == "entail"


def _is_paraphrase(row) -> bool:
    if row["cos_mpnet"] < COS_HI or row["cos_minilm"] < COS_HI:
        return False
    deberta_entail = _bidirectional_entail(row["nli_deberta_fwd"], row["nli_deberta_bwd"])
    roberta_entail = _bidirectional_entail(row["nli_roberta_fwd"], row["nli_roberta_bwd"])
    return deberta_entail or roberta_entail


def _is_overgenerative(row) -> bool:
    intended_len = len(str(row["intended_text"]))
    output_len = len(str(row["output_message"]))
    if intended_len == 0:
        return output_len > 0
    return (output_len / intended_len) >= OVERGEN_LEN_RATIO


def _classify_cause(row) -> str:
    """Apply the precedence order documented in the module docstring."""
    if _has_crit_flag(row):
        return "critical_substitution"
    if _is_formatting_only(row):
        return "formatting"
    if _is_paraphrase(row):
        return "paraphrase"
    if _is_overgenerative(row):
        return "overgenerative"
    return "other"


def _notes() -> str:
    return (
        "Filtered to realized_cer == 0 and label == 'drift' (a clean input the "
        "ensemble nonetheless called drift on). Precedence: critical_substitution "
        "> formatting > paraphrase > overgenerative > other -- each row is assigned "
        "the first cause it matches, not summed across causes. critical_substitution "
        f"= any of {list(CRIT_COLUMNS)} is True. formatting = output_message equals "
        "intended_text after lowercasing, punctuation removal, and whitespace "
        "collapse. paraphrase = cos_mpnet and cos_minilm both >= nli_metric.COS_HI "
        f"(currently {COS_HI}) and bidirectional entailment ('entail' both "
        "directions) on at least one of the two NLI models (deberta or roberta), "
        "reached only when no crit_* flag is set. overgenerative = output_message "
        f"at least {OVERGEN_LEN_RATIO}x intended_text's character length, reached "
        "only when none of the above matched. other = residual. Examples are capped "
        f"at {_MAX_EXAMPLES_PER_CAUSE} message_id(s) per cause."
    )


def categorize_zero_cer_drift(df: pd.DataFrame) -> dict:
    """Bin every zero-CER drift case into a documented cause taxonomy.

    Args:
        df: DataFrame with every column in `REQUIRED_COLUMNS` (see module
            docstring "Input contract").

    Returns:
        dict: `{total_zero_cer_drift, by_cause: {cause: count}, examples:
        {cause: [message_id, ...]} (each capped at
        `_MAX_EXAMPLES_PER_CAUSE`), notes}`. `by_cause` and `examples`
        always carry all five `CAUSES` as keys, even when their count is
        zero.

    Raises:
        ValueError: if `df` is missing a required column.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"categorize_zero_cer_drift: df is missing required column(s): {missing}")

    subset = df[(df["realized_cer"] == 0) & (df["label"] == "drift")]

    by_cause = {cause: 0 for cause in CAUSES}
    examples: dict[str, list] = {cause: [] for cause in CAUSES}

    for row in subset.itertuples(index=False):
        row_d = row._asdict()
        cause = _classify_cause(row_d)
        by_cause[cause] += 1
        if len(examples[cause]) < _MAX_EXAMPLES_PER_CAUSE:
            examples[cause].append(row_d["message_id"])

    return {
        "total_zero_cer_drift": int(len(subset)),
        "by_cause": by_cause,
        "examples": examples,
        "notes": _notes(),
    }


# =============================================================================
# Task B8: per-model / per-corpus breakdown + residual-bucket diagnostics.
#
# `categorize_zero_cer_drift` above answers "why" for the POOLED zero-CER
# drift set. Reviewers additionally asked (1) whether the cause mix differs
# by model or by corpus (AUTH/CRIT/CTRL), and (2) whether anything more can
# be said about the largest bucket, "other", beyond "none of the above
# matched". Everything below is pure post-processing on top of
# `categorize_zero_cer_drift` and the same already-computed signal columns
# -- no new signals, no model calls, no re-inference. `categorize_zero_cer_drift`
# itself is untouched; this section only adds new, additive functions.
#
# Honesty label: every function below is fully AUTOMATED and RULE-BASED --
# deterministic string normalization plus threshold comparisons on
# precomputed cos_*/nli_*/crit_* columns. None of this is a human
# adjudication or a model judgment call, and the categorization_method field
# on every returned dict says so explicitly so the label can't be quoted out
# of context as if a clinician or an LLM judge had made the call.
#
# Scope reminder the manuscript must repeat: this whole audit (both
# `categorize_zero_cer_drift` and everything below) is scoped to
# `realized_cer == 0` -- the decoder's ACTUAL character error on that
# attempt -- not `cer_target == 0` -- the noise ARM the attempt was sampled
# from. The two overlap heavily (most realized-0 drift rows do come from the
# target-0% arm, where zero realized error is the expected outcome) but are
# not the same set: a nontrivial minority of realized_cer==0 rows come from
# a nonzero-target-CER arm where the injected noise happened not to survive
# to the final decoded message. The audited set is therefore BROADER than
# "the target-0% arm" and must never be described as if it were limited to
# it.
# =============================================================================

_RESIDUAL_CAUSE = "other"
_RESIDUAL_CAUSE_LABEL = "unresolved_or_mixed"

CATEGORIZATION_METHOD = "automated_rule_based"


def _rename_residual(by_cause: dict) -> dict:
    """Relabel the `categorize_zero_cer_drift` residual key for reporting:
    "other" (an internal, imprecise name) becomes "unresolved_or_mixed" -- an
    honest label that these rows are not shown to share one mechanism, only
    to lack one of the four positively identified ones. Does not touch
    `categorize_zero_cer_drift`'s own return contract; only applied here."""
    renamed = dict(by_cause)
    renamed[_RESIDUAL_CAUSE_LABEL] = renamed.pop(_RESIDUAL_CAUSE)
    return renamed


def _cause_counts_only(df: pd.DataFrame) -> dict:
    """`categorize_zero_cer_drift`, stripped to total + renamed by_cause
    (examples/notes are pooled-level detail the per-group tables below
    don't need repeated at every row)."""
    result = categorize_zero_cer_drift(df)
    return {
        "total_zero_cer_drift": result["total_zero_cer_drift"],
        "by_cause": _rename_residual(result["by_cause"]),
    }


def zero_cer_breakdown(df: pd.DataFrame) -> dict:
    """Tabulate `categorize_zero_cer_drift`'s cause taxonomy by model and by
    corpus, with the residual cause renamed for reporting.

    Args:
        df: DataFrame with every column in `REQUIRED_COLUMNS` PLUS `model`
            and `corpus`.

    Returns:
        dict: `{total_zero_cer_drift, by_cause, by_model: {model:
        {total_zero_cer_drift, by_cause}}, by_corpus: {corpus:
        {total_zero_cer_drift, by_cause}}, categorization_method, notes}`.
        `by_cause` (at every level) always carries all five causes as keys,
        with "other" relabeled "unresolved_or_mixed". Grouping runs over the
        FULL `df` passed in -- each group is still internally filtered to
        `realized_cer == 0 & label == "drift"` by `categorize_zero_cer_drift`,
        so a model/corpus with zero qualifying rows still appears with
        all-zero counts rather than being omitted.

    Raises:
        ValueError: if `df` is missing a required column (including `model`
            or `corpus`).
    """
    missing = [c for c in REQUIRED_COLUMNS + ("model", "corpus") if c not in df.columns]
    if missing:
        raise ValueError(f"zero_cer_breakdown: df is missing required column(s): {missing}")

    overall = _cause_counts_only(df)
    by_model = {str(model): _cause_counts_only(group) for model, group in df.groupby("model", sort=True)}
    by_corpus = {str(corpus): _cause_counts_only(group) for corpus, group in df.groupby("corpus", sort=True)}

    return {
        "total_zero_cer_drift": overall["total_zero_cer_drift"],
        "by_cause": overall["by_cause"],
        "by_model": by_model,
        "by_corpus": by_corpus,
        "categorization_method": CATEGORIZATION_METHOD,
        "notes": (
            "Same taxonomy and precedence as categorize_zero_cer_drift "
            "(critical_substitution > formatting > paraphrase > "
            "overgenerative > unresolved_or_mixed), tabulated separately by "
            "model and by corpus (AUTH/CRIT/CTRL); categorize_zero_cer_drift's "
            "'other' key is relabeled 'unresolved_or_mixed' at every level of "
            "this breakdown. Each model/corpus total is the count of "
            "realized_cer==0 & label=='drift' rows within that group, not a "
            "share of the pooled total. Categorization is fully automated / "
            "rule-based (see categorize_zero_cer_drift): deterministic string "
            "normalization and threshold comparisons on precomputed signal "
            "columns, not a human adjudication or a model judgment call."
        ),
    }


# -----------------------------------------------------------------------
# Residual ("unresolved_or_mixed") bucket diagnostic split -- EXPLORATORY.
# -----------------------------------------------------------------------

# Margin around COS_HI within which a cosine value counts as "borderline"
# for the diagnostic overlay below -- exploratory, not part of
# categorize_zero_cer_drift's own taxonomy or precedence.
BORDERLINE_COS_MARGIN = 0.05

# Character-count threshold (post `_normalize_text`) at or below which
# output_message counts as "near-empty" -- suggestive of a decode/parsing
# artifact rather than a substantive reply.
NEAR_EMPTY_OUTPUT_CHARS = 3

_NLI_COLUMNS = ("nli_deberta_fwd", "nli_deberta_bwd", "nli_roberta_fwd", "nli_roberta_bwd")

# Exclusive, precedence-ordered diagnostic split of the residual bucket. A
# row only reaches "other" in categorize_zero_cer_drift after already
# failing critical_substitution/formatting/overgenerative and then failing
# the paraphrase gate too -- which happens either because an embedder's
# cosine sat below COS_HI, or because both cosines cleared COS_HI but
# neither NLI model entailed bidirectionally. These three categories are
# therefore exhaustive for genuine residual-bucket rows and always sum to
# `total_unresolved`.
UNRESOLVED_DIAGNOSTIC_CAUSES = (
    "nli_contradiction_driven",
    "cosine_driven_low_similarity",
    "entailment_gate_failed_high_cosine",
)


def _is_nli_contradiction(row) -> bool:
    return any(row[col] == "contradict" for col in _NLI_COLUMNS)


def _cosine_gate_failed(row) -> bool:
    return row["cos_mpnet"] < COS_HI or row["cos_minilm"] < COS_HI


def _is_borderline_cosine(row) -> bool:
    lo, hi = COS_HI - BORDERLINE_COS_MARGIN, COS_HI + BORDERLINE_COS_MARGIN
    return (lo <= row["cos_mpnet"] <= hi) or (lo <= row["cos_minilm"] <= hi)


def _is_near_empty_output(row) -> bool:
    return len(_normalize_text(row["output_message"])) <= NEAR_EMPTY_OUTPUT_CHARS


def _diagnose_unresolved(row) -> str:
    """Exclusive, precedence-ordered attribution of why a single residual
    ("other") row failed the paraphrase gate: nli_contradiction_driven (an
    NLI model explicitly asserted "contradict" in at least one direction,
    a stronger model-asserted opposite-meaning claim) takes precedence over
    cosine_driven_low_similarity (no contradiction signal, but at least one
    embedder's cosine sat below COS_HI) takes precedence over
    entailment_gate_failed_high_cosine (both cosines cleared COS_HI, so
    cosine was not the blocker, but neither NLI model entailed
    bidirectionally)."""
    if _is_nli_contradiction(row):
        return "nli_contradiction_driven"
    if _cosine_gate_failed(row):
        return "cosine_driven_low_similarity"
    return "entailment_gate_failed_high_cosine"


def unresolved_breakdown(df: pd.DataFrame) -> dict:
    """EXPLORATORY finer diagnostic split of the residual ("other" /
    "unresolved_or_mixed") bucket -- the rows `categorize_zero_cer_drift`
    could not positively attribute to critical_substitution, formatting,
    paraphrase, or overgenerative. Uses only signals already present in the
    labeled parquet (cos_*, nli_*, output_message); no model calls, no
    re-inference, no new columns.

    Two layers are reported:
      - `diagnostic_split`: EXCLUSIVE, precedence-ordered attribution of
        why each row failed the paraphrase gate (see `_diagnose_unresolved`
        docstring for the precedence). These three counts always sum to
        `total_unresolved`.
      - `overlay_flags`: two INDEPENDENT counts (not mutually exclusive with
        each other or with `diagnostic_split`) reviewers asked about
        directly: `borderline_cosine_near_threshold` (at least one of
        cos_mpnet/cos_minilm sits within `BORDERLINE_COS_MARGIN` of COS_HI --
        this could plausibly have cleared the paraphrase cosine gate under a
        slightly different threshold) and `empty_or_near_empty_output`
        (output_message, once normalized, is <= `NEAR_EMPTY_OUTPUT_CHARS`
        characters -- suggestive of a decode/parsing artifact rather than a
        substantive reply).

    This function does not resolve the residual cause -- it only
    characterizes it further using signals already computed upstream, and
    is exploratory rather than a validated taxonomy.

    Args:
        df: DataFrame with every column in `REQUIRED_COLUMNS` (see
            `categorize_zero_cer_drift`'s "Input contract").

    Returns:
        dict: `{total_unresolved, diagnostic_split: {cause: count},
        overlay_flags: {flag: count}, examples: {cause: [message_id, ...]}
        (capped at `_MAX_EXAMPLES_PER_CAUSE` per diagnostic_split cause),
        categorization_method, exploratory, notes}`.

    Raises:
        ValueError: if `df` is missing a required column.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"unresolved_breakdown: df is missing required column(s): {missing}")

    subset = df[(df["realized_cer"] == 0) & (df["label"] == "drift")]

    diagnostic_split = {cause: 0 for cause in UNRESOLVED_DIAGNOSTIC_CAUSES}
    examples: dict[str, list] = {cause: [] for cause in UNRESOLVED_DIAGNOSTIC_CAUSES}
    n_borderline = 0
    n_near_empty = 0
    total_unresolved = 0

    for row in subset.itertuples(index=False):
        row_d = row._asdict()
        if _classify_cause(row_d) != _RESIDUAL_CAUSE:
            continue
        total_unresolved += 1
        cause = _diagnose_unresolved(row_d)
        diagnostic_split[cause] += 1
        if len(examples[cause]) < _MAX_EXAMPLES_PER_CAUSE:
            examples[cause].append(row_d["message_id"])
        if _is_borderline_cosine(row_d):
            n_borderline += 1
        if _is_near_empty_output(row_d):
            n_near_empty += 1

    return {
        "total_unresolved": total_unresolved,
        "diagnostic_split": diagnostic_split,
        "overlay_flags": {
            "borderline_cosine_near_threshold": n_borderline,
            "empty_or_near_empty_output": n_near_empty,
        },
        "examples": examples,
        "categorization_method": CATEGORIZATION_METHOD,
        "exploratory": True,
        "notes": (
            "EXPLORATORY. Scoped to the realized_cer==0 & label=='drift' rows "
            "categorize_zero_cer_drift assigns to the residual 'other' cause "
            "('unresolved_or_mixed' in zero_cer_breakdown's renamed output) -- "
            "rows that already failed critical_substitution, formatting, and "
            "overgenerative, and then failed the paraphrase gate too. "
            "diagnostic_split is exclusive and sums to total_unresolved: "
            "nli_contradiction_driven > cosine_driven_low_similarity > "
            "entailment_gate_failed_high_cosine. overlay_flags are "
            "independent counts, not a partition, and may overlap each other "
            "or any diagnostic_split cause: borderline_cosine_near_threshold "
            f"= at least one of cos_mpnet/cos_minilm within "
            f"{BORDERLINE_COS_MARGIN} of nli_metric.COS_HI (currently "
            f"{COS_HI}); empty_or_near_empty_output = output_message "
            "normalized (lowercased, punctuation stripped, whitespace "
            f"collapsed) to <= {NEAR_EMPTY_OUTPUT_CHARS} characters. Neither "
            "diagnostic layer resolves the residual cause -- both only "
            "describe it further using signals already computed upstream. "
            "Categorization is fully automated / rule-based, not a human "
            "adjudication or a model judgment call."
        ),
    }


def _build_digest(df: pd.DataFrame) -> dict:
    """Assemble the full B8 digest dict (breakdown + unresolved diagnostic +
    the realized-CER-vs-target-CER scope note) from a labeled attempts
    DataFrame. Split out from `main` so it is independently testable/callable
    without going through argv/file I/O."""
    audited = df[(df["realized_cer"] == 0) & (df["label"] == "drift")]
    n_audited = int(len(audited))
    n_target_zero = int((audited["cer_target"] == 0).sum()) if "cer_target" in audited.columns else None

    if n_target_zero is None:
        scope_note = (
            "Audited set is realized_cer == 0 (the decoder's actual character "
            "error on that attempt), which is BROADER than cer_target == 0 "
            "(the noise arm the attempt was sampled from) -- cer_target was "
            "not present in this input to confirm the split numerically, but "
            "the distinction still holds: a realized_cer==0 row can come from "
            "any noise arm if the injected corruption happened not to survive "
            "to the final decoded message."
        )
    else:
        n_target_nonzero = n_audited - n_target_zero
        pct_target_zero = (n_target_zero / n_audited) if n_audited else 0.0
        scope_note = (
            f"Audited set is realized_cer == 0 (the decoder's actual "
            f"character error on that attempt), NOT cer_target == 0 (the "
            f"noise arm the attempt was sampled from). Of the {n_audited} "
            f"realized_cer==0 & label=='drift' rows audited here, "
            f"{n_target_zero} ({pct_target_zero:.1%}) also came from the "
            f"cer_target==0 arm (the expected/typical case: no noise was "
            f"injected, so zero realized error is unsurprising), and "
            f"{n_target_nonzero} came from a NONZERO-target-CER arm where "
            "injected noise happened not to survive to realized_cer (e.g. a "
            "corruption that round-tripped losslessly through decode/repair "
            "and left zero measured character error despite a nonzero noise "
            "target). The audited set is therefore BROADER than 'the "
            "target-0% arm' and must not be described as limited to it."
        )

    breakdown = zero_cer_breakdown(df)
    unresolved = unresolved_breakdown(df)

    return {
        "scope_note": scope_note,
        "overall": {
            "total_zero_cer_drift": breakdown["total_zero_cer_drift"],
            "by_cause": breakdown["by_cause"],
        },
        "by_model": breakdown["by_model"],
        "by_corpus": breakdown["by_corpus"],
        "unresolved_breakdown": unresolved,
        "categorization_method": CATEGORIZATION_METHOD,
        "notes": (
            "Categorization throughout this digest is fully automated / "
            "rule-based (deterministic string normalization and threshold "
            "comparisons on precomputed cos_*/nli_*/crit_* signal columns) -- "
            "it is NOT a human adjudication and NOT a model/LLM judgment "
            "call. See idrift.analysis.zero_cer_audit's module docstring and "
            "categorize_zero_cer_drift for the exact rules and precedence."
        ),
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--attempts", default="output/intermediate/attempts_v2_labeled.parquet")
    ap.add_argument("--out", default="output/zero_cer_breakdown_digest.json")
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.attempts)
    digest = _build_digest(df)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(digest, indent=2))

    print(f"scope note: {digest['scope_note']}")
    print(f"total zero-CER drift: {digest['overall']['total_zero_cer_drift']}")
    print(f"by_cause: {digest['overall']['by_cause']}")
    print("by_corpus:")
    for corpus, d in digest["by_corpus"].items():
        print(f"  {corpus}: total={d['total_zero_cer_drift']} by_cause={d['by_cause']}")
    print("by_model:")
    for model, d in digest["by_model"].items():
        print(f"  {model}: total={d['total_zero_cer_drift']} by_cause={d['by_cause']}")
    ub = digest["unresolved_breakdown"]
    print("unresolved breakdown:")
    print(f"  total_unresolved={ub['total_unresolved']}")
    print(f"  diagnostic_split={ub['diagnostic_split']}")
    print(f"  overlay_flags={ub['overlay_flags']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
