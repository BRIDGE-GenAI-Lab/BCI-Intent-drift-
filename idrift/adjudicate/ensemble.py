"""Multi-evaluator ensemble with inter-evaluator agreement (revision Task 3.3).

Reviewers flagged that a SINGLE automated evaluator (one NLI checkpoint,
one embedding checkpoint, feeding `idrift.adjudicate.taxonomy.label`)
creates evaluator-model dependence: whatever systematic failure mode that
one model has quietly biases every downstream faithful/degraded/drift
label, and the pipeline has no way to show this is not happening. This
module does not "fix" that by picking a supposedly better single model --
it makes the dependence checkable by running SEVERAL independent
evaluators over the same (intended, output) pair and reporting how often
they agree, so robustness is demonstrated empirically (inter-evaluator
agreement), not assumed from one model's output.

`ensemble_label(intended, output, evaluators)` is the whole module's
public contract: run every evaluator, majority-vote the final label, and
report `agreement` = the fraction of evaluators that landed on the
majority label. `evaluators` is a list of `NamedEvaluator` instances --
"named evaluator callables" per the brief: an object that is both callable
(`evaluator(intended, output) -> label`) and carries a `.name` used as the
`votes` dict key, so the ensemble is a plain, injectable list rather than
a hardcoded pair. That is what makes adding a third NLI model, a second
open-source embedding backbone, or an O2-hosted LLM judge a CONFIG change
(append one more `NamedEvaluator` to the list passed in) rather than a
rewrite of this module.

Majority vote + tie-break policy
-----------------------------------------
Each evaluator returns one of this study's three taxonomy labels
(`idrift.adjudicate.taxonomy` -- exactly {"faithful", "degraded", "drift"}).
The label with the most votes wins outright whenever there is a unique
top vote count. A TIE (more than one label sharing the max vote count --
READILY REACHABLE with an even evaluator count, e.g. a 2-2 split across
this study's 4 default evaluators, and also possible with an odd count
across 3 distinct labels, e.g. 1-1-1) is resolved according to the
`tie_break` argument, one of:

    "primary" (DEFAULT) -- defers to the pre-registered PRIMARY
        evaluator: the first evaluator in the `evaluators` list whose own
        vote is among the tied labels (in the common 2-label tie this is
        simply `evaluators[0]`'s own vote, since every evaluator's vote is
        necessarily one of the -- at most two -- tied labels; walking the
        list rather than reading `evaluators[0]` directly keeps the rule
        well defined even in a 3-plus-way tie where the first evaluator's
        own vote is not one of the tied labels). Deterministic and
        documented, and it inherits the primary evaluator's own label
        rather than systematically pushing every tie toward one taxonomy
        value.
    "drift_averse" -- the tied label with the LOWEST severity rank wins
        (faithful beats degraded beats drift): a lower-bound sensitivity
        knob.
    "severity" -- the tied label with the HIGHEST severity rank wins
        (drift beats degraded beats faithful):

            _SEVERITY_RANK = {"faithful": 0, "degraded": 1, "drift": 2}

        This was this module's FORMER, drift-favoring DEFAULT. A code
        review flagged that always resolving ties toward "drift" -- this
        study's PRIMARY OUTCOME -- systematically inflates the measured
        drift rate, since 2-2 splits are readily reachable with 4
        evaluators; that is a methodological bias for a measurement study,
        so "severity" is now kept only as an explicit, opt-in upper-bound
        sensitivity knob, never the default.

Phase 5 runs all three as a tie-direction SENSITIVITY analysis: "primary"
for the headline estimate, "drift_averse" and "severity" bounding it from
below and above. A label outside the three-value taxonomy (only possible
if a caller wires in a non-conforming evaluator, e.g. an ad hoc stub in a
test) is handled consistently by the two severity-based knobs (folded
into the same sort key -- see `_tie_break_severity` /
`_tie_break_drift_averse`) and, under "primary," is eligible like any
other tied label. This rule is exercised directly by
`tests/adjudicate/test_ensemble.py`.

Every tie is also surfaced in the return value (`"tie"`, `"tied_labels"`),
not silently resolved and dropped, so downstream code can route ambiguous
items to human adjudication or re-run the sensitivity analysis above.

`ensemble_label`'s "severity" and "drift_averse" tie-breaks never depend
on evaluator order or dict iteration order for their RESULT: both sort on
`(severity, label)`, not encounter order, so calling either twice with the
evaluators list in a different order returns the identical label (see
`test_severity_tie_break_is_deterministic_regardless_of_evaluator_order`).
"primary" is the deliberate exception: reversing the list changes which
evaluator is primary, and can therefore change the reported label -- see
`test_primary_tie_break_follows_the_first_evaluator_in_the_list`.

Duplicate evaluator names
-------------------------
Two evaluators sharing a `.name` used to silently overwrite one vote in
the `votes` dict (dict construction is last-write-wins), dropping a real
vote and skewing `agreement`. `ensemble_label` now raises `ValueError` up
front whenever two evaluators share a name, rather than silently losing a
vote.

Agreement
---------
`agreement = (count of evaluators whose label equals the final majority
label) / (total evaluators)`. This is computed against the FINAL label
(after tie-break, if any), not against an arbitrarily chosen "first"
label, so it always answers "what fraction of the ensemble backs the
label we actually reported" -- e.g. 2 of 3 evaluators agreeing on "drift"
reports `agreement = 2/3` (approximately 0.667).

`default_evaluators()` and laziness
--------------------------------------
`default_evaluators()` builds a real, independent multi-evaluator set: two
distinct NLI checkpoints (different base architectures and separately
fine-tuned MNLI weights, not the same model loaded twice) x two distinct
sentence-embedding checkpoints, each pairing wired through
`idrift.adjudicate.taxonomy.label` (Task 3.1) with a SHARED fluency
function (fluency is one grammaticality proxy for the whole pipeline, not
an ensembled dimension -- see `fluency.py`'s own module docstring for why
gpt2 was pinned once). This is a real, inspectable config: adding a third
NLI model, a third embedding model, or an O2-hosted open LLM judge is
exactly one more `make_model_fns(...)` call (or one more hand-written
evaluator function) appended to the list `default_evaluators()` returns,
never a change to `ensemble_label`'s voting logic.

Heavy imports (`torch`, `transformers`, `sentence_transformers`) are kept
strictly inside `default_evaluators()` (and the `nli_metric`/`fluency`
factories it calls), never at this module's top level, mirroring
`idrift.adjudicate.nli_metric.make_model_fns` and
`idrift.adjudicate.fluency.make_fluency_fn`: importing `ensemble.py` (or
calling `ensemble_label` with stub evaluators, as every unit test here
does) never touches those packages.

Deterministic; American spelling; no em dashes (double hyphens for
asides, per house style).
"""
from __future__ import annotations

from collections import Counter

_SEVERITY_RANK = {"faithful": 0, "degraded": 1, "drift": 2}


class NamedEvaluator:
    """A named, callable evaluator: `evaluator(intended, output) -> label`.

    Wraps a plain `(intended, output) -> str` function with a `.name` used
    as the `ensemble_label` `votes` dict key. This is the "named evaluator
    callable" the brief specifies -- a single object that is both
    identifiable (for reporting per-evaluator votes) and directly callable
    (so `ensemble_label` does not need to know whether it is holding a
    stub lambda or a real NLI+embedding-backed evaluator).
    """

    def __init__(self, name: str, fn) -> None:
        self.name = name
        self._fn = fn

    def __call__(self, intended: str, output: str) -> str:
        return self._fn(intended, output)

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid only
        return f"NamedEvaluator({self.name!r})"


_TIE_BREAKS = ("primary", "drift_averse", "severity")


def _tie_break_severity(tied_labels) -> str:
    """"severity" policy (the FORMER, drift-favoring default -- see module
    docstring): among `tied_labels` (>= 2 labels sharing the max vote
    count), the one with the HIGHEST severity rank wins. Sorts on
    `(severity_rank, label)` and returns the LAST entry (highest severity,
    alphabetically last among any further tie), so this study's taxonomy
    severity order -- drift > degraded > faithful -- decides the result
    rather than the order evaluators happened to run in. A label outside
    the known taxonomy (severity rank absent) sorts below every known
    label and is broken alphabetically against other unknowns."""
    return max(tied_labels, key=lambda label: (_SEVERITY_RANK.get(label, -1), label))


def _tie_break_drift_averse(tied_labels) -> str:
    """"drift_averse" policy: the mirror image of "severity" -- among
    `tied_labels`, the one with the LOWEST severity rank wins (faithful
    beats degraded beats drift), for a lower-bound sensitivity estimate. A
    label outside the known taxonomy sorts above every known label (so it
    is never preferred over a recognized, less-severe label) and is broken
    alphabetically against other unknowns."""
    return min(tied_labels, key=lambda label: (_SEVERITY_RANK.get(label, len(_SEVERITY_RANK)), label))


def _tie_break_primary(tied_labels, votes: dict, evaluators: list) -> str:
    """"primary" policy (the DEFAULT -- see module docstring): defers to
    the pre-registered PRIMARY evaluator, i.e. the first evaluator in
    `evaluators` (list order, not dict/Counter order) whose OWN vote is
    among `tied_labels`. In the common case this fix targets (an even
    split across exactly two tied labels, e.g. a 2-2 faithful/drift
    split), this is simply `evaluators[0]`'s vote, since every evaluator's
    vote is necessarily one of the (at most two) tied labels. Walking the
    list rather than reading `evaluators[0]` directly keeps the rule well
    defined even in a 3-plus-way tie where the very first evaluator's own
    vote happens not to be one of the tied labels."""
    tied = set(tied_labels)
    for evaluator in evaluators:
        vote = votes[evaluator.name]
        if vote in tied:
            return vote
    return tied_labels[0]  # pragma: no cover -- unreachable: every tied label was cast by some evaluator


def ensemble_label(intended: str, output: str, evaluators: list, tie_break: str = "primary") -> dict:
    """Run every evaluator in `evaluators` over (intended, output),
    majority-vote the final label, and report inter-evaluator agreement.

    Args:
        intended: the ground-truth / reference message.
        output: the model's reconstructed message.
        evaluators: a non-empty list of `NamedEvaluator` instances (or any
            object that is both callable as `(intended, output) -> str`
            and carries a `.name` attribute, unique across the list). Each
            evaluator independently maps (intended, output) to one of this
            study's taxonomy labels (typically by calling
            `idrift.adjudicate.taxonomy.label` with its own injected
            cos_fn/nli_fn/fluency_fn, see `default_evaluators`, though
            `ensemble_label` itself has no dependency on how a given
            evaluator computes its answer).
        tie_break: which policy resolves a tie for the top vote count --
            one of "primary" (default), "drift_averse", or "severity".
            See the module docstring for the full description of each and
            why "primary," not "severity," is the default.

    Returns:
        {"label": str, "votes": {evaluator_name: label, ...},
         "agreement": float, "tie": bool, "tied_labels": list} where
        "label" is the majority label (tie-broken per `tie_break` if
        needed), "agreement" is the fraction of evaluators whose vote
        equals that label, "tie" is True when more than one label shared
        the top vote count, and "tied_labels" lists every label that
        shared the top vote count (a single-element list when "tie" is
        False).

    Raises:
        ValueError: if `evaluators` is empty (agreement is undefined with
            zero evaluators), if `tie_break` is not a recognized policy,
            or if two evaluators share a `.name` (silently overwriting one
            vote would drop it and skew `agreement`).
    """
    if not evaluators:
        raise ValueError("ensemble_label requires at least one evaluator")
    if tie_break not in _TIE_BREAKS:
        raise ValueError(f"tie_break must be one of {_TIE_BREAKS}, got {tie_break!r}")

    name_counts = Counter(evaluator.name for evaluator in evaluators)
    duplicate_names = sorted(name for name, count in name_counts.items() if count > 1)
    if duplicate_names:
        raise ValueError(
            f"ensemble_label requires unique evaluator names, got duplicate(s): {duplicate_names}"
        )

    votes = {evaluator.name: evaluator(intended, output) for evaluator in evaluators}

    counts = Counter(votes.values())
    max_count = max(counts.values())
    tied_labels = sorted(label for label, count in counts.items() if count == max_count)
    tie = len(tied_labels) > 1

    if not tie:
        majority_label = tied_labels[0]
    elif tie_break == "primary":
        majority_label = _tie_break_primary(tied_labels, votes, evaluators)
    elif tie_break == "drift_averse":
        majority_label = _tie_break_drift_averse(tied_labels)
    else:
        majority_label = _tie_break_severity(tied_labels)

    agreement = max_count / len(votes)

    return {
        "label": majority_label,
        "votes": votes,
        "agreement": agreement,
        "tie": tie,
        "tied_labels": tied_labels,
    }


def default_evaluators() -> list:
    """Build the default multi-evaluator ensemble: two independent NLI
    checkpoints x two independent sentence-embedding checkpoints (4
    evaluators total), each wired through `idrift.adjudicate.taxonomy.label`
    with one shared fluency function. See module docstring for why this
    counts as a real, injectable config rather than a hardcoded pair, and
    for why the fluency function is shared (not ensembled) across all four.

    Model choices:
        - NLI: "microsoft/deberta-large-mnli" (this pipeline's existing
          default, `idrift.adjudicate.nli_metric.make_model_fns`'s default)
          and "roberta-large-mnli" -- a separately fine-tuned MNLI
          checkpoint on a different base architecture (DeBERTa's
          disentangled attention vs. RoBERTa's standard transformer
          encoder), so the two NLI evaluators are genuinely independent,
          not the same model loaded twice. "facebook/bart-large-mnli" (the
          brief's other suggested option) was reachable too (confirmed
          during this task) but roberta-large-mnli was chosen as the
          brief's first-listed alternative and because it was already
          straightforward to fetch in this environment.
        - Embedding: "all-mpnet-base-v2" (this pipeline's existing default)
          and "all-MiniLM-L6-v2" -- a distinct sentence-transformers
          checkpoint (smaller, different training recipe), both already
          present in this machine's local Hugging Face cache.

    Note:
        Heavy imports (`torch`, `transformers`, `sentence_transformers`)
        happen only when THIS function runs, via
        `idrift.adjudicate.nli_metric.make_model_fns` and
        `idrift.adjudicate.fluency.make_fluency_fn`. Importing
        `idrift.adjudicate.ensemble` does not import or require any of
        those packages.
    """
    from idrift.adjudicate.fluency import make_fluency_fn
    from idrift.adjudicate.nli_metric import make_model_fns
    from idrift.adjudicate.taxonomy import label as taxonomy_label

    nli_models = ["microsoft/deberta-large-mnli", "roberta-large-mnli"]
    emb_models = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]

    fluency_fn = make_fluency_fn()

    def _make_evaluator_fn(cos_fn, nli_fn):
        def evaluator_fn(intended, output):
            return taxonomy_label(intended, output, cos_fn, nli_fn, fluency_fn)["label"]

        return evaluator_fn

    evaluators = []
    for nli_model in nli_models:
        for emb_model in emb_models:
            cos_fn, nli_fn = make_model_fns(nli_model=nli_model, emb_model=emb_model)
            name = f"{nli_model.rsplit('/', 1)[-1]}+{emb_model}"
            evaluators.append(NamedEvaluator(name, _make_evaluator_fn(cos_fn, nli_fn)))

    return evaluators
