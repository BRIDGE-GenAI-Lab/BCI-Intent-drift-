"""Tests for the multi-evaluator ensemble (revision Task 3.3; tie-break
bias fix from the Task 3.3 code review).

Reviewers flagged that a SINGLE automated evaluator creates evaluator-model
dependence: its failure modes systematically bias every downstream label.
`ensemble_label()` combines several INDEPENDENT evaluator callables (in
production: different NLI backbones x different embedding backbones, each
wired through the same `idrift.adjudicate.taxonomy.label` decision logic,
see `default_evaluators()`) and majority-votes a final label, reporting
inter-evaluator agreement so robustness is demonstrated, not assumed.

A SEPARATE review pass then flagged the tie-break itself: the original
default resolved every tie via a fixed severity rank (drift > degraded >
faithful), so any tie touching "drift" -- this study's PRIMARY OUTCOME --
always resolved to "drift," inflating the measured drift rate. The
default is now "primary" (defers to the pre-registered primary evaluator,
not a severity rank); "severity" (the old rule) and "drift_averse" (its
mirror image) remain available as explicit `tie_break=` sensitivity knobs
for Phase 5's tie-direction sensitivity analysis. See ensemble.py's module
docstring for the full rule descriptions.

Every evaluator here is a STUB (a `NamedEvaluator` wrapping a canned
lambda) so these tests are fast, deterministic, and never touch the real
NLI/embedding/fluency models -- `test_default_evaluators_*` below only
checks that `default_evaluators` stays import-safe and lazy, it never
actually calls it.
"""
import pytest

from idrift.adjudicate.ensemble import NamedEvaluator, ensemble_label


def _stub(name, label):
    """A named evaluator callable that ignores its (intended, output)
    arguments and always returns the same canned label -- exactly the kind
    of stub the brief specifies for these tests."""
    return NamedEvaluator(name, lambda intended, output: label)


def test_majority_voting_picks_the_majority_label():
    evaluators = [_stub("a", "drift"), _stub("b", "drift"), _stub("c", "faithful")]
    result = ensemble_label("intended", "output", evaluators)

    assert result["label"] == "drift"


def test_votes_records_each_evaluator_label():
    evaluators = [_stub("a", "drift"), _stub("b", "faithful"), _stub("c", "degraded")]
    result = ensemble_label("intended", "output", evaluators)

    assert result["votes"] == {"a": "drift", "b": "faithful", "c": "degraded"}


def test_agreement_is_fraction_matching_the_majority():
    # 2 of 3 evaluators agree on "drift" -> agreement = 2/3 = 0.667.
    evaluators = [_stub("a", "drift"), _stub("b", "drift"), _stub("c", "faithful")]
    result = ensemble_label("intended", "output", evaluators)

    assert result["agreement"] == pytest.approx(2 / 3)


def test_unanimous_agreement_is_one_and_is_not_a_tie():
    evaluators = [_stub("a", "faithful"), _stub("b", "faithful")]
    result = ensemble_label("intended", "output", evaluators)

    assert result["label"] == "faithful"
    assert result["agreement"] == pytest.approx(1.0)
    assert result["tie"] is False
    assert result["tied_labels"] == ["faithful"]


def test_two_two_tie_defaults_to_primary_evaluator_not_forced_drift():
    # Regression test for the reviewed defect: the OLD default tie-break
    # used a fixed severity rank (drift > degraded > faithful), so every
    # tie touching "drift" silently resolved to "drift" -- this study's
    # PRIMARY OUTCOME -- inflating the measured drift rate. The new
    # default ("primary") instead defers to the pre-registered PRIMARY
    # evaluator ("a", first in the list): a 2-2 faithful/drift split must
    # return "faithful" (a's own vote), NOT "drift", and must surface the
    # tie so downstream code can route it to human adjudication.
    evaluators = [
        _stub("a", "faithful"),
        _stub("b", "drift"),
        _stub("c", "faithful"),
        _stub("d", "drift"),
    ]
    result = ensemble_label("intended", "output", evaluators)

    assert result["label"] == "faithful"
    assert result["tie"] is True
    assert sorted(result["tied_labels"]) == ["drift", "faithful"]
    assert result["agreement"] == pytest.approx(0.5)


def test_tie_break_sensitivity_knobs_on_the_same_2_2_split():
    # Same 2-2 split as above: the explicit sensitivity knobs let Phase 5
    # bound the result in both directions. "drift_averse" resolves toward
    # the least severe tied label (faithful); "severity" is the OLD
    # drift-favoring rule, kept only as an explicit, opt-in sensitivity
    # analysis (never the default).
    evaluators = [
        _stub("a", "faithful"),
        _stub("b", "drift"),
        _stub("c", "faithful"),
        _stub("d", "drift"),
    ]

    drift_averse = ensemble_label("intended", "output", evaluators, tie_break="drift_averse")
    severity = ensemble_label("intended", "output", evaluators, tie_break="severity")

    assert drift_averse["label"] == "faithful"
    assert severity["label"] == "drift"


def test_three_way_tie_defaults_to_primary_evaluator():
    evaluators = [_stub("a", "faithful"), _stub("b", "degraded"), _stub("c", "drift")]
    result = ensemble_label("intended", "output", evaluators)

    assert result["label"] == "faithful"
    assert result["tie"] is True
    assert sorted(result["tied_labels"]) == ["degraded", "drift", "faithful"]
    assert result["agreement"] == pytest.approx(1 / 3)


def test_three_way_tie_with_severity_tie_break_matches_old_behavior():
    # The "severity" sensitivity knob preserves the OLD (pre-fix)
    # drift-favoring rule exactly, for Phase 5's tie-direction sensitivity
    # analysis.
    evaluators = [_stub("a", "faithful"), _stub("b", "degraded"), _stub("c", "drift")]
    result = ensemble_label("intended", "output", evaluators, tie_break="severity")

    assert result["label"] == "drift"


def test_severity_tie_break_is_deterministic_regardless_of_evaluator_order():
    forward = [_stub("a", "faithful"), _stub("b", "drift")]
    backward = [_stub("b", "drift"), _stub("a", "faithful")]

    forward_label = ensemble_label("i", "o", forward, tie_break="severity")["label"]
    backward_label = ensemble_label("i", "o", backward, tie_break="severity")["label"]

    assert forward_label == backward_label == "drift"


def test_primary_tie_break_follows_the_first_evaluator_in_the_list():
    # Unlike "severity", the default "primary" rule is intentionally
    # order-dependent: it defers to whichever evaluator is registered
    # first (the pre-registered PRIMARY evaluator), not a fixed severity
    # rank. Reversing the list reverses which evaluator is primary, and
    # therefore can reverse the reported label.
    forward = [_stub("a", "faithful"), _stub("b", "drift")]
    backward = [_stub("b", "drift"), _stub("a", "faithful")]

    forward_label = ensemble_label("i", "o", forward)["label"]
    backward_label = ensemble_label("i", "o", backward)["label"]

    assert forward_label == "faithful"
    assert backward_label == "drift"


def test_degraded_vs_faithful_severity_tie_break_favors_degraded():
    evaluators = [_stub("a", "faithful"), _stub("b", "degraded")]
    result = ensemble_label("intended", "output", evaluators, tie_break="severity")

    assert result["label"] == "degraded"


def test_ensemble_label_is_deterministic():
    evaluators = [_stub("a", "drift"), _stub("b", "faithful"), _stub("c", "faithful")]
    first = ensemble_label("intended", "output", evaluators)
    second = ensemble_label("intended", "output", evaluators)

    assert first == second


def test_empty_evaluators_raises():
    with pytest.raises(ValueError):
        ensemble_label("intended", "output", [])


def test_invalid_tie_break_raises():
    evaluators = [_stub("a", "faithful"), _stub("b", "drift")]

    with pytest.raises(ValueError):
        ensemble_label("intended", "output", evaluators, tie_break="not_a_real_policy")


def test_duplicate_evaluator_names_raise_value_error():
    # Minor gotcha from the review: two evaluators sharing a `.name` used
    # to silently overwrite one vote in the `votes` dict (last write
    # wins), dropping a real vote and skewing `agreement`. This is now a
    # hard error instead of a silent data-loss bug.
    evaluators = [_stub("a", "faithful"), _stub("a", "drift")]

    with pytest.raises(ValueError):
        ensemble_label("intended", "output", evaluators)


def test_named_evaluator_is_callable_and_carries_a_name():
    ev = _stub("stub_a", "drift")

    assert ev.name == "stub_a"
    assert ev("anything", "goes") == "drift"


def test_default_evaluators_is_lazy_and_module_imports_without_heavy_deps():
    # Importing idrift.adjudicate.ensemble must never require torch /
    # transformers / sentence_transformers to be installed (mirrors
    # nli_metric.make_model_fns / fluency.make_fluency_fn); the heavy
    # imports live inside default_evaluators() itself, not at module scope.
    import idrift.adjudicate.ensemble as ensemble_module

    assert hasattr(ensemble_module, "default_evaluators")
    assert callable(ensemble_module.default_evaluators)
