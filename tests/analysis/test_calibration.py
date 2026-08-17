import numpy as np
import pandas as pd

from idrift.analysis.calibration import ece, auroc_conf_faithful, reliability


# --- Brief tests (verbatim) --------------------------------------------------

def test_perfect_calibration_low_ece():
    conf = np.array([0.0, 0.0, 1.0, 1.0])
    correct = np.array([0, 0, 1, 1])
    assert ece(conf, correct, bins=2) < 1e-6


def test_auroc_confident_when_faithful():
    conf = np.array([10, 20, 80, 90])
    faithful = np.array([0, 0, 1, 1])
    assert auroc_conf_faithful(conf, faithful) == 1.0


# --- Additional regression coverage ------------------------------------------

def test_ece_normalizes_0_100_scale_same_as_0_1():
    # Same calibration pattern as test_perfect_calibration_low_ece, but on
    # a 0-100 verbalized-confidence scale -- _norm must detect and rescale
    # it, so the ECE should come out identical either way.
    conf_pct = np.array([0.0, 0.0, 100.0, 100.0])
    correct = np.array([0, 0, 1, 1])
    assert ece(conf_pct, correct, bins=2) < 1e-6


def test_ece_high_for_confidently_wrong():
    # Confidently wrong in both directions: high confidence but wrong,
    # low confidence but also wrong -- badly miscalibrated, ECE should be
    # large (not near zero).
    conf = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([0, 0, 0, 0])
    assert ece(conf, correct, bins=10) > 0.5


def test_auroc_conf_faithful_nan_when_single_class():
    # AUROC is undefined when every row is faithful (or every row is
    # drift/degraded) -- must return NaN rather than raising or silently
    # returning a meaningless number.
    conf = np.array([10, 20, 80, 90])
    faithful = np.array([1, 1, 1, 1])
    result = auroc_conf_faithful(conf, faithful)
    assert np.isnan(result)


def test_reliability_returns_one_row_per_nonempty_bin_with_expected_columns():
    conf = np.array([0.05, 0.15, 0.95])
    correct = np.array([0, 1, 1])
    r = reliability(conf, correct, bins=10)
    assert isinstance(r, pd.DataFrame)
    assert list(r.columns) == ["bin_mid", "acc", "conf", "n"]
    # 3 rows fall into 3 distinct bins (0.0-0.1, 0.1-0.2, 0.9-1.0)
    assert len(r) == 3
    assert r["n"].sum() == 3


def test_reliability_includes_confidence_exactly_at_one():
    # A confidence of exactly 1.0 must land in the last bin (inclusive
    # right edge), not be silently dropped from every bin.
    conf = np.array([1.0])
    correct = np.array([1])
    r = reliability(conf, correct, bins=10)
    assert r["n"].sum() == 1
