import pandas as pd

from idrift.adjudicate.human_export import stratified_sample


# --- Regression: stratified_sample must not drop the `by` column ------------
# Task 13 review: `groupby(by, group_keys=False).apply(lambda x: x.sample(...))`
# runs, under the installed pandas (3.0.3), with `include_groups=False`
# semantics by default -- the grouping column(s) are excluded from the
# sub-frame handed to the lambda, so the returned sample silently lost the
# `by` column entirely.

def test_stratified_sample_preserves_by_column_and_expected_size():
    df = pd.DataFrame(
        {
            "model_class": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "x": range(9),
        }
    )
    out = stratified_sample(df, n=6, by="model_class", seed=0)

    # the grouping column must still be present in the result
    assert "model_class" in out.columns

    # equal quota per stratum: n // nunique == 6 // 3 == 2 per group, capped
    # by each group's own availability (3 available per group here, so no cap)
    assert len(out) == 6
    counts = out["model_class"].value_counts()
    assert set(counts.index) == {"a", "b", "c"}
    assert (counts == 2).all()


def test_stratified_sample_caps_quota_by_group_availability():
    # group "rare" only has 1 row available, less than the equal quota
    df = pd.DataFrame(
        {
            "model_class": ["common", "common", "common", "common", "rare"],
            "x": range(5),
        }
    )
    out = stratified_sample(df, n=4, by="model_class", seed=0)

    assert "model_class" in out.columns
    counts = out["model_class"].value_counts()
    assert counts["rare"] == 1  # capped by availability, not the n//2 quota
    assert counts["common"] == 2
