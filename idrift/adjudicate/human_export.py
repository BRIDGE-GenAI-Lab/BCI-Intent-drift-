import pandas as pd


def stratified_sample(df, n, by, seed):
    """Draw a stratified sample of n rows, using an equal quota per group.

    Each stratum (group defined by `by`) contributes an equal share of the
    sample -- `n // (number of distinct groups)`, capped by that group's own
    size -- rather than a share proportional to the group's size in `df`.

    Note: sampling is done by iterating groups and concatenating (not
    `groupby(...).apply(...)`), because under the installed pandas
    (3.0.3) `DataFrameGroupBy.apply` excludes the grouping column(s) from
    the sub-frame passed to the callable by default, and `include_groups=True`
    is no longer an accepted override -- either way, the naive apply
    approach silently drops the `by` column from the result.

    Args:
        df: DataFrame to sample from
        n: Total number of rows to sample
        by: Column name to stratify by
        seed: Random seed for reproducibility

    Returns:
        DataFrame: Stratified sample of approximately n rows, including
            the `by` column
    """
    quota = max(1, n // df[by].nunique())
    parts = [group.sample(min(len(group), quota), random_state=seed) for _, group in df.groupby(by)]
    return pd.concat(parts)


def to_rater_csv(sample, path):
    """Export a blinded sample to CSV for human raters.

    Drops model identifiers, shuffles rows, and adds an empty rater_label column.

    Args:
        sample: DataFrame with columns message_id, intended_text, noisy_text, output_text
        path: Output CSV path
    """
    blind = sample[["message_id", "intended_text", "noisy_text", "output_text"]].sample(frac=1, random_state=0)
    blind.assign(rater_label="").to_csv(path, index=False)
