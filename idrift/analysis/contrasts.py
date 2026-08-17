"""Model-class contrasts: paired bootstrap differences and Benjamini-Hochberg
multiple-testing correction across the primary statistical family.
"""
import numpy as np
from statsmodels.stats.multitest import multipletests


def paired_bootstrap(df, class_a, class_b, metric_fn, n=2000, seed=0):
    """Bootstrap the difference in `metric_fn` between two model classes.

    Independently resamples (with replacement) the `class_a` and `class_b`
    subsets of `df` `n` times, computing
    `metric_fn(resample_a) - metric_fn(resample_b)` each time. Every
    resample draw is seeded from a single local `np.random.default_rng(seed)`
    -- never the global `np.random` state -- so a call with the same `df`,
    `class_a`, `class_b`, `metric_fn`, `n`, and `seed` reproduces bit-for-bit
    identical output regardless of what else has touched `np.random`
    elsewhere in the process.

    Args:
        df: DataFrame with a `model_class` column identifying each row's
            class.
        class_a: value of `model_class` selecting the first arm.
        class_b: value of `model_class` selecting the second arm.
        metric_fn: callable, DataFrame -> scalar metric, applied separately
            to each bootstrap resample of each arm.
        n: number of bootstrap resamples.
        seed: seed for the local RNG driving all resampling draws.

    Returns:
        dict: {
            "diff": float, mean bootstrap difference (class_a - class_b),
            "ci": (float, float), the (2.5th, 97.5th) percentile bootstrap
                CI on the difference,
            "p": float, two-sided bootstrap p-value (twice the smaller of
                the two one-sided tail proportions on either side of zero),
        }
    """
    rng = np.random.default_rng(seed)
    a = df[df.model_class == class_a]
    b = df[df.model_class == class_b]
    diffs = []
    for _ in range(n):
        ia = a.sample(len(a), replace=True, random_state=int(rng.integers(1e9)))
        ib = b.sample(len(b), replace=True, random_state=int(rng.integers(1e9)))
        diffs.append(metric_fn(ia) - metric_fn(ib))
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "diff": float(diffs.mean()),
        "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "p": float(p),
    }


def bh(pvals):
    """Benjamini-Hochberg FDR correction over a family of p-values.

    This is the house standard for the "primary family" multiplicity
    correction referenced throughout this study's design doc (model-class
    x aim contrasts).

    Args:
        pvals: sequence of raw p-values.

    Returns:
        list[float]: BH-adjusted p-values, same order and length as `pvals`.
    """
    return list(multipletests(pvals, method="fdr_bh")[1])
