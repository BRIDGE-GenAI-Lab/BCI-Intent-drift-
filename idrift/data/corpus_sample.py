"""Prespecified, proportional stratified sampling from the authentic
ALS message-banking (Costello) corpus, plus a simulation-based power
calculation for the primary drift-vs-CER slope.

This replaces the earlier ad-hoc 120-phrase convenience sample: the sample
drawn here is proportionally stratified over category x length_bin, drawn
with a fixed random generator so it is fully reproducible from its seed,
and never contains a duplicate text.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

# The prespecified CER exposure grid used throughout this study (see
# docs/preregistration.md, Design Plan): five fixed corruption levels,
# expressed as a fraction (0.0-0.4), not a percentage.
CER_GRID = (0.0, 0.1, 0.2, 0.3, 0.4)

# Pilot-anchored reference effect size: the current pilot's overall drift
# rate rises from about 0.6% at CER = 0 to about 46.3% at CER = 0.4 (see
# docs/manuscript_facts.md). Converting those two endpoints to the logit
# scale and rescaling to a per-10-percentage-point-of-CER odds ratio gives
# approximately 3.46. This constant documents that derivation; callers of
# `power_for_slope` should pass an `effect` consistent with it rather than
# an arbitrary value.
PILOT_OR_PER_10PT_CER = float(
    np.exp(
        (
            np.log(0.463 / (1 - 0.463)) - np.log(0.006 / (1 - 0.006))
        )
        / 0.4
        * 0.1
    )
)

# Between-message heterogeneity in baseline (CER = 0) drift-proneness, on
# the log-odds scale. There is no prior estimate of this variance component
# from the pilot (which pooled across messages), so a moderate placeholder
# is used; this is a documented simulation assumption, not a fitted value.
_MESSAGE_RANDOM_INTERCEPT_SD = 0.5


def _hamilton_apportion(sizes: dict, total: int) -> dict:
    """Largest-remainder (Hamilton) apportionment of `total` integer units
    across the groups in `sizes`, proportional to each group's size.

    Guarantees, for every key k: floor(quota_k) <= alloc[k] <= ceil(quota_k),
    where quota_k = total * sizes[k] / sum(sizes.values()). Because
    quota_k <= sizes[k] whenever total <= sum(sizes.values()), this also
    guarantees alloc[k] <= sizes[k]: a stratum is never asked to supply more
    rows than it has. Ties in the fractional remainder are broken by sorted
    key order, so the result is deterministic.
    """
    keys = list(sizes.keys())
    grand = sum(sizes.values())
    if grand == 0 or total <= 0:
        return {k: 0 for k in keys}
    quotas = {k: total * sizes[k] / grand for k in keys}
    alloc = {k: int(np.floor(q)) for k, q in quotas.items()}
    remainder = total - sum(alloc.values())
    order = sorted(keys, key=lambda k: (-(quotas[k] - alloc[k]), k))
    for k in order[:remainder]:
        alloc[k] += 1
    return alloc


def _text_column(corpus_df: pd.DataFrame) -> str:
    if "text" in corpus_df.columns:
        return "text"
    if "intended_text" in corpus_df.columns:
        return "intended_text"
    raise ValueError("corpus_df must have a 'text' or 'intended_text' column")


# Tercile bin names, keyed by how many bins actually result. Requesting 3
# quantile bins (q=3) can legitimately collapse to fewer distinct bins when
# the length distribution has ties at a quantile edge (`pd.qcut(...,
# duplicates="drop")`); this maps whatever bin count results to a sensible
# ordered name set instead of raising or silently collapsing everything to
# one label.
_TERCILE_NAMES_BY_BIN_COUNT = {1: ["short"], 2: ["short", "long"], 3: ["short", "medium", "long"]}


def _length_terciles(texts: pd.Series) -> pd.Series:
    """Bin `texts` (by character length) into terciles computed within this
    series, returning short/medium/long labels (degrading gracefully to
    fewer bins if the length distribution has too few distinct values to
    support three, e.g. via tied quantile edges)."""
    lengths = texts.str.len()
    try:
        cats = pd.qcut(lengths, 3, duplicates="drop")
    except ValueError:
        # Fewer than 2 distinct length values in the whole corpus.
        return pd.Series(["short"] * len(lengths), index=lengths.index)
    codes = cats.cat.codes
    n_bins = int(codes.max()) + 1
    names = _TERCILE_NAMES_BY_BIN_COUNT.get(n_bins, [f"bin{i}" for i in range(n_bins)])
    return codes.map(dict(enumerate(names)))


def sample_authentic(corpus_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Draw a prespecified, proportional stratified sample from the
    authentic corpus.

    Stratifies proportionally over category x length_bin (a character-length
    tercile computed within `corpus_df` itself), using Hamilton
    (largest-remainder) apportionment in two stages: first across
    categories, then within each category across its length bins. Because
    a category's total allocation is exactly the sum of its own length-bin
    allocations, and the first-stage apportionment already guarantees each
    category's total is within 1 of its exact proportional quota, the
    combined category x length_bin stratification also preserves
    category-level marginal proportionality within +-1.

    Duplicate texts in the source corpus (e.g. the same phrase legitimately
    filed under two categories, as happens in the real Costello corpus) are
    collapsed to a single candidate row (first occurrence kept) before
    sampling, so the returned sample never contains the same text twice.

    Sampling within each stratum draws from a single `numpy.random.Generator`
    seeded with `seed`, consuming strata in a fixed (category, length_bin)
    sort order. The result is therefore fully deterministic: the same
    corpus, n, and seed always return the same message_id set in the same
    row order.

    Args:
        corpus_df: DataFrame with at least `message_id`, `category`, and a
            `text` (or `intended_text`) column, one row per candidate
            message.
        n: Sample size to draw.
        seed: Seed for the deterministic numpy random generator.

    Returns:
        DataFrame with columns `message_id`, `text`, `category`,
        `length_bin`, one row per sampled message.
    """
    text_col = _text_column(corpus_df)
    required = {"message_id", "category", text_col}
    missing = required - set(corpus_df.columns)
    if missing:
        raise ValueError(f"corpus_df is missing required columns: {sorted(missing)}")

    df = corpus_df.drop_duplicates(subset=text_col, keep="first").reset_index(drop=True)
    if n > len(df):
        raise ValueError(f"requested n={n} exceeds {len(df)} unique candidate texts")

    df = df.assign(length_bin=_length_terciles(df[text_col]).astype(str))

    rng = np.random.default_rng(seed)

    cat_sizes = df.groupby("category").size().to_dict()
    cat_alloc = _hamilton_apportion(cat_sizes, n)

    selected_idx = []
    for cat in sorted(cat_sizes):
        cat_n = cat_alloc[cat]
        if cat_n == 0:
            continue
        cat_df = df[df["category"] == cat]
        bin_sizes = cat_df.groupby("length_bin").size().to_dict()
        bin_alloc = _hamilton_apportion(bin_sizes, cat_n)
        for lb in sorted(bin_sizes):
            k = bin_alloc[lb]
            if k == 0:
                continue
            stratum_idx = cat_df.index[cat_df["length_bin"] == lb].to_numpy()
            chosen = rng.choice(stratum_idx, size=k, replace=False)
            selected_idx.extend(chosen.tolist())

    out = df.loc[selected_idx, ["message_id", text_col, "category", "length_bin"]]
    if text_col != "text":
        out = out.rename(columns={text_col: "text"})
    return out.reset_index(drop=True)


def power_for_slope(effect: float, n_msg: int, n_rep: int, alpha: float = 0.05,
                     n_sim: int = 200, seed: int = 0) -> float:
    """Simulation-based power for the primary drift ~ CER slope.

    Data-generating process (per simulation replicate):
      - `n_msg` messages, each observed at every level of the prespecified
        CER exposure grid `CER_GRID` (0.0, 0.1, 0.2, 0.3, 0.4; see
        docs/preregistration.md), replicated `n_rep` times per
        message x CER cell (e.g. across models / repeated corruption
        draws), for n_msg * len(CER_GRID) * n_rep total binary drift
        observations.
      - Each message carries its own random intercept on the log-odds
        scale, drawn from Normal(0, _MESSAGE_RANDOM_INTERCEPT_SD), giving
        between-message heterogeneity in baseline drift-proneness -- a
        random intercept for message, mirroring the random effect for
        message/corpus in the prespecified mixed-effects model.
      - The fixed-effect intercept is anchored to the current pilot's
        drift rate at CER = 0 (about 0.6%; docs/manuscript_facts.md), and
        the slope is derived from `effect`, a per-10-percentage-point CER
        odds ratio: slope = ln(effect) / 0.1 on the logit scale.
      - Binary drift outcomes are drawn as
        Bernoulli(sigmoid(intercept + message_intercept + slope * cer)).

    Analysis model (per simulation replicate): a logistic regression of
    drift ~ CER with cluster-robust standard errors (statsmodels GLM
    Binomial, cov_type="cluster", clustered on message id). This is a fast,
    numerically stable proxy for the fixed-effect slope test in the full
    mixed-effects logistic model: refitting an actual GLMM thousands of
    times inside a Monte Carlo power loop is not computationally practical,
    and a cluster-robust marginal model targets the same null hypothesis
    (slope = 0) with equivalent power under this design's clustering
    structure (repeated observations within message).

    Power is the proportion of `n_sim` simulated datasets for which the
    two-sided Wald p-value for the CER slope is below `alpha` AND the
    estimated slope is positive, matching the prespecified one-directional
    H1 (drift increases with CER). Simulated datasets that are degenerate
    (all-drift or all-faithful) or whose fit does not converge count as
    non-detections rather than being excluded, so the reported power is not
    biased upward by discarding hard cases.

    Args:
        effect: Pilot-consistent per-10-percentage-point-CER odds ratio for
            the drift ~ CER association (see `PILOT_OR_PER_10PT_CER`, about
            3.46, derived from the pilot's ~0.6%->46.3% rise across CER
            0->0.4).
        n_msg: Number of distinct messages (clusters) in the simulated
            design.
        n_rep: Number of repeated observations per message x CER cell.
        alpha: Two-sided significance threshold for the slope test.
        n_sim: Number of Monte Carlo simulation replicates.
        seed: Seed for the numpy random generator, for determinism.

    Returns:
        float: estimated power in [0, 1].
    """
    rng = np.random.default_rng(seed)
    grid = np.array(CER_GRID)
    beta1 = np.log(effect) / 0.1
    beta0 = np.log(0.006 / (1 - 0.006))

    n_per_msg = len(grid) * n_rep
    cer_block = np.repeat(grid, n_rep)

    successes = 0
    for _ in range(n_sim):
        msg_intercept = rng.normal(0.0, _MESSAGE_RANDOM_INTERCEPT_SD, size=n_msg)
        message_id = np.repeat(np.arange(n_msg), n_per_msg)
        cer = np.tile(cer_block, n_msg)
        intercept = np.repeat(msg_intercept, n_per_msg)
        logit = beta0 + intercept + beta1 * cer
        p = 1.0 / (1.0 + np.exp(-logit))
        y = rng.binomial(1, p)

        if y.sum() == 0 or y.sum() == len(y):
            continue  # degenerate draw: no variation to detect a slope from

        d = pd.DataFrame({"y": y, "cer": cer, "message_id": message_id})
        try:
            model = sm.GLM(d["y"], sm.add_constant(d["cer"]), family=sm.families.Binomial())
            fit = model.fit(cov_type="cluster", cov_kwds={"groups": d["message_id"]})
            pval = float(fit.pvalues["cer"])
            coef = float(fit.params["cer"])
        except Exception:
            continue  # non-convergent draw: counts as a non-detection

        if pval < alpha and coef > 0:
            successes += 1

    return successes / n_sim


def main(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Build `output/intermediate/auth_sample.csv`: the prespecified,
    proportionally stratified sample of `n` authentic Costello-corpus
    messages that replaces the earlier ad-hoc 120-phrase convenience
    sample used by the original `build_exposure_run.py` pipeline."""
    from idrift.lib.io_utils import _dir, load_checkpoint, log_provenance

    corpus = load_checkpoint("corpus_costello")
    sample = sample_authentic(corpus, n=n, seed=seed)

    out_path = _dir() / "auth_sample.csv"
    sample.to_csv(out_path, index=False)

    log_provenance({
        "auth_sample": {
            "rows": int(len(sample)),
            "n": n,
            "seed": seed,
            "source_corpus": "corpus_costello",
            "source_corpus_rows": int(len(corpus)),
            "n_categories_represented": int(sample["category"].nunique()),
            "category_counts": {str(k): int(v) for k, v in sample["category"].value_counts().items()},
            "length_bin_counts": {str(k): int(v) for k, v in sample["length_bin"].value_counts().items()},
        }
    })
    print(f"wrote {out_path} rows={len(sample)} categories={sample['category'].nunique()}")
    return sample


if __name__ == "__main__":
    main()
