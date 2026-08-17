"""Turn the completed 16-model panel16 physician ratings into the
manuscript's pending WEIGHTED numbers (Task 7, reviewer #1 follow-up).

Run AFTER both physicians (and, for any disagreed items, the third
adjudicator) return their filled ``panel_stratified``/``panel_zerocer``
sheets from ``build_panel16_sheet.py`` (Task 6). Like ``analyze_panel.py``
(the original 7-model composer), this module adds no new statistics of its
own beyond the weighted extension -- it composes already-tested pieces:

  * ``interrater`` / ``consensus_labels`` (this module) -> inter-rater
    agreement and human/adjudicator resolution, modeled on
    ``analyze_panel.py``'s own functions of the same name but NOT reusing
    them directly -- see "Label-text mismatch" below for why (house
    convention keeps kappa unweighted, see ``auto_vs_human_weighted``)
  * ``reconcile.validation_frame``           -> automated-vs-human join
    (the structural guard against dropping disagreement-resolved items)
  * ``validation_metrics.confusion`` / ``class_metrics`` WEIGHTED by each
    item's ``sampling_weight`` (Task 6's per-stratum inverse-sampling
    weight) -> class-specific agreement reweighted back to the FULL
    labeled population each stratum was drawn from, not the raw
    (disproportionate) rated-sample counts
  * ``misclassification.corrected_prevalence`` -> the Rogan-Gladen
    misclassification-corrected drift estimate, per corpus, using the
    WEIGHTED confusion matrix

Why weighting matters here specifically
----------------------------------------
Unlike the original 7-model panel (drawn close to a simple random sample),
the 16-model panel is a PRESPECIFIED, DISPROPORTIONATE stratified sample by
design: every nonempty model x corpus x cer_target x automated-label cell
contributes up to ``n_per_cell`` items regardless of that cell's own
population (see ``build_panel16_sheet.py``). Treating the rated sample's
raw counts as representative of the full labeled population would misstate
both the automated-vs-human agreement and the corrected drift estimate --
a small, hard-to-review cell (e.g. one rare model x high-CER x drift
combination) would count exactly as much as a huge, easy cell. Every step
below that touches class-specific agreement or the Rogan-Gladen correction
uses the item's ``sampling_weight`` (carried through from
``build_panel16_sheet.py``'s ``_KEY_DO_NOT_SHARE/key.csv``, the analyst-
only authoritative source -- not whatever a rater's own returned sheet
still happens to carry in that column) instead of a bare item count.

Small-n guard (mirrors the pooled-vs-per-corpus decision already made in
``analyze_panel.py``/``misclassification.py``): a corpus-specific weighted
confusion matrix is only trusted when that corpus's own validation subset
has at least ``min_stratum_n`` items AND nonzero support for every one of
the 3 taxonomy classes; otherwise it falls back to the POOLED weighted
matrix (all corpora combined) for that corpus, with a note recorded in the
output (see ``corrected_drift_by_corpus16``).

Label-text mismatch with analyze_panel.py (found during this task, fixed
here rather than shipped)
----------------------------------------------------------------------------
``analyze_panel.py``'s ``map_human_label``/``_HUMAN_MAP`` were written for
``build_rating_sheet.py``'s 7-model instrument, whose fourth rating-scale
value is the literal string ``"Message-critical drift"``
(``build_rating_sheet.RATING_LABELS``). ``build_panel16_sheet.py`` (Task 6)
independently defined its OWN four-label scale for the 16-model instrument,
``RATING_LABELS = ("Faithful", "Degraded", "Drift", "Message-critical")``
-- one word shorter. Calling ``analyze_panel.map_human_label("Message-
critical")`` (verified directly) returns ``(None, None)``: the exact-string
``_HUMAN_MAP`` lookup misses on the missing trailing " drift", so every
"Message-critical"-rated item -- precisely the safety-critical class this
whole panel exists to validate -- would be silently treated as an
unrecognized rating and excluded from every downstream statistic. This
module therefore defines its OWN ``map_human_label``/``_HUMAN_MAP``/
``interrater``/``consensus_labels`` (below), matching ``build_panel16_
sheet.RATING_LABELS`` exactly (while still also accepting "Message-critical
drift", in case a physician or a future revision of the instrument's wording
uses the longer form), rather than importing and reusing ``analyze_panel.
py``'s versions unmodified. ``analyze_panel.py`` itself is untouched -- it
is still exactly correct for the 7-model instrument's own label text -- this
is a parallel, panel16-specific implementation, not a shared-code fix.

Physicians rate BOTH ``panel_stratified/sheet.csv`` and
``panel_zerocer/sheet.csv`` (Task 6) as one continuous rating task; each
rater/adjudicator's completed sheet(s) may be passed as one path or a list
of paths (e.g. one completed CSV per panel) -- they are concatenated by
``item_id`` before any reconciliation (``_load_sheets``).

Outputs a JSON digest (``panel16_results.json``) whose fields map onto the
manuscript's pending 16-model panel statements. Until physician ratings
exist, this module can only be smoke-tested on a synthetic rated sheet
(see ``tests/test_analyze_panel16.py``) -- no inference is run here.

Usage (agreement only, before adjudication):
    uv run python -m idrift.adjudicate.analyze_panel16 \\
        --rater-a .../panel_stratified/rater_a.csv .../panel_zerocer/rater_a.csv \\
        --rater-b .../panel_stratified/rater_b.csv .../panel_zerocer/rater_b.csv \\
        --key output/human_rating_v3plus/_KEY_DO_NOT_SHARE/key.csv \\
        --out output/human_rating_v3plus/panel16_results.json

Full run (adds the weighted corrected drift estimate):
    ... --adjudicator .../adjudicator_a.csv .../adjudicator_z.csv \\
        --cohort output/intermediate/attempts_v3plus_labeled.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from idrift.adjudicate.misclassification import corrected_prevalence
from idrift.adjudicate.reconcile import cohen_kappa, load_rater_csv, merge_ratings, validation_frame
from idrift.adjudicate.validation_metrics import CLASS_ORDER, class_metrics, confusion

# Judgment call (documented, not empirically tuned -- no rated data exists
# yet): the minimum raw validation-item count a corpus needs before its own
# per-corpus weighted confusion matrix is trusted for the Rogan-Gladen
# inversion. See `_stratum_confusion_is_stable`.
DEFAULT_MIN_STRATUM_N = 30

_TARGET_CORPORA = ("AUTH", "CRIT", "CTRL", "ALL")

# Matches `build_panel16_sheet.RATING_LABELS` exactly ("Message-critical",
# NOT "Message-critical drift" -- see the module docstring's "Label-text
# mismatch" section for why this is its own map, not a reuse of
# `analyze_panel._HUMAN_MAP`). Also accepts the longer form defensively.
_HUMAN_MAP = {
    "faithful": ("faithful", False),
    "degraded": ("degraded", False),
    "drift": ("drift", False),
    "message-critical": ("drift", True),
    "message-critical drift": ("drift", True),
}


def map_human_label(raw: object) -> tuple:
    """Map one physician label string (the panel16 four-label scale --
    `build_panel16_sheet.RATING_LABELS`) to (3-class label, critical flag).

    Case- and whitespace-insensitive. An unrecognized or missing value maps
    to (None, None) so callers can exclude it rather than silently
    miscount it. See the module docstring for why this is a panel16-local
    map rather than a reuse of `analyze_panel.map_human_label`.
    """
    if not isinstance(raw, str):
        return (None, None)
    return _HUMAN_MAP.get(raw.strip().lower(), (None, None))


def _mapped_series(rater: pd.DataFrame) -> pd.DataFrame:
    """Return item_id -> (class3, critical) for a completed rater sheet."""
    m = rater[["item_id", "rating"]].copy()
    mapped = m["rating"].map(map_human_label)
    m["class3"] = mapped.map(lambda t: t[0])
    m["critical"] = mapped.map(lambda t: t[1])
    return m


def interrater(rater_a: pd.DataFrame, rater_b: pd.DataFrame) -> dict:
    """Inter-rater agreement between the two physicians on the 3-class
    label. Modeled on `analyze_panel.interrater` (identical shape/keys),
    but using this module's own `map_human_label` (see module docstring).

    Restricted to items both rated with a recognized label. Reports the raw
    percent agreement, overall Cohen's kappa, and a per-class one-vs-rest
    kappa.
    """
    a = _mapped_series(rater_a).rename(columns={"class3": "a"})
    b = _mapped_series(rater_b).rename(columns={"class3": "b"})
    merged = a[["item_id", "a"]].merge(b[["item_id", "b"]], on="item_id", how="inner")
    both = merged.dropna(subset=["a", "b"])
    n = len(both)
    result = {
        "n_items_both_rated": n,
        "raw_percent_agreement": float((both["a"] == both["b"]).mean()) if n else float("nan"),
        "cohen_kappa_overall": cohen_kappa(both["a"], both["b"]) if n else float("nan"),
        "cohen_kappa_by_class": {},
    }
    for c in CLASS_ORDER:
        if n:
            result["cohen_kappa_by_class"][c] = cohen_kappa(both["a"] == c, both["b"] == c)
        else:
            result["cohen_kappa_by_class"][c] = float("nan")
    return result


def consensus_labels(
    rater_a: pd.DataFrame, rater_b: pd.DataFrame, adjudicator: pd.DataFrame | None
) -> pd.DataFrame:
    """One consensus 3-class human label per item. Modeled on `analyze_
    panel.consensus_labels` (identical shape/keys/logic), but using this
    module's own `map_human_label` (see module docstring).

    Agreement -> the shared label. Disagreement (or single-rated) -> the
    adjudicator's label if present. Returns columns item_id, human, critical,
    resolution ('agreed' | 'adjudicated' | 'unresolved').
    """
    a = _mapped_series(rater_a).set_index("item_id")
    b = _mapped_series(rater_b).set_index("item_id")
    adj = _mapped_series(adjudicator).set_index("item_id") if adjudicator is not None else None

    all_ids = sorted(set(a.index) | set(b.index))
    rows = []
    for iid in all_ids:
        av = a["class3"].get(iid)
        bv = b["class3"].get(iid)
        acrit = a["critical"].get(iid)
        bcrit = b["critical"].get(iid)
        if av is not None and av == bv:
            rows.append((iid, av, bool(acrit) and bool(bcrit), "agreed"))
        elif adj is not None and adj["class3"].get(iid) is not None:
            rows.append((iid, adj["class3"].get(iid), bool(adj["critical"].get(iid)), "adjudicated"))
        else:
            rows.append((iid, None, None, "unresolved"))
    return pd.DataFrame(rows, columns=["item_id", "human", "critical", "resolution"])


def _load_sheets(paths) -> pd.DataFrame:
    """Load and concatenate one rater's completed sheet(s) into one table.

    Physicians rate `panel_stratified/sheet.csv` and `panel_zerocer/
    sheet.csv` as one continuous task; a rater's completed work may come
    back as a single combined CSV or as one CSV per panel. Accepts a
    single path or a list of paths, either way.

    Args:
        paths: a single path (str/Path) or a list of paths, each loadable
            by `reconcile.load_rater_csv`.

    Returns:
        DataFrame: the concatenated rater sheet(s), unmodified columns.

    Raises:
        ValueError: if the same item_id appears more than once across the
            given sheet(s) -- an accidental duplicate rating (or the same
            file passed twice) should fail loudly, not silently keep an
            arbitrary one of the duplicates.
    """
    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    frames = [load_rater_csv(p) for p in path_list]
    combined = pd.concat(frames, ignore_index=True)
    dupes = combined["item_id"][combined["item_id"].duplicated()]
    if len(dupes):
        raise ValueError(f"duplicate item_id(s) across rater sheet(s): {sorted(set(dupes))}")
    return combined


def auto_vs_human_weighted(consensus: pd.DataFrame, key: pd.DataFrame) -> dict:
    """Class-specific automated-vs-human agreement + confusion, WEIGHTED by
    each item's `sampling_weight` from the key.

    The KEY's `sampling_weight` is authoritative (not whatever a rater's
    own returned sheet still carries in that column, which a rater could
    edit or drop without affecting the true per-stratum weight). Same join
    discipline as `analyze_panel.auto_vs_human` (`validation_frame`,
    `on_missing="raise"`, so no resolved item is silently dropped from the
    comparison). Cohen's kappa is reported UNWEIGHTED: `reconcile.
    cohen_kappa` is out of this task's scope (see the Task 7 report), so
    inter-rater and auto-vs-human agreement kappa stay the house-standard
    unweighted statistic, exactly as `analyze_panel.py` reports them; only
    the class-specific sensitivity/specificity/PPV/NPV/F1 and confusion
    matrix (and, downstream, the Rogan-Gladen correction) are weighted.

    Args:
        consensus: output of `analyze_panel.consensus_labels` (columns
            `item_id`, `human`, `critical`, `resolution`).
        key: the panel16 unblinding key (`_KEY_DO_NOT_SHARE/key.csv`),
            with at least `item_id`, `label` (the automated 3-class
            label), `sampling_weight`, `corpus`, `orig_message_id`.

    Returns:
        dict with weighted class metrics/confusion plus two internal keys
        (`_confusion_frame_pooled`, `_val_frame`) meant to be popped by the
        caller and threaded into `corrected_drift_by_corpus16`.
    """
    resolved = consensus.dropna(subset=["human"]).copy()
    auto = key.rename(columns={"label": "auto_label"})[
        ["item_id", "auto_label", "sampling_weight", "corpus", "cer_target", "orig_message_id"]
    ]
    vf = validation_frame(resolved, auto, on_missing="raise", auto_col="auto_label")
    human, autolab, weights = vf["human"], vf["auto_label"], vf["sampling_weight"]

    cm = confusion(human, autolab, labels=CLASS_ORDER, weights=weights)
    metrics = class_metrics(human, autolab, labels=CLASS_ORDER, weights=weights)

    val_frame = vf.rename(
        columns={"auto_label": "auto", "orig_message_id": "message_id", "sampling_weight": "weight"}
    )[["item_id", "message_id", "human", "auto", "weight", "corpus", "cer_target"]]

    return {
        "n_resolved": int(len(resolved)),
        "n_unresolved": int(consensus["resolution"].eq("unresolved").sum()),
        "sum_sampling_weight": float(weights.sum()),
        "overall_percent_agreement_unweighted": float((human.values == autolab.values).mean()),
        "cohen_kappa_overall_unweighted": cohen_kappa(human, autolab),
        "cohen_kappa_by_class_unweighted": {c: cohen_kappa(human == c, autolab == c) for c in CLASS_ORDER},
        "class_metrics_weighted": metrics.to_dict(orient="index"),
        "confusion_weighted_human_rows_auto_cols": cm.to_dict(orient="index"),
        "_confusion_frame_pooled": cm,
        "_val_frame": val_frame,
    }


def _stratum_confusion_is_stable(val_corpus: pd.DataFrame, min_stratum_n: int) -> bool:
    """True iff `val_corpus` (a validation subset restricted to one corpus)
    is large enough, and has nonzero support for every one of the 3
    taxonomy classes in its own human labels, to trust a corpus-specific
    Rogan-Gladen confusion-matrix inversion computed from it alone.

    Below this bar, `corrected_drift_by_corpus16` falls back to the POOLED
    weighted matrix (all corpora combined) for that corpus -- mirroring the
    pooled-vs-per-corpus decision `analyze_panel.corrected_drift_by_corpus`
    already made for the unweighted panel (there, EVERY corpus used the
    pooled matrix because even the largest corpus's smallest validation
    subset, CTRL n=70, made a per-corpus 3x3 inversion unstable). Here each
    corpus is evaluated independently against the same stability bar
    rather than assumed unstable outright: a corpus missing a whole
    taxonomy class from its own validation subset cannot have that class's
    confusion-matrix row estimated at all (the same zero-support case
    `misclassification._build_confusion_matrix` guards with an identity-row
    fallback), which is precisely the unstable case this guard exists to
    catch before it reaches that fallback silently.

    Args:
        val_corpus: the validation subset restricted to one corpus, with
            at least a `human` column.
        min_stratum_n: raw item-count floor (not weighted) -- a documented
            judgment call, not empirically tuned (no rated data exists
            yet to tune it against).

    Returns:
        bool: True iff `len(val_corpus) >= min_stratum_n` and every class
        in `CLASS_ORDER` has at least one item in `val_corpus["human"]`.
    """
    if len(val_corpus) < min_stratum_n:
        return False
    supports = val_corpus["human"].value_counts()
    return all(supports.get(c, 0) > 0 for c in CLASS_ORDER)


def corrected_drift_by_corpus16(
    pooled_confusion_frame: pd.DataFrame,
    cohort: pd.DataFrame,
    val_frame: pd.DataFrame,
    seed: int = 0,
    min_stratum_n: int = DEFAULT_MIN_STRATUM_N,
) -> dict:
    """Rogan-Gladen misclassification-corrected drift prevalence per corpus,
    using the WEIGHTED confusion matrix -- unlike `analyze_panel.
    corrected_drift_by_corpus`'s raw-count matrix, and with a per-corpus
    stability guard `analyze_panel.py` never needed (its per-corpus
    inversion was tried once, found unstable everywhere, and rejected
    outright in favor of always pooling; here each corpus gets its own
    stability check instead of a blanket pooled default).

    Args:
        pooled_confusion_frame: the weighted human(rows) x auto(cols)
            confusion DataFrame estimated on the FULL validation subset
            (all corpora combined) -- the fallback matrix for any corpus
            that fails `_stratum_confusion_is_stable`.
        cohort: full-cohort automated-label rows, columns `["corpus",
            "auto_label", "message_id"]` (one row per generation).
        val_frame: resolved validation items, columns `["corpus", "human",
            "auto", "message_id", "weight"]` (one row per item; see
            `auto_vs_human_weighted`).
        seed: bootstrap seed, passed to `misclassification.
            corrected_prevalence`.
        min_stratum_n: passed to `_stratum_confusion_is_stable`.

    Returns:
        dict: one entry per corpus in `("AUTH", "CRIT", "CTRL", "ALL")`,
        each `misclassification.corrected_prevalence`'s result dict plus
        `validation_n` (this corpus's own validation item count, for
        diagnosing why the stability guard fired or not),
        `bootstrap_val_n` (the validation item count actually fed to the
        bootstrap CI -- equals `validation_n` when the per-corpus matrix
        was stable, or the full pooled `len(val_frame)` when it fell back;
        see the point-estimate/CI pairing note below), and
        `confusion_source_note` (whether the per-corpus or pooled matrix
        was used, and why).

    Point-estimate/CI pairing (the bug this docstring exists to prevent
    from recurring): `corrected_prevalence`'s point estimate is computed
    from `confusion_from_human` alone, but its bootstrap CI is computed by
    RESAMPLING `val_frame` and rebuilding a fresh confusion matrix from
    that resampled data every iteration (see `misclassification.
    _bootstrap_message_clustered`). If `confusion_from_human` and
    `val_frame` describe two DIFFERENT error models -- e.g. the point
    estimate uses the pooled matrix (because this corpus's own subset was
    too small/unstable) but the CI still resamples only this corpus's own
    small subset -- the point estimate and its own CI are no longer
    computed from the same underlying model, and the point estimate can
    land outside its own reported CI. This is exactly the failure mode
    `analyze_panel.py`'s module docstring already documents encountering
    and rejecting for the unweighted 7-/16-model panel (there, it was the
    reason a per-corpus confusion matrix was abandoned entirely in favor
    of always pooling). Whenever `confusion_frame` is the POOLED matrix
    below (the `elif corp == "ALL"` branch, and the small-n fallback
    `else` branch), `val_for_bootstrap` is set to the FULL POOLED
    `val_frame` -- never this corpus's own `val_c` -- so the point
    estimate and the CI always resample the SAME error model. Only when a
    corpus passes `_stratum_confusion_is_stable` and gets its OWN
    confusion matrix does it also get its OWN `val_c` for the bootstrap.
    """
    out = {}
    for corp in _TARGET_CORPORA:
        cohort_c = cohort if corp == "ALL" else cohort[cohort["corpus"] == corp]
        auto_counts = cohort_c["auto_label"].value_counts().to_dict()
        if not auto_counts:
            out[corp] = {"note": "insufficient data"}
            continue

        val_c = val_frame if corp == "ALL" else val_frame[val_frame["corpus"] == corp]

        if corp != "ALL" and _stratum_confusion_is_stable(val_c, min_stratum_n):
            confusion_frame = confusion(
                val_c["human"], val_c["auto"], labels=CLASS_ORDER, weights=val_c["weight"]
            )
            val_for_bootstrap = val_c
            note = f"per-corpus weighted confusion matrix (validation n={len(val_c)} >= min_stratum_n={min_stratum_n})"
        elif corp == "ALL":
            confusion_frame = pooled_confusion_frame
            val_for_bootstrap = val_frame
            note = "pooled weighted confusion matrix (this IS the full pooled validation set)"
        else:
            confusion_frame = pooled_confusion_frame
            # BUG FIX (coordinator review): must use the SAME pooled
            # val_frame the pooled confusion_frame itself came from, not
            # this corpus's own small val_c -- otherwise the point
            # estimate (pooled) and the bootstrap CI (resampling only the
            # small per-corpus subset) describe different error models,
            # and the point estimate can fall outside its own CI. See the
            # docstring's "Point-estimate/CI pairing" section.
            val_for_bootstrap = val_frame
            note = (
                f"insufficient per-corpus validation n or a zero-support class "
                f"(n={len(val_c)}, min_stratum_n={min_stratum_n}); fell back to the pooled weighted matrix "
                f"for BOTH the point estimate and the bootstrap CI (same error model)"
            )

        res = corrected_prevalence(
            auto_counts=auto_counts,
            confusion_from_human=confusion_frame,
            n_boot=1000,
            seed=seed,
            target="drift",
            cohort_frame=cohort_c[["message_id", "auto_label"]],
            val_frame=val_for_bootstrap[["message_id", "human", "auto", "weight"]],
        )
        res["validation_n"] = int(len(val_c))
        res["bootstrap_val_n"] = int(len(val_for_bootstrap))
        res["confusion_source_note"] = note
        out[corp] = res
    return out


_TARGET_CER_LEVELS = ("0.0", "0.1", "0.2", "0.3", "0.4", "ALL")


def corrected_drift_by_cer16(
    pooled_confusion_frame: pd.DataFrame,
    cohort: pd.DataFrame,
    val_frame: pd.DataFrame,
    seed: int = 0,
    min_stratum_n: int = DEFAULT_MIN_STRATUM_N,
) -> dict:
    """Rogan-Gladen misclassification-corrected drift prevalence per native
    target-CER level (0.0/0.1/0.2/0.3/0.4), using the WEIGHTED confusion
    matrix -- the CER-stratum analogue of `corrected_drift_by_corpus16`
    (mirrors it exactly; see that function's docstring for the full
    rationale, including the point-estimate/CI pairing discipline below).
    Added to validate the primary drift-vs-target-CER dose-response against
    physician-consensus labels AT EACH CER level, not just as one pooled
    correction (reviewer #1 follow-up).

    Args:
        pooled_confusion_frame: the weighted human(rows) x auto(cols)
            confusion DataFrame estimated on the FULL validation subset
            (all CER levels combined) -- the fallback matrix for any level
            that fails `_stratum_confusion_is_stable`.
        cohort: full-cohort automated-label rows, columns `["cer_target",
            "auto_label", "message_id"]` (one row per generation).
        val_frame: resolved validation items, columns `["cer_target",
            "human", "auto", "message_id", "weight"]` (one row per item;
            see `auto_vs_human_weighted`).
        seed: bootstrap seed, passed to `misclassification.
            corrected_prevalence`.
        min_stratum_n: passed to `_stratum_confusion_is_stable`.

    Returns:
        dict: one entry per level in `("0.0", "0.1", "0.2", "0.3", "0.4",
        "ALL")`, each `misclassification.corrected_prevalence`'s result
        dict plus `validation_n`, `bootstrap_val_n`, and
        `confusion_source_note` -- identical shape to one
        `corrected_drift_by_corpus16` entry.

    Point-estimate/CI pairing (same bug class `corrected_drift_by_corpus16`
    documents guarding against -- see that docstring's "Point-estimate/CI
    pairing" section in full): whenever `confusion_frame` is the POOLED
    matrix below, `val_for_bootstrap` is set to the FULL POOLED `val_frame`
    -- never this level's own `val_c` -- so the point estimate and the CI
    always resample the SAME error model.
    """
    out = {}
    for level in _TARGET_CER_LEVELS:
        cohort_c = cohort if level == "ALL" else cohort[cohort["cer_target"].round(1).astype(str) == level]
        auto_counts = cohort_c["auto_label"].value_counts().to_dict()
        if not auto_counts:
            out[level] = {"note": "insufficient data"}
            continue

        val_c = val_frame if level == "ALL" else val_frame[val_frame["cer_target"].round(1).astype(str) == level]

        if level != "ALL" and _stratum_confusion_is_stable(val_c, min_stratum_n):
            confusion_frame = confusion(
                val_c["human"], val_c["auto"], labels=CLASS_ORDER, weights=val_c["weight"]
            )
            val_for_bootstrap = val_c
            note = f"per-cer weighted confusion matrix (validation n={len(val_c)} >= min_stratum_n={min_stratum_n})"
        elif level == "ALL":
            confusion_frame = pooled_confusion_frame
            val_for_bootstrap = val_frame
            note = "pooled weighted confusion matrix (this IS the full pooled validation set)"
        else:
            confusion_frame = pooled_confusion_frame
            # Same pairing discipline as corrected_drift_by_corpus16: the
            # pooled point estimate must be bootstrapped from the SAME
            # pooled val_frame, never this level's own small val_c.
            val_for_bootstrap = val_frame
            note = (
                f"insufficient per-cer validation n or a zero-support class "
                f"(n={len(val_c)}, min_stratum_n={min_stratum_n}); fell back to the pooled weighted matrix "
                f"for BOTH the point estimate and the bootstrap CI (same error model)"
            )

        res = corrected_prevalence(
            auto_counts=auto_counts,
            confusion_from_human=confusion_frame,
            n_boot=1000,
            seed=seed,
            target="drift",
            cohort_frame=cohort_c[["message_id", "auto_label"]],
            val_frame=val_for_bootstrap[["message_id", "human", "auto", "weight"]],
        )
        res["validation_n"] = int(len(val_c))
        res["bootstrap_val_n"] = int(len(val_for_bootstrap))
        res["confusion_source_note"] = note
        out[level] = res
    return out


def run(
    rater_a_paths,
    rater_b_paths,
    key_path,
    out_path,
    adjudicator_paths=None,
    cohort_path=None,
    seed: int = 0,
    min_stratum_n: int = DEFAULT_MIN_STRATUM_N,
) -> dict:
    """End-to-end panel16 analysis; writes `out_path` and returns the digest.

    Args:
        rater_a_paths, rater_b_paths: rater A/B's completed sheet(s), each
            a single path or a list of paths (see `_load_sheets`).
        key_path: path to `_KEY_DO_NOT_SHARE/key.csv`, or an already-loaded
            DataFrame (accepted directly so tests can build one in memory).
        out_path: destination `panel16_results.json`.
        adjudicator_paths: optional third-rater sheet(s) for disagreed
            items, same shape as `rater_a_paths`.
        cohort_path: optional full-cohort parquet path (or an already-
            loaded DataFrame) for the weighted corrected-drift estimate.
            Skipped (with a note) if omitted or if any item is still
            unresolved.
        seed: bootstrap seed.
        min_stratum_n: passed to `corrected_drift_by_corpus16`.

    Returns:
        dict: the full digest (also written to `out_path` as JSON).
    """
    _slim = lambda d: d[[c for c in ["item_id", "rating", "notes"] if c in d.columns]]
    a = _slim(_load_sheets(rater_a_paths))
    b = _slim(_load_sheets(rater_b_paths))
    adj = _slim(_load_sheets(adjudicator_paths)) if adjudicator_paths else None
    key = key_path if isinstance(key_path, pd.DataFrame) else pd.read_csv(key_path)

    merged = merge_ratings(a, b)
    digest = {
        "n_items": int(len(set(a["item_id"]) | set(b["item_id"]))),
        "raw_pair_agreement": float(merged["agree"].mean()),
        "interrater": interrater(a, b),
    }

    cons = consensus_labels(a, b, adj)
    digest["resolution_counts"] = cons["resolution"].value_counts().to_dict()

    avh = auto_vs_human_weighted(cons, key)
    confusion_frame_pooled = avh.pop("_confusion_frame_pooled")
    val_frame = avh.pop("_val_frame")
    digest["automated_vs_human_weighted"] = avh

    if cohort_path is not None and digest["resolution_counts"].get("unresolved", 0) == 0:
        cohort = cohort_path if isinstance(cohort_path, pd.DataFrame) else pd.read_parquet(cohort_path)
        cohort = cohort.rename(columns={"label": "auto_label"})
        if "message_id" not in cohort.columns and "orig_message_id" in cohort.columns:
            cohort["message_id"] = cohort["orig_message_id"]
        digest["corrected_drift_by_corpus_weighted"] = corrected_drift_by_corpus16(
            confusion_frame_pooled, cohort, val_frame, seed=seed, min_stratum_n=min_stratum_n,
        )
        digest["corrected_drift_by_cer_weighted"] = corrected_drift_by_cer16(
            confusion_frame_pooled, cohort, val_frame, seed=seed, min_stratum_n=min_stratum_n,
        )
    else:
        digest["corrected_drift_by_corpus_weighted"] = {
            "note": "skipped: provide a cohort and resolve all disagreements (need adjudicator) first"
        }
        digest["corrected_drift_by_cer_weighted"] = {
            "note": "skipped: provide a cohort and resolve all disagreements (need adjudicator) first"
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(digest, indent=2, default=str))
    return digest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rater-a", nargs="+", required=True,
                    help="rater A's completed sheet(s) (panel_stratified + panel_zerocer, if rated separately)")
    ap.add_argument("--rater-b", nargs="+", required=True, help="rater B's completed sheet(s)")
    ap.add_argument("--adjudicator", nargs="+", default=None, help="third-rater sheet(s) for disagreed items")
    ap.add_argument("--key", required=True, help="output/human_rating_v3plus/_KEY_DO_NOT_SHARE/key.csv")
    ap.add_argument("--cohort", default=None,
                    help="full-cohort parquet (attempts_v3plus_labeled.parquet) for the weighted corrected estimate")
    ap.add_argument("--out", default="output/human_rating_v3plus/panel16_results.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-stratum-n", type=int, default=DEFAULT_MIN_STRATUM_N)
    args = ap.parse_args(argv)

    digest = run(
        args.rater_a, args.rater_b, args.key, args.out,
        adjudicator_paths=args.adjudicator, cohort_path=args.cohort, seed=args.seed,
        min_stratum_n=args.min_stratum_n,
    )
    ir = digest["interrater"]
    print(f"items {digest['n_items']} | inter-rater kappa {ir['cohen_kappa_overall']:.3f} "
          f"(raw {ir['raw_percent_agreement']:.3f})")
    avh = digest["automated_vs_human_weighted"]
    print(f"weighted auto-vs-human (unweighted kappa {avh['cohen_kappa_overall_unweighted']:.3f}) "
          f"(resolved {avh['n_resolved']}, unresolved {avh['n_unresolved']}, "
          f"sum_sampling_weight {avh['sum_sampling_weight']:.1f})")
    cd = digest["corrected_drift_by_corpus_weighted"]
    if "note" not in cd:
        for corp in _TARGET_CORPORA:
            r = cd.get(corp, {})
            if "corrected" in r:
                print(f"  corrected drift [{corp}]: naive {r['naive']:.3f} -> "
                      f"corrected {r['corrected']:.3f} CI {r['ci'][0]:.3f}-{r['ci'][1]:.3f} "
                      f"({r['confusion_source_note']})")
    else:
        print(cd["note"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
