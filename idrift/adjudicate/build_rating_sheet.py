"""Blinded dual-rater rating-sheet generator (manuscript_facts.md Sec. 7, layer 3).

Builds the human ground-truth instrument: a stratified sample of labeled
attempts (CER grid x model_class) PLUS every message-critical item, blinded
to model identity / CER / the automated label, for two independent human
raters (rater A = Alon, rater B). Reuses `human_export.stratified_sample`
for the stratified draw rather than re-implementing quota sampling.

Selection is de-duplicated by a natural attempt key
(message_id, model_id, cer_target) -- the real attempts_labeled.parquet has
zero duplicate rows on that triple (verified against the baseline 5,020-row
run), so no synthetic row-id column is needed. Blinded item ids ("R0001", ...)
are assigned by sorting on that natural key (a canonical, seed-independent
base order) and then shuffling deterministically with `--seed`, so the same
input data + seed always reproduce the same item_id assignment.

Usage:
    uv run python -m idrift.adjudicate.build_rating_sheet \\
        --attempts output/attempts_labeled.parquet \\
        --n-stratified 500 --seed 0 --outdir output/human_rating

Rev Task 4.1 adds two flags that resolve two reviewer fatal points: the
manuscript claimed "every message-critical output" was human-adjudicated
when the sheet was actually a capped, stratified enrichment of the
message-critical pool; and a 395-vs-378 discrepancy in which the automated-
vs-human agreement metric silently ran on fewer items than the human panel
actually rated (see `idrift.adjudicate.reconcile.validation_frame` for the
structural fix to that second problem).

    --all-critical          include EVERY message-critical item (a full
                             census, ignoring --n-critical).
    --power-target N        size the non-critical stratified draw so the
                             grand total AIMS FOR approximately N (an upper
                             target; the realized total may be modestly
                             lower due to per-stratum quota granularity --
                             see `size_stratified_draw_for_power_target`).

    uv run python -m idrift.adjudicate.build_rating_sheet \\
        --attempts output/attempts_v2_labeled.parquet \\
        --all-critical --power-target 1000 --seed 0 --outdir output/human_rating_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from idrift.adjudicate.human_export import stratified_sample

RATING_LABELS = ("Faithful", "Degraded", "Drift", "Message-critical drift")

BLINDED_COLUMNS = ["item_id", "intended_message", "system_output", "rating", "notes"]


def _attempt_key(df: pd.DataFrame) -> pd.Series:
    """Stable natural key identifying one attempt: (message_id, model_id, cer_target).

    Verified unique (0 duplicates) on the real baseline attempts_labeled.parquet
    (5,020 rows, 4 models x 5 CER levels). `cer_target` is rounded before
    stringifying so float repr noise cannot split what should be one key.
    """
    return (
        df["message_id"].astype(str)
        + "::"
        + df["model_id"].astype(str)
        + "::"
        + df["cer_target"].round(6).astype(str)
    )


def size_stratified_draw_for_power_target(n_critical_selected: int, power_target: int) -> int:
    """Map a prespecified TOTAL human-rated sample size to the size of the
    non-critical stratified draw (rev Task 4.1).

    The message-critical items are drawn first and are non-negotiable (every
    one of them under `--all-critical`, or a fixed cap otherwise). Whatever
    budget remains after that is what the non-critical stratified (AUTH/CTRL)
    draw should aim for, so the grand total (critical + stratified) AIMS FOR
    approximately `power_target`:

        n_stratified = max(0, power_target - n_critical_selected)

    `power_target` is an upper target, not a guarantee: the reused
    `stratified_sample` allocates `n // n_strata` per stratum, so the
    realized stratified draw -- and therefore the grand total -- can fall
    modestly short of the requested size whenever that per-stratum quota
    does not divide evenly. Callers who need an exact total should treat
    `power_target` as a floor-seeking approximation, not an exact count.

    Floored at 0 -- never negative -- because the critical draw is fixed and
    the stratified draw exists only to top up the total; when the critical
    draw alone already meets or exceeds the target, no additional
    non-critical items are drawn.

    Args:
        n_critical_selected: Number of message-critical items already
            selected (e.g. `len(critical)` after applying `all_critical` /
            `n_critical`).
        power_target: Prespecified total human-rated sample size (Task 1.1
            power calculation), e.g. sized so class-specific metrics have
            usable confidence intervals. Treated as an upper target, not an
            exact count -- see the per-stratum-quota-granularity note above.

    Returns:
        int: target size for the non-critical stratified draw.
    """
    return max(0, power_target - n_critical_selected)


def select_items(
    df: pd.DataFrame,
    n_stratified: int,
    seed: int,
    n_critical: int | None = None,
    all_critical: bool = False,
    power_target: int | None = None,
) -> pd.DataFrame:
    """Select the union of a stratified sample and the message-critical items.

    Stratifies on CER grid x model_class jointly (one combined `_stratum`
    column) via the existing `stratified_sample` quota sampler. The stratified
    draw is taken from the NON-critical rows only (`category !=
    "message_critical"`), so the stratified and message-critical selections
    are disjoint by construction -- the union is genuinely `n_critical +
    n_stratified` (up to the sampler's per-stratum quota granularity), never
    an overlap-shrunk total. Adds a `_selection_source` column recording
    whether each row was drawn from the stratified sample, the
    message-critical set, or (defensively) both -- "both" should not occur
    given the disjoint pools, but the label is kept for the summary print,
    not for the rater-facing sheet.

    Args:
        df: Labeled attempts DataFrame (schema.AttemptRecord columns).
        n_stratified: Target size of the stratified draw (before the
            message-critical union; the final selection is usually larger).
            Ignored when `power_target` is given (see below).
        seed: Seed for the stratified draw (passed through to
            `stratified_sample`).
        n_critical: If given and smaller than the available message-critical
            pool, stratified-sample that many message-critical items (across
            CER x model_class) instead of taking every one. This keeps the
            human rating burden feasible when the full model x CER matrix
            makes "every message-critical output" run to thousands of rows.
            Default None keeps the original behavior (all message-critical).
            Ignored when `all_critical` is True.
        all_critical: If True, include EVERY message-critical item present in
            `df` and ignore `n_critical` entirely (rev Task 4.1: resolves the
            reviewer's fatal point that the manuscript claimed "every
            message-critical output" was adjudicated when the sheet had
            actually been built from a capped, stratified enrichment of the
            message-critical pool, not a full census of it).
        power_target: If given, `n_stratified` is NOT used directly; the
            non-critical stratified draw is instead sized by
            `size_stratified_draw_for_power_target` so the grand total
            (critical + stratified) aims for approximately this prespecified
            total sample size -- an upper target, not a guarantee: the
            realized total may be modestly lower due to per-stratum quota
            granularity in the reused sampler. When the sized draw is 0 (the
            critical draw alone already meets or exceeds the target), no
            stratified sampling is performed at all.

    Returns:
        DataFrame: de-duplicated union, original attempts columns plus
            `_attempt_key` and `_selection_source`.
    """
    work = df.copy()
    work["_attempt_key"] = _attempt_key(work)
    work["_stratum"] = work["cer_target"].round(6).astype(str) + "|" + work["model_class"].astype(str)

    critical_pool = work[work["category"] == "message_critical"]
    # Stratified draw is taken from the NON-critical rows only, never from
    # `work` as a whole. Drawing from the full frame let the stratified
    # sampler re-select rows already claimed by the critical pool, so the
    # union (after drop_duplicates) landed BELOW n_critical + n_stratified --
    # the two pools were not actually disjoint. Restricting the source pool
    # here makes them disjoint by construction, so the union is genuinely
    # n_critical + n_stratified (up to the sampler's own per-stratum quota
    # granularity -- see size_stratified_draw_for_power_target).
    noncritical_pool = work[work["category"] != "message_critical"]
    if all_critical:
        critical = critical_pool
    elif n_critical is not None and len(critical_pool) > n_critical:
        # enrich but cap: stratified draw across CER x model_class (distinct seed)
        critical = stratified_sample(critical_pool, n=n_critical, by="_stratum", seed=seed + 1)
    else:
        critical = critical_pool

    effective_n_stratified = n_stratified
    if power_target is not None:
        effective_n_stratified = size_stratified_draw_for_power_target(
            n_critical_selected=len(critical), power_target=power_target,
        )

    if effective_n_stratified > 0 and not noncritical_pool.empty:
        stratified = stratified_sample(noncritical_pool, n=effective_n_stratified, by="_stratum", seed=seed)
    else:
        # stratified_sample's quota is floored at 1-per-stratum even for
        # n=0, so a target of 0 must skip the draw entirely rather than
        # calling it -- otherwise power_target could never be honored when
        # the critical draw alone already meets or exceeds it. Also skipped
        # when there are no non-critical rows left to draw from at all.
        stratified = work.iloc[0:0]

    strat_keys = set(stratified["_attempt_key"])
    crit_keys = set(critical["_attempt_key"])

    combined = pd.concat([stratified, critical]).drop_duplicates(subset="_attempt_key").copy()

    def _source(key: str) -> str:
        # Disjoint by construction (see noncritical_pool above): a key can
        # no longer be in both sets. "both" is kept as a defensive label
        # (not removed) so print_summary's breakdown stays meaningful even
        # if that invariant is ever violated by a future change.
        in_s, in_c = key in strat_keys, key in crit_keys
        if in_s and in_c:
            return "both"
        return "message_critical" if in_c else "stratified"

    combined["_selection_source"] = combined["_attempt_key"].map(_source)
    return combined


def assign_item_ids(selected: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign stable, shuffled blinded item ids ("R0001", ...).

    Sorts on the natural `_attempt_key` first (a canonical order independent
    of the incoming row order / selection order), then shuffles that
    canonical frame deterministically with `seed`. Same data + same seed ->
    same item_id assignment, every time.
    """
    canonical = selected.sort_values("_attempt_key").reset_index(drop=True)
    shuffled = canonical.sample(frac=1, random_state=seed).reset_index(drop=True)
    shuffled = shuffled.copy()
    shuffled["item_id"] = [f"R{i + 1:04d}" for i in range(len(shuffled))]
    return shuffled


def make_blinded_sheet(items: pd.DataFrame) -> pd.DataFrame:
    """Build the rater-facing blinded sheet: ONLY item_id/intended_message/
    system_output plus blank rating/notes columns. No model_id, cer, or
    final_label leaks through.
    """
    sheet = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "intended_message": items["intended_text"],
            "system_output": items["output_text"],
            "rating": "",
            "notes": "",
        }
    )
    return sheet[BLINDED_COLUMNS]


def make_key(items: pd.DataFrame) -> pd.DataFrame:
    """Build the unblinding key: item_id -> original attempt id, model_id,
    cer, category, automated final_label (plus model_class / selection
    source for extra provenance).
    """
    key = pd.DataFrame(
        {
            "item_id": items["item_id"],
            "orig_attempt_id": items["_attempt_key"],
            "model_id": items["model_id"],
            "model_class": items["model_class"],
            "cer_target": items["cer_target"],
            "category": items["category"],
            "automated_final_label": items.get("final_label"),
            "selection_source": items["_selection_source"],
        }
    )
    return key


def _esc(s: object) -> str:
    """Minimal HTML-escape for inline embedding (no external templating lib)."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def make_rating_form_html(items: pd.DataFrame) -> str:
    """Render a self-contained offline HTML rating form for the same blinded
    items: radio buttons for the four ordinal labels, a notes box per item,
    and a "download my responses as CSV" button (inline JS, no external
    assets). The CSVs remain the primary path; this is a convenience.
    """
    rows_html = []
    for r in items.itertuples():
        radios = "\n".join(
            f'<label><input type="radio" name="rating_{_esc(r.item_id)}" value="{_esc(lbl)}"> {_esc(lbl)}</label>'
            for lbl in RATING_LABELS
        )
        rows_html.append(
            f"""
<div class="item" data-item-id="{_esc(r.item_id)}">
  <h3>{_esc(r.item_id)}</h3>
  <p><strong>Intended message:</strong> {_esc(r.intended_text)}</p>
  <p><strong>System output:</strong> {_esc(r.output_text)}</p>
  <div class="ratings">{radios}</div>
  <textarea class="notes" placeholder="Notes (optional)"></textarea>
</div>"""
        )

    items_block = "\n".join(rows_html)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Intent Drift: dual-rater rating form</title>
<style>
body {{ font-family: Georgia, serif; max-width: 800px; margin: 2em auto; padding: 0 1em; }}
.item {{ border-bottom: 1px solid #ccc; padding: 1em 0; }}
.ratings label {{ display: inline-block; margin-right: 1.2em; }}
textarea.notes {{ width: 100%; height: 3em; margin-top: 0.5em; }}
#download-btn {{ position: sticky; top: 0; background: #fff; padding: 0.5em 0; }}
</style>
</head>
<body>
<div id="download-btn">
<button type="button" onclick="downloadCsv()">Download my responses as CSV</button>
</div>
<form id="rating-form">
{items_block}
</form>
<script>
function csvEscape(s) {{
  s = String(s == null ? "" : s);
  if (s.indexOf(",") !== -1 || s.indexOf('"') !== -1 || s.indexOf("\\n") !== -1) {{
    s = '"' + s.replace(/"/g, '""') + '"';
  }}
  return s;
}}

function downloadCsv() {{
  var items = document.querySelectorAll(".item");
  var lines = ["item_id,rating,notes"];
  items.forEach(function (el) {{
    var itemId = el.getAttribute("data-item-id");
    var checked = el.querySelector('input[type="radio"]:checked');
    var rating = checked ? checked.value : "";
    var notes = el.querySelector(".notes").value;
    lines.push([csvEscape(itemId), csvEscape(rating), csvEscape(notes)].join(","));
  }});
  var blob = new Blob([lines.join("\\n")], {{ type: "text/csv" }});
  var url = URL.createObjectURL(blob);
  var a = document.createElement("a");
  a.href = url;
  a.download = "rating_sheet_responses.csv";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}
</script>
</body>
</html>
"""


def print_summary(key: pd.DataFrame) -> None:
    """Print the console summary: total items, stratified vs message-critical
    counts, per-CER counts, per-model counts -- all read from the key (the
    blinded sheet carries none of this).
    """
    n_total = len(key)
    src_counts = key["selection_source"].value_counts().to_dict()
    n_stratified = src_counts.get("stratified", 0) + src_counts.get("both", 0)
    n_critical = src_counts.get("message_critical", 0) + src_counts.get("both", 0)

    print(f"Total items: {n_total}")
    print(f"  from stratified sample: {n_stratified}")
    print(f"  message-critical (every item): {n_critical}")
    print(f"  in both (stratified draw happened to hit a critical item): {src_counts.get('both', 0)}")

    print("Per-CER counts:")
    for cer, n in sorted(key["cer_target"].value_counts().to_dict().items()):
        print(f"  CER {cer}: {n}")

    print("Per-model counts:")
    for model, n in key["model_id"].value_counts().to_dict().items():
        print(f"  {model}: {n}")


def build_rating_sheet(
    attempts_path,
    n_stratified: int,
    seed: int,
    outdir,
    n_critical: int | None = None,
    all_critical: bool = False,
    power_target: int | None = None,
) -> dict:
    """End-to-end: load parquet -> select items -> blind -> write the three
    artifacts (rating_sheet_raterA.csv, rating_sheet_raterB.csv, key.csv,
    rating_form.html) -> print the summary.

    Args:
        attempts_path: Path to a labeled attempts parquet (or a DataFrame --
            accepted directly so tests can build one in memory or via a temp
            parquet round trip).
        n_stratified: Target size of the stratified draw. Ignored when
            `power_target` is given.
        seed: Random seed (sampling + shuffle), for determinism.
        outdir: Output directory; created if missing.
        n_critical: Cap on message-critical items (stratified-sampled);
            ignored when `all_critical` is True.
        all_critical: Include EVERY message-critical item present in the
            labeled attempts (rev Task 4.1); see `select_items`.
        power_target: Prespecified total human-rated sample size; sizes the
            non-critical stratified draw via
            `size_stratified_draw_for_power_target` (rev Task 4.1); see
            `select_items`. An upper target -- the realized total aims for
            approximately `power_target` but may be modestly lower due to
            per-stratum quota granularity.

    Returns:
        dict: paths of the four written artifacts, keyed by name.
    """
    df = attempts_path if isinstance(attempts_path, pd.DataFrame) else pd.read_parquet(attempts_path)

    selected = select_items(
        df, n_stratified=n_stratified, seed=seed, n_critical=n_critical,
        all_critical=all_critical, power_target=power_target,
    )
    items = assign_item_ids(selected, seed=seed)

    blinded = make_blinded_sheet(items)
    key = make_key(items)
    form_html = make_rating_form_html(items)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    path_a = out / "rating_sheet_raterA.csv"
    path_b = out / "rating_sheet_raterB.csv"
    path_key = out / "key.csv"
    path_html = out / "rating_form.html"

    blinded.to_csv(path_a, index=False)
    blinded.to_csv(path_b, index=False)
    key.to_csv(path_key, index=False)
    path_html.write_text(form_html)

    print_summary(key)

    return {
        "rater_a": path_a,
        "rater_b": path_b,
        "key": path_key,
        "rating_form_html": path_html,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Build blinded dual-rater rating sheets from a labeled attempts parquet."
    )
    ap.add_argument("--attempts", required=True, help="path to a labeled attempts parquet")
    ap.add_argument(
        "--n-stratified", type=int, default=500,
        help="target size of the stratified sample across CER grid x model_class",
    )
    ap.add_argument(
        "--n-critical", type=int, default=None,
        help="cap on message-critical items (stratified-sampled); default None = every one; "
        "ignored when --all-critical is set",
    )
    ap.add_argument(
        "--all-critical", action="store_true",
        help="include EVERY message-critical item present in the labeled attempts, ignoring "
        "--n-critical entirely (rev Task 4.1: a full census, not a capped enrichment)",
    )
    ap.add_argument(
        "--power-target", type=int, default=None,
        help="prespecified TOTAL human-rated sample size (Task 1.1 power calc); sizes the "
        "non-critical stratified draw as max(0, power_target - n_critical_selected), overriding "
        "--n-stratified; aims for approximately this total (an upper target -- the realized "
        "total may be modestly lower due to per-stratum quota granularity)",
    )
    ap.add_argument("--seed", type=int, default=0, help="seed for sampling and blinded shuffling")
    ap.add_argument("--outdir", default="output/human_rating", help="output directory")
    args = ap.parse_args(argv)

    build_rating_sheet(
        args.attempts, n_stratified=args.n_stratified, seed=args.seed,
        outdir=args.outdir, n_critical=args.n_critical,
        all_critical=args.all_critical, power_target=args.power_target,
    )


if __name__ == "__main__":
    main()
