"""Adapt the v2 labeled attempts parquet to the schema build_rating_sheet.py expects.

build_rating_sheet.py was written for the v1 labeled schema and reads
``model_id`` / ``model_class`` / ``output_text`` / ``category`` / ``final_label``.
The v2 labeled parquet (``output/intermediate/attempts_v2_labeled.parquet``,
533,400 generations, 7 models) instead spells these ``model`` / ``output_message``
/ ``corpus`` / ``label`` and has no ``category`` or ``model_class`` column.

This module is a thin, column-only mapping (no row filtering, no relabeling):

    model_id     <- model                (exact model tag, e.g. "phi4:mini")
    model_class  <- model                (exact tag; keeps all 7 models as strata)
    output_text  <- output_message
    category     <- "message_critical" where corpus == "CRIT", else the corpus
    final_label  <- label                (automated ensemble label, for the key)

The one substantive transform is the attempt key. build_rating_sheet.py keys an
"attempt" on (message_id, model_id, cer_target) and de-duplicates on it. In v2
the synthetic corruption is drawn independently per ``replicate_idx``, so the 20
replicates of a given (message, model, CER) cell are DISTINCT generations
(79.5% of cells carry more than one distinct output). Collapsing them on the v1
key would silently discard 19 of every 20 labeled generations and rate only one
corruption draw per cell. To keep each generation its own rate-able item, we
fold the replicate index into ``message_id`` (``costello_2164#r6``); the builder
then sees one unique attempt per generation and its stratified draw samples
across the real (message x model x CER x replicate) population the automated
labels were actually applied to. The original message id and replicate index are
preserved in separate columns for the unblinding key / downstream joins.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Columns build_rating_sheet.py reads, mapped from their v2 names. Everything the
# builder touches is listed here so a schema drift on either side fails loudly.
_V2_REQUIRED = [
    "model",
    "message_id",
    "corpus",
    "cer_target",
    "replicate_idx",
    "intended_text",
    "output_message",
    "label",
]


def adapt(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the columns build_rating_sheet.py expects.

    Pure column mapping plus the replicate-aware attempt key; no rows are
    dropped and no labels are changed.
    """
    missing = [c for c in _V2_REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"v2 labeled parquet is missing expected columns: {missing}")

    out = df.copy()
    out["orig_message_id"] = out["message_id"].astype(str)
    # Fold replicate into the id so (message_id, model_id, cer_target) is unique
    # per generation (see module docstring).
    out["message_id"] = out["orig_message_id"] + "#r" + out["replicate_idx"].astype(str)
    out["model_id"] = out["model"]
    out["model_class"] = out["model"]
    out["output_text"] = out["output_message"]
    out["category"] = out["corpus"].where(out["corpus"] != "CRIT", "message_critical")
    out["final_label"] = out["label"]
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="in_path",
        default="output/intermediate/attempts_v2_labeled.parquet",
        help="v2 labeled attempts parquet",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default="output/intermediate/attempts_v2_for_rating.parquet",
        help="destination parquet in the v1 rating-sheet schema",
    )
    args = ap.parse_args(argv)

    df = pd.read_parquet(args.in_path)
    mapped = adapt(df)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    mapped.to_parquet(args.out_path, index=False)
    print(f"wrote {args.out_path}: {len(mapped):,} rows, {mapped.shape[1]} cols")
    print("unique attempt keys (message_id, model_id, cer_target):",
          f"{(mapped['message_id'] + mapped['model_id'] + mapped['cer_target'].astype(str)).nunique():,}")
    print("category counts:")
    print(mapped["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
