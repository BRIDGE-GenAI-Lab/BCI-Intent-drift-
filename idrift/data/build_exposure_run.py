"""Build the exposure matrix from the real ALS confusion matrix + corpora.

Composes tested components: corpus (Task 4), probe set (Task 5), the real
per-subject confusion matrices (materialize_confusion), and build_exposure
(Task 7). Produces output/intermediate/exposure_{smoke,full}.parquet.

Design notes:
- Intended text is UPPERCASED before noise injection because the bigP3BCI
  speller grid is uppercase (A-Z, 1-9, space). The original casing is kept in
  `intended_display`; `intended_text` (the adjudication reference M) is the
  uppercased form so realized CER is measured against the true noise reference.
- Characters not on the 36-cell grid (punctuation, '0') pass through
  uncorrupted, because they are not selectable on the speller. This slightly
  lowers realized CER for punctuation-heavy messages (a stated, minor effect).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from idrift.data.build_exposure import build_exposure
from idrift.lib.io_utils import load_checkpoint, log_provenance, _dir


def _load_corpus_and_probe(n_corpus: int | None, n_probe: int | None, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    costello = load_checkpoint("corpus_costello")[["message_id", "corpus", "category", "intended_text"]]
    if n_corpus is not None and n_corpus < len(costello):
        costello = costello.iloc[rng.choice(len(costello), n_corpus, replace=False)].reset_index(drop=True)

    probe_path = _dir() / "probe_set.json"
    probe = pd.DataFrame(json.loads(probe_path.read_text()))
    probe = probe.assign(corpus="probe")[["message_id", "corpus", "category", "intended_text"]]
    if n_probe is not None and n_probe < len(probe):
        probe = probe.iloc[rng.choice(len(probe), n_probe, replace=False)].reset_index(drop=True)

    df = pd.concat([costello, probe], ignore_index=True)
    df["intended_display"] = df["intended_text"]
    df["intended_text"] = df["intended_text"].str.upper()  # M reference in grid casing
    return df


def _load_confusion() -> tuple[np.ndarray, list[str]]:
    """Load the POOLED overall confusion matrix + grid alphabet.

    Per-subject matrices are too sparse (a subject spells few distinct letters) to
    corrupt arbitrary text, so the noise structure is driven by the pooled matrix
    (all 36 grid characters, 2537 real ALS selections). Per-study matrices are used
    for a robustness/sensitivity analysis in the full pipeline, not here.
    """
    d = _dir()
    overall = np.load(d / "confusion_overall.npy")
    alphabet = json.loads((d / "grid_alphabet.json").read_text())
    return overall, alphabet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    args = ap.parse_args()

    if args.mode == "smoke":
        n_corpus, n_probe, cer_grid, n_seeds = 12, 8, [0.0, 0.2, 0.4], 1
    else:
        # tractable first-pass full run: 120 sampled Costello messages + ALL probe items,
        # 5-level CER grid, 1 seed, pooled ALS confusion. ~1255 exposure rows.
        n_corpus, n_probe, cer_grid, n_seeds = 120, None, [0.0, 0.1, 0.2, 0.3, 0.4], 1

    corpus_df = _load_corpus_and_probe(n_corpus, n_probe, seed=0)
    overall, alphabet = _load_confusion()
    subjects = ["pooled_ALS"]
    conf_subset = {"pooled_ALS": overall}

    exposure = build_exposure(corpus_df, conf_subset, cer_grid, n_seeds, alphabet)

    out = f"exposure_{args.mode}"
    from idrift.lib.io_utils import save_checkpoint
    save_checkpoint(exposure, out)

    # realized CER by target level (sanity)
    realized = exposure.groupby("cer_target")["actual_cer"].mean().round(3).to_dict()
    log_provenance({
        f"exposure_{args.mode}": {
            "rows": int(len(exposure)),
            "n_messages": int(corpus_df["message_id"].nunique()),
            "subjects": subjects,
            "cer_grid": cer_grid,
            "n_seeds": n_seeds,
            "realized_mean_cer_by_target": {str(k): v for k, v in realized.items()},
        }
    })

    print(f"mode={args.mode} rows={len(exposure)} messages={corpus_df['message_id'].nunique()} "
          f"subjects={subjects}")
    print("realized mean CER by target:", realized)
    print("\nsample (intended -> noisy) at CER 0.2:")
    samp = exposure[exposure.cer_target == 0.2].head(4)
    for _, r in samp.iterrows():
        print(f"  [{r.actual_cer:.2f}] {r.intended_text!r} -> {r.noisy_text!r}")


if __name__ == "__main__":
    main()
