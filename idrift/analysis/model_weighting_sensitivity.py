"""Generation-weighted vs equal-model-weighted pooled drift (reviewer major #7).

Every pooled drift rate elsewhere in this manuscript is GENERATION-weighted:
each of the 20 models contributes its own number of generations to the
pooled denominator (243,000 for 17 models, 40,500/40,326 for the 3
reduced-replicate reasoning models), so a model run on the full grid
implicitly counts ~6x more than a reduced-replicate model in any pooled
rate. This module reports the alternative EQUAL-model-weight rate (each of
the 20 models contributes one vote, its own drift rate, unweighted by how
many generations it ran) as a sensitivity check, so a reader can see how
much the panel's own composition (17 full-grid + 3 reduced-replicate)
could move the headline number.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_CORPORA = ("AUTH", "CRIT", "CTRL")


def run(parquet_path: str, out_path: str) -> dict:
    df = pd.read_parquet(parquet_path, columns=["model", "corpus", "label"])
    generation_weighted = {}
    model_weighted_equal = {}
    per_model_auth_rate = {}
    for corpus in _CORPORA:
        sub = df[df["corpus"] == corpus]
        generation_weighted[corpus] = float((sub["label"] == "drift").mean() * 100.0)
        per_model = sub.groupby("model")["label"].apply(lambda s: float((s == "drift").mean() * 100.0))
        model_weighted_equal[corpus] = float(per_model.mean())
        if corpus == "AUTH":
            per_model_auth_rate = {k: float(v) for k, v in per_model.items()}
    digest = {
        "generation_weighted": generation_weighted,
        "model_weighted_equal": model_weighted_equal,
        "per_model_auth_rate": per_model_auth_rate,
        "n_models": len(per_model_auth_rate),
        "notes": (
            "generation_weighted pools all labeled generations regardless of model "
            "(each model contributes proportional to its own generation count, 243,000 "
            "for 17 models vs 40,500/40,326 for the 3 reduced-replicate reasoning models); "
            "model_weighted_equal is the unweighted mean of the 20 per-model rates, one vote per "
            "model regardless of how many generations it contributed. Reported as a sensitivity "
            "check on panel composition, not a claim that either weighting is more correct."
        ),
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(digest, indent=2))
    return digest


def main() -> None:
    digest = run("output/intermediate/attempts_v3plus_labeled.parquet", "output/model_weighting_sensitivity_digest.json")
    print(f"generation_weighted: {digest['generation_weighted']}")
    print(f"model_weighted_equal: {digest['model_weighted_equal']}")
    print(f"n_models: {digest['n_models']}")


if __name__ == "__main__":
    main()
