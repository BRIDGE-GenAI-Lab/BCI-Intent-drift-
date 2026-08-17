"""Run the D5 interface substudy analysis on the O2-labeled P1-P5 outputs.

The raw-decode label frame is the union of the v2 main-run labels and the v3
delta labels; together they cover all 28,100 substudy exposure keys exactly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idrift.analysis.interface_substudy import run

PROJECT = Path(__file__).resolve().parents[1]
INTER = PROJECT / "output" / "intermediate"
KEYS = ["message_id", "corpus", "cer_target", "replicate_idx"]

t0 = time.time()
raw = pd.concat(
    [
        pd.read_parquet(INTER / "raw_decode_labels_v2.parquet"),
        pd.read_parquet(INTER / "raw_decode_v3_delta_labeled.parquet"),
    ],
    ignore_index=True,
).drop_duplicates(subset=KEYS)
print(f"raw decode labels: {len(raw):,} unique exposures", flush=True)

digest = run(
    substudy_labeled_path=INTER / "substudy_attempts_labeled.parquet",
    v3_labeled_path=INTER / "attempts_v3_labeled.parquet",
    v2_labeled_path=INTER / "attempts_v2_labeled.parquet",
    subset_parquet=INTER / "substudy_exposures.parquet",
    raw_decode_path=raw,
    out_path=str(PROJECT / "output" / "interface_substudy_digest.json"),
)

print(f"\ndone in {time.time() - t0:.0f}s")
print(f"rows total {digest['n_rows_total']:,} "
      f"(substudy {digest['n_rows_substudy']:,} + forced {digest['n_rows_forced_assembled']:,})")

print("\n=== per-condition rates (overall) ===")
for cond in digest["per_condition_rates"]["conditions_present"]:
    b = digest["per_condition_rates"]["by_condition"][cond]["overall"]
    print(f"  {cond:20s} n={b['n_total']:>7,}  declined {b['declined_rate']*100:5.1f}%  "
          f"faithful {b['rate_faithful']*100:5.1f}%  degraded {b['rate_degraded']*100:5.1f}%  "
          f"drift {b['rate_drift']*100:5.1f}%")

print("\n=== drift vs forced (message-clustered logit) ===")
dcm = digest["drift_condition_model"]
print(f"  converged={dcm.get('converged')} method={dcm.get('method')}")
for term, v in (dcm.get("odds_ratios") or {}).items():
    print(f"  {term:30s} {v}")

print("\n=== candidate recovery ===")
print(digest["candidate_recovery"])

print("\n=== benefit-harm ranked ===")
for r in digest["benefit_harm_by_condition"].get("ranked_by_net_metric", []):
    print(f"  {r['condition']:20s} {r['faithful_gained_per_drift_introduced']}")
