"""Rebuild only the v3 dependence-sensitivity digest after the GEE guard fix.

The published v3 digest carried a diverged exchangeable GEE (slope -3.6e22, SE 0)
that statsmodels reported as converged. That value drove the reported spread to
~1e22 and flipped the stability conclusion to False. With the plausibility guard
in place the fit falls back to an independence working correlation, as it did in
v2, and the digest can be rebuilt without re-running every other section.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idrift.analysis.run_v3 import build_dependence_digest

PROJECT = Path(__file__).resolve().parents[1]
LABELED = PROJECT / "output" / "intermediate" / "attempts_v3_labeled.parquet"
DIGEST = PROJECT / "output" / "dependence_sensitivity_v3_digest.json"
CONF = PROJECT / "output" / "confidence_v3_digest.json"
STATS = PROJECT / "output" / "stats_v3_digest.json"

t0 = time.time()
print(f"loading {LABELED.name} ...", flush=True)
df = pd.read_parquet(LABELED)
print(f"  {len(df):,} rows in {time.time() - t0:.0f}s", flush=True)

previous = json.loads(DIGEST.read_text())
anchor = previous["reference_slopes"]["univariate_message_clustered_anchor"]
univariate_anchor = {"est": anchor["slope"], "se": anchor["se"]}
primary_adjusted = previous["reference_slopes"]["primary_message_clustered_ADJUSTED"]

print("rebuilding dependence digest (GEE + DL pool + 2 bootstraps) ...", flush=True)
digest = build_dependence_digest(
    df, primary_slope_adjusted=primary_adjusted, univariate_anchor=univariate_anchor
)

DIGEST.write_text(json.dumps(digest, indent=2))
print(f"written in {time.time() - t0:.0f}s\n")

print("=== corrected estimators ===")
for name, est in digest["estimators"].items():
    slope = est.get("slope") if isinstance(est, dict) else est
    extra = est.get("cov_struct_used", "") if isinstance(est, dict) else ""
    print(f"  {name:22s} {slope:8.3f}   {extra}")
print(f"\nspread across estimators : {digest['slope_spread_across_estimators']:.4f}")
print(f"max % deviation vs anchor: {digest['max_pct_deviation_from_anchor']:.2f}%")
print(f"stable                   : {digest['conclusion']['stable_across_dependence_assumptions']}")
print(f"\nsummary: {digest['conclusion']['summary']}")
