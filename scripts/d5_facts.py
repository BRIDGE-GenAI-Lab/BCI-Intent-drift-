"""Extract the substudy facts the Results section will cite.

Adds a `per_condition_rates_by_cer` block to the digest by calling the tested
`per_condition_rates` on cer_target slices (no new estimation logic), and
prints the throughput view that counts declines in the denominator -- without
it, abstain_enabled's faithful rate is a survivor rate and overstates it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from idrift.analysis.digests import _default
from idrift.analysis.interface_substudy import assemble_forced, per_condition_rates

PROJECT = Path(__file__).resolve().parents[1]
INTER = PROJECT / "output" / "intermediate"
DIGEST = PROJECT / "output" / "interface_substudy_digest.json"

sub = pd.read_parquet(INTER / "substudy_attempts_labeled.parquet")
ids = sorted(set(pd.read_parquet(INTER / "substudy_exposures.parquet")["message_id"]))
forced = assemble_forced(
    ids, INTER / "attempts_v3_labeled.parquet", INTER / "attempts_v2_labeled.parquet"
)
allrows = pd.concat([sub, forced], ignore_index=True, sort=False)

digest = json.loads(DIGEST.read_text())

by_cer = {
    str(cer): per_condition_rates(slice_)
    for cer, slice_ in allrows.groupby("cer_target", sort=True)
}
digest["per_condition_rates_by_cer"] = by_cer
DIGEST.write_text(json.dumps(digest, indent=2, default=_default, sort_keys=True))

print("=== abstention is targeted? declined rate by target CER ===")
for cer, blk in by_cer.items():
    b = blk["by_condition"]["abstain_enabled"]["overall"]
    print(f"  CER {cer}: declined {b['declined_rate']*100:5.1f}%   "
          f"drift among answered {b['rate_drift']*100:5.1f}%")

print("\n=== throughput: faithful per 100 exposures (declines in denominator) ===")
order = digest["per_condition_rates"]["conditions_present"]
for cond in order:
    b = digest["per_condition_rates"]["by_condition"][cond]["overall"]
    thr = b["n_faithful"] / b["n_total"] * 100
    dr_all = b["n_drift"] / b["n_total"] * 100
    print(f"  {cond:20s} faithful/100 {thr:5.1f}   drift/100 {dr_all:5.1f}   "
          f"(survivor faithful {b['rate_faithful']*100:5.1f}%)")

print("\n=== drift by target CER, forced vs each condition ===")
print(f"  {'CER':<6}" + "".join(f"{c[:12]:>14}" for c in order))
for cer, blk in by_cer.items():
    row = f"  {cer:<6}"
    for cond in order:
        b = blk["by_condition"][cond]["overall"]
        row += f"{b['rate_drift']*100:13.1f}%"
    print(row)

print("\n=== benefit-harm, pooled per condition ===")
for r in digest["benefit_harm_by_condition"]["by_condition_pooled"]["by_group"]:
    print(f"  {r['condition']:20s} " + "  ".join(
        f"{k.replace('rate_',''):>16s} {r[k]*100:5.1f}%"
        for k in ("rate_rescue", "rate_silent_failure", "rate_llm_induced_harm") if k in r
    ) + f"   net {r['faithful_gained_per_drift_introduced']:.2f}")

print("\nkeys in a benefit-harm row:", sorted(
    digest["benefit_harm_by_condition"]["by_condition_pooled"]["by_group"][0]))
