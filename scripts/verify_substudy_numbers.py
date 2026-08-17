"""Check every substudy number in the manuscript and supplement against the digest.

Each claim is recomputed from output/interface_substudy_digest.json and then
searched for as a literal string in the manuscript sources. A claim that is
not found is either wrong or phrased differently; both need a human look.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
d = json.loads((PROJECT / "output" / "interface_substudy_digest.json").read_text())

TEXT = {
    p.name: p.read_text()
    for p in sorted((PROJECT / "manuscript").glob("*.md"))
    if p.name not in {"manuscript.md"} and not p.name.startswith("._")
}

pcr = d["per_condition_rates"]["by_condition"]
bycer = d["per_condition_rates_by_cer"]
ors = d["drift_condition_model"]["main_effects_reference"]["condition_odds_ratios"]
bh = {r["condition"]: r for r in d["benefit_harm_by_condition"]["by_condition_pooled"]["by_group"]}
cr = d["candidate_recovery"]
order = d["per_condition_rates"]["conditions_present"]

checks: list[tuple[str, str]] = []


def add(label, value):
    checks.append((label, value))


add("total generations", f"{d['n_rows_total']:,}")
add("per-condition generations", f"{d['per_condition_rates']['by_condition']['forced']['overall']['n_total']:,}")
add("n scored outputs", f"{d['drift_condition_model']['n_obs']:,}")
add("n declines excluded", f"{d['drift_condition_model']['n_declined_excluded']:,}")
add("n message clusters", str(d["drift_condition_model"]["n_clusters"]))

for c in order:
    b = pcr[c]["overall"]
    add(f"{c}: drift of answered", f"{b['rate_drift']*100:.1f}%")
    add(f"{c}: drift per 100 attempted", f"{b['n_drift']/b['n_total']*100:.1f}")
    add(f"{c}: faithful per 100 attempted", f"{b['n_faithful']/b['n_total']*100:.1f}")
    if c in ors:
        o = ors[c]
        add(f"{c}: OR", f"{o['odds_ratio']:.2f}")
        add(f"{c}: CI", f"{o['ci_lower']:.2f}-{o['ci_upper']:.2f}")
    add(f"{c}: net benefit-harm", f"{bh[c]['faithful_gained_per_drift_introduced']:.2f}")

for cer in bycer:
    ab = bycer[cer]["by_condition"]["abstain_enabled"]["overall"]
    add(f"abstain decline rate @CER {cer}", f"{ab['declined_rate']*100:.1f}%")

for c in ("forced", "copy_when_uncertain", "expansion", "abstain_enabled"):
    b = bycer["0.4"]["by_condition"][c]["overall"]
    add(f"{c}: drift @CER 0.4", f"{b['rate_drift']*100:.1f}%")

add("candidate primary faithful", f"{cr['primary_faithful_rate']*100:.1f}%")
add("candidate any faithful", f"{cr['any_candidate_faithful_rate']*100:.1f}%")
add("candidate lift", f"{cr['recovery_lift']*100:.1f}")
add("candidate lift @CER 0.2", f"{cr['by_cer_target']['0.2']['recovery_lift']*100:.1f}")

print(f"{'STATUS':<8} {'CLAIM':<42} {'VALUE':<12} FILES")
print("-" * 92)
ok = missing = 0
for label, value in checks:
    hits = sorted(n.replace(".md", "") for n, t in TEXT.items() if value in t)
    if hits:
        ok += 1
        print(f"{'OK':<8} {label:<42} {value:<12} {','.join(hits)}")
    else:
        missing += 1
        print(f"{'ABSENT':<8} {label:<42} {value:<12} -")
print("-" * 92)
print(f"{ok} found in text, {missing} not stated")
