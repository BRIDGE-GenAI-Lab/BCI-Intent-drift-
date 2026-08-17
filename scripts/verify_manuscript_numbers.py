"""Cross-check every load-bearing manuscript number against its digest of record.

Two passes. Forward: each digest value must appear somewhere in the manuscript sources.
Reverse: known superseded values (v2 numbers replaced by v3) must appear nowhere.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MS = PROJECT / "manuscript"


def load(name):
    return json.loads((PROJECT / "output" / name).read_text())


stats = load("stats_v3_digest.json")
bh = load("benefit_harm_v3_digest.json")
multi = load("multinomial_v3_digest.json")
conf = load("confidence_v3_digest.json")
dep = load("dependence_sensitivity_v3_digest.json")
zero = load("zero_cer_breakdown_v3_digest.json")

TEXT = {
    p.name: p.read_text()
    for p in sorted(MS.glob("*.md"))
    if p.name != "manuscript.md" and not p.name.startswith("._")
}

CHECKS: list[tuple[str, str, tuple[str, ...]]] = []


def check(label, value, *variants):
    CHECKS.append((label, value, variants or (value,)))


def pct(x, d=1):
    return f"{x:.{d}f}"


# scale
n_rows = stats["_meta"]["n_rows"]
check("labeled generations", f"{n_rows:,}", f"{n_rows:,}")
check("AUTH corpus messages", str(stats["_meta"]["n_auth_messages"]),
      str(stats["_meta"]["n_auth_messages"]))

# drift curve, values already in percent
for corpus, row in stats["drift_curve"]["drift_by_corpus_x_targetcer"].items():
    for cer, rate in sorted(row.items()):
        check(f"drift {corpus} @CER {cer}", pct(rate), pct(rate), pct(rate, 2).rstrip("0").rstrip("."))

# calibration
check("meta ECE", pct(stats["calibration"]["meta_ece"]["pooled"], 2),
      pct(stats["calibration"]["meta_ece"]["pooled"], 2))
check("meta AUROC", pct(stats["calibration"]["meta_auroc"]["pooled"], 2),
      pct(stats["calibration"]["meta_auroc"]["pooled"], 2))

# zero-CER
z = zero["overall"]["total_zero_cer_drift"]
check("zero-CER drift count", f"{z:,}", f"{z:,}")

# benefit-harm
for key, label in [("rate_rescue", "rescue"), ("rate_silent_failure", "silent failure"),
                   ("rate_llm_induced_harm", "LLM-induced harm")]:
    v = bh["overall"][key]
    v = v * 100 if v < 1 else v
    check(f"benefit-harm {label}", pct(v), pct(v), pct(v, 2))
for cer, v in sorted(bh["auth_net_by_cer"].items()):
    if cer != "0.0":
        check(f"AUTH net benefit @CER {cer}", pct(v, 2), pct(v, 2))

# confidently wrong
cw = conf["confidently_wrong_rate"]["pooled"]["confidently_wrong_rate"] * 100
check("confidently-wrong rate", pct(cw), pct(cw))

# slope
check("primary slope (adjusted)", pct(dep["reference_slopes"]["primary_message_clustered_ADJUSTED"], 2),
      pct(dep["reference_slopes"]["primary_message_clustered_ADJUSTED"], 2))

# multinomial drift-vs-degraded, realized-CER term
for term in multi["contrast_drift_vs_degraded"]["terms"]:
    if "cer" in term["term"].lower():
        check(f"drift-vs-degraded OR ({term['term']})", pct(term["odds_ratio"], 2),
              pct(term["odds_ratio"], 2))

SUPERSEDED = {
    "533,400": "v2 generation count, replaced by full-corpus v3",
    "4.79": "v2 slope, replaced by 5.41",
    "8.21": "v2 drift-vs-degraded OR, replaced by 10.69",
    "1462": "v2 zero-CER count, replaced by v3",
    "1,462": "v2 zero-CER count, replaced by v3",
    "0.32 [0.24": "v2 ECE CI",
    "27%": "v2 confidently-wrong, replaced by 26.3%",
}

print(f"{'STATUS':<9} {'CLAIM':<40} {'DIGEST':<10} FILES")
print("-" * 96)
ok = absent = 0
for label, value, variants in CHECKS:
    hits = sorted({name for name, t in TEXT.items() for v in variants if v in t})
    if hits:
        ok += 1
        print(f"{'OK':<9} {label:<40} {value:<10} {','.join(h.replace('.md','') for h in hits)}")
    else:
        absent += 1
        print(f"{'ABSENT':<9} {label:<40} {value:<10} -")

print("-" * 96)
print(f"{ok} confirmed in text, {absent} not stated\n")

print("SUPERSEDED-VALUE SWEEP (these must not appear)")
print("-" * 96)
stale = 0
for token, why in SUPERSEDED.items():
    hits = sorted({name for name, t in TEXT.items() if token in t})
    if hits:
        stale += 1
        print(f"STALE     {token:<14} {why:<46} {','.join(h.replace('.md','') for h in hits)}")
print(f"{'no stale values found' if not stale else f'{stale} stale value(s) present'}")
