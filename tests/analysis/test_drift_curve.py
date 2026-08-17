import numpy as np, pandas as pd
from idrift.analysis.drift_curve import drift_rate_by_cer, critical_threshold
def test_drift_rate_monotone_synthetic():
    rows = []
    for cer in [0.0,0.1,0.2,0.3,0.4]:
        for i in range(100):
            rows.append({"cer_target":cer,"final_label":"drift" if np.random.default_rng(i+int(cer*100)).random()<cer else "faithful",
                         "category":"message_critical"})
    df = pd.DataFrame(rows)
    r = drift_rate_by_cer(df)
    assert r[0.0] <= r[0.4]
def test_threshold_returns_grid_value():
    df = pd.DataFrame([{"cer_target":c,"final_label":"drift" if c>=0.2 else "faithful","category":"message_critical"} for c in [0.0,0.1,0.2,0.3]])
    assert critical_threshold(df, tol=0.5) == 0.2
