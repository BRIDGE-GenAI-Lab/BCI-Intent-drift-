"""Background: label the raw corrupted decodes (noisy_text) for the benefit-harm matrix (B1)."""
import time, sys
t0=time.time()
from idrift.analysis.benefit_harm import label_raw_decode
out = label_raw_decode(
    "output/intermediate/attempts_v2_labeled.parquet",
    "output/intermediate/raw_decode_labels_v2.parquet",
    device="mps", batch_size=64,
    checkpoint_path="output/intermediate/raw_decode_signals_ckpt.parquet",
)
print(f"DONE raw-decode labels: {len(out)} rows in {time.time()-t0:.0f}s", flush=True)
import pandas as pd
print("raw_label counts:\n", pd.read_parquet('output/intermediate/raw_decode_labels_v2.parquet')['raw_label'].value_counts().to_string(), flush=True)
