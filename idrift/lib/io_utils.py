import hashlib
import json
import os
from pathlib import Path
import pandas as pd

def _dir() -> Path:
    d = Path(os.environ.get("IDRIFT_INTERMEDIATE", "output/intermediate"))
    d.mkdir(parents=True, exist_ok=True)
    return d

def save_checkpoint(df: pd.DataFrame, name: str) -> Path:
    p = _dir() / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p

def load_checkpoint(name: str) -> pd.DataFrame:
    return pd.read_parquet(_dir() / f"{name}.parquet")

def sha256_file(path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()

def log_provenance(entries: dict, name: str = "provenance.json"):
    p = _dir() / name
    existing = json.loads(p.read_text()) if p.exists() else {}
    existing.update(entries)
    p.write_text(json.dumps(existing, indent=2))
    return p
