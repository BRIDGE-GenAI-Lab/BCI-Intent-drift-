import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from idrift.data.build_exposure import build_exposure

AB = list("abcdefghijklmnopqrstuvwxyz ")


def test_grid_and_columns():
    corpus = pd.DataFrame([{"message_id": "m1", "corpus": "costello", "category": "phone",
                             "intended_text": "call my daughter"}])
    conf = {"F_03": np.full((27, 27), 1 / 27)}
    df = build_exposure(corpus, conf, cer_grid=[0.0, 0.2], n_seeds=3, alphabet=AB)
    assert len(df) == 1 * 2 * 3 * 1
    assert (df.loc[df.cer_target == 0.0, "actual_cer"] == 0.0).all()
    assert set(["noisy_text", "actual_cer", "source_subject"]).issubset(df.columns)


def test_determinism():
    # Reproducibility across independent process-level calls is a hard constraint:
    # the per-row seed must come from a STABLE hash, not Python's salted
    # built-in hash() (which varies run-to-run under PYTHONHASHSEED randomization).
    corpus = pd.DataFrame([
        {"message_id": "m1", "corpus": "costello", "category": "phone",
         "intended_text": "call my daughter"},
        {"message_id": "m2", "corpus": "costello", "category": "food",
         "intended_text": "i am hungry"},
    ])
    conf = {"F_03": np.full((27, 27), 1 / 27), "F_07": np.full((27, 27), 1 / 27)}

    df_a = build_exposure(corpus, conf, cer_grid=[0.0, 0.2, 0.4], n_seeds=4, alphabet=AB)
    df_b = build_exposure(corpus, conf, cer_grid=[0.0, 0.2, 0.4], n_seeds=4, alphabet=AB)

    assert df_a.equals(df_b)


def test_determinism_across_processes():
    """Verify stable output under different PYTHONHASHSEED values across processes.

    The in-process test above would pass even if the code regressed to salted hash().
    This test runs the exposure build in two separate processes with different
    PYTHONHASHSEED values and asserts identical noisy_text column hashes.
    """
    snippet = (
        "import numpy as np, pandas as pd, hashlib;"
        "from idrift.data.build_exposure import build_exposure;"
        "AB=list('abcdefghijklmnopqrstuvwxyz ');"
        "corpus=pd.DataFrame([{'message_id':'m1','corpus':'costello','category':'phone','intended_text':'call my daughter'}]);"
        "conf={'F_03':np.full((27,27),1/27)};"
        "df=build_exposure(corpus,conf,cer_grid=[0.2],n_seeds=3,alphabet=AB);"
        "print(hashlib.sha256('|'.join(df['noisy_text']).encode()).hexdigest())"
    )

    # Get repo root (three levels up from this test file)
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def run(seed):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        out = subprocess.run(
            [sys.executable, "-c", snippet],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
        )
        assert out.returncode == 0, f"Subprocess failed: {out.stderr}"
        return out.stdout.strip()

    # Run with two different PYTHONHASHSEED values; output must match
    hash_seed_0 = run(0)
    hash_seed_999 = run(999)
    assert hash_seed_0 == hash_seed_999, (
        f"Hashes differ across PYTHONHASHSEED: {hash_seed_0} vs {hash_seed_999}"
    )
