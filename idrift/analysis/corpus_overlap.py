"""Distinct-message overlap across AUTH/CRIT/CTRL (reviewer major #8).

Why this module exists
-----------------------
A reviewer asked for an explicit accounting of message-identity overlap
across the three corpora (AUTH, CRIT, CTRL) that feed the pooled primary
analyses -- a flow diagram or table, in the reviewer's words. This module
computes that accounting directly from `message_id` set membership on the
current 20-model panel.

The actual structure (verify, don't assume)
---------------------------------------------
It is tempting to assume CRIT (the 131 hand-authored critical-substitution
probes) is a subset of AUTH (the 2,168 authentic BCI messages), with CTRL
(the 131 matched controls) disjoint from both. The real structure is the
OPPOSITE: CTRL's 131 `costello_*` message ids are a full SUBSET of AUTH's
2,168 ids (CTRL items are matched controls drawn from the same authentic
message pool), while CRIT's 131 `probe_*` ids never appear in AUTH or CTRL
(the hand-authored critical probes are a wholly separate item set). This
matches the already-documented, reviewer-accepted finding for the original
7-model panel (`scratchpad/harden/a1_dedup.py`'s module docstring) and
reproduces identically on the current 20-model panel
(`attempts_v3plus_labeled.parquet`). So `total_unique_messages` = AUTH +
CRIT (2,168 + 131 = 2,299), not AUTH + CTRL -- CTRL contributes zero new
message identities.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def run(parquet_path, out_path: str) -> dict:
    """Compute distinct-message overlap across AUTH/CRIT/CTRL and write it
    to `out_path` as JSON.

    Args:
        parquet_path: path to the labeled attempts parquet (columns
            `message_id`, `corpus`), or an already-loaded DataFrame with
            those columns.
        out_path: path to write the JSON digest to.

    Returns:
        dict: distinct-message counts per corpus, the CRIT-subset-of-AUTH
        check, CTRL's overlap with AUTH and CRIT, and the total distinct
        message count across all three corpora.
    """
    df = (
        parquet_path
        if isinstance(parquet_path, pd.DataFrame)
        else pd.read_parquet(parquet_path, columns=["message_id", "corpus"])
    )
    auth_ids = set(df.loc[df["corpus"] == "AUTH", "message_id"])
    crit_ids = set(df.loc[df["corpus"] == "CRIT", "message_id"])
    ctrl_ids = set(df.loc[df["corpus"] == "CTRL", "message_id"])
    digest = {
        "auth_n_unique": len(auth_ids),
        "crit_n_unique": len(crit_ids),
        "ctrl_n_unique": len(ctrl_ids),
        "crit_subset_of_auth": crit_ids <= auth_ids,
        "ctrl_subset_of_auth": ctrl_ids <= auth_ids,
        "crit_overlap_with_auth_n": len(crit_ids & auth_ids),
        "ctrl_overlap_with_auth_n": len(ctrl_ids & auth_ids),
        "ctrl_overlap_with_crit_n": len(ctrl_ids & crit_ids),
        "total_unique_messages": len(auth_ids | crit_ids | ctrl_ids),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(digest, indent=2))
    return digest


if __name__ == "__main__":
    print(run("output/intermediate/attempts_v3plus_labeled.parquet", "output/corpus_overlap_digest.json"))
