"""Batched full-cohort labeling runner (revision Task 5.0, productionizing
Phase 3 Task 3.3 Step 5).

Calling `ensemble.ensemble_label` once per (intended, output) pair over the
full 533,400-row `output/intermediate/o2_v2_outputs.parquet` is infeasible:
each call triggers ~7 unbatched model forward passes (2 NLI models x 2
directions + 2 embedding models + 1 fluency model), an estimated ~100
GPU-hours at full scale. This module splits that per-item pipeline into two
layers so the expensive part runs ONCE, batched and de-duplicated, and the
cheap part (majority-vote decision logic) can be re-run for free:

- `compute_signals`: the ONLY layer that touches a model. Runs each of the
  six independent model passes (fluency/gpt2, cosine/mpnet, cosine/minilm,
  NLI/deberta, NLI/roberta, critical-rules/spaCy) ONCE over the
  DE-DUPLICATED set of unique texts or text pairs that pass needs, batched
  where the model supports it (NLI, embeddings), and checkpoints
  incrementally so a multi-hour run is resumable after a crash.
- `derive_labels`: PURE Python, no models. Rebuilds the exact 4
  `default_evaluators()` decisions from the precomputed signals via
  lookup-closures wired through the UNMODIFIED `taxonomy.label` +
  `ensemble.ensemble_label`, so tau-recalibration and tie-direction
  sensitivity (Phase 5) are free re-derivations, never re-inference.

`run` orchestrates both layers end to end: load -> compute_signals
(checkpointed) -> write a signals-only checkpoint parquet -> derive_labels
-> concat original + signal + label columns -> write the labeled parquet.

Reuse, not reimplementation
----------------------------
Every DECISION (what counts as a meaning difference, how ties resolve, what
a critical rule flags) still runs through the exact same functions the
per-item path uses: `idrift.adjudicate.taxonomy.label` (with its new
`crit=` pass-through, see that module), `idrift.adjudicate.ensemble.
ensemble_label`/`NamedEvaluator`, and `idrift.adjudicate.critical_rules.
detect`. This module never reimplements any of that logic -- it only
reimplements the INFERENCE calls (`nli_metric.make_model_fns`'s single-pair
pipeline call, `fluency.make_fluency_fn`'s single-text forward pass) as
batched, de-duplicated equivalents, and the load-bearing equivalence test
(`tests/adjudicate/test_label_runner.py::test_7_...`) proves the batched
inference reproduces the exact same per-row values the single-pair path
would have produced.

Checkpoint layout
-----------------
`compute_signals(df, checkpoint_path=...)` treats `checkpoint_path` as a
single parquet file holding whichever of the six signal BLOCKS have been
computed so far (fluency, cos_mpnet, cos_minilm, nli_deberta, nli_roberta,
crit). After each block finishes, the accumulated signals frame (all
columns computed so far, for every row) is written to `checkpoint_path`,
overwriting the previous checkpoint. A re-invocation with the same
`checkpoint_path` reads it back, skips any block whose columns are already
present (and the checkpoint has the same row count as `df`), and only runs
the remaining blocks. A crash mid-block loses at most that one block's
progress, never an already-completed block.

Deterministic; American spelling; no em dashes (double hyphens for
asides, per house style).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from idrift.adjudicate.ensemble import NamedEvaluator, ensemble_label
from idrift.adjudicate.taxonomy import label as taxonomy_label

# ---------------------------------------------------------------------------
# Shared column/name conventions -- MUST mirror `ensemble.default_evaluators()`
# exactly (same NLI/embedding model lists, same iteration order, same name
# format) so `derive_labels` reconstructs identical evaluators.
# ---------------------------------------------------------------------------

_NLI_MODELS = ["microsoft/deberta-large-mnli", "roberta-large-mnli"]
_EMB_MODELS = ["all-mpnet-base-v2", "all-MiniLM-L6-v2"]

_NLI_MODEL_COLS = {
    "microsoft/deberta-large-mnli": ("nli_deberta_fwd", "nli_deberta_bwd"),
    "roberta-large-mnli": ("nli_roberta_fwd", "nli_roberta_bwd"),
}
_EMB_MODEL_COLS = {
    "all-mpnet-base-v2": "cos_mpnet",
    "all-MiniLM-L6-v2": "cos_minilm",
}

# Same nested-loop order `default_evaluators()` uses: NLI outer, embedding inner.
_EVALUATOR_ORDER = [(nli_model, emb_model) for nli_model in _NLI_MODELS for emb_model in _EMB_MODELS]

_CRIT_KEY_TO_COL = {
    "negation_flip": "crit_negation_flip",
    "numeral_change": "crit_numeral_change",
    "recipient_change": "crit_recipient_change",
    "urgency_change": "crit_urgency_change",
    "actionable_omission": "crit_actionable_omission",
}
_CRIT_COLUMNS = list(_CRIT_KEY_TO_COL.values())

_SIGNAL_COLUMNS = (
    ["nli_deberta_fwd", "nli_deberta_bwd", "nli_roberta_fwd", "nli_roberta_bwd", "cos_mpnet", "cos_minilm", "fluency_raw"]
    + _CRIT_COLUMNS
)

# Reproduces `nli_metric.make_model_fns`'s function-local `_CANON` dict
# verbatim (it is not importable -- it is a local variable inside that
# factory function, not a module-level name) so the batched NLI path below
# canonicalizes labels identically to the per-item `nli_fn` it replaces.
_CANON = {"entailment": "entail", "contradiction": "contradict", "neutral": "neutral"}


def _evaluator_name(nli_model: str, emb_model: str) -> str:
    """Same name format `default_evaluators()` uses for its `NamedEvaluator`s."""
    return f"{nli_model.rsplit('/', 1)[-1]}+{emb_model}"


_VOTE_COLUMNS = [f"vote__{_evaluator_name(n, e)}" for n, e in _EVALUATOR_ORDER]


def _clean_venv_sidecars() -> None:
    """Delete exFAT/network-volume macOS AppleDouble `._*` sidecar files
    under `.venv` before any model load. These sidecars crash
    transformers/sentence-transformers imports on this workspace's volume
    (see the Task 1.3 report); must run before EVERY model-loading entry
    point in this module, not just once at import time."""
    venv = Path(".venv")
    if not venv.exists():
        return
    for path in venv.rglob("._*"):
        try:
            path.unlink()
        except OSError:
            pass


def _resolve_device(device: str) -> str:
    """Resolve a requested device string against actual availability,
    falling back to cpu if the requested accelerator is not present (so a
    test environment without mps/cuda does not hard-crash)."""
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        return "mps"
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    if device in ("mps", "cuda"):
        return "cpu"
    return device


# ---------------------------------------------------------------------------
# Step B: compute_signals -- the only layer that touches a model.
# ---------------------------------------------------------------------------


def _compute_fluency_block(df: pd.DataFrame) -> pd.DataFrame:
    """`fluency_raw` = `fluency.make_fluency_fn("gpt2")` applied to
    `output_message`, de-duplicated over unique output texts (the fluency
    function itself is reused verbatim, unbatched, per the brief -- the win
    here is skipping repeated forward passes on a text seen more than once,
    not tensor-batching gpt2's variable-length forward pass)."""
    from idrift.adjudicate.fluency import make_fluency_fn

    fluency_fn = make_fluency_fn("gpt2")
    outputs = df["output_message"].tolist()
    cache: dict[str, float] = {}
    for text in outputs:
        if text not in cache:
            cache[text] = fluency_fn(text)
    return pd.DataFrame({"fluency_raw": [cache[text] for text in outputs]}, index=df.index)


def _compute_cos_block(df: pd.DataFrame, emb_model: str, device: str, batch_size: int, col_name: str) -> pd.DataFrame:
    """Batched, de-duplicated cosine similarity for one embedding
    checkpoint: embed the UNION of unique intended/output texts once, then
    look up each row's pair and compute cosine via L2-normalize + dot
    product -- mathematically identical to `nli_metric.make_model_fns`'s
    per-pair `util.cos_sim(va, vb)` (see the module docstring's `_CANON`
    note for why the equivalence is proved by the test, not asserted here).
    """
    import torch
    from sentence_transformers import SentenceTransformer

    dev = _resolve_device(device)
    model = SentenceTransformer(emb_model, device=dev)
    model.eval()

    intended = df["intended_text"].tolist()
    output = df["output_message"].tolist()
    unique_texts = sorted(set(intended) | set(output))
    text_to_idx = {text: idx for idx, text in enumerate(unique_texts)}

    with torch.no_grad():
        embeddings = model.encode(
            unique_texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False
        )
        idx_i = torch.tensor([text_to_idx[t] for t in intended], device=embeddings.device)
        idx_o = torch.tensor([text_to_idx[t] for t in output], device=embeddings.device)
        emb_i = torch.nn.functional.normalize(embeddings[idx_i], dim=1)
        emb_o = torch.nn.functional.normalize(embeddings[idx_o], dim=1)
        cos = (emb_i * emb_o).sum(dim=1)

    return pd.DataFrame({col_name: cos.cpu().numpy().astype(float)}, index=df.index)


def _classify_pairs(pairs: list[tuple[str, str]], tokenizer, model, device: str, batch_size: int) -> list[str]:
    """Batched, de-duplicated NLI classification over ordered (premise,
    hypothesis) pairs, reproducing `nli_metric.make_model_fns`'s
    `nli_fn(p, h)` exactly: argmax over the model's own logits, mapped
    through `model.config.id2label` (the same mapping the HF
    text-classification pipeline reads internally) and canonicalized via
    `_CANON` (identical to `nli_metric`'s function-local dict)."""
    import torch

    unique_pairs = sorted(set(pairs))
    pair_to_label: dict[tuple[str, str], str] = {}
    id2label = model.config.id2label

    for start in range(0, len(unique_pairs), batch_size):
        batch = unique_pairs[start : start + batch_size]
        premises = [p for p, _ in batch]
        hypotheses = [h for _, h in batch]
        encoded = tokenizer(premises, hypotheses, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        pred_ids = logits.argmax(dim=-1).tolist()
        for pair, pred_id in zip(batch, pred_ids):
            pair_to_label[pair] = _CANON.get(id2label[pred_id].lower(), "neutral")

    return [pair_to_label[pair] for pair in pairs]


def _compute_nli_block(
    df: pd.DataFrame, nli_model: str, device: str, batch_size: int, fwd_col: str, bwd_col: str
) -> pd.DataFrame:
    """Batched, de-duplicated bidirectional NLI for one checkpoint:
    `*_fwd = nli(intended, output)`, `*_bwd = nli(output, intended)`, each
    computed once per unique ordered pair via `_classify_pairs`."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model).to(dev)
    model.eval()

    intended = df["intended_text"].tolist()
    output = df["output_message"].tolist()
    fwd_pairs = list(zip(intended, output))
    bwd_pairs = list(zip(output, intended))

    fwd_labels = _classify_pairs(fwd_pairs, tokenizer, model, dev, batch_size)
    bwd_labels = _classify_pairs(bwd_pairs, tokenizer, model, dev, batch_size)

    return pd.DataFrame({fwd_col: fwd_labels, bwd_col: bwd_labels}, index=df.index)


def _compute_crit_block(df: pd.DataFrame) -> pd.DataFrame:
    """`crit_*` flags = `critical_rules.detect(intended, output)`, called
    ONCE per unique (intended, output) pair (spaCy parsing is CPU-bound and
    not tensor-batchable through the reused `detect()` function itself, so
    de-duplication is the whole win here, not GPU batching)."""
    from idrift.adjudicate.critical_rules import detect

    pairs = list(zip(df["intended_text"].tolist(), df["output_message"].tolist()))
    cache: dict[tuple[str, str], dict] = {}
    for pair in pairs:
        if pair not in cache:
            cache[pair] = detect(*pair)
    rows = [cache[pair] for pair in pairs]
    flags = pd.DataFrame(rows, index=df.index)
    return flags.rename(columns={key: col for key, col in _CRIT_KEY_TO_COL.items()})[_CRIT_COLUMNS]


# (block_name, columns, builder) -- order chosen arbitrarily; each block is
# independent and checkpointed after it completes.
_BLOCK_SPECS = [
    ("fluency", ["fluency_raw"], lambda df, device, batch_size: _compute_fluency_block(df)),
    ("cos_mpnet", ["cos_mpnet"], lambda df, device, batch_size: _compute_cos_block(df, "all-mpnet-base-v2", device, batch_size, "cos_mpnet")),
    ("cos_minilm", ["cos_minilm"], lambda df, device, batch_size: _compute_cos_block(df, "all-MiniLM-L6-v2", device, batch_size, "cos_minilm")),
    ("nli_deberta", ["nli_deberta_fwd", "nli_deberta_bwd"], lambda df, device, batch_size: _compute_nli_block(df, "microsoft/deberta-large-mnli", device, batch_size, "nli_deberta_fwd", "nli_deberta_bwd")),
    ("nli_roberta", ["nli_roberta_fwd", "nli_roberta_bwd"], lambda df, device, batch_size: _compute_nli_block(df, "roberta-large-mnli", device, batch_size, "nli_roberta_fwd", "nli_roberta_bwd")),
    ("crit", _CRIT_COLUMNS, lambda df, device, batch_size: _compute_crit_block(df)),
]


def _checkpoint(signals: pd.DataFrame, checkpoint_path: str | None) -> None:
    if not checkpoint_path:
        return
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(path)


def _free_accelerator_cache(device: str) -> None:
    """Release the accelerator allocator's cached memory after each signal
    block so peak resident memory stays near a single model rather than the
    accumulated sum of every block's model (each block loads its own model,
    which the allocator otherwise keeps cached across blocks). Purely a
    memory-hygiene step -- freeing cache never changes the computed signals,
    so it is covered by the existing batched-vs-per-item equivalence test."""
    import gc

    gc.collect()
    try:
        import torch

        dev = _resolve_device(device)
        if dev == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif dev == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def compute_signals(
    df: pd.DataFrame, *, device: str = "mps", batch_size: int = 64, checkpoint_path: str | None = None
) -> pd.DataFrame:
    """Run the real models over `df`, batched and de-duplicated, returning
    one row of raw signal columns per input row (same order/index as `df`):
    `nli_deberta_fwd/bwd`, `nli_roberta_fwd/bwd`, `cos_mpnet`, `cos_minilm`,
    `fluency_raw`, and the five `crit_*` bool columns.

    Resumable: if `checkpoint_path` already holds a signals frame with the
    same row count as `df`, any block whose columns are already present is
    skipped; only the remaining blocks run. After each block completes, the
    accumulated frame is written back to `checkpoint_path`, so a crash loses
    at most one in-progress block. See the module docstring for the full
    checkpoint layout.

    Deterministic: `torch.no_grad()`, eval mode, no sampling anywhere;
    the same `df` always produces the same signals.
    """
    _clean_venv_sidecars()

    signals = pd.DataFrame(index=df.index)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        cached = pd.read_parquet(checkpoint_path)
        if len(cached) == len(df):
            cached.index = df.index
            signals = cached

    for _name, columns, builder in _BLOCK_SPECS:
        if all(col in signals.columns for col in columns):
            continue
        block = builder(df, device, batch_size)
        for col in columns:
            signals[col] = block[col].values
        _checkpoint(signals, checkpoint_path)
        _free_accelerator_cache(device)

    return signals[_SIGNAL_COLUMNS]


# ---------------------------------------------------------------------------
# Step C: derive_labels -- pure, no models.
# ---------------------------------------------------------------------------


def derive_labels(
    signals_df: pd.DataFrame, *, tau: float = 0.5, tie_break: str = "primary", use_critical_rules: bool = True
) -> pd.DataFrame:
    """Reconstruct the 4 `default_evaluators()` labels for every row from
    precomputed signals, calling the UNMODIFIED `taxonomy.label` (with its
    `crit=` pass-through) and `ensemble.ensemble_label` -- NO model calls,
    NO spaCy. `signals_df` must carry `intended_text`, `output_message`,
    and every column `compute_signals` produces (see that function).

    Returns one row per input row (same order/index as `signals_df`) with
    columns: `label`, one `vote__<evaluator_name>` per of the 4 default
    evaluators, `agreement`, `tie`, `tied_labels` (stable "|"-joined sorted
    string), and `fluency_raw` carried through.

    `tau` controls the fluency gate WITHOUT touching `taxonomy.TAU`: each
    row's closure returns `1.0 if fluency_raw >= tau else 0.0`, so
    `taxonomy.label`'s own fixed `>= TAU (0.5)` check on that 0/1 value
    reduces to `fluency_raw >= tau`. This is how tau-recalibration becomes a
    free re-derivation instead of a re-run of the fluency model.

    `use_critical_rules` (default `True`, exactly the prior behavior): when
    `True`, each row's real computed `crit_*` flags are passed to
    `taxonomy.label` as before. When `False`, every row is instead given an
    all-`False` crit dict, regardless of what the five `crit_*` columns
    actually hold (they need not even be present in `signals_df` in that
    case) -- so the returned label is resolved purely from the meaning
    channel `taxonomy.label` otherwise uses: bidirectional NLI contradiction
    or the cosine threshold, gated by fluency (see `taxonomy._meaning_
    differs`). This answers a reviewer's circularity objection -- the five
    `crit_*` rule detectors (negation flip, numeral change, recipient
    change, urgency change, actionable omission) are also the five
    characteristics used to define the study's message-critical (CRIT)
    probe set, so a CRIT-vs-CTRL drift excess measured with those same rules
    baked into the outcome is confounded with the exposure definition. Set
    `use_critical_rules=False` to re-derive a rule-free outcome and re-run
    the matched comparison on it (see `scratchpad/harden/a2_critfree.py`).
    """
    records = []
    for row in signals_df.itertuples(index=False):
        row_d = row._asdict()
        intended_text = row_d["intended_text"]
        output_message = row_d["output_message"]
        fluency_raw = row_d["fluency_raw"]
        if use_critical_rules:
            crit = {key: bool(row_d[col]) for key, col in _CRIT_KEY_TO_COL.items()}
        else:
            crit = {key: False for key in _CRIT_KEY_TO_COL}

        def fluency_fn(_text: str, _fluency_raw: float = fluency_raw, _tau: float = tau) -> float:
            return 1.0 if _fluency_raw >= _tau else 0.0

        evaluators = []
        for nli_model, emb_model in _EVALUATOR_ORDER:
            fwd_col, bwd_col = _NLI_MODEL_COLS[nli_model]
            cos_col = _EMB_MODEL_COLS[emb_model]
            fwd_val = row_d[fwd_col]
            bwd_val = row_d[bwd_col]
            cos_val = row_d[cos_col]

            def cos_fn(_a: str, _b: str, _cos_val: float = cos_val) -> float:
                return _cos_val

            def nli_fn(p: str, _h: str, _intended: str = intended_text, _fwd: str = fwd_val, _bwd: str = bwd_val) -> str:
                # taxonomy.label calls nli_fn(intended, output) then
                # nli_fn(output, intended); keying on p == intended_text
                # selects the matching precomputed direction (when
                # intended == output both directions are equal so either
                # is correct).
                return _fwd if p == _intended else _bwd

            def evaluator_fn(
                intended: str,
                output: str,
                _cos_fn=cos_fn,
                _nli_fn=nli_fn,
                _fluency_fn=fluency_fn,
                _crit=crit,
            ) -> str:
                return taxonomy_label(intended, output, _cos_fn, _nli_fn, _fluency_fn, crit=_crit)["label"]

            name = _evaluator_name(nli_model, emb_model)
            evaluators.append(NamedEvaluator(name, evaluator_fn))

        result = ensemble_label(intended_text, output_message, evaluators, tie_break=tie_break)

        record = {"label": result["label"]}
        for nli_model, emb_model in _EVALUATOR_ORDER:
            name = _evaluator_name(nli_model, emb_model)
            record[f"vote__{name}"] = result["votes"][name]
        record["agreement"] = result["agreement"]
        record["tie"] = result["tie"]
        record["tied_labels"] = "|".join(sorted(result["tied_labels"]))
        record["fluency_raw"] = fluency_raw
        records.append(record)

    columns = ["label"] + _VOTE_COLUMNS + ["agreement", "tie", "tied_labels", "fluency_raw"]
    return pd.DataFrame(records, columns=columns, index=signals_df.index)


# ---------------------------------------------------------------------------
# Step D: run -- orchestrate compute_signals -> derive_labels end to end.
# ---------------------------------------------------------------------------


def _print_summary(result: pd.DataFrame) -> None:
    n = len(result)
    dist = result["label"].value_counts(normalize=True).to_dict()
    mean_agreement = float(result["agreement"].mean())
    tie_rate = float(result["tie"].mean())
    print(f"label_runner.run: {n} rows labeled")
    print(f"label distribution: {dist}")
    print(f"mean agreement: {mean_agreement:.4f}")
    print(f"tie rate: {tie_rate:.4f}")


def run(
    input_parquet: str,
    output_parquet: str,
    *,
    device: str = "mps",
    batch_size: int = 64,
    checkpoint_dir: str | None = None,
    tau: float = 0.5,
    tie_break: str = "primary",
) -> pd.DataFrame:
    """Orchestrate the full pipeline: load `input_parquet` ->
    `compute_signals` (checkpointed under `checkpoint_dir`) -> write a
    signals-only checkpoint parquet (`attempts_v2_signals.parquet`, next to
    `input_parquet`) -> `derive_labels` -> concat original + signal + label
    columns -> write `output_parquet`. Prints a small summary (row count,
    label distribution, mean agreement, tie rate).

    Does NOT run the full 533k-row cohort by itself -- this is meant to be
    invoked directly on the full parquet by the controller's background job;
    call it on a small slice to verify it end to end.
    """
    _clean_venv_sidecars()

    df = pd.read_parquet(input_parquet)

    checkpoint_path = None
    if checkpoint_dir is not None:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = str(Path(checkpoint_dir) / "attempts_v2_signals_checkpoint.parquet")

    signals = compute_signals(df, device=device, batch_size=batch_size, checkpoint_path=checkpoint_path)

    signals_out_path = Path(input_parquet).parent / "attempts_v2_signals.parquet"
    signals.to_parquet(signals_out_path)

    merged_for_labels = pd.concat(
        [df[["intended_text", "output_message"]].reset_index(drop=True), signals.reset_index(drop=True)], axis=1
    )
    merged_for_labels.index = df.index

    labels = derive_labels(merged_for_labels, tau=tau, tie_break=tie_break)
    labels_for_concat = labels.drop(columns=["fluency_raw"])  # already present in signals

    result = pd.concat(
        [df.reset_index(drop=True), signals.reset_index(drop=True), labels_for_concat.reset_index(drop=True)], axis=1
    )
    result.index = df.index
    result.to_parquet(output_parquet)

    _print_summary(result)
    return result
