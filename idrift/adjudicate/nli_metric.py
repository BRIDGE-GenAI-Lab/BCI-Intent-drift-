"""NLI + cosine automated ordinal labeler.

Maps (intended, output) pairs to ordinal labels: faithful|degraded|drift.
Uses bidirectional NLI to detect contradictions (drift), bidirectional entailment (faithful),
and cosine threshold to disambiguate between faithful and degraded.
"""

COS_HI = 0.75


def label_row(intended, output, cos_fn, nli_fn):
    """Label an (intended, output) pair with an ordinal label.

    Args:
        intended: The intended statement (string)
        output: The model output (string)
        cos_fn: Injected function (str, str) -> float for cosine similarity
        nli_fn: Injected function (str, str) -> str for NLI label
                Returns one of: "entail", "contradict", "neutral"

    Returns:
        One of: "faithful", "degraded", "drift"

    Rules:
        - If NLI detects contradiction in either direction: "drift"
        - If both directions entail each other: "faithful"
        - Otherwise use cosine threshold: >= COS_HI -> "faithful", else "degraded"
    """
    fwd = nli_fn(intended, output)
    bwd = nli_fn(output, intended)

    # Contradiction in either direction -> drift
    if "contradict" in (fwd, bwd):
        return "drift"

    # Bidirectional entailment -> faithful
    if fwd == "entail" and bwd == "entail":
        return "faithful"

    # Use cosine threshold to decide between faithful and degraded
    if cos_fn(intended, output) >= COS_HI:
        return "faithful"
    else:
        return "degraded"


def make_model_fns(nli_model="microsoft/deberta-large-mnli", emb_model="all-mpnet-base-v2"):
    """Create real model functions using sentence-transformers and transformers.

    Args:
        nli_model: HuggingFace model ID for NLI (DeBERTa-MNLI)
        emb_model: HuggingFace model ID for embeddings (all-mpnet-base-v2)

    Returns:
        Tuple of (cos_fn, nli_fn) ready to pass to label_row()

    Note:
        Heavy imports (sentence_transformers, transformers) are kept inside this function
        to allow importing nli_metric.py for testing without installing those packages.
    """
    import torch
    from sentence_transformers import SentenceTransformer, util
    from transformers import pipeline

    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    emb = SentenceTransformer(emb_model, device=dev)
    nli = pipeline("text-classification", model=nli_model, top_k=1, device=dev)

    def cos_fn(a, b):
        """Compute cosine similarity between two texts."""
        va, vb = emb.encode([a, b])
        return float(util.cos_sim(va, vb))

    _CANON = {"entailment": "entail", "contradiction": "contradict", "neutral": "neutral"}

    def nli_fn(p, h):
        """Classify NLI relation from premise p to hypothesis h (proper text-pair input)."""
        res = nli({"text": p, "text_pair": h})
        top = res[0]
        if isinstance(top, list):          # some versions nest per-input results
            top = top[0]
        return _CANON.get(top["label"].lower(), "neutral")

    return cos_fn, nli_fn
