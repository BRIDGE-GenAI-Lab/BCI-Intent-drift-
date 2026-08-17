import numpy as np
from idrift.data.noise_model import inject, inject_with_metadata
from idrift.lib.cer import cer

AB = list("abcdefghijklmnopqrstuvwxyz ")
CONF = np.full((27, 27), 1 / 27)   # uniform confusion for the test


def test_zero_cer_is_identity():
    assert inject("call my daughter", 0.0, CONF, AB, seed=1) == "call my daughter"


def test_determinism():
    a = inject("call my daughter", 0.3, CONF, AB, seed=42)
    b = inject("call my daughter", 0.3, CONF, AB, seed=42)
    assert a == b


def test_hits_target_cer_on_average():
    txt = "i would like to watch the evening news"
    errs = [cer(txt, inject(txt, 0.2, CONF, AB, seed=s)) for s in range(200)]
    assert abs(np.mean(errs) - 0.2) < 0.05          # within 5 CER points on average


def test_negative_cer_target_is_also_identity():
    # brief's guard is `cer_target <= 0`, not `== 0`; lock in the `<` side too
    assert inject("call my daughter", -0.1, CONF, AB, seed=1) == "call my daughter"


def test_substitution_never_leaves_char_unchanged():
    # ins_del_rate=0 + cer_target=1.0 forces every in-alphabet char through the
    # substitution branch on every draw; confirm the confusion-row draw never
    # reproduces the original character (row[idx[ch]] must stay zeroed out).
    txt = "abcdefghijklmnopqrstuvwxyz"
    for seed in range(20):
        out = inject(txt, 1.0, CONF, AB, seed=seed, ins_del_rate=0.0)
        assert len(out) == len(txt)
        assert all(o != t for o, t in zip(out, txt))


def test_empty_confusion_row_falls_back_to_uniform_substitution():
    # A sparse/unseen char (all-diagonal confusion row) must STILL be substituted
    # to a different char so the manipulated CER is realized -- not passed through.
    ab = list("abc")
    ident = np.eye(3)                      # every row all-diagonal -> empty after zeroing
    # force the substitution branch every step (cer_target=1, no indels)
    out = inject("abcabcabc", 1.0, ident, ab, seed=3, ins_del_rate=0.0)
    assert len(out) == len("abcabcabc")
    assert all(o != t for o, t in zip(out, "abcabcabc"))   # fallback still changes each char


def test_empty_row_fallback_hits_target_cer():
    # With an all-diagonal (useless) confusion matrix, realized CER should still
    # track the target thanks to the uniform fallback (previously it collapsed to ~0).
    ab = list("abcdefghijklmnopqrstuvwxyz ")
    ident = np.eye(27)
    txt = "the quick brown fox jumps over"
    errs = [cer(txt, inject(txt, 0.3, ident, ab, seed=s)) for s in range(200)]
    assert abs(np.mean(errs) - 0.3) < 0.05


def test_inject_with_metadata_matches_plain_inject_output():
    # rev Task 1.2: `inject_with_metadata` is a sibling of `inject`, not a
    # replacement -- it must reproduce the exact same noisy_text for the
    # same arguments (same RNG draw order), across zero-corruption,
    # ordinary, and saturated cer_target, and across a text containing a
    # char with an empty confusion row (the fallback path).
    txt = "call my daughter and tell her i am hungry"
    for cer_target in (0.0, 0.2, 0.5, 1.0):
        for seed in range(10):
            plain = inject(txt, cer_target, CONF, AB, seed=seed)
            meta = inject_with_metadata(txt, cer_target, CONF, AB, seed=seed)
            assert meta["noisy_text"] == plain
            assert meta["sub_count"] + meta["ins_count"] + meta["del_count"] == len(
                meta["error_positions"]
            )
            assert all(0 <= p < len(txt) for p in meta["error_positions"])


def test_inject_with_metadata_empty_row_fallback_matches_plain_inject():
    ab = list("abc")
    ident = np.eye(3)
    for seed in range(20):
        plain = inject("abcabcabc", 1.0, ident, ab, seed=seed, ins_del_rate=0.0)
        meta = inject_with_metadata("abcabcabc", 1.0, ident, ab, seed=seed, ins_del_rate=0.0)
        assert meta["noisy_text"] == plain
