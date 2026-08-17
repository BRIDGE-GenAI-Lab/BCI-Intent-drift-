import numpy as np
from idrift.data.confusion_matrix import build_confusion, overall_cer

def test_perfect_decoder_is_identity():
    pairs = [("a","a"),("b","b"),("a","a")]
    M = build_confusion(pairs, ["a","b"])
    assert np.allclose(M, np.eye(2))
    assert overall_cer(pairs) == 0.0

def test_row_normalized_and_cer():
    pairs = [("a","a"),("a","b")]           # 'a' decoded a,b once each
    M = build_confusion(pairs, ["a","b"])
    assert np.allclose(M[0], [0.5,0.5])
    assert overall_cer(pairs) == 0.5

def test_symbol_type_agnostic_ints():
    # confusion_matrix must work in integer grid-code space too (bigP3BCI adapter emits ints)
    pairs = [(1, 1), (2, 3), (1, 1)]
    M = build_confusion(pairs, [1, 2, 3])
    assert np.allclose(M[0], [1.0, 0.0, 0.0])
    assert np.allclose(M[1], [0.0, 0.0, 1.0])
    assert overall_cer(pairs) == 1/3
