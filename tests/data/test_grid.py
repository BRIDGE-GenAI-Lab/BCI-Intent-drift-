from idrift.data.grid import CHARMAP, GRID_ALPHABET, code_to_char


def test_charmap_shape():
    assert len(CHARMAP) == 36
    assert CHARMAP[-1] == " "
    assert GRID_ALPHABET == list(CHARMAP)


def test_code_to_char_bounds():
    assert code_to_char(1) == "A"
    assert code_to_char(26) == "Z"
    assert code_to_char(27) == "1"
    assert code_to_char(35) == "9"
    assert code_to_char(36) == " "


def test_code_to_char_out_of_range_raises():
    for bad in (0, -1, 37, 100):
        try:
            code_to_char(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for code {bad}")
