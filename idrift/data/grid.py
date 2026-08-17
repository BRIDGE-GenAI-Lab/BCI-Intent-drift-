"""The canonical bigP3BCI 6x6 speller grid (Farwell-Donchin/BCI2000 default).

Studies F/L/N reference a 6x6 speller (36 cells), row-major, 1-based grid
codes 1..36 (verified against observed codes 1..32 in the source EDFs):

    row1  A B C D E F
    row2  G H I J K L
    row3  M N O P Q R
    row4  S T U V W X
    row5  Y Z 1 2 3 4
    row6  5 6 7 8 9 _   (the 36th cell, "_", is SPACE)

`CHARMAP[i-1]` gives the character for 1-based grid code `i`.
"""

CHARMAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789" + " "
GRID_ALPHABET = list(CHARMAP)


def code_to_char(code: int) -> str:
    """Map a 1-based 6x6 grid stimulus code (1..36) to its speller character."""
    if not (1 <= code <= 36):
        raise ValueError(f"grid code out of range 1..36: {code!r}")
    return CHARMAP[code - 1]
