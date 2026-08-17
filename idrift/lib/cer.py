import Levenshtein

def cer(reference: str, hypothesis: str) -> float:
    if len(reference) == 0:
        return 0.0
    return Levenshtein.distance(reference, hypothesis) / len(reference)
