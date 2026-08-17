import re

_SYS = {
 "postedit":"You correct spelling errors in text typed by a brain-computer interface. Return only the corrected sentence.",
 "autocomplete":"You complete the current word/sentence from a partial brain-computer-interface typed string. Return only the completed sentence.",
 "expansion":"You expand a few brain-computer-interface-selected keywords into the full sentence the user intended. Return only that sentence.",
}

CONFIDENCE_SUFFIX = "\nThen on a new line write 'Confidence: N' where N is 0-100, your confidence the sentence matches the user's intent."

def build_prompt(depth, noisy_text, variant="v1"):
    if depth not in _SYS: raise ValueError(depth)
    user = f"Input: {noisy_text}\nOutput:" + CONFIDENCE_SUFFIX
    return _SYS[depth], user

def parse_confidence(text):
    m = re.search(r"[Cc]onfidence:\s*([0-9]{1,3})", text)
    return float(m.group(1)) if m else None
