from idrift.adjudicate.nli_metric import label_row


def cos(a, b):
    return 0.95 if a.replace("not ", "") == b.replace("not ", "") else 0.2


def nli(p, h):
    # negation flip is high-cosine but a contradiction
    return "contradict" if (("not" in p) != ("not" in h)) else "entail"


def test_negation_flip_is_drift_not_faithful():
    assert label_row("I do not want suction", "I do want suction", cos, nli) == "drift"


def test_true_paraphrase_is_faithful():
    def nli2(p, h):
        return "entail"

    assert label_row("call my doctor", "Please call my doctor.", lambda a, b: 0.9, nli2) == "faithful"


def test_garbled_low_cosine_is_degraded():
    def nli3(p, h):
        return "neutral"

    assert label_row("call my doctor", "xq zzt", lambda a, b: 0.1, nli3) == "degraded"
