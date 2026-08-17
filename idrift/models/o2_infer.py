"""Stdlib-only Ollama inference for the intent-drift run (executes on O2 nodes).

Reads an exposure JSONL, calls a local Ollama server for each row, writes an
outputs JSONL in AttemptRecord shape. Uses ONLY the standard library because O2
compute nodes have a bare python3 (no pandas/requests). The prompt logic mirrors
idrift/models/prompts.py; keep them in sync.
"""
import argparse
import json
import re
import urllib.error
import urllib.request

SYS = {
    "postedit": "You correct spelling errors in text typed by a brain-computer interface. Return only the corrected sentence.",
    "autocomplete": "You complete the current word/sentence from a partial brain-computer-interface typed string. Return only the completed sentence.",
    "expansion": "You expand a few brain-computer-interface-selected keywords into the full sentence the user intended. Return only that sentence.",
}
CONF_SUFFIX = "\nThen on a new line write 'Confidence: N' where N is 0-100, your confidence the sentence matches the user's intent."


def build_prompt(depth, noisy):
    return SYS[depth], f"Input: {noisy}\nOutput:" + CONF_SUFFIX


def parse_conf(text):
    m = re.search(r"[Cc]onfidence:\s*([0-9]{1,3})", text)
    return float(m.group(1)) if m else None


def _post(host, payload):
    req = urllib.request.Request(host + "/api/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["response"]


def ollama(system, prompt, model, temperature, host):
    # think=False disables chain-of-thought for reasoning models (e.g. qwen3.5),
    # which otherwise emit thousand-token traces. num_predict caps runaway output.
    # Non-reasoning models (llama/qwen2.5/gemma/phi) reject the `think` field with a
    # 400 -> retry without it.
    base = {
        "model": model, "system": system, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": 128},
    }
    try:
        return _post(host, {**base, "think": False})
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "ignore")
        except Exception:
            pass
        if "think" in detail.lower():
            return _post(host, base)  # model does not support thinking
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--depth", default="postedit")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    a = ap.parse_args()

    rows = [json.loads(line) for line in open(a.exposure) if line.strip()]
    with open(a.out, "w") as out:
        for i, e in enumerate(rows):
            system, prompt = build_prompt(a.depth, e["noisy_text"])
            try:
                raw = ollama(system, prompt, a.model, a.temperature, a.host)
            except Exception as ex:  # keep going; record the failure
                raw = f"__ERROR__ {ex}"
            rec = dict(e)
            rec.update({
                "model_id": a.model, "model_class": "local", "depth": a.depth,
                "temperature": a.temperature, "prompt_id": a.depth + "_v1",
                "output_text": raw.split("Confidence:")[0].strip(),
                "verbalized_conf": parse_conf(raw), "logprob": None,
            })
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"[{i + 1}/{len(rows)}] cer={e.get('cer_target')} "
                  f"-> {rec['output_text'][:60]!r}", flush=True)


if __name__ == "__main__":
    main()
