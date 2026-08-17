#!/usr/bin/env python3
"""O2 inference runner v2 (stdlib only; runs on the cluster).

Reads the replicate-aware exposure (exposure_v2 jsonl) and a frozen prompt bank, applies one
prompt id to each noisy stream, calls the local Ollama server, parses a JSON reconstruction
plus a verbalized confidence, and writes one output row per exposure row with full provenance.
Resume-safe: rows already present in --out (keyed by message_id|cer_target|replicate_idx) are
skipped. A run-level provenance record (Ollama version, model digest, host) is written once to
<out>.provenance.json.

Usage:
  python3 o2_runner_v2.py --exposure exposure_v2_full.jsonl --prompts prompts.json \
      --prompt P0 --model gemma4:31b --condition postedit --temperature 0.0 \
      --out outputs_v2_P0_gemma4_31b.jsonl --host http://127.0.0.1:11520
"""
import argparse
import json
import os
import re
import socket
import urllib.error
import urllib.request

_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def _post(host, payload, timeout=180):
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())["response"]


def ollama(system, prompt, model, temperature, host):
    """Generate; send think=false first (reasoning models), retry without it on a think error."""
    base = {
        "model": model, "system": system, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": 160},
    }
    try:
        return _post(host, {**base, "think": False})
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")
        if "think" in detail.lower():
            return _post(host, base)
        raise


def parse_reply(raw, abstain_sentinel):
    """Extract {message, confidence, abstained} from a model reply (JSON-first, regex fallback)."""
    message, confidence = None, None
    m = _JSON_OBJ.search(raw or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            message = obj.get("message")
            c = obj.get("confidence")
            confidence = float(c) if c is not None and str(c).strip() != "" else None
        except (ValueError, TypeError):
            pass
    if message is None:                       # fallback: whole reply is the message
        message = (raw or "").strip()
    abstained = isinstance(message, str) and message.strip() == abstain_sentinel
    return {"output_message": message, "confidence": confidence, "abstained": abstained}


def run_provenance(model, host, out_path):
    """Capture run-level provenance once (best-effort; never fatal)."""
    prov = {"model": model, "host": host, "hostname": socket.gethostname()}
    try:
        req = urllib.request.Request(
            host.rstrip("/") + "/api/show",
            data=json.dumps({"name": model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            info = json.loads(r.read().decode())
        det = info.get("details", {})
        prov.update({
            "quantization": det.get("quantization_level"),
            "parameter_size": det.get("parameter_size"),
            "family": det.get("family"),
            "template_sha_present": bool(info.get("template")),
        })
    except Exception as e:  # provenance is best-effort
        prov["show_error"] = str(e)
    prov["ollama_version"] = os.environ.get("OLLAMA_VERSION", "unknown")
    with open(out_path + ".provenance.json", "w") as fh:
        json.dump(prov, fh, indent=2)
    return prov


def key(row):
    return f"{row['message_id']}|{row['cer_target']}|{row['replicate_idx']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--prompt", default="P0")
    ap.add_argument("--model", required=True)
    ap.add_argument("--condition", default="postedit")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    args = ap.parse_args()

    bank = json.load(open(args.prompts))
    template = bank["prompts"][args.prompt]
    sentinel = bank.get("abstain_sentinel", "[ABSTAIN]")
    prompt_sha = bank.get("bank_sha256", "")
    system = "You reconstruct messages typed on a noisy assistive speller. Reply only with the requested JSON."

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                done.add(key(json.loads(line)))
            except Exception:
                pass
    prov = run_provenance(args.model, args.host, args.out)
    print(f"model={args.model} prompt={args.prompt} condition={args.condition} "
          f"quant={prov.get('quantization')} resume_skip={len(done)}", flush=True)

    n, err = 0, 0
    with open(args.out, "a") as out:
        for line in open(args.exposure):
            if not line.strip():
                continue
            row = json.loads(line)
            if key(row) in done:
                continue
            try:
                raw = ollama(system, template + row["noisy_text"], args.model, args.temperature, args.host)
                rec = parse_reply(raw, sentinel)
            except Exception as e:
                raw, rec, err = f"__ERROR__ {e}", {"output_message": None, "confidence": None, "abstained": False}, err + 1
            out.write(json.dumps({
                "model": args.model, "prompt_id": args.prompt, "prompt_sha256": prompt_sha,
                "condition": args.condition, "temperature": args.temperature,
                "message_id": row["message_id"], "corpus": row["corpus"],
                "cer_target": row["cer_target"], "replicate_idx": row["replicate_idx"],
                "realized_cer": row.get("realized_cer"), "intended_text": row["text"],
                "noisy_text": row["noisy_text"], "raw_output": raw, **rec,
            }) + "\n")
            n += 1
            if n % 500 == 0:
                out.flush()
                print(f"  {args.model}/{args.prompt}: {n} done ({err} errors)", flush=True)
    print(f"DONE {args.model}/{args.prompt} rows={n} errors={err}", flush=True)


if __name__ == "__main__":
    main()
