#!/bin/bash
#SBATCH -p short
#SBATCH -t 0-04:00
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -J idrift_create2
#SBATCH -o /n/scratch/users/a/alg7274/idrift/create_remaining.%j.log
set -uo pipefail
# Register the 5 newest GGUFs NOT yet in the store (e4b + 12b already done).
# FAST-FIRST ordering: small models first so a slow 31B import cannot starve them.
# The 31GB gemma4:31b goes LAST and gets whatever walltime remains.
export OLLAMA_MODELS=/n/scratch/users/a/alg7274/ollama_models
export OLLAMA_HOST=127.0.0.1:11456
G=/n/scratch/users/a/alg7274/gguf
mkdir -p /n/scratch/users/a/alg7274/idrift

~/ollama/bin/ollama serve > /n/scratch/users/a/alg7274/idrift/create2_serve.${SLURM_JOB_ID}.log 2>&1 &
OPID=$!
trap 'kill $OPID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do curl -sf http://127.0.0.1:11456/api/tags >/dev/null 2>&1 && { echo "serve up ($i)"; break; }; sleep 2; done

create() {
  local tag="$1" gguf="$2"
  # skip if already registered
  if ~/ollama/bin/ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$tag"; then
    echo "SKIP $tag (already present)"; return
  fi
  if [ ! -f "$G/$gguf" ]; then echo "MISSING $G/$gguf"; return; fi
  printf 'FROM %s\n' "$G/$gguf" > /tmp/idrift2_${SLURM_JOB_ID}.Modelfile
  echo "=== creating $tag from $gguf ($(date +%H:%M:%S)) ==="
  ~/ollama/bin/ollama create "$tag" -f /tmp/idrift2_${SLURM_JOB_ID}.Modelfile && echo "OK $tag $(date +%H:%M:%S)" || echo "FAIL $tag"
}

# fast-first
create phi4:mini         microsoft_Phi-4-mini-instruct-Q8_0.gguf
create mistral-small:24b Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf
create phi4:14b          phi-4-Q8_0.gguf
create qwen3.5:27b-q4    Qwen3.5-27B-Q4_K_M.gguf
# giant last
create gemma4:31b        gemma-4-31B-it-Q8_0.gguf

echo "== ollama list =="; ~/ollama/bin/ollama list
echo "CREATE_REMAINING_DONE $(date)"
