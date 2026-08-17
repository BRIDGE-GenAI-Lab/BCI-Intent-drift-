#!/bin/bash
#SBATCH -p short
#SBATCH -t 0-01:30
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -J idrift_create
#SBATCH -o /n/scratch/users/a/alg7274/idrift/create_newest.%j.log
set -uo pipefail
# Register the newest GGUFs as Ollama models (CPU import job; no GPU needed).
export OLLAMA_MODELS=/n/scratch/users/a/alg7274/ollama_models
export OLLAMA_HOST=127.0.0.1:11455
G=/n/scratch/users/a/alg7274/gguf
mkdir -p /n/scratch/users/a/alg7274/idrift

~/ollama/bin/ollama serve > /n/scratch/users/a/alg7274/idrift/create_serve.${SLURM_JOB_ID}.log 2>&1 &
OPID=$!
trap 'kill $OPID 2>/dev/null || true' EXIT
for i in $(seq 1 60); do curl -sf http://127.0.0.1:11455/api/tags >/dev/null 2>&1 && { echo "serve up ($i)"; break; }; sleep 2; done

# tag <TAB> gguf-relative-path
create() {
  local tag="$1" gguf="$2"
  if [ ! -f "$G/$gguf" ]; then echo "MISSING $G/$gguf"; return; fi
  printf 'FROM %s\n' "$G/$gguf" > /tmp/idrift_${SLURM_JOB_ID}.Modelfile
  echo "=== creating $tag from $gguf ==="
  ~/ollama/bin/ollama create "$tag" -f /tmp/idrift_${SLURM_JOB_ID}.Modelfile && echo "OK $tag" || echo "FAIL $tag"
}

create gemma4:e4b        gemma-4-E4B-it-Q8_0.gguf
create gemma4:12b        gemma-4-12b-it-Q8_0.gguf
create gemma4:31b        gemma-4-31B-it-Q8_0.gguf
create phi4:mini         microsoft_Phi-4-mini-instruct-Q8_0.gguf
create phi4:14b          phi-4-Q8_0.gguf
create qwen3.5:27b-q4    Qwen3.5-27B-Q4_K_M.gguf
create mistral-small:24b Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf

echo "== ollama list =="; ~/ollama/bin/ollama list
echo "CREATE_NEWEST_DONE $(date)"
