#!/bin/bash
#SBATCH -p gpu_sun
#SBATCH --account=sun_hs285_contrib
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -J idrift_v2_smoke
#SBATCH -o /n/scratch/users/a/alg7274/idrift/v2_smoke_%j.out
set -uo pipefail
export OLLAMA_MODELS=/n/scratch/users/a/alg7274/ollama_models
export OLLAMA_HOST=127.0.0.1:11599
export OLLAMA_VERSION=$(~/ollama/bin/ollama --version 2>/dev/null | head -1)
~/ollama/bin/ollama serve > /n/scratch/users/a/alg7274/idrift/v2_smoke_serve_${SLURM_JOB_ID}.log 2>&1 &
for i in $(seq 1 30); do curl -sf http://127.0.0.1:11599/api/tags >/dev/null 2>&1 && break; sleep 2; done
~/ollama/bin/ollama run gemma4:e4b "hi" --keepalive 30m >/dev/null 2>&1 || true
python3 ~/idrift/o2_runner_v2.py --exposure ~/idrift/exposure_v2_smoke.jsonl \
  --prompts ~/idrift/prompts.json --prompt P0 --model gemma4:e4b --condition postedit \
  --temperature 0.0 --out /n/scratch/users/a/alg7274/idrift/outputs_v2_SMOKE.jsonl \
  --host http://127.0.0.1:11599
echo "SMOKE_DONE rows=$(wc -l < /n/scratch/users/a/alg7274/idrift/outputs_v2_SMOKE.jsonl)"
