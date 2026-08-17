#!/bin/bash
#SBATCH -p gpu_sun
#SBATCH --account=sun_hs285_contrib
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J idrift_v2
#SBATCH --array=0-41
#SBATCH -o /n/scratch/users/a/alg7274/idrift/v2_%A_%a.out
set -uo pipefail
# Primary condition: 7 current-generation models x 6 exposure shards = 42 tasks, prompt P0.
# task id -> model = id/6, shard = id%6. Resume-safe runner; distinct port per task.
MODELS=(gemma4:e4b gemma4:12b gemma4:31b phi4:mini phi4:14b qwen3.5:27b-q4 mistral-small:24b)
MID=$((SLURM_ARRAY_TASK_ID / 6))
SHARD=$((SLURM_ARRAY_TASK_ID % 6))
MODEL=${MODELS[$MID]}
PORT=$((11600 + SLURM_ARRAY_TASK_ID))
SAFE=$(echo "$MODEL" | tr ':/' '__')
export OLLAMA_MODELS=/n/scratch/users/a/alg7274/ollama_models
export OLLAMA_HOST=127.0.0.1:$PORT
export OLLAMA_VERSION=$(~/ollama/bin/ollama --version 2>&1 | awk "/version/{print $NF; exit}")
OUT=/n/scratch/users/a/alg7274/idrift/outputs_v2_P0_${SAFE}_part${SHARD}.jsonl
echo "task=$SLURM_ARRAY_TASK_ID model=$MODEL shard=$SHARD port=$PORT host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"

~/ollama/bin/ollama serve > /n/scratch/users/a/alg7274/idrift/v2_serve_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log 2>&1 &
SPID=$!
for i in $(seq 1 30); do curl -sf http://127.0.0.1:$PORT/api/tags >/dev/null 2>&1 && break; sleep 2; done
~/ollama/bin/ollama run "$MODEL" "hi" --keepalive 120m > /dev/null 2>&1 || echo "warmup nonzero (continuing)"

python3 ~/idrift/o2_runner_v2.py \
  --exposure ~/idrift/exposure_v2.part${SHARD}.jsonl \
  --prompts ~/idrift/prompts.json --prompt P0 \
  --model "$MODEL" --condition postedit --temperature 0.0 \
  --out "$OUT" --host http://127.0.0.1:$PORT

kill $SPID 2>/dev/null || true
echo "DONE task=$SLURM_ARRAY_TASK_ID rows=$(wc -l < "$OUT" 2>/dev/null)"
