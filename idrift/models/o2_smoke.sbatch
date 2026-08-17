#!/bin/bash
#SBATCH -p gpu_sun
#SBATCH --account=sun_hs285_contrib
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J idrift_smoke
#SBATCH -o /n/scratch/users/a/alg7274/idrift/smoke_%j.out
set -e
# gpu_sun = 4x L40S @ 48G each; the 53G bf16 27B model needs 2 GPUs to fit in VRAM.
export OLLAMA_MODELS=/n/scratch/users/a/alg7274/ollama_models
export OLLAMA_HOST=127.0.0.1:11434
MODEL=qwen3.5:27b-bf16   # explicit text model (NOT the -vl vision variant)
mkdir -p /n/scratch/users/a/alg7274/idrift

echo "GPUs: $CUDA_VISIBLE_DEVICES"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "starting ollama serve..."
~/ollama/bin/ollama serve > /n/scratch/users/a/alg7274/idrift/serve_${SLURM_JOB_ID}.log 2>&1 &
SERVE_PID=$!
sleep 20
~/ollama/bin/ollama list
echo "warming up model $MODEL (forces load across both GPUs)..."
~/ollama/bin/ollama run "$MODEL" "hi" --keepalive 60m > /dev/null 2>&1 || echo "warmup returned nonzero (continuing)"

python3 ~/idrift/o2_infer.py \
  --exposure ~/idrift/exposure_smoke.jsonl \
  --out /n/scratch/users/a/alg7274/idrift/outputs_smoke.jsonl \
  --model "$MODEL" --depth postedit --temperature 0.0

kill $SERVE_PID 2>/dev/null || true
echo "DONE rows=$(wc -l < /n/scratch/users/a/alg7274/idrift/outputs_smoke.jsonl)"
